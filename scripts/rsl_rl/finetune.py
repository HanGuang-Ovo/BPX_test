"""Weights-only task transfer; intentionally distinct from resuming training."""

import math
from pathlib import Path

import torch


def _adapt_actor_input(state, expected, adapter):
    """Explicit warm start for the old privileged, single-frame standing actor.

    Source: [v(3), omega(3), gravity(3), command(3), q(12), dq(12), action(12)].
    Target: each of the last six terms has ten frames, oldest -> newest.
    The unavailable linear velocity is fixed at zero. Every older-frame weight
    starts at zero, so the initial actor exactly matches the source at v=0.
    This does not preserve the old actor's response to nonzero linear velocity.
    """
    if adapter is None:
        return state, None
    if adapter != "stand48_to_blind450":
        raise ValueError(f"Unknown actor input adapter: {adapter!r}")
    source = state["actor.0.weight"]
    target = expected["actor.0.weight"]
    # Already-compatible Hold checkpoints remain valid explicit initializers.
    if source.shape == target.shape:
        return state, None
    if (source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]
            or source.shape[1] != 48 or target.shape[1] != 450):
        raise ValueError(
            f"stand48_to_blind450 requires Actor 48 -> 450 with matching hidden width; "
            f"received {tuple(source.shape)} -> {tuple(target.shape)}")
    latest = []
    offset = 0
    for width in (3, 3, 3, 12, 12, 12):
        latest.extend(range(offset + 9 * width, offset + 10 * width))
        offset += 10 * width
    weight = source.new_zeros(target.shape)
    weight[:, latest] = source[:, 3:]
    adapted = {**state, "actor.0.weight": weight}
    return adapted, {"name": adapter, "source_width": 48, "target_width": 450,
                     "source_linear_velocity": "fixed_zero", "history_order": "term_major_oldest_to_newest",
                     "latest_frame_columns": latest}


def load_finetune_weights(runner, checkpoint_path, *, allow_critic_reset=False, reset_critic=False,
                         actor_input_adapter=None):
    """Transfer compatible weights; optionally rebuild a changed privileged Critic.

    Optimizer moments, learning rate, iteration count and curriculum are never
    copied. Actor incompatibilities always fail before mutating either network.
    A changed Critic is reset in full only when explicitly permitted. Task splits
    can also force a fresh Critic when its dimensions happen to match.
    Input adaptation is opt-in; ordinary incompatible actors are still rejected.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Fine-tuning checkpoint does not exist: {path}")
    if runner.empirical_normalization:
        raise ValueError("Weights-only transfer currently requires empirical_normalization=False")
    saved = torch.load(path, map_location=runner.device, weights_only=True)
    state = saved["model_state_dict"]
    policy = runner.alg.policy
    expected = policy.state_dict()
    state, adapter_info = _adapt_actor_input(state, expected, actor_input_adapter)
    if adapter_info is not None and "obs_norm_state_dict" in saved:
        raise ValueError("stand48_to_blind450 requires an unnormalized source checkpoint")
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatched = {key: (tuple(state[key].shape), tuple(value.shape))
                  for key, value in expected.items() if key in state and state[key].shape != value.shape}
    critic_keys = {key for key in set(expected) | set(state) if key.startswith("critic.")}
    critic_changed = bool(set(missing) & critic_keys or set(unexpected) & critic_keys
                          or set(mismatched) & critic_keys)
    rejected = ((set(missing) | set(unexpected) | set(mismatched))
                - (critic_keys if allow_critic_reset or reset_critic else set()))
    if rejected:
        actor_dim = expected["actor.0.weight"].shape[1]
        critic_dim = expected["critic.0.weight"].shape[1]
        raise ValueError(f"Incompatible fine-tuning policy: missing={missing}, unexpected={unexpected}, "
                         f"checkpoint/target shapes={mismatched}. Target requires Actor {actor_dim} / "
                         f"Critic {critic_dim}; allow_critic_reset={allow_critic_reset}.")
    critic_reset = critic_changed or reset_critic
    transfer = {key: value for key, value in state.items()
                if not (critic_reset and key.startswith("critic."))}
    # Fresh Critic parameters remain exactly as initialized, including matching
    # hidden layers: mixing a new input layer with old value layers is avoided.
    policy.load_state_dict({**expected, **transfer}, strict=True)
    noise = runner.cfg["policy"]["init_noise_std"]
    with torch.no_grad():
        if hasattr(policy, "std"):
            policy.std.fill_(noise)
        elif hasattr(policy, "log_std"):
            policy.log_std.fill_(math.log(noise))
    runner.current_learning_iteration = 0
    return {"mode": "weights_only", "checkpoint": str(path), "source_iteration": saved.get("iter"),
            "optimizer_loaded": False, "initial_noise_std": noise,
            "critic_loaded": not critic_reset, "critic_reset": critic_reset,
            "actor_input_adapter": adapter_info}
