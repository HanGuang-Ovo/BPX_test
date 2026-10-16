#!/usr/bin/env python3
"""Export a saved feedforward RSL-RL ActorCritic without launching Isaac Sim.

Run in the training Python environment (torch, rsl_rl, PyYAML, onnx).
The saved agent.yaml is required; normalized or recurrent policies are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from rsl_rl.modules import ActorCritic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = checkpoint.parent / "params" / "agent.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config.get("empirical_normalization", False):
        raise ValueError("Normalized policies require the training-side normalizer exporter")
    policy_config = dict(config["policy"])
    if policy_config.pop("class_name") != "ActorCritic":
        raise ValueError("Only feedforward ActorCritic checkpoints are supported")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = saved["model_state_dict"]
    actor_input = weights["actor.0.weight"].shape[1]
    critic_input = weights["critic.0.weight"].shape[1]
    actor_layers = [key for key in weights if key.startswith("actor.") and key.endswith(".weight")]
    last_layer = max(actor_layers, key=lambda key: int(key.split(".")[1]))
    action_dim = weights[last_layer].shape[0]
    policy = ActorCritic(actor_input, critic_input, action_dim, **policy_config)
    policy.load_state_dict(weights, strict=True)
    policy.eval()
    output = (args.output or checkpoint.parent / "exported" / checkpoint.stem).resolve()
    output.mkdir(parents=True, exist_ok=True)
    actor = policy.actor.cpu().eval()
    scripted = torch.jit.script(actor)
    scripted.save(str(output / "policy.pt"))
    torch.onnx.export(
        actor, torch.zeros(1, actor_input), str(output / "policy.onnx"),
        input_names=["obs"], output_names=["actions"], opset_version=17, dynamo=False,
    )
    generator = torch.Generator().manual_seed(42)
    with torch.inference_mode():
        observations = torch.randn(100, actor_input, generator=generator)
        error = float((scripted(observations) - policy.act_inference(observations)).abs().max())
    if error > 1e-6:
        raise RuntimeError(f"TorchScript export disagrees with checkpoint actor: {error}")
    metadata = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "iteration": saved["iter"], "actor_input": actor_input, "action_dimension": action_dim,
        "policy_config": policy_config, "empirical_normalization": False,
        "torchscript_actor_max_abs_error": error,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Exported iteration {saved['iter']}: {actor_input} -> {action_dim}, actor error={error:g}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
