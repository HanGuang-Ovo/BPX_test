"""Task transfer must preserve new PPO settings and reject incompatible inputs."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.modules import ActorCritic

spec = importlib.util.spec_from_file_location("bpx_finetune_test", Path(__file__).resolve().parents[1] / "rsl_rl/finetune.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

history_spec = importlib.util.spec_from_file_location(
    "bpx_finetune_history_test", Path(__file__).resolve().parents[2] / "bpx_runtime/history.py")
history_module = importlib.util.module_from_spec(history_spec)
history_spec.loader.exec_module(history_module)


def runner(actor_width=450, critic_width=480):
    policy = ActorCritic(actor_width, critic_width, 12, actor_hidden_dims=[128, 128, 128],
                         critic_hidden_dims=[128, 128, 128], activation="elu", init_noise_std=.35)
    return SimpleNamespace(alg=SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters(), lr=3.e-4)),
                           device="cpu", empirical_normalization=False, cfg={"policy": {"init_noise_std": .35}},
                           current_learning_iteration=0)


def test_compatible_weights_keep_fresh_optimizer_and_exploration(tmp_path):
    old, new = runner(), runner()
    old.alg.policy.std.data.fill_(.9)
    path = tmp_path / "old.pt"
    torch.save({"model_state_dict": old.alg.policy.state_dict(), "iter": 4999,
                "optimizer_state_dict": {"should_never_be_read": True}}, path)
    info = module.load_finetune_weights(new, path)
    assert info["source_iteration"] == 4999
    assert new.current_learning_iteration == 0 and not new.alg.optimizer.state
    assert new.alg.optimizer.param_groups[0]["lr"] == 3.e-4
    torch.testing.assert_close(new.alg.policy.actor[0].weight, old.alg.policy.actor[0].weight)
    torch.testing.assert_close(new.alg.policy.std, torch.full((12,), .35))
    assert info["critic_loaded"] and not info["critic_reset"]


def test_incompatible_actor_rejected_without_mutating_target(tmp_path):
    old, new = runner(48), runner()
    before = new.alg.policy.actor[0].weight.clone()
    path = tmp_path / "wrong.pt"
    torch.save({"model_state_dict": old.alg.policy.state_dict()}, path)
    with pytest.raises(ValueError, match="Actor 450"):
        module.load_finetune_weights(new, path)
    torch.testing.assert_close(before, new.alg.policy.actor[0].weight)


def test_changed_privileged_critic_requires_explicit_reset(tmp_path):
    old, new = runner(), runner(critic_width=528)
    before = {key: value.clone() for key, value in new.alg.policy.state_dict().items()}
    path = tmp_path / "old.pt"
    torch.save({"model_state_dict": old.alg.policy.state_dict()}, path)
    with pytest.raises(ValueError, match="Critic 528"):
        module.load_finetune_weights(new, path)
    for key, value in new.alg.policy.state_dict().items():
        torch.testing.assert_close(value, before[key])
    info = module.load_finetune_weights(new, path, allow_critic_reset=True)
    assert info["critic_reset"] and not info["critic_loaded"]
    for key, value in new.alg.policy.state_dict().items():
        if key.startswith("actor."):
            torch.testing.assert_close(value, old.alg.policy.state_dict()[key])
        elif key.startswith("critic."):
            torch.testing.assert_close(value, before[key])
    assert not new.alg.optimizer.state and new.current_learning_iteration == 0


def test_critic_reset_never_permits_an_incompatible_actor(tmp_path):
    old, new = runner(actor_width=480), runner(critic_width=528)
    before = {key: value.clone() for key, value in new.alg.policy.state_dict().items()}
    path = tmp_path / "wrong_actor.pt"
    torch.save({"model_state_dict": old.alg.policy.state_dict()}, path)
    with pytest.raises(ValueError, match="Actor 450"):
        module.load_finetune_weights(new, path, allow_critic_reset=True)
    for key, value in new.alg.policy.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_task_split_forces_fresh_critic_even_when_shapes_match(tmp_path):
    old, new = runner(critic_width=528), runner(critic_width=528)
    before = {key: value.clone() for key, value in new.alg.policy.state_dict().items()}
    path = tmp_path / "same_shape.pt"
    torch.save({"model_state_dict": old.alg.policy.state_dict()}, path)
    info = module.load_finetune_weights(new, path, reset_critic=True)
    assert info["critic_reset"] and not info["critic_loaded"]
    for key, value in new.alg.policy.state_dict().items():
        if key.startswith("critic."):
            torch.testing.assert_close(value, before[key])
        elif key.startswith("actor."):
            torch.testing.assert_close(value, old.alg.policy.state_dict()[key])


def test_stand48_adapter_matches_source_at_zero_velocity_with_arbitrary_older_history(tmp_path):
    old, new = runner(actor_width=48, critic_width=48), runner(critic_width=528)
    before = {key: value.clone() for key, value in new.alg.policy.state_dict().items()}
    path = tmp_path / "stand48.pt"
    torch.save({"model_state_dict": old.alg.policy.state_dict(), "iter": 1998}, path)
    info = module.load_finetune_weights(new, path, reset_critic=True,
                                        actor_input_adapter="stand48_to_blind450")
    assert info["source_iteration"] == 1998 and info["critic_reset"]
    assert info["actor_input_adapter"]["source_linear_velocity"] == "fixed_zero"
    # Use the deployment history writer, rather than duplicating the adapter's
    # feature offsets. Changing older frames must not affect initial actions.
    history = history_module.ObservationHistory(10, include_base_lin_vel=False)
    for _ in range(10):
        latest = torch.randn(45)
        packed = history.append(latest.numpy())
    with torch.no_grad():
        expected = old.alg.policy.act_inference(torch.cat((torch.zeros(3), latest)).unsqueeze(0))
        actual = new.alg.policy.act_inference(torch.from_numpy(packed).unsqueeze(0))
    torch.testing.assert_close(actual, expected, atol=1.e-6, rtol=1.e-5)
    for key, value in new.alg.policy.state_dict().items():
        if key.startswith("critic."):
            torch.testing.assert_close(value, before[key])
        elif key.startswith("actor.") and key != "actor.0.weight":
            torch.testing.assert_close(value, old.alg.policy.state_dict()[key])
    assert not new.alg.optimizer.state and new.current_learning_iteration == 0


@pytest.mark.parametrize("source_width,target_width", [(45, 450), (48, 480), (480, 450)])
def test_stand_adapter_rejects_other_layouts_before_mutating_weights(tmp_path, source_width, target_width):
    old, new = runner(actor_width=source_width), runner(actor_width=target_width, critic_width=528)
    before = {key: value.clone() for key, value in new.alg.policy.state_dict().items()}
    path = tmp_path / "wrong_layout.pt"
    torch.save({"model_state_dict": old.alg.policy.state_dict()}, path)
    with pytest.raises(ValueError, match="Actor 48 -> 450"):
        module.load_finetune_weights(new, path, reset_critic=True,
                                     actor_input_adapter="stand48_to_blind450")
    for key, value in new.alg.policy.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_stand_adapter_keeps_native_history_checkpoint_compatible(tmp_path):
    old, new = runner(critic_width=528), runner(critic_width=528)
    path = tmp_path / "hold450.pt"
    torch.save({"model_state_dict": old.alg.policy.state_dict()}, path)
    info = module.load_finetune_weights(new, path, reset_critic=True,
                                        actor_input_adapter="stand48_to_blind450")
    assert info["actor_input_adapter"] is None and info["critic_reset"]
    for key, value in new.alg.policy.actor.state_dict().items():
        torch.testing.assert_close(value, old.alg.policy.actor.state_dict()[key])


def test_stand_adapter_rejects_normalized_source_without_mutating_target(tmp_path):
    old, new = runner(actor_width=48), runner(critic_width=528)
    path = tmp_path / "normalized.pt"
    torch.save({"model_state_dict": old.alg.policy.state_dict(), "obs_norm_state_dict": {}}, path)
    before = {key: value.clone() for key, value in new.alg.policy.state_dict().items()}
    with pytest.raises(ValueError, match="unnormalized"):
        module.load_finetune_weights(new, path, reset_critic=True,
                                     actor_input_adapter="stand48_to_blind450")
    for key, value in new.alg.policy.state_dict().items():
        torch.testing.assert_close(value, before[key])
