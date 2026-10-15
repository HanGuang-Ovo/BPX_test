"""Stair PPO v4; Actor matches walking checkpoints, Critic uses privileged inputs."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BpxStairsBasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 48
    max_iterations = 4000
    save_interval = 50
    experiment_name = "bpx_stairs"
    run_name = "stairs_move"
    # Actor 输入保持兼容;旧 480 维 Critic 不能载入新 528 维价值网络。
    finetune_allow_critic_reset: bool = True
    # 默认从原行走基线重新微调,不默认继承尚未通过停驻验收的楼梯策略。
    # CLI --finetune_checkpoint 可覆盖;--resume 或 --from_scratch 不使用此默认值。
    finetune_checkpoint: str | None = (
        "logs/rsl_rl/bpx_rough_history_asymmetric/2026-09-23_11-17-22/model_4999.pt"
    )
    empirical_normalization = False
    clip_actions = None
    logger = "tensorboard"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic", init_noise_std=.08, noise_std_type="scalar",
        actor_hidden_dims=[128, 128, 128], critic_hidden_dims=[128, 128, 128], activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO", value_loss_coef=1., use_clipped_value_loss=True,
        clip_param=.2, entropy_coef=.001, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3.e-4, schedule="adaptive", gamma=.995, lam=.95,
        desired_kl=.01, max_grad_norm=1., normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None, rnd_cfg=None,
    )
