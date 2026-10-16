"""Conservative mixed-stair fine-tuning with the existing network contract."""

from isaaclab.utils import configclass

from .bpx_rough_history_ppo_cfg import BpxRoughHistoryPPORunnerCfg


@configclass
class BpxStairsHistoryPPORunnerCfg(BpxRoughHistoryPPORunnerCfg):
    experiment_name = "bpx_stairs_history"
    run_name = "up_down_flat_from_scratch"
    # 26 scalar tags: task outcome/level/speed, flat quality and PPO health.
    # Set agent.tensorboard_scalar_filter=null to restore full logging.
    tensorboard_scalar_filter: str | None = (
        r"^(Curriculum/terrain_levels/("
        r"(up|down|flat)/(success_rate|failure_rate|timeout_rate|actual_speed|command_speed)"
        r"|(up|down)/terrain_level|command_stage|terrain_sampling_phase"
        r"|flat/(stable_fraction|max_unstable_time))"
        r"|Train/(mean_reward|mean_episode_length)"
        r"|Policy/mean_noise_std|Loss/(value_function|learning_rate))$"
    )

    def __post_init__(self):
        # From-scratch training: use longer rollouts for the delayed stair
        # landing signal and adaptive learning-rate control.
        self.num_steps_per_env = 48
        self.max_iterations = 10000
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.schedule = "adaptive"
        self.algorithm.desired_kl = 0.01
        self.algorithm.num_learning_epochs = 5
        self.algorithm.num_mini_batches = 4
        self.algorithm.clip_param = 0.2
        self.algorithm.entropy_coef = 0.005
        self.algorithm.gamma = 0.99
        self.algorithm.lam = 0.95
        self.algorithm.max_grad_norm = 1.0
