"""Default config for running PPOWithBYOL via IsaacLab RSL-RL harness.

Use with train_byol.py by overriding the agent entry point to this config class.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPObyolRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 1
    device = "cuda:0"
    num_steps_per_env = 1024
    max_iterations = 3000000 // (1024 * 8)
    clip_actions = 1.0
    experiment_name = "ppo_byol_ctx"

    # map obs groups (same set for actor/critic)
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}

    @configclass
    class policy(RslRlPpoActorCriticCfg):
        # Must match symbols imported in OnPolicyRunnerBYOL (eval-based)
        class_name = "ActorCriticAug"
        # Gaussian policy with learnable log-std
        init_noise_std = 0.6
        noise_std_type = "log"
        actor_obs_normalization = False
        critic_obs_normalization = False
        # BYOL-aware augmentation
        use_prev_action = False
        ctx_mode = "film"
        ctx_dim = 64    # Must match algorithm.byol_z_dim
        feat_dim = 64

    @configclass
    class algorithm(RslRlPpoAlgorithmCfg):
        # Must match symbols imported in OnPolicyRunnerBYOL (eval-based)
        class_name = "PPOWithBYOL"
        # PPO hparams
        learning_rate = 0.0005
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4
        num_learning_epochs = 5
        clip_param = 0.2
        entropy_coef = 0.0
        value_loss_coef = 1.0
        desired_kl = 0.01
        max_grad_norm = 1.0
        normalize_advantage_per_mini_batch = True

        # BYOL knobs (forwarded to PPOWithBYOL.__init__)
        share_byol_encoder = True
        byol_lambda = 0.2
        byol_window = 32
        byol_batch = -1
        byol_tau_start = 0.99
        byol_tau_end = 0.999
        byol_z_dim = 64   # Must match policy.ctx_dim
        byol_proj_dim = 128
        byol_max_shift = 0
        byol_noise_std = 0.01
        byol_time_warp_scale = 0.01
        byol_feat_drop = 0.05
        byol_frame_drop = 0.05
        byol_use_actions = False

