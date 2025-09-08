"""Default config for running PPOWithBYOL via IsaacLab RSL-RL harness.

Use with train_byol.py by overriding the agent entry point to this config class.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPObyolRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32 # default is 24
    max_iterations = 1500 # this is default 

    """ Override for BYOL augmentation """
    @configclass
    class policy(RslRlPpoActorCriticCfg):
        # default parameters
        class_name = "ActorCriticAug"
        init_noise_std = 1.0
        noise_std_type = "scalar"
        actor_obs_normalization = False
        critic_obs_normalization = False

        # NEW
        use_prev_action = False
        ctx_mode = "film" # choose from 'film', 'concat', 'none'
        ctx_dim = 64    # Must match algorithm.byol_z_dim
        feat_dim = 128  # ObsEncoder output dim. None => observation space dim

    @configclass
    class algorithm(RslRlPpoAlgorithmCfg):
        class_name = "PPOWithBYOL"

        # PPO hparams (keep defaults)

        # BYOL knobs (forwarded to PPOWithBYOL.__init__)
        enable_byol = True
        share_byol_encoder = True
        byol_lambda = 0.2
        byol_window = 16
        byol_batch = 512                # -1 -> auto: 2*minibatch_size capped at 512
        byol_tau_start = 0.99
        byol_tau_end = 0.999
        byol_z_dim = 64   # Must match policy.ctx_dim
        byol_proj_dim = 128 
        byol_max_shift = 0
        byol_noise_std = 0.02
        byol_time_warp_scale = 0.01
        byol_feat_drop = 0.1
        byol_frame_drop = 0.1
        byol_use_actions = True

