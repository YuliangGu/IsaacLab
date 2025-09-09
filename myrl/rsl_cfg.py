"""Default config for running PPOWithBYOL via IsaacLab RSL-RL harness.

Use with train_byol.py by overriding the agent entry point to this config class.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPObyolRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Default config for running PPOWithBYOL via IsaacLab RSL-RL. """
    """ Override for BYOL augmentation """

    # extra LR multipliers
    lr_mult_encoder: float = 0.5  # LR multiplier for encoder params (shared by actor/critic)
    lr_mult_byol: float = 1.0      # LR multiplier for BYOL params

    @configclass
    class policy(RslRlPpoActorCriticCfg):
        # default ActorCritic hparams

        # BYOL extras
        use_prev_action: bool = False
        ctx_mode: str = "concat"  # 'film'|'concat'|'none'
        ctx_dim: int = 64          # Must match algorithm.byol_z_dim
        feat_dim: int = 256        # ObsEncoder output dim
        ctx_to_critic: bool = True  # Whether to also feed context to critic

    @configclass
    class algorithm(RslRlPpoAlgorithmCfg):
        # PPO hparams (keep defaults)

        # BYOL knobs (forwarded to PPOWithBYOL.__init__)
        enable_byol: bool = True
        share_byol_encoder: bool = True
        byol_lambda: float = 0.05
        byol_window: int = 16
        byol_batch: int = 512
        byol_tau_start: float = 0.99
        byol_tau_end: float = 0.999
        byol_z_dim: int = 64   # Must match policy.ctx_dim
        byol_proj_dim: int = 128
        byol_max_shift: int = 2
        byol_noise_std: float = 0.05
        byol_time_warp_scale: float = 0.0  # NOTE: this operation can be slow
        byol_feat_drop: float = 0.1
        byol_frame_drop: float = 0.1
        byol_use_actions: bool = True

        # Extra augmentations (all optional; default disabled)
        byol_ch_drop: float = 0.05
        byol_time_mask_prob: float = 0.1
        byol_time_mask_span: int = 2
        byol_gain_std: float = 0.05
        byol_bias_std: float = 0.02
        byol_smooth_prob: float = 0.25
        byol_smooth_kernel: int = 3
        byol_mix_strength: float = 0.0
