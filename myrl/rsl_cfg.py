"""Default config for running PPOWithBYOL via IsaacLab RSL-RL.

Use with train_byol.py by overriding the agent entry point to this config class.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPObyolRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Default config for running PPOWithBYOL via IsaacLab RSL-RL. """
    """ Override for BYOL augmentation """

    # extra LR multipliers
    lr_mult_encoder: float = 0.5  # LR multiplier for encoder params (shared by actor/critic): should be slower than base LR
    lr_mult_byol: float = 1.5      # LR multiplier for BYOL params: should be same or faster than base LR

    @configclass
    class policy(RslRlPpoActorCriticCfg):
        # default ActorCritic hparams

        # BYOL extras
        rpo_actor: bool = True
        rpo_alpha: float = 0.3
        ctx_mode: str = "concat"  # 'film'|'concat'|'none'
        ctx_dim: int = 128          # Must match algorithm.byol_z_dim
        feat_dim: int = 128        # ObsEncoder output dim

    @configclass
    class algorithm(RslRlPpoAlgorithmCfg):
        # PPO hparams (keep defaults)

        # BYOL knobs (forwarded to PPOWithBYOL.__init__)
        enable_byol: bool = True
        share_byol_encoder: bool = True
        byol_lambda: float = 0.1
        byol_window: int = 16
        byol_batch: int = -1
        byol_tau_start: float = 0.98
        byol_tau_end: float = 0.998
        byol_z_dim: int = 128   # Must match policy.ctx_dim
        byol_proj_dim: int = 128
        byol_update_proportion: float = 0.5

        # BYOL augmentations
        byol_delay: int = 2
        byol_gaussian_jitter_std: float = 0.02
        byol_causal_padding_proportion: float = 0.1
        byol_frame_drop: float = 0.1

        # Aggregator for BYOL context: 'last' | 'mean' | 'attn'
        byol_ctx_agg: str = "attn"

