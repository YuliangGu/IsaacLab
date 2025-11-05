"""Default config for running PPOWithBYOL via IsaacLab RSL-RL.

Use with ``myrl.algorithms.train_byol`` by overriding the agent entry point to this config class.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPObyolRunnerCfg(RslRlOnPolicyRunnerCfg):
    """ overlay config for PPO with BYOL. NOTE: overrides needs to be done in train_byol.py """

    # extra LR multipliers
    lr_mult_encoder: float = 1.0  # LR multiplier for encoder params 
    lr_mult_byol: float = 1.5      # LR multiplier for BYOL params

    @configclass
    class policy(RslRlPpoActorCriticCfg):
        # default ActorCritic hparams

        # BYOL extras
        rpo_actor: bool = False
        rpo_alpha: float = 0.3
        ctx_mode: str = "film"  # 'film'|'concat'|'none'
        ctx_dim: int = 64          # Must match algorithm.byol_z_dim
        feat_dim: int = 256        # ObsEncoder output dim

    @configclass
    class algorithm(RslRlPpoAlgorithmCfg):

        # BYOL knobs (forwarded to PPOWithBYOL.__init__)
        enable_byol: bool = True
        share_byol_encoder: bool = True
        byol_lambda: float = 0.1
        byol_window: int = 16
        byol_batch: int = -1
        use_transformer: bool = False
        byol_tau_start: float = 0.98
        byol_tau_end: float = 0.998
        byol_z_dim: int = 64                        # Must match policy.ctx_dim
        byol_proj_dim: int = 128
        byol_update_proportion: float = 0.5
        byol_intrinsic_coef: float = 0.05

        # BYOL augmentations
        byol_delay: int = 2
        byol_gaussian_jitter_std: float = 0.05
        byol_frame_drop: float = 0.1

        # Aggregator for BYOL context: 'last' | 'mean' | 'attn'
        byol_ctx_agg: str = "mean"
