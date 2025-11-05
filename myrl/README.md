# myrl: BYOL-Augmented PPO for Isaac Lab (RSL-RL)

This package adds a BYOL-style sequential representation learning head to PPO in Isaac Lab,
plus a runner that wires everything together.

- Algorithm: `myrl.ppo_BYOL.PPOWithBYOL`
- Policy: `myrl.modules.ActorCriticAug` (shared encoder with optional FiLM/concat context)
- Runner: `myrl.runner_BYOL.OnPolicyRunnerBYOL`

## Features

- Temporal BYOL over windowed observation sequences with optional action conditioning and GRU aggregators (`last`/`mean`/`attn`)
- Shared or dedicated encoders for policy vs. BYOL plus per-group LR multipliers for the optimizer
- Context/belief injection into both policy and critic via FiLM or concat, with previous-action features
- Rich augmentation suite: jitter, time-warp, feature/frame/channel drop, mix, smooth, and mask augmentations
- Runtime diagnostics (BYOL mismatch, EMA) and optional BYOL-driven curriculum suggestions in the runner

## Install

From the repository root:

```
pip install -e ./myrl
```

Requires:
- Python 3.8+
- PyTorch 2.0+
- `rsl-rl-lib` 3.0.1+

## Usage

### Train via Isaac Lab script

Use the training script under `myrl/algorithms/train_byol.py`.

- To run (BYOL default on):

```
isaaclab.bat -p myrl\algorithms\train_byol.py --task  Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs=1000 --headless
```


### Key algorithm knobs (agent.algorithm)

- `enable_byol` / `share_byol_encoder`: toggle BYOL and whether it reuses the policy encoder
- `byol_lambda`, `byol_window`, `byol_batch`, `byol_tau_start`/`byol_tau_end`, `byol_z_dim`, `byol_proj_dim`: core loss and architecture settings
- `byol_ctx_agg`: select GRU aggregation for the BYOL context (`last`, `mean`, `attn`)
- Augmentations: `byol_noise_std`, `byol_time_warp_scale`, `byol_feat_drop`, `byol_frame_drop`, `byol_max_shift`
- Extra augmentations: `byol_ch_drop`, `byol_time_mask_prob`/`byol_time_mask_span`, `byol_gain_std`, `byol_bias_std`, `byol_smooth_prob`/`byol_smooth_kernel`, `byol_mix_strength`
- Optimizer multipliers (forwarded via PPO kwargs): `lr_mult_encoder`, `lr_mult_byol`

### Policy knobs (agent.policy)

- `class_name=ActorCriticAug`
- `ctx_mode`: `film` | `concat` | `none`
- `ctx_dim`: match `byol_z_dim`
- `feat_dim`: encoder output dim
- `use_prev_action`: include previous action via an action encoder

## Diagnostics

The algorithm logs additional metrics when BYOL is enabled:
- `byol_mismatch`: small-sample BYOL target mismatch
- Runner adds `byol_ema` (EMA of mismatch) and `terrain_level_suggested` (if enabled)

## TODOs:
- 

## License
This package follows the Isaac Lab project licensing.
