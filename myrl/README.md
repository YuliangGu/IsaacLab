# myrl: BYOL-Augmented PPO for Isaac Lab (RSL-RL)

This package adds a BYOL-style sequential representation learning head to PPO in Isaac Lab,
plus a runner that wires everything together.

- Algorithm: `myrl.ppo_BYOL.PPOWithBYOL`
- Policy: `myrl.modules.ActorCriticAug` (shared encoder with optional FiLM/concat context)
- Runner: `myrl.runner_BYOL.OnPolicyRunnerBYOL`

## Features

- Temporal BYOL over windowed observation (and optional action) sequences
- Optional encoder sharing between PPO and BYOL (toggleable)
- Context/belief injection into the policy (optional:critic) via FiLM or concat
- Lightweight diagnostics: BYOL mismatch, etc.
- BYOL-driven curriculum suggestion (EMA-based) in the runner (optional)

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

Use the training script under `scripts/reinforcement_learning/rsl_rl/train_byol.py`.

- To run (BYOL default on):

```
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train_byol.py --task  Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs=1000 --headless
```


### Key algorithm knobs (agent.algorithm)

- `enable_byol` (bool): turn BYOL loss on/off
- `share_byol_encoder` (bool): share obs encoder between PPO and BYOL
- `byol_lambda` (float): BYOL loss weight
- `byol_window` (int): sequence window length
- `byol_batch` (int): BYOL batch size per update
- `byol_tau_start` / `byol_tau_end` (float): EMA momentum range for target nets
- `byol_z_dim` (int): latent/context dim (must match policy.ctx_dim)
- Augmentations: `byol_noise_std`, `byol_time_warp_scale`, `byol_feat_drop`, `byol_frame_drop`, `byol_max_shift`

### Policy knobs (agent.policy)

- `class_name=ActorCriticAug`
- `ctx_mode`: `film` | `concat` | `none`
- `ctx_dim`: match `byol_z_dim`
- `feat_dim`: encoder output dim
- `use_prev_action`: include previous action via an action encoder
- `ctx_to_critic`: inject context to critic

## Diagnostics

The algorithm logs additional metrics when BYOL is enabled:
- `byol_mismatch`: small-sample BYOL target mismatch
- Runner adds `byol_ema` (EMA of mismatch) and `terrain_level_suggested` (if enabled)

## TODOs:
- 

## License
This package follows the Isaac Lab project licensing.

