# myrl: BYOL-Augmented PPO for Isaac Lab (RSL-RL)

This package adds a BYOL-style sequential representation learning head to PPO in Isaac Lab,
plus a runner that wires everything together.

- Algorithm: `myrl.ppo_BYOL.PPOWithBYOL`
- Policy: `myrl.modules.ActorCriticAug` (shared encoder with optional FiLM/concat context)
- Runner: `myrl.runner_BYOL.OnPolicyRunnerBYOL`

## Features

- Temporal BYOL over windowed observation (and optional action) sequences
- Optional encoder sharing between PPO and BYOL (toggleable)
- Context/belief injection into the policy via FiLM or concat
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

- Enable BYOL (default on):

```
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train_byol.py \
  --task=Isaac-Ant-v0 --num_envs=50 --headless
```

- Explicitly use our BYOL agent config alias:

```
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train_byol.py \
  --agent=myrl_byol --task=Isaac-Ant-v0 --num_envs=50 --headless
```

- Disable BYOL (run pure PPO through the same pipeline):

```
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train_byol.py \
  --no-byol --task=Isaac-Ant-v0 --num_envs=50 --headless
```

- Print effective BYOL/policy knobs:

```
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train_byol.py \
  --byol_debug --task=Isaac-Ant-v0 --num_envs=50 --headless
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

## Diagnostics

The algorithm logs additional metrics when BYOL is enabled:
- `ctx_norm_mean`, `ctx_norm_std`: context L2 norm stats
- `ctx_cos_prev`: temporal cosine similarity of successive contexts
- `byol_mismatch`: small-sample BYOL target mismatch
- Runner adds `byol_ema` (EMA of mismatch) and `terrain_level_suggested` (if enabled)

## Notes

- For PPO baselines: disable BYOL and consider using `ActorCritic` instead of `ActorCriticAug`.
- With shared encoders, keep `byol_lambda` modest and consider detaching BYOL gradients if you observe interference.
- For curriculum integration, see `OnPolicyRunnerBYOL.log` where BYOL EMA is mapped to a suggested terrain level.

## License

This package follows the Isaac Lab project licensing.

