"""
Quick smoke test for PPOWithBYOL + ActorCriticAug without a real env.

- Builds a dummy observation container matching rsl-rl's expected shapes
  and runs a short rollout through storage, then updates once.
- Verifies that BYOL belief is wired and the auxiliary loss can be computed.

Run:
  python -m myrl.test_ppo_byol
"""

# Ensure repository root is on sys.path so `import myrl` works when run directly
import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

# from __future__ import annotations

import os
import torch

from myrl.modules import ActorCriticAug
from myrl.ppo_BYOL import PPOWithBYOL

def _make_obs(N: int, D: int, device: torch.device):
    obs = {
        "policy": torch.randn(N, D, device=device),
        "critic": torch.randn(N, D, device=device),
    }
    return obs


def main():
    device = torch.device("cpu")
    torch.manual_seed(0)

    # Dummy shapes
    N = 8          # num envs
    T = 500         # steps per env
    D = 48         # obs dim
    A = 6          # action dim

    # Build a template obs container (used by storage allocation)
    obs0 = _make_obs(N, D, device)

    # Policy with shared encoder + belief hooks
    policy = ActorCriticAug(
        obs=obs0,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=A,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        ctx_mode="film",
        ctx_dim=64,     # NOTE: must match BYOL z_dim
        use_prev_action=False,
        feat_dim=64,
    ).to(device)

    # PPO + BYOL (enable small BYOL loss)
    algo = PPOWithBYOL(
        policy=policy,
        share_byol_encoder=True,
        byol_lambda=0.1,
        byol_window=16,
        byol_batch=128,
        byol_tau_start=0.99,
        byol_tau_end=0.999,
        byol_z_dim=64,
        byol_proj_dim=128,
        byol_max_shift=0,
        byol_noise_std=0.01,
        byol_time_warp_scale=0.0,
        byol_feat_drop=0.05,
        byol_frame_drop=0.05,
    )

    # Initialize rollout storage (like runner does)
    algo.init_storage(
        training_type="rl",
        num_envs=N,
        num_transitions_per_env=T,
        obs=obs0,
        actions_shape=[A],
    )
    

    # Rollout loop (no real env): feed random transitions
    obs = obs0
    with torch.no_grad():
        for t in range(T):
            with torch.no_grad():
                actions = algo.act(obs)

            # next obs, rewards, dones
            next_obs = _make_obs(N, D, device)
            rewards = torch.randn(N, 1, device=device)
            dones = torch.zeros(N, 1, dtype=torch.bool, device=device)
            if t == T - 1:
                dones[:] = True
            extras = {}

            algo.process_env_step(next_obs, rewards, dones, extras)
            obs = next_obs

    algo.compute_returns(obs)
    logs = algo.update()
    print("PPOWithBYOL update logs:")
    for k, v in logs.items():
        print(f"{k}: {v}")
    # print("OK - PPOWithBYOL update completed.")
    # for k, v in logs.items():
    #     print(f"{k}: {v}")


if __name__ == "__main__":
    main()
