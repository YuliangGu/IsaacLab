"""
Smoke test for OnPolicyRunnerBYOL with a tiny dummy vector env.

Run:
  python -m myrl.test_runner_byol
"""

from __future__ import annotations

import os
import sys
from typing import Tuple, Dict, Any


# Ensure repository root is on sys.path so `import myrl` works when run directly
import os, sys
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)
    
import torch
from tensordict import TensorDict

from myrl.runner_BYOL import OnPolicyRunnerBYOL


class DummyVecEnv:
    """Minimal VecEnv-like environment compatible with OnPolicyRunnerBYOL.

    - Observations: dict-like (TensorDict) with keys 'policy' and 'critic'
    - Rewards/Dones: torch tensors shaped [N, 1]
    - step() returns empty extras dict
    """

    def __init__(self, num_envs: int, obs_dim: int, act_dim: int, max_episode_length: int, device: torch.device):
        self.num_envs = int(num_envs)
        self.num_actions = int(act_dim)
        self.max_episode_length = int(max_episode_length)
        self.device = device
        self._t = 0

    def _obs(self) -> TensorDict:
        obs = TensorDict(
            {
                "policy": torch.randn(self.num_envs, self._obs_dim, device=self.device),
                "critic": torch.randn(self.num_envs, self._obs_dim, device=self.device),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )
        return obs

    def get_observations(self) -> TensorDict:
        # lazily infer obs dim from first call if not set
        if not hasattr(self, "_obs_dim"):
            self._obs_dim = 32
        return self._obs()

    def step(self, actions: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        self._t += 1
        obs = self._obs()
        rewards = torch.randn(self.num_envs, 1, device=self.device)
        dones = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        if self._t >= self.max_episode_length:
            dones[:] = True
            self._t = 0
        extras: Dict[str, Any] = {}
        return obs, rewards, dones, extras


def main():
    device = torch.device("cpu")
    torch.manual_seed(0)

    # Tiny dummy env
    N, D, A, T = 4, 32, 6, 16
    env = DummyVecEnv(num_envs=N, obs_dim=D, act_dim=A, max_episode_length=T, device=device)

    # Training config for runner
    train_cfg = {
        "num_steps_per_env": T,
        "save_interval": 10_000,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "policy": {
            "class_name": "ActorCriticAug",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "ctx_mode": "film",
            "ctx_dim": 64,
            "feat_dim": 64,
            "use_prev_action": False,
        },
        "algorithm": {
            "class_name": "PPOWithBYOL",
            # BYOL settings (PPO hyperparams use defaults inside PPOWithBYOL)
            "share_byol_encoder": True,
            "byol_lambda": 0.0,            # disable BYOL loss for fast smoke test
            "byol_window": 8,
            "byol_batch": 64,
            "byol_tau_start": 0.99,
            "byol_tau_end": 0.999,
            "byol_z_dim": 64,
            "byol_proj_dim": 64,
            "byol_max_shift": 0,
            "byol_noise_std": 0.0,
            "byol_time_warp_scale": 0.0,
            "byol_feat_drop": 0.0,
            "byol_frame_drop": 0.0,
            "byol_use_actions": False,
        },
    }

    runner = OnPolicyRunnerBYOL(env, train_cfg, log_dir=None, device=device)
    runner.learn(num_learning_iterations=1)
    print("OK - OnPolicyRunnerBYOL finished one update.")


if __name__ == "__main__":
    # Ensure repo root is on sys.path for module imports when run directly
    _pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    main()

