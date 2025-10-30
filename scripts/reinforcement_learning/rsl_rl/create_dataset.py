"""Collect rollouts from a trained RSL-RL PPO policy and save them to disk.

Typical usage after training completes::

    # activate Isaac Lab's python first (e.g. `./isaaclab.bat -p`)
    python scripts/reinforcement_learning/rsl_rl/create_dataset.py \
        --task Isaac-Velocity-Rough-Unitree-Go2-v0 \
        --checkpoint logs/rsl_rl/unitree_go2_rough/2025-10-30_14-17-10/model_150.pt \
        --output datasets/go2_rough_rollouts.pt \
        --num-steps 20000

The saved ``.pt`` file stores tensors shaped ``[T, N, ...]`` (time, envs, features),
ready to be consumed by ``myrl.utils_core.sample_byol_windows`` for BYOL augmentation.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from isaaclab.app import AppLauncher
import cli_args 

from tensordict import TensorDictBase

parser = argparse.ArgumentParser(description="Collect rollouts from a trained RSL-RL agent.")
parser.add_argument("--task", type=str, required=True, help="Gym registry id for the task to replay.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="RL agent configuration entry point (Hydra).",
)
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="File path where the rollout dataset (.pt) will be written.",
)
parser.add_argument(
    "--num-steps",
    type=int,
    default=5000,
    help="Number of environment steps to record (per environment frame).",
)
parser.add_argument(
    "--num-episodes",
    type=int,
    default=None,
    help="Optional cap on the number of finished episodes to record.",
)
parser.add_argument(
    "--num-envs",
    type=int,
    default=None,
    help="Override number of environments when collecting rollouts.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric (USD I/O) for compatibility with older scenes.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed to replicate environment randomisation.",
)
parser.add_argument(
    "--quiet",
    action="store_true",
    help="Suppress per-iteration logging during collection.",
)
parser.add_argument(
    "--flush-interval",
    type=int,
    default=0,
    help="Future use: reserved for streamed writers (currently unused).",
)

# append RSL-RL cli arguments and AppLauncher arguments
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# propagate additional CLI overrides to Hydra
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  

from isaaclab.envs import (  
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  
from isaaclab.utils.dict import print_dict  

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  

import isaaclab_tasks  
from isaaclab_tasks.utils import get_checkpoint_path  
from isaaclab_tasks.utils.hydra import hydra_task_config  

from rsl_rl.runners import OnPolicyRunner  


def _resolve_checkpoint(agent_cfg: RslRlBaseRunnerCfg) -> str:
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _ensure_parent_dir(path: Path) -> None:
    """Create parent directories for the target path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _stack(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Stack a non-empty list of tensors along a new first dimension."""
    if not tensors:
        raise RuntimeError("Attempted to stack an empty tensor list.")
    return torch.stack(tensors, dim=0)


def _flatten_obs(obs: Any) -> torch.Tensor:
    """Extract policy observations as a dense tensor.

    Accepts raw tensors, TensorDicts, or nested dictionaries. Preference order:
    1. Direct tensors (returned unchanged).
    2. Objects containing a ``policy`` entry (recursively flattened).
    3. Concatenation of any tensor leaves discovered in the container.
    """
    if torch.is_tensor(obs):
        return obs

    if isinstance(obs, TensorDictBase):
        # convert to standard dictionary for recursive handling
        return _flatten_obs(obs.to_dict())

    if isinstance(obs, dict):
        if "policy" in obs:
            return _flatten_obs(obs["policy"])

        leaves = list(_iter_tensor_leaves(obs.values()))
        if not leaves:
            raise TypeError("Observation container does not contain any tensor leaves.")

        # ensure 2-D [num_envs, feat] tensors before concatenation
        formatted = [
            leaf if leaf.dim() > 1 else leaf.unsqueeze(-1)
            for leaf in leaves
        ]
        if len(formatted) == 1:
            return formatted[0]
        return torch.cat(formatted, dim=-1)

    raise TypeError(f"Unsupported observation type: {type(obs).__name__}")


def _iter_tensor_leaves(values: Iterable[Any]) -> Iterable[torch.Tensor]:
    """Yield tensor leaves from a possibly nested container."""
    for value in values:
        if torch.is_tensor(value):
            yield value
        elif isinstance(value, TensorDictBase):
            yield from _iter_tensor_leaves(value.to_dict().values())
        elif isinstance(value, dict):
            yield from _iter_tensor_leaves(value.values())


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # apply CLI overrides to the Hydra config objects
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed

    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    resume_path = _resolve_checkpoint(agent_cfg)
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    if not args_cli.quiet:
        print(f"[INFO] Loading checkpoint: {resume_path}")

    # build environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # construct runner (no log dir needed for inference)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # policy callable that already handles context/normalization
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # initial observations
    obs = env.get_observations().to(agent_cfg.device)

    obs_tensor = _flatten_obs(obs)
    num_envs = int(obs_tensor.shape[0])
    obs_dim = int(obs_tensor.shape[-1])

    target_steps = max(1, int(args_cli.num_steps))
    episode_cap = args_cli.num_episodes if args_cli.num_episodes is not None else float("inf")

    # buffers: each entry is [num_envs, ...]
    obs_buf: List[torch.Tensor] = []
    next_obs_buf: List[torch.Tensor] = []
    act_buf: List[torch.Tensor] = []
    rew_buf: List[torch.Tensor] = []
    done_buf: List[torch.Tensor] = []
    episode_id_buf: List[torch.Tensor] = []
    step_in_episode_buf: List[torch.Tensor] = []
    episode_metrics: List[Dict[str, Any]] = []

    # track episode identity per environment
    episode_ids = torch.arange(num_envs, dtype=torch.long)
    next_episode_id = int(episode_ids.max().item()) + 1
    step_in_episode = torch.zeros(num_envs, dtype=torch.long)
    finished_episodes = 0
    collected_steps = 0

    if not args_cli.quiet:
        print_dict(
            {
                "num_envs": num_envs,
                "obs_dim": obs_dim,
                "target_steps": target_steps,
                "episode_cap": episode_cap if episode_cap != float("inf") else "unbounded",
            },
            nesting=3,
        )

    # rollout loop
    while collected_steps < target_steps and finished_episodes < episode_cap:
        obs_tensor = _flatten_obs(obs)

        with torch.inference_mode():
            actions = policy(obs)
        if not torch.is_tensor(actions):
            raise TypeError("Inference policy returned non-tensor actions.")

        # step environment
        next_obs, rewards, dones, extras = env.step(actions)
        next_obs = next_obs.to(agent_cfg.device)

        next_obs_tensor = _flatten_obs(next_obs)
        reward_tensor = rewards.detach()
        done_tensor = dones.to(dtype=torch.bool)

        # store CPU copies for serialization
        obs_buf.append(obs_tensor.detach().cpu().clone())
        next_obs_buf.append(next_obs_tensor.detach().cpu().clone())
        act_buf.append(actions.detach().cpu())
        rew_buf.append(reward_tensor.cpu())
        done_buf.append(done_tensor.cpu())
        episode_id_buf.append(episode_ids.clone())
        step_in_episode_buf.append(step_in_episode.clone())

        collected_steps += num_envs

        # gather episode stats (if provided)
        if isinstance(extras, dict):
            ep_info = extras.get("episode") or extras.get("log")
            if ep_info:
                if isinstance(ep_info, TensorDictBase):
                    ep_info = ep_info.to_dict()
                done_indices = torch.nonzero(done_tensor, as_tuple=False).squeeze(-1)
                for idx in done_indices.tolist():
                    metrics: Dict[str, Any] = {}
                    for k, v in ep_info.items():
                        if torch.is_tensor(v):
                            if v.numel() == 0:
                                metrics[k] = None
                                continue
                            if v.dim() > 0 and idx >= v.shape[0]:
                                metrics[k] = v.detach().cpu().tolist()
                                continue
                            try:
                                item = v[idx]
                            except IndexError:
                                metrics[k] = v.detach().cpu().tolist()
                                continue
                            metrics[k] = item.item() if item.numel() == 1 else item.detach().cpu().tolist()
                        elif isinstance(v, (list, tuple)):
                            if len(v) == 0:
                                metrics[k] = None
                            elif idx < len(v):
                                metrics[k] = v[idx]
                            else:
                                metrics[k] = list(v)
                        else:
                            metrics[k] = v

                    entry = {
                        "episode_id": int(episode_ids[idx].item()),
                        "env_index": idx,
                        "metrics": metrics,
                    }
                    episode_metrics.append(entry)

        # update episode bookkeeping
        done_indices = torch.nonzero(done_tensor, as_tuple=False).squeeze(-1)
        if done_indices.numel() > 0:
            finished_episodes += int(done_indices.numel())
            for idx in done_indices.tolist():
                episode_ids[idx] = next_episode_id
                next_episode_id += 1
                step_in_episode[idx] = 0
        step_in_episode += 1

        obs = next_obs

    env.close()

    if not obs_buf:
        raise RuntimeError("No rollout frames were collected. Check num-steps/episodes settings.")

    data = {
        "observations": _stack(obs_buf),
        "next_observations": _stack(next_obs_buf),
        "actions": _stack(act_buf),
        "rewards": _stack(rew_buf),
        "dones": _stack(done_buf),
        "episode_ids": _stack(episode_id_buf),
        "step_in_episode": _stack(step_in_episode_buf),
        "meta": {
            "checkpoint": resume_path,
            "task": args_cli.task,
            "agent_cfg": agent_cfg.to_dict(),
            "env_cfg": env_cfg.to_dict(),
            "num_envs": num_envs,
            "collected_env_steps": collected_steps,
            "finished_episodes": finished_episodes,
            "created_utc": datetime.utcnow().isoformat(),
        },
        "episode_metrics": episode_metrics,
    }

    output_path = Path(args_cli.output)
    _ensure_parent_dir(output_path)
    torch.save(data, output_path)

    if not args_cli.quiet:
        print(
            f"[INFO] Saved rollout dataset ({collected_steps} env-steps, "
            f"{finished_episodes} episodes) to: {output_path}"
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
