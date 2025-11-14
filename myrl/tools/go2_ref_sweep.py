#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Focused hyper-parameter sweep for the 11-13 15:57 baseline on Go2 rough terrain."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List


# Narrow grid centred around the known-good 2025-11-13_15-57 configuration.
SWEEP_GRID: Dict[str, List[Any]] = {
    # Loss scaling.
    "agent.algorithm.byol_lambda": [0.15, 0.20, 0.25],
    # Amount of encoder reuse for the online predictor.
    "agent.algorithm.byol_update_proportion": [0.35, 0.45],
    # EMA target stickiness.
    "agent.algorithm.byol_tau_end": [0.995, 0.999],
    # Temporal context length.
    "agent.algorithm.byol_window": [16, 24],
    # Context injection mode for the policy encoder.
    "agent.policy.ctx_mode": ["film", "concat"],
}

# Baseline-overrides we always want active for these sweeps.
BASE_OVERRIDES = [
    # Keep encoder shared with PPO as in the 11-13 run.
    "agent.algorithm.share_byol_encoder=false",
    # Stick to the original stochastic augmentations.
    "agent.algorithm.byol_gaussian_jitter_std=0.05",
    "agent.algorithm.byol_frame_drop=0.1",
]


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _slugify(key: str, value: Any) -> str:
    suffix = key.split(".")[-1]
    if isinstance(value, bool):
        val = "t" if value else "f"
    else:
        val = str(value).replace(".", "p").replace("-", "m")
    return f"{suffix}{val}"


def _iter_combos(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    for combo in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def _build_command(
    train_script: Path,
    base_cli: List[str],
    overrides: List[str] | None = None,
) -> List[str]:
    cmd = [sys.executable, str(train_script)]
    cmd.extend(base_cli)
    if overrides:
        cmd.extend(overrides)
    return cmd


def _cmd_to_str(cmd: List[str]) -> str:
    return " ".join(shlex.quote(token) for token in cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the curated Go2 rough BYOL sweep used during hyper-parameter search."
    )
    parser.add_argument("--task", default="Isaac-Velocity-Rough-Unitree-Go2-v0")
    parser.add_argument(
        "--agent-entry",
        default="rsl_rl_cfg_entry_point",
        help="Gym registry key for the agent config (e.g. 'rsl_rl_cfg_entry_point').",
    )
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--experiment-name", default="go2_ref_sweep")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-root", type=Path, default=Path("sweeps/go2_refined"))
    parser.add_argument("--train-script", type=Path, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument(
        "--extra-override",
        dest="extra_overrides",
        action="append",
        default=[],
        help="Additional agent overrides (key=value) applied to every job.",
    )
    parser.add_argument(
        "--train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra CLI options forwarded verbatim to train_byol.py (must appear last).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_script = (
        args.train_script
        if args.train_script is not None
        else Path(__file__).resolve().parents[1] / "algorithms" / "train_byol.py"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.log_root / timestamp
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "manifest.json"

    base_cli: List[str] = [
        "--task",
        args.task,
        "--agent",
        args.agent_entry,
        "--experiment_name",
        args.experiment_name,
    ]
    if args.num_envs is not None:
        base_cli += ["--num_envs", str(args.num_envs)]
    if args.device:
        base_cli += ["--device", args.device]
    if args.max_iterations is not None:
        base_cli += ["--max_iterations", str(args.max_iterations)]
    if args.headless:
        base_cli.append("--headless")
    base_cli += list(args.train_args)

    jobs = []
    job_id = 0
    grid = SWEEP_GRID
    combos = list(_iter_combos(grid))
    for combo in combos:
        shared_override_values: List[str] = [f"{k}={_format_value(v)}" for k, v in combo.items()]
        shared_override_values.extend(BASE_OVERRIDES)
        shared_override_values.extend(args.extra_overrides)
        run_name_bits = [_slugify(k, v) for k, v in combo.items()]
        for seed in args.seeds:
            job_id += 1
            run_name = f"go2ref_{job_id:04d}_{'_'.join(run_name_bits)}_s{seed}"
            job_overrides = list(shared_override_values)
            job_overrides.append(f"agent.seed={seed}")
            cli_overrides: List[str] = []
            for item in job_overrides:
                cli_overrides.extend(["--agent-override", item])
            run_cli = base_cli + ["--seed", str(seed), "--run_name", run_name]
            run_cli.extend(cli_overrides)
            cmd = _build_command(train_script, run_cli)
            jobs.append(
                {
                    "id": job_id,
                    "seed": seed,
                    "combo": combo,
                    "agent_overrides": job_overrides,
                    "command": cmd,
                    "command_str": _cmd_to_str(cmd),
                    "run_name": run_name,
                }
            )

    results = []
    if args.dry_run:
        print("[DRY-RUN] Planned commands:")
        for job in jobs:
            print(job["command_str"])
            entry = job.copy()
            entry["status"] = "dry-run"
            results.append(entry)
    else:
        env = os.environ.copy()
        queue = deque(jobs)
        active: List[Dict[str, Any]] = []
        while queue or active:
            while queue and len(active) < max(1, args.max_parallel):
                job = queue.popleft()
                print(f"[SWEEP] Launching run {job['id']} -> {job['command_str']}")
                proc = subprocess.Popen(job["command"], env=env)
                job["process"] = proc
                job["start_time"] = time.time()
                active.append(job)
            time.sleep(1.0)
            for job in list(active):
                proc = job["process"]
                ret = proc.poll()
                if ret is None:
                    continue
                duration = time.time() - job["start_time"]
                status = "completed" if ret == 0 else "failed"
                print(f"[SWEEP] Run {job['id']} exited with code {ret} after {duration:.1f}s ({status}).")
                entry = {
                    "id": job["id"],
                    "seed": job["seed"],
                    "combo": job["combo"],
                    "command_str": job["command_str"],
                    "run_name": job["run_name"],
                    "returncode": ret,
                    "status": status,
                    "duration_sec": round(duration, 2),
                }
                results.append(entry)
                active.remove(job)

    manifest = {
        "timestamp": timestamp,
        "task": args.task,
        "agent_entry": args.agent_entry,
        "grid": grid,
        "base_overrides": BASE_OVERRIDES,
        "extra_overrides": args.extra_overrides,
        "seeds": args.seeds,
        "runs": results,
        "train_script": str(train_script),
        "base_cli": base_cli,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SWEEP] Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
