#!/usr/bin/env python
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Utility to launch a parameter sweep over ``myrl.algorithms.train_byol``."""

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

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


DEFAULT_GRID: Dict[str, List[Any]] = {
    "agent.algorithm.byol_lambda": [0.05, 0.1, 0.2],
    "agent.algorithm.byol_window": [16, 24],
    "agent.algorithm.byol_update_proportion": [0.25, 0.5],
    "agent.algorithm.byol_gaussian_jitter_std": [0.0, 0.05],
}


def _load_grid(path: Path | None) -> Dict[str, List[Any]]:
    if path is None:
        return DEFAULT_GRID
    if not path.exists():
        raise FileNotFoundError(f"Grid file '{path}' does not exist.")
    ext = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if ext in {".json", ".js"}:
            grid = json.load(f)
        elif ext in {".yml", ".yaml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to read YAML grid files.")
            grid = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported grid file extension '{ext}'. Use .json or .yaml.")
    if not isinstance(grid, dict):
        raise ValueError("Grid file must define a mapping of override -> list of values.")
    norm_grid: Dict[str, List[Any]] = {}
    for key, values in grid.items():
        if isinstance(values, (list, tuple)):
            vals = list(values)
        else:
            vals = [values]
        if not vals:
            raise ValueError(f"Grid entry '{key}' has no candidate values.")
        norm_grid[str(key)] = vals
    return norm_grid


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _slugify(key: str, value: Any) -> str:
    suffix = key.split(".")[-1]
    val = str(value).replace(".", "p").replace("/", "-")
    return f"{suffix}{val}"


def _product(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    for combo in product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def _build_command(
    train_script: Path,
    base_args: List[str],
    overrides: List[str],
) -> List[str]:
    cmd = [sys.executable, str(train_script)]
    cmd.extend(base_args)
    cmd.extend(overrides)
    return cmd


def _cmd_to_str(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch BYOL sweeps via train_byol.")
    parser.add_argument("--task", required=True, help="Isaac Lab task name (e.g. IsaacLabUnitreeGo2RoughEnv).")
    parser.add_argument("--agent-entry", type=str, default=None, help="Optional agent registry entry to pass through.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override --num_envs for train_byol.")
    parser.add_argument("--device", type=str, default=None, help="Forwarded to train_byol's --device.")
    parser.add_argument("--experiment-name", type=str, default="byol_sweep", help="Experiment name used by train_byol.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="Seeds to iterate over.")
    parser.add_argument(
        "--grid-file",
        type=Path,
        default=None,
        help="JSON/YAML file describing the sweep grid. Defaults cover common BYOL knobs.",
    )
    parser.add_argument(
        "--fixed-overrides",
        nargs="*",
        default=[],
        help="Hydra overrides applied to every run (e.g. agent.algorithm.byol_delay=3).",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum concurrent train_byol processes. Defaults to sequential execution.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without launching training.")
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("sweeps/byol"),
        help="Directory to store sweep manifests.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=None,
        help="Path to train_byol.py. Auto-resolves to myrl/algorithms/train_byol.py when omitted.",
    )
    parser.add_argument(
        "--train-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional CLI args forwarded verbatim to train_byol (must come last).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_script = (
        args.train_script
        if args.train_script is not None
        else Path(__file__).resolve().parents[1] / "algorithms" / "train_byol.py"
    )
    grid = _load_grid(args.grid_file)
    combos = list(_product(grid))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.sweep_root / timestamp
    sweep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = sweep_dir / "manifest.json"

    base_cli: List[str] = ["--task", args.task]
    if args.agent_entry:
        base_cli += ["--agent", args.agent_entry]
    if args.num_envs is not None:
        base_cli += ["--num_envs", str(args.num_envs)]
    if args.device is not None:
        base_cli += ["--device", args.device]
    if args.experiment_name:
        base_cli += ["--experiment_name", args.experiment_name]

    jobs = []
    job_id = 0
    for combo in combos:
        override_pairs = [f"{k}={_format_value(v)}" for k, v in combo.items()]
        for seed in args.seeds:
            job_id += 1
            run_name_bits = [_slugify(k, v) for k, v in combo.items()]
            run_name = f"sweep_{job_id:04d}_{'_'.join(run_name_bits)}_seed{seed}"
            overrides = override_pairs + list(args.fixed_overrides)
            overrides.append(f"agent.seed={seed}")
            run_cli = base_cli + ["--seed", str(seed), "--run_name", run_name]
            run_cli += list(args.train_args)
            cmd = _build_command(train_script, run_cli, overrides)
            jobs.append(
                {
                    "id": job_id,
                    "seed": seed,
                    "combo": combo,
                    "command": cmd,
                    "command_str": _cmd_to_str(cmd),
                    "run_name": run_name,
                }
            )

    results = []
    if args.dry_run:
        print("[DRY-RUN] Commands to execute:")
        for job in jobs:
            print(job["command_str"])
            job_result = job.copy()
            job_result["status"] = "dry-run"
            results.append(job_result)
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
                print(f"[SWEEP] Run {job['id']} finished with code {ret} after {duration:.1f}s ({status}).")
                job_result = {
                    "id": job["id"],
                    "seed": job["seed"],
                    "combo": job["combo"],
                    "command_str": job["command_str"],
                    "run_name": job["run_name"],
                    "returncode": ret,
                    "status": status,
                    "duration_sec": round(duration, 2),
                }
                results.append(job_result)
                active.remove(job)

    manifest = {
        "task": args.task,
        "agent_entry": args.agent_entry,
        "seeds": args.seeds,
        "grid": grid,
        "fixed_overrides": args.fixed_overrides,
        "train_script": str(train_script),
        "timestamp": timestamp,
        "runs": results,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[SWEEP] Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
