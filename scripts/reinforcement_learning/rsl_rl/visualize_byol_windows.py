#!/usr/bin/env python3
# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize BYOL window embeddings with simple dimensionality reduction."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import torch

# ensure repo root is on sys.path when running directly
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from myrl.utils_core import sample_byol_windows  # noqa: E402


def _flatten_windows(window: torch.Tensor) -> torch.Tensor:
    """Flatten [B, W, D] tensors into [B, W*D] for downstream PCA."""
    return window.reshape(window.shape[0], -1)


def _pca_project(x: torch.Tensor, n_components: int = 2) -> torch.Tensor:
    """Compute a low-rank PCA projection with Torch primitives."""
    x = x.to(torch.float32)
    x_centered = x - x.mean(dim=0, keepdim=True)
    q = min(n_components + 5, x_centered.shape[1])
    U, S, V = torch.pca_lowrank(x_centered, q=q)
    basis = V[:, :n_components]
    projected = x_centered @ basis
    return projected.cpu()


def _scatter_views(
    proj_v1: torch.Tensor,
    proj_v2: torch.Tensor,
    picks: Iterable[Tuple[int, int]],
    output_path: Path,
    title: str,
) -> None:
    """Create a scatter plot overlaying two augmented views."""
    plt.figure(figsize=(8, 6))
    plt.scatter(proj_v1[:, 0], proj_v1[:, 1], s=12, alpha=0.6, label="view1", color="#1f77b4")
    plt.scatter(proj_v2[:, 0], proj_v2[:, 1], s=12, alpha=0.6, label="view2", color="#ff7f0e")
    plt.legend()
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Project BYOL windows to 2-D for quick inspection.")
    parser.add_argument("--dataset", type=str, required=True, help="Rollout dataset path (.pt).")
    parser.add_argument("--output", type=str, default="byol_windows_pca.png", help="Output image file.")
    parser.add_argument("--window", type=int, default=32, help="Temporal window length (W).")
    parser.add_argument("--batch", type=int, default=512, help="Number of window pairs to visualize (B).")
    parser.add_argument("--max-shift", type=int, default=1, help="Temporal shift parameter for sampling.")
    parser.add_argument("--noise-std", type=float, default=0.02, help="Gaussian feature noise std.")
    parser.add_argument("--feat-drop", type=float, default=0.05, help="Per-frame feature dropout probability.")
    parser.add_argument("--frame-drop", type=float, default=0.05, help="Per-sequence frame dropout probability.")
    parser.add_argument("--time-warp", type=float, default=0.0, help="Enable time warp augmentation (0 disables).")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for sampling.")
    args = parser.parse_args()

    data = torch.load(args.dataset, map_location="cpu")
    obs = data["observations"].float()
    dones = data["dones"].bool()
    actions = data.get("actions")
    if actions is not None:
        actions = actions.float()

    windows = sample_byol_windows(
        obs=obs,
        dones=dones,
        W=args.window,
        B=args.batch,
        max_shift=args.max_shift,
        noise_std=args.noise_std,
        feat_drop=args.feat_drop,
        frame_drop=args.frame_drop,
        time_warp_scale=args.time_warp,
        device=torch.device(args.device),
        actions=actions,
    )

    if actions is None:
        v1, v2, picks, count = windows
    else:
        (v1, _), (v2, _), picks, count = windows

    v1_flat = _flatten_windows(v1.cpu())
    v2_flat = _flatten_windows(v2.cpu())

    proj_v1 = _pca_project(v1_flat)
    proj_v2 = _pca_project(v2_flat)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    title = f"PCA on BYOL windows (B={count}, W={min(args.window, obs.shape[0])})"
    _scatter_views(proj_v1, proj_v2, picks, output_path, title)

    print(f"[INFO] Saved PCA visualization to {output_path}")


if __name__ == "__main__":
    main()
