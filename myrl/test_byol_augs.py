#!/usr/bin/env python3
# Copyright (c) 2022-2025
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal BYOL augmentation smoke-test on a saved rollout dataset.

Example:
    python myrl/test_byol_augs.py --dataset datasets/go2_rough_rollouts.pt --window 16 --batch 512
"""

from __future__ import annotations

import argparse
import os
import sys
import pickle

import numpy as np
import torch

# add repository root so that ``import myrl`` works when run directly
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from myrl.utils_core import sample_byol_windows, BYOLSeq, ObsEncoder, ActionEncoder


def _load_tensor(data: dict, key: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = data[key]
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick BYOL augmentation check on rollout dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the saved rollout .pt file.")
    parser.add_argument("--window", type=int, default=16, help="Temporal window length (W).")
    parser.add_argument("--batch", type=int, default=256, help="Number of window pairs to sample (B).")
    parser.add_argument("--max-shift", type=int, default=1, help="Max relative time shift between paired views.")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Gaussian noise std for feature jitter.")
    parser.add_argument("--feat-drop", type=float, default=0.05, help="Per-frame feature dropout probability.")
    parser.add_argument("--frame-drop", type=float, default=0.05, help="Per-sequence frame dropout probability.")
    parser.add_argument("--time-warp", type=float, default=0.0, help="Time warp scale (0 disables).")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for augmentation computations.")
    args = parser.parse_args()

    try:
        data = torch.load(args.dataset, map_location="cpu")
    except (pickle.UnpicklingError, RuntimeError) as exc:
        msg = str(exc)
        if "Weights only load failed" in msg or "weights_only" in msg:
            data = torch.load(args.dataset, map_location="cpu", weights_only=False)
        else:
            raise

    obs = _load_tensor(data, "observations", dtype=torch.float32)
    dones = _load_tensor(data, "dones").to(dtype=torch.bool)
    actions = data.get("actions")
    if actions is not None:
        actions = actions.to(torch.float32)

    device = torch.device(args.device)

    obs_encoder = ObsEncoder(obs_dim=obs.shape[-1], feat_dim=256).to(device=device)
    action_encoder = None
    if actions is not None:
        action_encoder = ActionEncoder(act_dim=actions.shape[-1], feat_dim=256).to(device=device)

    model = BYOLSeq(
        obs_encoder=obs_encoder,
        action_encoder=action_encoder,
        z_dim=128,
        proj_dim=128,
        tau=0.99,
        use_information_bottleneck=True,
    ).to(device=device)
    optimizer = torch.optim.AdamW(
        model.online_parameters(
            include_obs_encoder=True, include_action_encoder=actions is not None
        ),
        lr=3e-4,
        weight_decay=1e-4,
    )

    train_steps = max(4, min(16, 4096 // max(1, args.batch)))
    model.train()
    for step in range(train_steps):
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
            device=device,
            actions=actions,
        )
        if actions is None:
            v1, v2, _, _ = windows
        else:
            (v1o, v1a), (v2o, v2a), _, _ = windows
            v1 = (v1o, v1a)
            v2 = (v2o, v2a)
        loss = model.loss(v1, v2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.ema_update()
        if (step + 1) == train_steps or step % 4 == 0:
            print(f"[BYOL] step {step + 1}/{train_steps}: loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        z_view1 = model.f_online(v1)
        z_view2 = model.f_online(v2)

    embeddings = torch.cat((z_view1, z_view2), dim=0)
    pcs = _pca_project(embeddings, n_components=2).numpy()
    split = z_view1.shape[0]

    import matplotlib.pyplot as plt  # Imported lazily so training-only runs stay lightweight.

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pcs[:split, 0], pcs[:split, 1], s=12, alpha=0.6, label="view 1")
    ax.scatter(pcs[split:, 0], pcs[split:, 1], s=12, alpha=0.6, label="view 2")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("BYOL representations (PCA)")
    ax.legend(loc="best")
    fig.tight_layout()

    dataset_stem = os.path.splitext(os.path.basename(args.dataset))[0]
    out_path = f"{dataset_stem}_byol_pca.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[BYOL] saved PCA scatter to {out_path}")

    # if actions is None:
    #     v1, v2, picks, count = windows
    #     print(f"obs view1 shape = {tuple(v1.shape)}, view2 shape = {tuple(v2.shape)}")
    # else:
    #     (v1o, v1a), (v2o, v2a), picks, count = windows
    #     print(f"obs view1 shape = {tuple(v1o.shape)}, actions view1 shape = {tuple(v1a.shape)}")
    #     print(f"obs view2 shape = {tuple(v2o.shape)}, actions view2 shape = {tuple(v2a.shape)}")

    # print(f"sampled windows = {count}, unique picks = {len(picks)}")
    # print("first 5 picks:", picks[:5])


if __name__ == "__main__":
    main()
