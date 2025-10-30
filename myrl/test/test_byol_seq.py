import torch
import os
import sys

try:
    from myrl.utils_core import BYOLSeq, ObsEncoder, ActionEncoder, sample_byol_windows
except ModuleNotFoundError:
    # Ensure project root on sys.path when launching via path runners
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from myrl.utils_core import BYOLSeq, ObsEncoder, ActionEncoder, sample_byol_windows


def test_byol_seq_shapes_and_grads():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # small synthetic setup
    B, W = 8, 6
    D_obs, D_act = 20, 5
    z_dim, proj_dim, feat_dim = 16, 32, 24

    obs_enc = ObsEncoder(D_obs, feat_dim=feat_dim).to(device)
    act_enc = ActionEncoder(D_act, feat_dim=feat_dim).to(device)
    model = BYOLSeq(obs_enc, z_dim=z_dim, proj_dim=proj_dim, action_encoder=act_enc).to(device)

    # two random views
    v1 = (
        torch.randn(B, W, D_obs, device=device),
        torch.randn(B, W, D_act, device=device),
    )
    v2 = (
        torch.randn(B, W, D_obs, device=device),
        torch.randn(B, W, D_act, device=device),
    )

    # forward and loss
    loss = model.loss(v1, v2)
    assert loss.ndim == 0, "Loss should be a scalar"

    # backward: online params get grads, target params do not
    model.train()
    model.zero_grad(set_to_none=True)
    loss.backward()

    # online has grads
    online_grads = [p.grad is not None for p in (
        list(model.f_online.parameters())
        + list(model.g_online.parameters())
        + list(model.q_online.parameters())
    )]
    assert all(online_grads), "Online parameters should receive gradients"

    # target has no grads
    target_grads = [p.grad is not None for p in (
        list(model.f_targ.parameters()) + list(model.g_targ.parameters())
    )]
    assert not any(target_grads), "Target parameters must not receive gradients"


def test_sample_byol_windows_respects_dones():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # rollout-shaped tensors
    T, N, D = 20, 4, 12
    W, B = 6, 16
    obs = torch.randn(T, N, D, device=device)

    # fabricate dones that create short segments; ensure no window crosses a done
    dones = torch.zeros(T, N, dtype=torch.bool, device=device)
    dones[5, :] = True  # boundary at t=5 for all envs
    dones[13, 2] = True # extra boundary for env 2

    v1, v2, picks, K = sample_byol_windows(
        obs=obs,
        dones=dones,
        W=W,
        B=B,
        max_shift=0,
        noise_std=0.01,
        feat_drop=0.0,
        frame_drop=0.0,
        time_warp_scale=0.0,
        device=device,
        actions=None,
    )

    assert K == len(picks) and K > 0, "Should sample at least one valid window"
    assert v1.shape[:2] == (K, W) and v2.shape[:2] == (K, W), "Window shapes must be [K, W, D]"

    # Verify each (start, env) stays within a single episode segment
    for (s, n) in picks:
        seg = dones[:, n].nonzero(as_tuple=False).squeeze(-1)
        # find first done >= s within the W-span
        if seg.numel() > 0:
            first_done_after_s = seg[seg >= s]
            if first_done_after_s.numel() > 0:
                assert int(first_done_after_s[0].item()) >= s + W - 1, "Window crosses episode boundary"


if __name__ == "__main__":
    test_byol_seq_shapes_and_grads()
    test_sample_byol_windows_respects_dones()
    print("BYOLSeq tests passed.")