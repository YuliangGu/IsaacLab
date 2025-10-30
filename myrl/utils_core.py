import math
import torch, torch.nn as nn, numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List

def l2norm(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L2-normalization."""
    return F.normalize(x, dim=-1)

def layer_init(layer: nn.Linear, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal + bias init for Tanh MLPs."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Light MLP head (used in BYOL)."""
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        # nn.BatchNorm1d(512), # BatchNorm for heterogeneous data
        nn.LayerNorm(512),  # use LayerNorm instead of BatchNorm
        nn.ReLU(inplace=True),
        nn.Linear(512, out_dim),
    )

class ObsEncoder(nn.Module):
    """Compact encoder for flat states."""
    def __init__(self, obs_dim: int, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)), nn.ELU(),
            layer_init(nn.Linear(128, feat_dim)), nn.ELU(),
        )
        self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ActionEncoder(nn.Module):
    """Map actions to the same feature space as observations."""
    def __init__(self, act_dim: int, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(act_dim, feat_dim)), nn.ELU(),
        )
        self.out_dim = feat_dim

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation: z' = (1 + gamma(c)) ⊙ z + beta(c).
    Stable, simple conditioning for belief/context vectors.

    See: https://arxiv.org/abs/1709.07871
    """
    def __init__(self, ctx_dim: int, feat_dim: int):
        super().__init__()
        self.gamma = nn.Linear(ctx_dim, feat_dim)
        self.beta  = nn.Linear(ctx_dim, feat_dim)
        nn.init.zeros_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, z: torch.Tensor, c: Optional[torch.Tensor]) -> torch.Tensor:
        if c is None: return z
        return z * (1.0 + self.gamma(c)) + self.beta(c)

class SeqEncoder(nn.Module):
    """
    Sequence encoder: (ObsEncoder [ + ActionEncoder ]) -> GRU -> z_T
    Accepts input as either:
        - obs_seq:  Tensor[B, W, D_obs]
        - (obs_seq, act_seq): Tuple[Tensor[B,W,D_obs], Tensor[B,W,D_act]]
    Returns: Tensor[B, z_dim]
    """
    def __init__(self, obs_encoder: ObsEncoder, z_dim: int, action_encoder: Optional[ActionEncoder] = None):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.action_encoder = action_encoder
        self.gru = nn.GRU(input_size=self.obs_encoder.out_dim,
                          hidden_size=z_dim,
                          num_layers=1,
                          dropout=0.1,
                          bidirectional=False,
                          batch_first=True)

    def _encode_frames(self, obs_seq: torch.Tensor, act_seq: Optional[torch.Tensor]) -> torch.Tensor:
        B, W, D = obs_seq.shape
        # Ensure inputs are normal (non-inference) tensors so GRU can save
        # them for backward. Use a clone outside inference_mode to
        # re-materialize fresh tensors for Autograd.
        with torch.inference_mode(False):
            x_view = obs_seq.reshape(B * W, D)
            x = x_view.clone()  # normal tensor for autograd
            feats = self.obs_encoder(x).reshape(B, W, -1)  # [B,W,F]
            if act_seq is not None and self.action_encoder is not None:
                A = act_seq.shape[-1]
                a_view = act_seq.reshape(B * W, A)
                a = a_view.clone()  # normal tensor for autograd
                actf = self.action_encoder(a).reshape(B, W, -1)
                feats = feats + actf  # same feature space, simple additive fusion
        return feats

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if isinstance(x, tuple):
            obs_seq, act_seq = x
        else:
            obs_seq, act_seq = x, None
        feats = self._encode_frames(obs_seq, act_seq)       # [B,W,F]
        # Clone to make sure GRU sees a normal tensor for saved tensors
        feats = feats.clone()
        _, hT = self.gru(feats)                             # hT: [1,B,z]
        return hT.squeeze(0)                                # [B,z]
    
# class SeqEncoder(nn.Module):
#     def __init__(self, obs_encoder: nn.Module, z_dim: int, action_encoder: Optional[nn.Module] = None):
#         super().__init__()
#         self.obs_encoder = obs_encoder      # shared with policy (reference)
#         self.action_encoder = action_encoder
#         self.gru = nn.GRU(input_size=self.obs_encoder.out_dim, hidden_size=z_dim, batch_first=True)

#     def _encode_frames(self, obs_seq, act_seq):
#         B, W, D = obs_seq.shape
#         # Ensure we run outside inference mode when building inputs for autograd
#         with torch.inference_mode(False):
#             # Strong re-materialization: allocate new tensors and copy data
#             x_view = obs_seq.reshape(B * W, D)
#             x = torch.empty_like(x_view, device=x_view.device, dtype=x_view.dtype)
#             x.copy_(x_view)
#             feats = self.obs_encoder(x).reshape(B, W, -1)
#             if (act_seq is not None) and (self.action_encoder is not None):
#                 A = act_seq.shape[-1]
#                 a_view = act_seq.reshape(B * W, A)
#                 a = torch.empty_like(a_view, device=a_view.device, dtype=a_view.dtype)
#                 a.copy_(a_view)
#                 actf = self.action_encoder(a).reshape(B, W, -1)
#                 feats = feats + actf
#         return feats
    
#     def forward(self, x):
#         if isinstance(x, tuple): obs_seq, act_seq = x
#         else:                    obs_seq, act_seq = x, None
#         feats = self._encode_frames(obs_seq, act_seq)   # [B,W,F]
#         feats = feats.clone()  # ensure a non-inference, fresh tensor for Autograd
#         _, hT = self.gru(feats)
#         return hT.squeeze(0)


class BYOLSeq(nn.Module):
    """
    Temporal BYOL on vector sequences with optional action conditioning.

    Original BYOL paper: https://arxiv.org/abs/2006.07733
    """
    def __init__(self,
                 obs_encoder: ObsEncoder,
                 z_dim: int = 128,
                 proj_dim: int = 128,
                 tau: float = 0.996,
                 action_encoder: Optional[ActionEncoder] = None):
        super().__init__()
        self.f_online = SeqEncoder(obs_encoder, z_dim=z_dim, action_encoder=action_encoder)
        self.g_online = mlp(z_dim, proj_dim)
        self.q_online = mlp(proj_dim, proj_dim)

        import copy
        self.f_targ = copy.deepcopy(self.f_online)
        self.g_targ = copy.deepcopy(self.g_online)
        for p in list(self.f_targ.parameters()) + list(self.g_targ.parameters()):
            p.requires_grad_(False)

        self.tau = float(tau)

        # print full model summary
        print("--------------------------------------------------")
        print(f"[BYOLSeq] Initialized BYOLSeq model with:")
        print(f"BYOLSeq: {self}")

    def train(self, mode: bool = True):
        # Override to keep target networks in eval mode
        super().train(mode)
        self.f_targ.eval()
        self.g_targ.eval()
        return self
    
    @torch.no_grad()
    def ema_update(self) -> None:
        """EMA update for target encoders."""
        for online, targ in ((self.f_online, self.f_targ), (self.g_online, self.g_targ)):
            for p_o, p_t in zip(online.parameters(), targ.parameters()):
                p_t.data.mul_(self.tau).add_(p_o.data, alpha=1.0 - self.tau)

    def _forward_pair(self,
                      v1: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                      v2: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
        # Online
        z1 = self.f_online(v1)          # [B, z]
        z2 = self.f_online(v2)
        p1 = self.g_online(z1)          # [B, p]
        p2 = self.g_online(z2)
        h1 = self.q_online(p1)          # [B, p]
        h2 = self.q_online(p2)
        
        # Targets (no grad)
        with torch.no_grad():
            z1t = self.f_targ(v1)
            z2t = self.f_targ(v2)
            p1t = self.g_targ(z1t)
            p2t = self.g_targ(z2t)
        return h1, h2, p1t, p2t

    def loss(self,
             v1: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
             v2: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        h1, h2, p1t, p2t = self._forward_pair(v1, v2)
        loss_12 = (l2norm(h1) - l2norm(p2t)).pow(2).sum(-1).mean()
        loss_21 = (l2norm(h2) - l2norm(p1t)).pow(2).sum(-1).mean()
        return loss_12 + loss_21

    @torch.no_grad()
    def mismatch_per_window(self,
                            v1: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                            v2: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        h1, h2, p1t, p2t = self._forward_pair(v1, v2)
        m12 = (l2norm(h1) - l2norm(p2t)).pow(2).sum(-1)
        m21 = (l2norm(h2) - l2norm(p1t)).pow(2).sum(-1)
        return 0.5 * (m12 + m21)

    def online_parameters(self, include_obs_encoder: bool, include_action_encoder: bool = False):
        params = (
            list(self.f_online.gru.parameters())
            + list(self.g_online.parameters())
            + list(self.q_online.parameters())
        )
        if include_obs_encoder:
            params = list(self.f_online.obs_encoder.parameters()) + params
        if include_action_encoder and (self.f_online.action_encoder is not None):
            params = list(self.f_online.action_encoder.parameters()) + params
        return params

# class BYOLSeq(nn.Module):
#     def __init__(self, obs_encoder: nn.Module, z_dim=128, proj_dim=128, tau=0.996,
#                  action_encoder: Optional[nn.Module]=None,
#                  train_backbone: bool = True):
#         super().__init__()
#         self.f_online = SeqEncoder(obs_encoder, z_dim=z_dim, action_encoder=action_encoder)
#         self.g_online = mlp(z_dim, proj_dim)
#         self.q_online = mlp(proj_dim, proj_dim)
#         import copy
#         self.f_targ = copy.deepcopy(self.f_online); self.g_targ = copy.deepcopy(self.g_online)
#         for p in list(self.f_targ.parameters()) + list(self.g_targ.parameters()): p.requires_grad_(False)
#         # target networks operate in eval mode
#         self.f_targ.eval(); self.g_targ.eval()

#         # Optionally freeze backbone (shared obs encoder + GRU)
#         self.train_backbone = bool(train_backbone)
#         if not self.train_backbone:
#             try:
#                 for p in self.f_online.obs_encoder.parameters():
#                     p.requires_grad_(False)
#             except Exception:
#                 pass
#             try:
#                 for p in self.f_online.gru.parameters():
#                     p.requires_grad_(False)
#             except Exception:
#                 pass

#         # Disable BN running-stat updates in online heads to avoid inplace updates on inference-flagged buffers.
#         try:
#             import torch.nn as _nn
#             for m in list(self.g_online.modules()) + list(self.q_online.modules()):
#                 if isinstance(m, _nn.BatchNorm1d):
#                     m.eval()  # disable running stats update
#                     m.track_running_stats = False
#         except Exception:
#             pass
#         self.tau = float(tau)

#     def online_parameters(
#         self,
#         include_obs_encoder: bool = False,
#         include_action_encoder: bool = False,
#         include_backbone_gru: bool = False,
#     ):
#         """Return trainable parameters of the online BYOL components.

#         Args:
#             include_obs_encoder: Include obs encoder params (set False when encoder is shared with policy/optimized elsewhere).
#             include_action_encoder: Include action encoder params if present.
#         """
#         params = list(self.g_online.parameters()) + list(self.q_online.parameters())
#         if include_obs_encoder:
#             params += list(self.f_online.obs_encoder.parameters())
#         if include_action_encoder and (self.f_online.action_encoder is not None):
#             params += list(self.f_online.action_encoder.parameters())
#         if include_backbone_gru:
#             params += list(self.f_online.gru.parameters())
#         return params
    
#     @torch.no_grad()
#     def ema_update(self):
#         for online, targ in ((self.f_online, self.f_targ), (self.g_online, self.g_targ)):
#             for p_o, p_t in zip(online.parameters(), targ.parameters()):
#                 p_t.data.mul_(self.tau).add_(p_o.data, alpha=1.0 - self.tau)

#     def _forward_pair(self, v1, v2):
#         # Ensure online path runs with autograd enabled (even if outer code used inference mode)
#         with torch.inference_mode(False):
#             with torch.enable_grad():
#                 # Re-materialize inputs inside non-inference context to avoid
#                 # 'inference tensors cannot be saved for backward'.
#                 if isinstance(v1, tuple):
#                     v1o, v1a = v1; v2o, v2a = v2
#                     x1 = torch.empty_like(v1o); x1.copy_(v1o)
#                     x2 = torch.empty_like(v2o); x2.copy_(v2o)
#                     if v1a is not None:
#                         a1 = torch.empty_like(v1a); a1.copy_(v1a)
#                     else:
#                         a1 = None
#                     if v2a is not None:
#                         a2 = torch.empty_like(v2a); a2.copy_(v2a)
#                     else:
#                         a2 = None
#                     v1_in = (x1, a1); v2_in = (x2, a2)
#                 else:
#                     x1 = torch.empty_like(v1); x1.copy_(v1)
#                     x2 = torch.empty_like(v2); x2.copy_(v2)
#                     v1_in = x1; v2_in = x2

#                 # Backprop through backbone only if train_backbone=True
#                 if self.train_backbone:
#                     z1, z2 = self.f_online(v1_in), self.f_online(v2_in)
#                 else:
#                     with torch.no_grad():
#                         z1, z2 = self.f_online(v1_in), self.f_online(v2_in)
#                 p1, p2 = self.g_online(z1), self.g_online(z2)
#                 h1, h2 = self.q_online(p1), self.q_online(p2)
#         with torch.no_grad():
#             p1t, p2t = self.g_targ(self.f_targ(v1)), self.g_targ(self.f_targ(v2))
#         return h1, h2, p1t, p2t
    
#     def loss(self, v1, v2):
#         h1, h2, p1t, p2t = self._forward_pair(v1, v2)
#         return ((l2norm(h1)-l2norm(p2t)).pow(2).sum(-1).mean()
#               + (l2norm(h2)-l2norm(p1t)).pow(2).sum(-1).mean())

#     @torch.no_grad()
#     def beliefs_from_rollout(
#         self,
#         obs: torch.Tensor,                 # [T, N, D_obs]
#         dones: torch.Tensor,               # [T, N]  (True where episode ended at t)
#         actions: Optional[torch.Tensor] = None,  # [T, N, A] or None
#     ) -> torch.Tensor:
#         """
#         Run the shared obs-encoder + BYOL GRU across the entire rollout and return
#         per-step beliefs: Tensor[T, N, z_dim]. Resets GRU hidden state on dones.
#         """
#         device = next(self.parameters()).device
#         T, N, D = obs.shape
#         # Encode all frames with the *shared* encoder (no grad for training-time cache)
#         x = obs.reshape(T * N, D).clone().to(device)
#         feats = self.f_online.obs_encoder(x).reshape(T, N, -1)  # [T,N,F]
#         if (actions is not None) and (self.f_online.action_encoder is not None):
#             A = actions.shape[-1]
#             a = actions.reshape(T * N, A).clone().to(device)
#             actf = self.f_online.action_encoder(a).reshape(T, N, -1)
#             feats = feats + actf
#         # Roll the GRU with resets on dones
#         z_dim = self.f_online.gru.hidden_size
#         h = torch.zeros(1, N, z_dim, device=device)
#         out = []
#         for t in range(T):
#             f = feats[t].unsqueeze(1)     # [N,1,F]
#             _, h = self.f_online.gru(f, h)
#             out.append(h.squeeze(0).clone())
#             if dones[t].any():
#                 mask = dones[t].to(torch.bool).to(device)
#                 h[:, mask, :] = 0.0
#         return torch.stack(out, dim=0)    # [T,N,Z]


@torch.no_grad()
def extract_policy_obs_and_dones(storage) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return policy observations and dones as dense tensors.

    Returns
    - obs:   [T, N, D]
    - dones: [T, N] (True where episode ended at t)
    """
    # Locate observations container
    obs_container = getattr(storage, "observations", None)
    if obs_container is None:
        obs_container = getattr(storage, "obs", None)
    if obs_container is None:
        raise AttributeError("Cannot find observations in storage (expected 'observations' or 'obs').")

    # Prefer the 'policy' view if present
    if hasattr(obs_container, "get"):
        try:
            obs = obs_container.get("policy", None)
        except TypeError:
            # some containers require only the key
            obs = obs_container.get("policy") if hasattr(obs_container, "get") else None
    else:
        obs = None
    if obs is None:
        obs = obs_container

    # If obs is a container, concatenate tensor leaves along last dim
    if not torch.is_tensor(obs):
        fields = []
        for k in list(obs.keys()):
            v = None
            if hasattr(obs, "get"):
                try:
                    v = obs.get(k)
                except Exception:
                    v = None
            if v is None:
                try:
                    v = obs[k]
                except Exception:
                    v = None
            if torch.is_tensor(v):
                if v.dim() == 2:
                    v = v.unsqueeze(-1)  # [T,N] -> [T,N,1]
                fields.append(v)
        if not fields:
            raise TypeError("No tensor leaves found in observation container")
        obs = torch.cat(fields, dim=-1)

    # Ensure shape [T,N,D]
    if obs.dim() == 2:
        obs = obs.unsqueeze(-1)

    # Dones / terminations
    dones = getattr(storage, "dones", None)
    if dones is None:
        dones = getattr(storage, "terminated", None)
    if dones is None:
        dones = torch.zeros(obs.shape[0], obs.shape[1], dtype=torch.bool, device=obs.device)
    if dones.dim() == 3 and dones.shape[-1] == 1:
        dones = dones.squeeze(-1)
    dones = dones.to(torch.bool)
    return obs, dones


# ========================= Augmentations & Sampling =======================

def _randn_like_gen(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    try:
        return torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=rng)
    except TypeError:
        return torch.randn(x.shape, device=x.device, dtype=x.dtype)

def _rand_gen(shape: Tuple[int, ...], device: torch.device, rng: torch.Generator) -> torch.Tensor:
    try:
        return torch.rand(shape, device=device, generator=rng)
    except TypeError:
        return torch.rand(shape, device=device)

def _feature_jitter(x: torch.Tensor, noise_std: float, feat_drop: float, rng: torch.Generator) -> None:
    """In-place: Gaussian noise + Bernoulli feature dropout per frame. x: [B,W,D]"""
    if noise_std > 0:
        x.add_(_randn_like_gen(x, rng) * noise_std)
    if feat_drop > 0:
        drop = _rand_gen(x.shape, x.device, rng) < feat_drop
        x.mul_(1.0 - drop.float())

def _frame_drop_inplace(x: torch.Tensor, p: float, rng: torch.Generator) -> None:
    """In-place 'temporal dropout': with prob p, copy previous frame at t."""
    if p <= 0: return
    B, T, _ = x.shape
    mask = _rand_gen((B, T), x.device, rng) < p
    for t in range(1, T):
        x[:, t][mask[:, t]] = x[:, t - 1][mask[:, t]]

def _temporal_shift(x: torch.Tensor, shift: int) -> torch.Tensor:
    """ CAUTION: This operation seems too strong for RL tasks. """
    if shift == 0: return x
    if shift > 0:
        return torch.cat([x[:, :1].expand(-1, shift, -1), x[:, :-shift]], dim=1)
    shift = -shift
    return torch.cat([x[:, shift:], x[:, -1:].expand(-1, shift, -1)], dim=1)

def _time_warp(x: torch.Tensor, max_scale: float, rng: torch.Generator) -> torch.Tensor:
    """
    Monotonic time warp: resample each sequence by a random global speed in [1-max_scale, 1+max_scale],
    then linearly re-interpolate back to the original length. x: [B,T,D] -> [B,T,D]

    NOTE: 
        This operation is quite expensive (slow).
    """
    if max_scale <= 0: return x
    B, T, D = x.shape
    speeds = (1.0 - max_scale) + (2.0 * max_scale) * torch.rand(B, device=x.device, generator=rng)
    # target time grid [0, T-1]
    t_out = torch.linspace(0, T - 1, T, device=x.device)
    out = torch.empty_like(x)
    for b in range(B):
        s = float(speeds[b].item())
        t_in = torch.arange(T, device=x.device) / s
        t_in = torch.clamp(t_in, 0, T - 1)
        # linear interpolation
        t0 = t_in.floor().long()
        t1 = torch.clamp(t0 + 1, 0, T - 1)
        w = (t_in - t0.float()).unsqueeze(-1)
        out[b] = (1 - w) * x[b, t0] + w * x[b, t1]
    return out

def _channel_dropout_inplace(x: torch.Tensor, p: float, rng: torch.Generator) -> None:
    """Drop entire feature channels consistently over time: mask shape [B,1,D]. In-place on x: [B,T,D]."""
    if p <= 0: 
        return
    B, T, D = x.shape
    mask = (_rand_gen((B, 1, D), x.device, rng) >= p).to(x.dtype)
    x.mul_(mask)

def _time_mask_inplace(x: torch.Tensor, span: int, prob: float, rng: torch.Generator) -> None:
    """Randomly zero a contiguous time span per sequence with probability `prob`.
    Args:
        x: [B,T,D]
        span: maximum span length to mask (clip to [1,T])
        prob: probability to apply one span mask per sequence
    """
    if prob <= 0 or span <= 0:
        return
    B, T, D = x.shape
    span = max(1, min(int(span), T))
    # for each sequence, decide whether to mask and pick a start
    apply = _rand_gen((B,), x.device, rng) < prob
    if not torch.any(apply):
        return
    starts = torch.randint(0, max(1, T - span + 1), (B,), device=x.device, generator=rng)
    for b in torch.nonzero(apply, as_tuple=False).flatten():
        s = int(starts[b].item())
        x[b, s:s+span, :] = 0

def _calibration_noise_inplace(x: torch.Tensor, gain_std: float, bias_std: float, rng: torch.Generator) -> None:
    """Apply per-channel gain/bias perturbation consistent over time: x <- (1+g)*x + b.
    g, b have shape [B,1,D]. In-place on x: [B,T,D]."""
    if gain_std <= 0 and bias_std <= 0:
        return
    B, T, D = x.shape
    if gain_std > 0:
        g = _randn_like_gen(x.new_empty((B, 1, D)), rng) * gain_std
    else:
        g = x.new_zeros((B, 1, D))
    if bias_std > 0:
        b = _randn_like_gen(x.new_empty((B, 1, D)), rng) * bias_std
    else:
        b = x.new_zeros((B, 1, D))
    x.mul_(1.0 + g).add_(b)

def _smooth_time_inplace(x: torch.Tensor, kernel_size: int, prob: float) -> None:
    """Optionally low-pass filter along time using a depthwise 1D conv with triangular kernel.
    Args:
        x: [B,T,D]
        kernel_size: odd int >=3; if <=0 no-op
        prob: apply smoothing with this probability
    """
    if kernel_size <= 1 or prob <= 0:
        return
    B, T, D = x.shape
    # coin flip once per batch to avoid per-sample branching
    if torch.rand((), device=x.device) >= prob:
        return
    K = int(kernel_size)
    if K % 2 == 0:
        K += 1
    # triangular kernel
    base = torch.arange(1, (K//2)+2, device=x.device, dtype=x.dtype)
    kernel = torch.cat([base, base[:-1].flip(0)])
    kernel = (kernel / kernel.sum()).view(1, 1, K)
    # depthwise conv over [B,D,T]
    x_t = x.permute(0, 2, 1)  # [B,D,T]
    weight = kernel.expand(D, 1, K)
    pad = K // 2
    x_t = torch.nn.functional.conv1d(x_t, weight, padding=pad, groups=D)
    x.copy_(x_t.permute(0, 2, 1))

@torch.no_grad()
def _valid_window_starts(dones: torch.Tensor, W: int) -> np.ndarray:
    """
    Return (start, env) for windows of length W that do not cross episode boundaries.
    'dones' is [T,N] with True when an episode ended at t.
    """
    T, N = dones.shape
    dn = dones.to(dtype=torch.bool, device="cpu").numpy()
    valid: List[Tuple[int,int]] = []
    for n in range(N):
        seg_start = 0
        for t in range(T):
            if dn[t, n]:
                seg_end = t
                # include the last valid start: s <= seg_end - W + 1
                stop = max(seg_start, seg_end - W + 2)
                for s in range(seg_start, stop):
                    valid.append((s, n))
                seg_start = t + 1
        seg_end = T - 1
        stop = max(seg_start, seg_end - W + 2)
        for s in range(seg_start, stop):
            valid.append((s, n))
    return np.array(valid, dtype=np.int64)

@torch.no_grad()
def sample_byol_windows(
    obs: torch.Tensor,                 # [T,N,D_obs]
    dones: torch.Tensor,               # [T,N]
    W: int, B: int, max_shift: int,
    noise_std: float, feat_drop: float, frame_drop: float,
    time_warp_scale: float,
    # Extra augmentations (all optional; default disabled)
    ch_drop: float = 0.0,
    time_mask_prob: float = 0.0,
    time_mask_span: int = 0,
    gain_std: float = 0.0,
    bias_std: float = 0.0,
    smooth_prob: float = 0.0,
    smooth_kernel: int = 0,
    mix_strength: float = 0.0,
    device: torch.device = torch.device("cpu"),
    actions: Optional[torch.Tensor] = None,  # [T,N,A] or None
) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        List[Tuple[int,int]],
        int,
    ]:
    """
    Sample B window pairs (v1, v2) with independent augs.
    
    Args:
        obs: [T,N,D_obs] float tensor of observations
        dones: [T,N] bool tensor with True where episode ended at t
        W: window length
        B: batch size (number of pairs)
        max_shift: max temporal shift (frames) between v1 and v2; 0 = no shift
        noise_std: per-frame Gaussian noise stddev
        feat_drop: per-frame Bernoulli feature dropout probability
        frame_drop: per-sequence Bernoulli frame-drop probability
        time_warp_scale: max time warp scale (0 = no warp)
        device: target device for output tensors
        actions: optional [T,N,A] float tensor of actions (if provided, also return action windows)

        ch_drop: per-sequence channel dropout probability (consistent over time)
        time_mask_prob: per-sequence probability to apply a random time-span mask
        time_mask_span: max length of time-span mask (in frames: use W//8, e.g., W=16 -> span=2)
        gain_std: per-sequence per-channel gain noise stddev (consistent over time)
        bias_std: per-sequence per-channel bias noise stddev (consistent over time)
        smooth_prob: per-batch probability to apply low-pass smoothing
        smooth_kernel: odd kernel size >=3 for low-pass smoothing along time
        mix_strength: strength of temporal mixing augmentation (0 = no mixing)
    """
    W_use = min(int(W), int(obs.shape[0]))
    starts = _valid_window_starts(dones, W_use)
    if len(starts) == 0:
        starts = np.array([(0, n) for n in range(obs.shape[1])], dtype=np.int64)
    replace = len(starts) < B
    idx = np.random.choice(len(starts), size=min(B, len(starts)), replace=replace)
    picks = starts[idx].tolist()

    v1o, v2o, v1a, v2a = [], [], [], []
    T_total = obs.shape[0]
    for (s, n) in picks:
        w1o = obs[s : s + W_use, n, :].clone()
        delta = np.random.randint(-max_shift, max_shift + 1) if max_shift > 0 else 0
        s2 = max(0, min(s + delta, T_total - W_use))
        w2o = obs[s2 : s2 + W_use, n, :].clone()
        v1o.append(w1o); v2o.append(w2o)
        if actions is not None:
            w1a = actions[s  : s  + W_use, n, :].clone()
            w2a = actions[s2 : s2 + W_use, n, :].clone()
            v1a.append(w1a); v2a.append(w2a)

    v1o = torch.stack(v1o, dim=0).to(device)  # [B,W_use,D_obs]
    v2o = torch.stack(v2o, dim=0).to(device)

    g1 = torch.Generator(device=device).manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())
    g2 = torch.Generator(device=device).manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())

    _feature_jitter(v1o, noise_std, feat_drop, g1)
    _feature_jitter(v2o, noise_std, feat_drop, g2)

    # calibration (per-channel gain/bias)
    _calibration_noise_inplace(v1o, gain_std, bias_std, g1)
    _calibration_noise_inplace(v2o, gain_std, bias_std, g2)

    # channel-consistent dropout
    _channel_dropout_inplace(v1o, ch_drop, g1)
    _channel_dropout_inplace(v2o, ch_drop, g2)

    # time-span masking
    _time_mask_inplace(v1o, time_mask_span, time_mask_prob, g1)
    _time_mask_inplace(v2o, time_mask_span, time_mask_prob, g2)

    # frame dropout (temporal consistency)
    _frame_drop_inplace(v1o, frame_drop, g1)
    _frame_drop_inplace(v2o, frame_drop, g2)

    if max_shift > 0:
        # random temporal shift (independent for v1 and v2)
        v1o = _temporal_shift(v1o, int(torch.randint(-max_shift, max_shift + 1, (1,)).item()))
        v2o = _temporal_shift(v2o, int(torch.randint(-max_shift, max_shift + 1, (1,)).item()))

    if time_warp_scale > 0:
        # time warp (independent for v1 and v2)
        v1o = _time_warp(v1o, time_warp_scale, g1)
        v2o = _time_warp(v2o, time_warp_scale, g2)

    if mix_strength > 0.0:
        # temporal mixing (independent for v1 and v2)
        v1o = _temporal_mixing(v1o, mix_strength, g1)
        v2o = _temporal_mixing(v2o, mix_strength, g2)

    # optional smoothing at the end
    _smooth_time_inplace(v1o, smooth_kernel, smooth_prob)
    _smooth_time_inplace(v2o, smooth_kernel, smooth_prob)

    if actions is None:
        return v1o, v2o, picks, len(picks)

    v1a = torch.stack(v1a, dim=0).to(device)  # [B,W,A]
    v2a = torch.stack(v2a, dim=0).to(device)
    return (v1o, v1a), (v2o, v2a), picks, len(picks)
