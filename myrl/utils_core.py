import math
import torch, torch.nn as nn, numpy as np
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

def l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)

def layer_init(layer: nn.Linear, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        # nn.BatchNorm1d(512), # BatchNorm for heterogeneous data
        nn.LayerNorm(256),  # use LayerNorm instead of BatchNorm
        nn.ReLU(inplace=True),
        nn.Linear(256, out_dim),
    )

class ObsEncoder(nn.Module):
    def __init__(self, obs_dim: int, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, feat_dim)),
            nn.ReLU(inplace=True),
        ) #Just a single linear layer
        self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer. See: https://arxiv.org/abs/1709.07871
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
    Sequence encoder: ObsEncoder -> GRU -> z_T
    Accepts input as:
        - obs_seq:  Tensor[B, W, D_obs]
    Returns: Tensor[B, z_dim]
    """
    def __init__(self, obs_encoder: ObsEncoder, z_dim: int, output_type: str = "last"):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.gru = nn.GRU(input_size=self.obs_encoder.out_dim,
                          hidden_size=z_dim,
                          batch_first=True)
        self.output_type = str(output_type)
        if self.output_type == "attn":
            self.attn = nn.Linear(z_dim, 1)

    def _encode_frames(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, W, D = obs_seq.shape # batch, window, obs_dim
        feats = self.obs_encoder(obs_seq.reshape(B * W, D)).reshape(B, W, -1)  # [B,W,F]
        return feats

    def forward(self, obs_seq: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        # is_causal is unused here; for compatibility with TransformerEncoder
        feats = self._encode_frames(obs_seq)       # [B,W,F]: batch, window, feat_dim
        out, hT = self.gru(feats)  # out: [B,W,z], hT: [1,B,z]
        
        if self.output_type == "last":
            return hT.squeeze(0)    # [B,z]
        elif self.output_type == "mean":
            return out.mean(dim=1)   # [B,z] 
        elif self.output_type == "attn":
            attn_weights = torch.softmax(self.attn(out).squeeze(-1), dim=-1)  # [B,W]
            z = (out * attn_weights.unsqueeze(-1)).sum(dim=1)                  # [B,z]
            return z
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")

# Causal Transformer Encoder (upgrade for SeqEncoder)
class TransformerEncoder(nn.Module):
    """ Transformer encoder for sequential data. """
    def __init__(self,
                 obs_encoder: ObsEncoder,
                 z_dim: int,
                 output_type: str = "last",
                 n_layers: int = 2,
                 n_heads: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.obs_encoder = obs_encoder
        if output_type not in ("last", "mean"):
            raise ValueError(f"Unsupported output_type for TransformerEncoder: {output_type}")
        self.output_type = str(output_type)
        encoder_layer = nn.TransformerEncoderLayer(d_model=z_dim,
                                                   nhead=n_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu',
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.input_proj = nn.Linear(self.obs_encoder.out_dim, z_dim)

    def _encode_frames(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, W, D = obs_seq.shape # batch, window, obs_dim
        feats = self.obs_encoder(obs_seq.reshape(B * W, D)).reshape(B, W, -1)  # [B,W,F]
        return feats
    
    def forward(self, obs_seq: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        feats = self._encode_frames(obs_seq)       # [B,W,F]: batch, window, feat_dim
        x = self.input_proj(feats)                 # [B,W,z_dim]
        
        if is_causal:
            W = x.size(1)
            mask = torch.triu(torch.ones((W, W), device=x.device), diagonal=1).bool()
        else:
            mask = None
        
        out = self.transformer_encoder(x, mask=mask)  # [B,W,z_dim]
        if self.output_type == "last":
            return out[:, -1, :]    # [B,z_dim]
        elif self.output_type == "mean":
            return out.mean(dim=1)   # [B,z_dim]
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")
        
class BYOLSeq(nn.Module):
    """
    BYOL for sequential data (BYOL-Seq).

    Args:
        obs_encoder: ObsEncoder module to encode individual observations
        z_dim:       Dimension of sequence representation
        proj_dim:    Dimension of projection head output
        tau:         EMA coefficient for target networks
        output_type: How to pool GRU outputs ('last', 'mean', 'attn')
        use_information_bottleneck: Whether to apply an information bottleneck on z

    Notes:
        output_type:
            'last' - use final hidden state h_T
            'mean' - average over all hidden states
            'attn' - learnable attention over hidden states
        use_information_bottleneck:
            If True, applies an information bottleneck on the sequence representation z. Use re-parameterization trick.
    """
    def __init__(self,
                 obs_encoder: ObsEncoder,
                 z_dim: int = 128,
                 proj_dim: int = 128,
                 tau: float = 0.996,
                 output_type: str = "mean",
                 use_information_bottleneck: bool = False,
                 use_transformer: bool = False):
        super().__init__()
        if use_transformer:
            self.f_online = TransformerEncoder(obs_encoder, z_dim=z_dim, output_type=output_type)
        else:
            self.f_online = SeqEncoder(obs_encoder, z_dim=z_dim, output_type=output_type)
        self.g_online = mlp(z_dim, proj_dim)
        self.q_online = mlp(proj_dim, proj_dim)
        self.use_information_bottleneck = bool(use_information_bottleneck)
        if self.use_information_bottleneck:
            self.ib_head = layer_init(nn.Linear(z_dim, 2 * z_dim), std=1.0)
            self.beta_ib = 1e-3
        else:
            self.ib_head = None
        self._last_ib_kl: Optional[torch.Tensor] = None

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

    def infer_ctx(self,
                  obs_seq: torch.Tensor) -> torch.Tensor:
        """Encode context from observation sequence. obs_seq: [B,W,D_obs] -> [B,z_dim]"""
        z = self.f_online(obs_seq)
        if self.use_information_bottleneck:
            z, _ = self._apply_information_bottleneck(z, sample=True)
        return z
    
    @torch.no_grad()
    def ema_update(self) -> None:
        """EMA update for target encoders."""
        for online, targ in ((self.f_online, self.f_targ), (self.g_online, self.g_targ)):
            for p_o, p_t in zip(online.parameters(), targ.parameters()):
                p_t.data.mul_(self.tau).add_(p_o.data, alpha=1.0 - self.tau)

    def _forward_pair(self,
                      v1: torch.Tensor,
                      v2: torch.Tensor):
        # Online
        z1 = self.f_online(v1)          # [B, z]
        z2 = self.f_online(v2)
        if self.use_information_bottleneck:
            z1, kl1 = self._apply_information_bottleneck(z1, sample=False)
            z2, kl2 = self._apply_information_bottleneck(z2, sample=False)
            self._last_ib_kl = 0.5 * (kl1 + kl2)
        else:
            self._last_ib_kl = z1.new_zeros(())
        p1 = self.g_online(z1)          # [B, p]
        p2 = self.g_online(z2)
        h1 = self.q_online(p1)          # [B, p]
        h2 = self.q_online(p2)
        
        # Targets (no grad)
        with torch.no_grad():
            # Non-causal info flows into target networks. Use IB to limit excessive leakage.
            z1t = self.f_targ(v1, is_causal=False)
            z2t = self.f_targ(v2, is_causal=False)
            if self.use_information_bottleneck:
                z1t, _ = self._apply_information_bottleneck(z1t, sample=True) 
                z2t, _ = self._apply_information_bottleneck(z2t, sample=True)
            p1t = self.g_targ(z1t)
            p2t = self.g_targ(z2t)
        return h1, h2, p1t, p2t

    def _apply_information_bottleneck(self, z: torch.Tensor, *, sample: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_information_bottleneck or self.ib_head is None:
            return z, z.new_zeros(())
        mu, logvar = self.ib_head(z).chunk(2, dim=-1)
        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_sample = mu + std * eps
            var = std.pow(2)
            kl = 0.5 * (mu.pow(2) + var - logvar - 1.0).sum(dim=-1).mean()
        else:
            z_sample = mu
            kl = mu.new_zeros(())
        return z_sample, kl

    def loss(self,
             v1: torch.Tensor,
             v2: torch.Tensor) -> torch.Tensor:
        h1, h2, p1t, p2t = self._forward_pair(v1, v2)
        loss_12 = (l2norm(h1) - l2norm(p2t)).pow(2).sum(-1).mean()
        loss_21 = (l2norm(h2) - l2norm(p1t)).pow(2).sum(-1).mean()
        loss = loss_12 + loss_21
        if self.use_information_bottleneck and self._last_ib_kl is not None:
            loss += self.beta_ib * self._last_ib_kl
        return loss

    @torch.no_grad()
    def mismatch_per_window(self,
                            v1: torch.Tensor,
                            v2: torch.Tensor) -> torch.Tensor:
        h1, h2, p1t, p2t = self._forward_pair(v1, v2)
        m12 = (l2norm(h1) - l2norm(p2t)).pow(2).sum(-1)
        m21 = (l2norm(h2) - l2norm(p1t)).pow(2).sum(-1)
        return 0.5 * (m12 + m21)

    def online_parameters(self, include_obs_encoder: bool):
        params = (
            list(self.f_online.gru.parameters())
            + list(self.g_online.parameters())
            + list(self.q_online.parameters())
        )
        if self.use_information_bottleneck and self.ib_head is not None:
            params = list(self.ib_head.parameters()) + params
        if include_obs_encoder:
            params = list(self.f_online.obs_encoder.parameters()) + params
        return params

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
Index = Union[slice, List[int]]

@dataclass
class ObsGroup:
    """Index map for [B, W, D_obs] observation tensor. This is for QuadrupedEnv obs structure."""
    base_lin_vel: Optional[Index] = None
    base_ang_vel: Optional[Index] = None
    proj_gravity: Optional[Index] = None
    cmd_vel: Optional[Index] = None
    joint_pos: Optional[Index] = None
    joint_vel: Optional[Index] = None
    prev_actions: Optional[Index] = None     # joint position commands

    # if any
    rpy: Optional[Index] = None
    privilege_obs: Optional[Index] = None

def _as_view(x: torch.Tensor, idx: Optional[Index]) -> torch.Tensor:
    return x if idx is None else x[..., idx]

def _yaw2Rot(theta: torch.Tensor) -> torch.Tensor:
    """Convert yaw angle(s) to 2D rotation matrix/matrices."""
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.zeros(theta.shape + (3, 3), device=theta.device, dtype=theta.dtype)
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    R[..., 2, 2] = 1.0      # Identity in 3D
    return R

def _apply_yaw_rot_inplace(x: torch.Tensor, obs_group: ObsGroup, yaw: torch.Tensor):
    R = _yaw2Rot(yaw)
    for g in ("base_lin_vel", "base_ang_vel", "cmd_vel", "proj_gravity"):
        idx = getattr(ObsGroup, g)
        v = _as_view(x, idx)
        if v is None:
            continue
        if v.shape[-1] >= 3:
            vxyz = v[..., :3]
            vxyz.copy_(torch.matmul(vxyz.unsqueeze(-2), R).squeeze(-2)) # [B,W,3]

def _sim_slip_inplace(x: torch.Tensor, obs_group: ObsGroup, scale_xy: torch.Tensor):
    # Simulate slip by shrinking base linear velocity in x-y plane. Keep z as is.
    if obs_group.base_lin_vel is None:
        return
    v = _as_view(x, obs_group.base_lin_vel)
    if v is None or v.shape[-1] < 2:
        return
    v[..., 0:2] *= scale_xy # [B,1] boradcast across W

def _sim_imu_drift_inplace(x: torch.Tensor, obs_group: ObsGroup, bias_lin: torch.Tensor, bias_ang: torch.Tensor):
    # Bias over the windows on imu channels
    if obs_group.base_lin_vel is not None:
        _as_view(x, obs_group.base_lin_vel)[..., 0:3] += bias_lin
    if obs_group.base_ang_vel is not None:
        _as_view(x, obs_group.base_ang_vel)[..., 0:3] += bias_ang

def _per_leg_calibration_inplace(x: torch.Tensor, obs_group: ObsGroup, gain_std: float, rng: torch.Generator):
    # Per-leg joint position calibration noise (NOTE: groups should be contiguous. TODO: check this)
    for name in ("joint_pos", "joint_vel"):
        idx = getattr(obs_group, name)
        v = _as_view(x, idx)
        if v is None:
            continue
        B, W, D = v.shape
        # assume 3 DoF per leg * 4 legs (12)
        L = 4 if D % 4 == 0 else 1
        G = torch.randn((B, 1, L), device=x.device, generator=rng, dtype=x.dtype) * gain_std
        if L == 4:
            leg_size = D // 4
            for leg in range(4):
                sl = slice(leg * leg_size, (leg + 1) * leg_size)
                v[..., sl] *= (1.0 + G[..., leg:leg+1])
        else:
            v *= (1.0 + torch.randn((B, 1, 1), device=x.device, generator=rng, dtype=x.dtype) * gain_std)

def _sim_deadzone_inplace(x: torch.Tensor, obs_group: ObsGroup, deadzone: float, step: float):
    # Simulate joint measurement deadzone
    v = _as_view(x, obs_group.joint_vel)
    if v is not None and deadzone > 0:
        mask = v.abs() < deadzone
        v[mask] = 0.0
    vq = _as_view(x, obs_group.joint_pos)
    if vq is not None and step > 0:
        vq.copy_((vq / step).round() * step)

def _sim_sensor_spike_inplace(x: torch.Tensor, obs_group: ObsGroup, spike_prob: float, spike_mag: float, rng: torch.Generator):
    # Simulate random sensor spikes
    for name in ("base_lin_vel", "base_ang_vel"):
        v = _as_view(x, getattr(obs_group, name))
        if v is None:
            continue
        B, W, D = v.shape
        mask = torch.rand((B, W, 1), device=x.device, generator=rng) < spike_prob
        noise = (torch.rand((B, W, D), device=x.device, generator=rng)) * spike_mag
        v.add_(mask * noise)

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

def _time_mask_inplace(x: torch.Tensor, span: int, prob: float, rng: torch.Generator) -> None:
    """Randomly zero a contiguous time span per sequence with probability `prob`.
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
def sample_byol_windows_phy(
    obs: torch.Tensor,                 # [T,N,D_obs]
    dones: torch.Tensor,               # [T,N]
    W: int, B: int,
    obs_group: ObsGroup,
    # general augmentations
    delay: int,
    gaussian_jitter_std: float,
    frame_drop: float,
    # physics-informed augmentations
    yaw_rot_max: float = math.radians(10.0),
    imu_drift_lin_std: float = 0.01,
    imu_drift_ang_std: float = 0.01,
    slip_lin_scale_range: Tuple[float,float] = (0.7, 1.0),
    joint_gain_std: float = 0.01,
    joint_deadzone: float = 0.0,
    joint_pos_step: float = 0.0,
    sensor_spike_prob: float = 0.01,
    sensor_spike_mag: float = 0.05,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int,int]], int]:
    """
    Sample B window pairs (v1, v2) with independent augs.
    """
    W_use = min(int(W), int(obs.shape[0]))
    starts = _valid_window_starts(dones, W_use)
    if len(starts) == 0:
        starts = np.array([(0, n) for n in range(obs.shape[1])], dtype=np.int64)
    replace = len(starts) < B
    idx = np.random.choice(len(starts), size=min(B, len(starts)), replace=replace)
    picks = starts[idx].tolist()

    v1o, v2o = [], []
    T_total = obs.shape[0]
    for (s, n) in picks:
        w1o = obs[s : s + W_use, n, :].clone()
        delta = np.random.randint(-delay, delay + 1) if delay > 0 else 0  # random delay shift
        s2 = max(0, min(s + delta, T_total - W_use))
        w2o = obs[s2 : s2 + W_use, n, :].clone()
        v1o.append(w1o); v2o.append(w2o)
    v1o = torch.stack(v1o, dim=0).to(device)  # [B,W_use,D_obs]
    v2o = torch.stack(v2o, dim=0).to(device)

    g1 = torch.Generator(device=device).manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())
    g2 = torch.Generator(device=device).manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())

    # general augmentations
    if gaussian_jitter_std > 0:
        _feature_jitter(v1o, gaussian_jitter_std, 0.0, g1)
        _feature_jitter(v2o, gaussian_jitter_std, 0.0, g2)
    _frame_drop_inplace(v1o, frame_drop, g1)
    _frame_drop_inplace(v2o, frame_drop, g2)

    # physics-informed augmentations
    def _apply_physics(vx, rng):
        B, W, D = vx.shape
        # sample window-wise random variables
        yaw = (torch.rand((B,1), generator=rng, device=device) * 2 - 1) * yaw_rot_max
        fxy = torch.empty((B,1), device=device).uniform_(*slip_lin_scale_range)
        b_lin = torch.randn((B,1,3), generator=rng, device=device) * imu_drift_lin_std
        b_ang = torch.randn((B,1,3), generator=rng, device=device) * imu_drift_ang_std

        _apply_yaw_rot_inplace(vx, obs_group, yaw)
        _sim_slip_inplace(vx, obs_group, fxy)
        _sim_imu_drift_inplace(vx, obs_group, b_lin, b_ang)
        _per_leg_calibration_inplace(vx, obs_group, joint_gain_std, rng)
        _sim_deadzone_inplace(vx, obs_group, joint_deadzone, joint_pos_step)
        _sim_sensor_spike_inplace(vx, obs_group, sensor_spike_prob, sensor_spike_mag, rng)
    
    _apply_physics(v1o, g1)
    _apply_physics(v2o, g2)

    return v1o, v2o, picks, len(picks)


# @torch.no_grad()
# def sample_byol_windows(
#     obs: torch.Tensor,                 # [T,N,D_obs]
#     dones: torch.Tensor,               # [T,N]
#     W: int, B: int, max_shift: int,
#     noise_std: float, feat_drop: float, frame_drop: float,
#     time_warp_scale: float,
#     # Extra augmentations (all optional; default disabled)
#     ch_drop: float = 0.0,
#     time_mask_prob: float = 0.0,
#     time_mask_span: int = 0,
#     gain_std: float = 0.0,
#     bias_std: float = 0.0,
#     smooth_prob: float = 0.0,
#     smooth_kernel: int = 0,
#     mix_strength: float = 0.0,
#     device: torch.device = torch.device("cpu"),
# ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int,int]], int]:
#     """
#     Sample B window pairs (v1, v2) with independent augs.
#     """
#     W_use = min(int(W), int(obs.shape[0]))
#     starts = _valid_window_starts(dones, W_use)
#     if len(starts) == 0:
#         starts = np.array([(0, n) for n in range(obs.shape[1])], dtype=np.int64)
#     replace = len(starts) < B
#     idx = np.random.choice(len(starts), size=min(B, len(starts)), replace=replace)
#     picks = starts[idx].tolist()

#     v1o, v2o = [], []
#     T_total = obs.shape[0]
#     for (s, n) in picks:
#         w1o = obs[s : s + W_use, n, :].clone()
#         delta = np.random.randint(-max_shift, max_shift + 1) if max_shift > 0 else 0
#         s2 = max(0, min(s + delta, T_total - W_use))
#         w2o = obs[s2 : s2 + W_use, n, :].clone()
#         v1o.append(w1o); v2o.append(w2o)

#     v1o = torch.stack(v1o, dim=0).to(device)  # [B,W_use,D_obs]
#     v2o = torch.stack(v2o, dim=0).to(device)

#     g1 = torch.Generator(device=device).manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())
#     g2 = torch.Generator(device=device).manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())

#     _feature_jitter(v1o, noise_std, feat_drop, g1)
#     _feature_jitter(v2o, noise_std, feat_drop, g2)

#     # calibration (per-channel gain/bias)
#     _calibration_noise_inplace(v1o, gain_std, bias_std, g1)
#     _calibration_noise_inplace(v2o, gain_std, bias_std, g2)

#     # channel-consistent dropout
#     _channel_dropout_inplace(v1o, ch_drop, g1)
#     _channel_dropout_inplace(v2o, ch_drop, g2)

#     # time-span masking
#     _time_mask_inplace(v1o, time_mask_span, time_mask_prob, g1)
#     _time_mask_inplace(v2o, time_mask_span, time_mask_prob, g2)

#     # frame dropout (temporal consistency)
#     _frame_drop_inplace(v1o, frame_drop, g1)
#     _frame_drop_inplace(v2o, frame_drop, g2)

#     if max_shift > 0:
#         # random temporal shift (independent for v1 and v2)
#         v1o = _temporal_shift(v1o, int(torch.randint(-max_shift, max_shift + 1, (1,)).item()))
#         v2o = _temporal_shift(v2o, int(torch.randint(-max_shift, max_shift + 1, (1,)).item()))

#     if time_warp_scale > 0:
#         # time warp (independent for v1 and v2)
#         v1o = _time_warp(v1o, time_warp_scale, g1)
#         v2o = _time_warp(v2o, time_warp_scale, g2)

#     if mix_strength > 0.0:
#         # temporal mixing (independent for v1 and v2)
#         v1o = _temporal_mixing(v1o, mix_strength, g1)
#         v2o = _temporal_mixing(v2o, mix_strength, g2)

#     # optional smoothing at the end
#     _smooth_time_inplace(v1o, smooth_kernel, smooth_prob)
#     _smooth_time_inplace(v2o, smooth_kernel, smooth_prob)

#     return v1o, v2o, picks, len(picks)
