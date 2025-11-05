import math
import torch, torch.nn as nn, numpy as np
import torch.nn.functional as F
from typing import Iterable, Optional, Tuple, List, Union, Dict
from dataclasses import dataclass, fields

def l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)

def layer_init(layer: nn.Linear, std: float = math.sqrt(2.0), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        # nn.BatchNorm1d(512),  # BatchNorm for heterogeneous data
        nn.LayerNorm(256),      # use LayerNorm instead of BatchNorm
        nn.ReLU(inplace=True),
        nn.Linear(256, out_dim),
    )

class ObsEncoder(nn.Module):
    def __init__(self, obs_dim: int, feat_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, feat_dim)),
            nn.ReLU(inplace=True),
        )
        self.out_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class FiLM(nn.Module):
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
    def __init__(self, obs_encoder: ObsEncoder, z_dim: int, output_type: str = "last"):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.gru = nn.GRU(input_size=self.obs_encoder.out_dim,
                          hidden_size=z_dim,
                          batch_first=True)
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 1.0)
                
        if output_type not in ("last", "mean", "attn"):
            raise ValueError(f"Unsupported output_type for SeqEncoder: {output_type}")
        self.output_type = str(output_type)
        if self.output_type == "attn":
            self.attn = layer_init(nn.Linear(z_dim, 1), std=0.1)

    def _encode_frames(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, W, D = obs_seq.shape
        feats = self.obs_encoder(obs_seq.reshape(B * W, D)).reshape(B, W, -1)  # [B,W,F]
        return feats

    def forward(self, obs_seq: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        feats = self._encode_frames(obs_seq)        # [B,W,F]
        out, hT = self.gru(feats)                   # out: [B,W,z], hT: [1,B,z]
        
        if self.output_type == "last":
            return hT.squeeze(0)    # [B,z]
        elif self.output_type == "mean":
            return out.mean(dim=1)   # [B,z] 
        elif self.output_type == "attn":
            attn_weights = torch.softmax(self.attn(out).squeeze(-1), dim=-1)     # [B,W]
            z = (out * attn_weights.unsqueeze(-1)).sum(dim=1)                    # [B,z]
            return z
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")
        
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
    """
    def __init__(self,
                 obs_encoder: ObsEncoder,
                 z_dim: int = 128,
                 proj_dim: int = 128,
                 tau: float = 0.996,
                 output_type: str = "mean",
                 use_transformer: bool = False):
        super().__init__()
        if use_transformer:
            self.f_online = TransformerEncoder(obs_encoder, z_dim=z_dim, output_type=output_type)
        else:
            self.f_online = SeqEncoder(obs_encoder, z_dim=z_dim, output_type=output_type)
        self.g_online = mlp(z_dim, proj_dim)
        self.q_online = mlp(proj_dim, proj_dim)

        import copy
        self.f_targ = copy.deepcopy(self.f_online)
        self.g_targ = copy.deepcopy(self.g_online)
        for p in list(self.f_targ.parameters()) + list(self.g_targ.parameters()):
            p.requires_grad_(False)

        self.tau = float(tau)

    def train(self, mode: bool = True):
        super().train(mode)
        self.f_targ.eval()
        self.g_targ.eval()
        return self

    def infer_ctx(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Infer context vector from observation sequence. obs_seq: [B,W,D_obs] -> [B,z_dim]"""
        return self.f_online(obs_seq)
    
    @torch.no_grad()
    def ema_update(self) -> None:
        for online, targ in ((self.f_online, self.f_targ), (self.g_online, self.g_targ)):
            for p_o, p_t in zip(online.parameters(), targ.parameters()):
                p_t.data.mul_(self.tau).add_(p_o.data, alpha=1.0 - self.tau)

    def _forward_pair(self, v1: torch.Tensor, v2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.f_online(v1)          # [B, z]
        z2 = self.f_online(v2)
        p1 = self.g_online(z1)          # [B, p]
        p2 = self.g_online(z2)
        h1 = self.q_online(p1)          # [B, p]
        h2 = self.q_online(p2)
    
        with torch.no_grad():        # target path
            z1t = self.f_targ(v1)
            z2t = self.f_targ(v2)
            p1t = self.g_targ(z1t)
            p2t = self.g_targ(z2t)
        return h1, h2, p1t, p2t

    def loss(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        h1, h2, p1t, p2t = self._forward_pair(v1, v2)
        loss_12 = (l2norm(h1) - l2norm(p2t)).pow(2).sum(-1).mean()
        loss_21 = (l2norm(h2) - l2norm(p1t)).pow(2).sum(-1).mean()
        return loss_12 + loss_21
    
    def loss_per_sample(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        h1, h2, p1t, p2t = self._forward_pair(v1, v2)
        loss_12 = (l2norm(h1) - l2norm(p2t)).pow(2).sum(-1)
        loss_21 = (l2norm(h2) - l2norm(p1t)).pow(2).sum(-1)
        return 0.5 * (loss_12 + loss_21)

    @torch.no_grad()
    def mismatch_per_window(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
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

def extract_flat_obs_dict(storage, group_names: Iterable[str]) -> Dict[str, torch.Tensor]:
    """Return {group: [T*N, D_g]} robustly, from storage.observations/obs."""
    obs_container = getattr(storage, "observations", None)
    if obs_container is None:
        obs_container = getattr(storage, "obs", None)
    if obs_container is None:
        raise AttributeError("No observations found in storage.")
    T = storage.num_transitions_per_env
    N = storage.num_envs
    out: Dict[str, torch.Tensor] = {}
    def _pull(container, key):
        v = None
        if hasattr(container, "get"):
            try: v = container.get(key)
            except Exception: v = None
        if v is None and isinstance(container, dict):
            v = container.get(key)
        return v
    for g in group_names:
        v = _pull(obs_container, g)
        if v is None:
            # try nested views
            for parent in ("policy", "critic"):
                parent_view = _pull(obs_container, parent)
                if parent_view is None:
                    continue
                vv = _pull(parent_view, g)
                if vv is None and isinstance(parent_view, dict) and g in parent_view:
                    vv = parent_view[g]
                if torch.is_tensor(vv):
                    v = vv; break
        if not torch.is_tensor(v):
            raise KeyError(f"Cannot locate observation group '{g}' in storage.")
        if v.dim() == 2:
            v = v.unsqueeze(-1)
        out[g] = v.flatten(0, 1)  # [T*N, D_g]
    return out

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
    prev_actions: Optional[Index] = None     

    # privilege_obs (not directly accessible to policy)
    privilege_obs: Optional[Dict[str, Index]] = None

UNITREE_GO2_ROUGH_OBS_GROUP = ObsGroup(
    base_lin_vel=slice(0, 3),
    base_ang_vel=slice(3, 6),
    proj_gravity=slice(6, 9),
    cmd_vel=slice(9, 12),
    joint_pos=slice(12, 24),
    joint_vel=slice(24, 36),
    prev_actions=slice(36, 48),
)

@dataclass
class PrivInfoGroup:
    terrain_level: Optional[str] = "Curriculum/terrain_levels"
    vel_error_xy: Optional[str] = "Metrics/base_velocity/error_vel_xy"
    vel_error_yaw: Optional[str] = "Metrics/base_velocity/error_vel_yaw"
    time_out: Optional[str] = "Episode_Termination/time_out"
    base_contact: Optional[str] = "Episode_Termination/base_contact"

def _extract_extra_value(extras: Dict, key: str):
    """Fetch value from extras using either flat or hierarchical keys."""
    if not isinstance(extras, dict):
        return None
    if key in extras:
        return extras[key]
    node = extras
    for token in key.split("/"):
        if isinstance(node, dict) and token in node:
            node = node[token]
        else:
            return None
    return node

def _value_to_tensor(value, num_envs: int, device: torch.device) -> Optional[torch.Tensor]:
    """Normalize scalars from extras into [N,1] tensors on the desired device."""
    if isinstance(value, torch.Tensor):
        data = value.to(device)
    elif isinstance(value, (float, int)):
        data = torch.full((num_envs,), float(value), device=device)
    else:
        return None
    if data.dim() == 0:
        data = data.expand(num_envs)
    if data.shape[0] != num_envs:
        data = data.reshape(num_envs, -1)
        data = data[:, 0]
    return data.view(num_envs, 1)


class PrivInfoBuffer:
    """Ring buffer that mirrors rollout storage for privileged per-step scalars."""
    def __init__(self, group: PrivInfoGroup, horizon: int, num_envs: int, device: torch.device):
        self.device = device
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.paths: Dict[str, str] = {}
        self.buffers: Dict[str, torch.Tensor] = {}
        for field in fields(PrivInfoGroup):
            path = getattr(group, field.name)
            if path is None:
                continue
            self.paths[field.name] = str(path)
            self.buffers[field.name] = torch.zeros(self.horizon, self.num_envs, 1, device=device, dtype=torch.float32)
        self.active = len(self.buffers) > 0

    def record(self, step_idx: int, extras: Dict) -> None:
        if not self.active:
            return
        if step_idx < 0 or step_idx >= self.horizon:
            return
        for name, key in self.paths.items():
            value = _extract_extra_value(extras, key)
            if value is None:
                continue
            data = _value_to_tensor(value, self.num_envs, self.device)
            if data is None:
                continue
            self.buffers[name][step_idx].copy_(data)

    def get(self, name: str) -> Optional[torch.Tensor]:
        return self.buffers.get(name)

    def zero_(self) -> None:
        for buf in self.buffers.values():
            buf.zero_()
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        return {name: buf.clone() for name, buf in self.buffers.items()}

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
        idx = getattr(obs_group, g)
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
    scale = scale_xy
    while scale.dim() < v.dim():
        scale = scale.unsqueeze(-1)
    v[..., 0:2] *= scale  # broadcast along window/time dims

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

def _randn_like(x: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    try:
        return torch.randn_like(x, generator=rng)
    except TypeError:
        return torch.randn(x.shape, device=x.device, dtype=x.dtype)

def _feature_jitter(x: torch.Tensor, noise_std: float, feat_drop: float, rng: torch.Generator) -> None:
    """In-place: Gaussian noise + Bernoulli feature dropout per frame. x: [B,W,D]"""
    if noise_std > 0:
        noise = _randn_like(x, rng) * noise_std
        x.add_(noise)
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
    obs_group: Optional[ObsGroup] = UNITREE_GO2_ROUGH_OBS_GROUP,
    # general augmentations
    delay: int = 0,
    gaussian_jitter_std: float = 0.0,
    frame_drop: float = 0.0,
    # physics-informed augmentations
    yaw_rot_max: float = math.radians(10.0),
    imu_drift_lin_std: float = 0.02,
    imu_drift_ang_std: float = 0.02,
    slip_lin_scale_range: Tuple[float,float] = (0.7, 1.0),
    joint_gain_std: float = 0.01,
    joint_deadzone: float = 0.0,
    joint_pos_step: float = 0.0,
    sensor_spike_prob: float = 0.05,
    sensor_spike_mag: float = 0.05,
    device: torch.device = torch.device("cpu"),
    priv: Optional["PrivInfoBuffer"] = None,
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
    env_ids, starts_v1, starts_v2 = [], [], []
    T_total = obs.shape[0]
    for (s, n) in picks:
        w1o = obs[s : s + W_use, n, :].clone()
        delta = np.random.randint(-delay, delay + 1) if delay > 0 else 0  # random delay shift
        s2 = max(0, min(s + delta, T_total - W_use))
        w2o = obs[s2 : s2 + W_use, n, :].clone()
        v1o.append(w1o); v2o.append(w2o)
        env_ids.append(n)
        starts_v1.append(s)
        starts_v2.append(s2)
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

    def _gather_priv_windows(starts: List[int]) -> Dict[str, torch.Tensor]:
        if priv is None or not getattr(priv, "active", False) or len(starts) == 0:
            return {}
        data: Dict[str, torch.Tensor] = {}
        for name, buf in priv.buffers.items():
            windows = []
            for env, s in zip(env_ids, starts):
                window = buf[s : s + W_use, env, :]  # [W_use,1]
                windows.append(window)
            data[name] = torch.stack(windows, dim=0).to(device)
        return data

    priv_v1 = _gather_priv_windows(starts_v1)
    priv_v2 = _gather_priv_windows(starts_v2)

    def _apply_physics(vx: torch.Tensor, rng: torch.Generator, priv_windows: Dict[str, torch.Tensor]) -> None:
        if obs_group is None:
            return
        B, W, _ = vx.shape
        dev = vx.device

        def _priv_strength(name: str) -> Optional[torch.Tensor]:
            buf = priv_windows.get(name)
            if buf is None:
                return None
            return torch.tanh(buf.mean(dim=1)).to(device=dev, dtype=vx.dtype)

        terrain = _priv_strength("terrain_level")
        vel_xy = _priv_strength("vel_error_xy")
        vel_yaw = _priv_strength("vel_error_yaw")
        timeout = _priv_strength("time_out")
        base_contact = _priv_strength("base_contact")

        # yaw perturbation follows yaw tracking error / terrain difficulty
        if yaw_rot_max > 0:
            yaw_scale = torch.full((B, 1), yaw_rot_max, device=dev, dtype=vx.dtype)
            if vel_yaw is not None:
                yaw_scale *= (0.4 + 0.6 * vel_yaw.abs().clamp_max(1.0))
            elif terrain is not None:
                yaw_scale *= (0.6 + 0.4 * terrain.abs().clamp_max(1.0))
            yaw_noise = (_rand_gen((B, 1), dev, rng) * 2.0 - 1.0) * yaw_scale  # [B,1]
            yaw = yaw_noise.expand(-1, W)
            _apply_yaw_rot_inplace(vx, obs_group, yaw)

        # slope-aware gravity tilt informed by terrain difficulty & impacts
        g_slice = _as_view(vx, obs_group.proj_gravity) if obs_group.proj_gravity is not None else None
        if g_slice is not None and g_slice.shape[-1] >= 2:
            tilt_ctrl = torch.zeros((B, 1), device=dev, dtype=vx.dtype)
            if terrain is not None:
                tilt_ctrl += terrain.relu().clamp_max(1.0)
            if base_contact is not None:
                tilt_ctrl += base_contact.relu().clamp_max(1.0)
            tilt_ctrl = torch.clamp(tilt_ctrl, 0.0, 1.0)
            if torch.any(tilt_ctrl > 0):
                tilt_scale = (0.01 + 0.04 * tilt_ctrl).unsqueeze(1)  # [B,1,1]
                tilt_noise = _randn_like(vx.new_zeros(B, W, 2), rng) * tilt_scale
                g_slice[..., :2] += tilt_noise

        # slip scaling informed by XY velocity tracking error / rough terrain
        slip_min, slip_max = slip_lin_scale_range
        slip_range = max(0.0, slip_max - slip_min)
        if slip_range > 0:
            slip_base = slip_min + slip_range * _rand_gen((B, 1), dev, rng)
            slip_ctrl = torch.zeros_like(slip_base, dtype=vx.dtype)
            if vel_xy is not None:
                slip_ctrl += vel_xy.abs().clamp_max(1.0)
            if terrain is not None:
                slip_ctrl += terrain.relu().clamp_max(1.0)
            slip_ctrl = torch.clamp(slip_ctrl, 0.0, 1.0)
            slip_scale = slip_base - (slip_base - slip_min) * slip_ctrl
            _sim_slip_inplace(vx, obs_group, slip_scale.expand(-1, W))

        # IMU drift driven by tracking errors / impending timeouts
        if imu_drift_lin_std > 0 or imu_drift_ang_std > 0:
            ctrl_lin = torch.ones((B, 1), device=dev, dtype=vx.dtype)
            ctrl_ang = torch.ones((B, 1), device=dev, dtype=vx.dtype)
            if vel_xy is not None:
                ctrl_lin += 0.5 * vel_xy.abs().clamp_max(1.0)
            if vel_yaw is not None:
                ctrl_ang += 0.5 * vel_yaw.abs().clamp_max(1.0)
            if timeout is not None:
                ctrl_lin += 0.25 * timeout.relu().clamp_max(1.0)
                ctrl_ang += 0.25 * timeout.relu().clamp_max(1.0)
            bias_lin = _randn_like(vx.new_zeros(B, 1, 3), rng) * imu_drift_lin_std
            bias_ang = _randn_like(vx.new_zeros(B, 1, 3), rng) * imu_drift_ang_std
            bias_lin *= ctrl_lin.unsqueeze(-1)
            bias_ang *= ctrl_ang.unsqueeze(-1)
            _sim_imu_drift_inplace(vx, obs_group, bias_lin, bias_ang)

        # More calibration noise once robot stability degrades
        if joint_gain_std > 0:
            gain_scale = 1.0
            if base_contact is not None:
                gain_scale += 0.5 * float(base_contact.abs().mean().item())
            if timeout is not None:
                gain_scale += 0.25 * float(timeout.relu().mean().item())
            _per_leg_calibration_inplace(vx, obs_group, joint_gain_std * gain_scale, rng)

        if joint_deadzone > 0 or joint_pos_step > 0:
            _sim_deadzone_inplace(vx, obs_group, joint_deadzone, joint_pos_step)

        # command dropout under poor tracking / timeouts
        cmd_slice = _as_view(vx, obs_group.cmd_vel) if obs_group.cmd_vel is not None else None
        if cmd_slice is not None:
            drop_ctrl = torch.zeros((B, 1), device=dev, dtype=vx.dtype)
            if timeout is not None:
                drop_ctrl += timeout.relu().clamp_max(1.0)
            if vel_xy is not None:
                drop_ctrl += vel_xy.abs().clamp_max(1.0)
            drop_prob = torch.clamp(0.05 + 0.25 * drop_ctrl, 0.0, 0.6)
            if torch.any(drop_prob > 0):
                drop_mask = (_rand_gen((B, W), dev, rng) < drop_prob).unsqueeze(-1)
                cmd_slice.mul_(1.0 - drop_mask.to(dtype=cmd_slice.dtype))

        # actuator stiction when impacts accumulate
        stiction_ctrl = torch.zeros((B, 1), device=dev, dtype=vx.dtype)
        if base_contact is not None:
            stiction_ctrl += base_contact.relu().clamp_max(1.0)
        if vel_xy is not None:
            stiction_ctrl += 0.5 * vel_xy.abs().clamp_max(1.0)
        stiction_scale = torch.clamp(1.0 - 0.4 * torch.clamp(stiction_ctrl, 0.0, 1.0), 0.2, 1.0)
        if torch.any(stiction_scale < 1.0):
            for name in ("joint_vel", "prev_actions"):
                idx = getattr(obs_group, name)
                v = _as_view(vx, idx)
                if v is None:
                    continue
                scale = stiction_scale
                while scale.dim() < v.dim():
                    scale = scale.unsqueeze(-1)
                v.mul_(scale)

        if sensor_spike_prob > 0 and sensor_spike_mag > 0:
            spike_prob = sensor_spike_prob
            spike_mag = sensor_spike_mag
            if base_contact is not None:
                spike_prob *= float((1.0 + base_contact.abs().mean()).clamp(max=2.0).item())
            if timeout is not None:
                spike_prob *= float((1.0 + timeout.relu().mean()).clamp(max=2.0).item())
            if vel_xy is not None:
                spike_mag *= float((1.0 + vel_xy.abs().mean()).clamp(max=2.0).item())
            spike_prob = min(spike_prob, 1.0)
            _sim_sensor_spike_inplace(vx, obs_group, spike_prob, spike_mag, rng)

    _apply_physics(v1o, g1, priv_v1)
    _apply_physics(v2o, g2, priv_v2)

    return v1o, v2o, picks, len(picks)
