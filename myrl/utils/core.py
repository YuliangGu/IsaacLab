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

class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverseFn.apply(x, lambd)

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
                          batch_first=True,
                          bidirectional=False) # experimental: bidirectional GRU
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
        if output_type not in ("last", "mean"):
            raise ValueError(f"Unsupported output_type for SeqEncoder: {output_type}")
        self.output_type = str(output_type)

    def _encode_frames(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, W, D = obs_seq.shape
        feats = self.obs_encoder(obs_seq.reshape(B * W, D)).reshape(B, W, -1)  # [B,W,F]
        return feats

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        feats = self._encode_frames(obs_seq)        # [B,W,F]
        out, hT = self.gru(feats)                   # out: [B,W,z], hT: [1,B,z]
        
        if self.output_type == "last":
            return hT.squeeze(0)    # [B,z]
        elif self.output_type == "mean":
            return out.mean(dim=1)   # [B,z]
        else:
            raise ValueError(f"Unknown output_type: {self.output_type}")
        
class BYOLSeq(nn.Module):
    """BYOL for sequential data (BYOL-Seq)."""
    def __init__(self,
                 obs_encoder: ObsEncoder,
                 z_dim: int = 128,
                 proj_dim: int = 128,
                 tau: float = 0.996,
                 output_type: str = "mean"):
        super().__init__()
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
        return self.f_online(obs_seq)  # [B,W,D_obs] -> [B,z_dim]
    
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
        seq_params: List[torch.Tensor] = []
        for name, param in self.f_online.named_parameters():
            if (not include_obs_encoder) and name.startswith("obs_encoder"):
                continue
            seq_params.append(param)
        params = seq_params + list(self.g_online.parameters()) + list(self.q_online.parameters())
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
    """Return {group: [T*N, D_g]} from storage.observations/obs."""
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
    vel_error_xy: Optional[str] = "privileged/vel_error_xy"
    vel_error_yaw: Optional[str] = "privileged/vel_error_yaw"
    time_out: Optional[str] = "Episode_Termination/time_out"
    base_contact: Optional[str] = "Episode_Termination/base_contact"

    # privileged metrics
    contact_slip: Optional[str] = "privileged/contact_slip"
    torque_rms: Optional[str] = "privileged/torque_rms"
    terrain_roughness: Optional[str] = "privileged/terrain_roughness"

def _extract_extra_value(extras: Dict, key: str):
    """Fetch value from extras using either flat or hierarchical keys."""
    if not isinstance(extras, dict):
        return None

    def _probe(container: Dict):
        if key in container:
            return container[key]
        node = container
        for token in key.split("/"):
            if isinstance(node, dict) and token in node:
                node = node[token]
            else:
                return None
        return node

    for candidate in (extras, extras.get("log")):
        if isinstance(candidate, dict):
            value = _probe(candidate)
            if value is not None:
                return value
    return None

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
        if data.numel() == 1:
            data = data.reshape(1).expand(num_envs)
        else:
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
    """Sticky frames: with probability p, replace frame t with frame t-1."""
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

def _actuator_lag_inplace(x: torch.Tensor, obs_group: ObsGroup, lag_prob: float, rng: torch.Generator):
    """Apply temporal dropout ONLY to the prev_actions slice to mimic actuator lag."""
    if lag_prob <= 0 or obs_group.prev_actions is None:
        return
    v = _as_view(x, obs_group.prev_actions)  # [B,W,D_a]
    if v is not None:
        _frame_drop_inplace(v, lag_prob, rng)

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
    joint_pos_step: float = 0.0,
    sensor_spike_prob: float = 0.05,
    sensor_spike_mag: float = 0.05,
    device: torch.device = torch.device("cpu"),
    priv: Optional["PrivInfoBuffer"] = None,
    return_weights: bool = False,

    slip_norm: float = 0.5,            # m/s
    torque_norm: float = 40.0,         # N·m RMS
    rough_norm: float = 0.05,          # meters RMS height
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, List[Tuple[int,int]], int],
    Tuple[torch.Tensor, torch.Tensor, List[Tuple[int,int]], int, torch.Tensor],
]:
    """
    Sample B window pairs (v1, v2) with independent augs.

    When `return_weights` is True, also return per-window weights derived from the
    privileged `vel_error_xy` and `vel_error_yaw` buffers (defaults to uniform
    weights when the signals are missing).
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

    # Gaussian feature jitter + temporal dropout + sticky frames
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

    weights: Optional[torch.Tensor] = None
    if return_weights:
        size = len(picks)
        if size == 0:
            weights = torch.empty(0, device=device)
        else:
            vel_err = priv_v1.get("vel_error_xy")
            yaw_err = priv_v1.get("vel_error_yaw")
            if vel_err is None and yaw_err is None:
                weights = torch.ones(size, device=device)
            else:
                comps = []
                if vel_err is not None:
                    comps.append(vel_err.mean(dim=(1, 2)))
                if yaw_err is not None:
                    comps.append(yaw_err.mean(dim=(1, 2)))
                score = torch.stack(comps, dim=0).sum(dim=0)
                score = torch.clamp(score, min=1e-6)
                weights = score / score.mean().clamp(min=1e-6)

    def _apply_physics(vx: torch.Tensor, rng: torch.Generator, priv_windows: Dict[str, torch.Tensor]) -> None:
        if obs_group is None:
            return
        B, W, _ = vx.shape
        dev = vx.device

        def _strength(name: str, norm: float) -> Optional[torch.Tensor]:
            buf = priv_windows.get(name)
            if buf is None:
                return None
            val = buf.mean(dim=1)
            return torch.tanh(val / max(1e-6, norm)).clamp(0, 1).to(device=dev, dtype=vx.dtype)

        s_slip  = _strength("contact_slip",    slip_norm)
        s_torque= _strength("torque_rms",      torque_norm)
        s_rough = _strength("terrain_roughness", rough_norm)

        # 1) Friction: slip scaling + mild yaw noise from slip
        if s_slip is not None:
            slip_min, slip_max = slip_lin_scale_range
            base = slip_max - (slip_max - slip_min) * s_slip     # more slip -> more shrink
            _sim_slip_inplace(vx, obs_group, base.expand(-1, W))
            # yaw jitter from slip (cap at yaw_rot_max)
            yaw_scale = yaw_rot_max * (0.2 + 0.8 * s_slip)       # [B,1]
            yaw = (_rand_gen((B, 1), dev, rng) * 2 - 1) * yaw_scale
            _apply_yaw_rot_inplace(vx, obs_group, yaw.expand(-1, W))

        # 2) Geometry: gravity tilt noise & small yaw from roughness
        if s_rough is not None:
            g_slice = _as_view(vx, obs_group.proj_gravity)
            if g_slice is not None and g_slice.shape[-1] >= 2:
                tilt_scale = (0.005 + 0.03 * s_rough).unsqueeze(1)      # [B,1,1]
                g_slice[..., :2] += _randn_like(vx.new_zeros(B, W, 2), rng) * tilt_scale
            yaw_scale = yaw_rot_max * (0.1 + 0.6 * s_rough)
            yaw = (_rand_gen((B, 1), dev, rng) * 2 - 1) * yaw_scale
            _apply_yaw_rot_inplace(vx, obs_group, yaw.expand(-1, W))

        # 3) IMU drift grows with slip & roughness (tracking gets worse)
        if imu_drift_lin_std > 0 or imu_drift_ang_std > 0:
            ctrl = ( (s_slip if s_slip is not None else 0.0)
                   + (s_rough if s_rough is not None else 0.0) )
            ctrl = torch.clamp(ctrl, 0, 1)
            bias_lin = _randn_like(vx.new_zeros(B, 1, 3), rng) * (imu_drift_lin_std * (0.5 + 0.5 * ctrl)).unsqueeze(-1)
            bias_ang = _randn_like(vx.new_zeros(B, 1, 3), rng) * (imu_drift_ang_std * (0.5 + 0.5 * ctrl)).unsqueeze(-1)
            _sim_imu_drift_inplace(vx, obs_group, bias_lin, bias_ang)
        
        # 4) Actuation: stiction + lag increases with torque RMS
        if s_torque is not None:
            stiction = torch.clamp(1.0 - 0.4 * s_torque, 0.3, 1.0)   # shrink v_joint & prev_actions
            for name in ("joint_vel", "prev_actions"):
                v = _as_view(vx, getattr(obs_group, name))
                if v is not None:
                    scale = stiction
                    while scale.dim() < v.dim():
                        scale = scale.unsqueeze(-1)
                    v.mul_(scale)
            # actuator lag as temporal dropout on prev_actions only
            _actuator_lag_inplace(vx, obs_group, lag_prob=0.05 + 0.25 * s_torque.mean(), rng=rng)
            # small quantization/deadzone
            _sim_deadzone_inplace(vx, obs_group, deadzone=0.0 + 0.02 * s_torque.mean().item(),
                                  step=joint_pos_step)
        
        # 5) Rare spikes stronger when slip is high (impacts/micro-bounces)
        slip_mean = float(s_slip.mean().item()) if s_slip is not None else 0.0
        slip_scale = min(2.0, 1.0 + slip_mean)
        spike_prob = sensor_spike_prob * slip_scale
        spike_mag  = sensor_spike_mag  * slip_scale
        _sim_sensor_spike_inplace(vx, obs_group, min(spike_prob, 1.0), spike_mag, rng)

    _apply_physics(v1o, g1, priv_v1)
    _apply_physics(v2o, g2, priv_v2)

    if return_weights:
        return v1o, v2o, picks, len(picks), weights
    return v1o, v2o, picks, len(picks)
