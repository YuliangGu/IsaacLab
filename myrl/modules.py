"""Actor-Critic and Augmented Actor-Critic models 
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization
from myrl.utils_core import ObsEncoder, ActionEncoder, FiLM

""" This is the original RSL-RL ActorCritic (copied here for extension) """
class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        # get the observation dimensions
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        """ NEW: store the obs and act dim for later use """
        self._obs_dim = num_actor_obs  # assume actor and critic share same encoder input dim
        self._act_dim = int(num_actions)

        # actor
        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        # actor observation normalization
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        print(f"Actor MLP: {self.actor}")

        # critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        # compute mean
        mean = self.actor(obs)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        return self.actor(obs)

    def evaluate(self, obs, **kwargs):
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        return self.critic(obs)

    def get_actor_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["critic"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """

        super().load_state_dict(state_dict, strict=strict)
        return True  # training resumes

class ActorCriticAug(ActorCritic):
    """
    Augmented ActorCritic with:
      - shared ObsEncoder feeding both heads
      - optional previous-action additive features
      - optional context 'c_t' via FiLM or concat
    """
    def __init__(self, *base_args,
                 use_prev_action: bool = False,
                 ctx_mode: str = "concat",      # 'none'|'concat'|'film'
                 ctx_dim: int = 0,
                 feat_dim: Optional[int] = 128,  # ObsEncoder output dim; defaults to obs dim
                 **kwargs):
        """Augmented ActorCritic compatible with rsl-rl.

        Args:
            use_prev_action (bool): Whether to use previous action as input.
            ctx_mode (str): Context mode to use ('none', 'concat', 'film').
            ctx_dim (int): Dimension of the context vector.
            feat_dim (Optional[int]): Feature dimension for the encoder. If None, defaults to observation dimension.

        Description:
            Features: x_t -> encoder -> z_t
            contexts: c_t (from BYOL GRU, set externally)
            fusion: 
                - none: z_t
                - concat: [z_t, c_t]
                - film: FiLM(z_t, c_t)
        """

        # Stash model hparams we need to rebuild heads after adding encoder
        actor_hidden_dims = kwargs.get("actor_hidden_dims", [256, 256, 256])
        critic_hidden_dims = kwargs.get("critic_hidden_dims", [256, 256, 256])
        activation = kwargs.get("activation", "elu")

        # Build base to populate dims, noise params, and group logic
        super().__init__(*base_args, **kwargs)

        self._feat_dim = feat_dim
        # Shared encoder and optional conditioners
        out_dim = int(self._obs_dim if feat_dim is None else feat_dim)
        self.encoder = ObsEncoder(self._obs_dim, feat_dim=out_dim)
        self.use_prev_action = bool(use_prev_action)
        self.ctx_mode = str(ctx_mode)
        self.ctx_dim = int(ctx_dim)

        # optional modules
        self.aenc = ActionEncoder(self._act_dim, self.encoder.out_dim) if self.use_prev_action else None
        self.film = FiLM(self.ctx_dim, self.encoder.out_dim) if (self.ctx_mode == "film" and self.ctx_dim > 0) else None

        # compute head input dim 
        head_in = self.encoder.out_dim + (self.ctx_dim if (self.ctx_mode == "concat" and self.ctx_dim > 0) else 0)

        # Rebuild actor/critic heads to consume shared features
        self.actor = MLP(head_in, self._act_dim, actor_hidden_dims, activation)
        self.critic = MLP(head_in, 1, critic_hidden_dims, activation)

        # Re-init observation normalizers to match feature dims
        if self.actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(head_in)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        if self.critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(head_in)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Runtime inputs set by algorithm (optional)
        self._ctx: Optional[torch.Tensor] = None
        self._prev_action: Optional[torch.Tensor] = None

        # print workflow diagram
        print("=================== PPO_BYOL WORKFLOW ====================")
        self._draw_model_diagram()
        print("========================================================")
    
    def _draw_model_diagram(self):
        """Print a detailed ASCII diagram of the end-to-end model workflow.

        Includes:
        - Policy path (shared ObsEncoder -> optional prev-action -> optional context via FiLM/concat -> heads)
        - BYOL path (sequence encoder + GRU -> online/target projection; context c_t comes from algorithm)
        """
        obs_dim = getattr(self, "_obs_dim", None)
        act_dim = getattr(self, "_act_dim", None)
        feat_dim = getattr(self.encoder, "out_dim", None) if hasattr(self, "encoder") else None
        ctx_dim = int(getattr(self, "ctx_dim", 0))
        head_in = feat_dim + (ctx_dim if (self.ctx_mode == "concat" and ctx_dim > 0) else 0)

        def fmt_dim(x):
            return "?" if x is None else str(int(x))

        lines: list[str] = []

        # Section: Policy path
        lines.append("POLICY PATH")
        lines.append(f"  x_t [N,{fmt_dim(obs_dim)}] --ObsEncoder--> z_t [N,{fmt_dim(feat_dim)}]")
        if self.use_prev_action:
            lines.append(f"  a_(t-1) [N,{fmt_dim(act_dim)}] --ActionEncoder--> e_a [N,{fmt_dim(feat_dim)}]")
            lines.append("                                  + (add) --> z_t'")
        if self.ctx_mode == "film" and self.film is not None and ctx_dim > 0:
            lines.append(f"  c_t [N,{fmt_dim(ctx_dim)}] --FiLM(z_t, c_t)--> z*_t [N,{fmt_dim(feat_dim)}]")
            head_in_now = feat_dim
        elif self.ctx_mode == "concat" and ctx_dim > 0:
            lines.append(f"  c_t [N,{fmt_dim(ctx_dim)}] --concat--> z*_t [N,{fmt_dim(head_in)}]")
            head_in_now = head_in
        else:
            lines.append("  (no context)  z*_t = z_t")
            head_in_now = feat_dim

        lines.append(f"  z*_t [N,{fmt_dim(head_in_now)}] --ActorMLP--> μ, σ -> a_t")
        lines.append(f"  z*_t [N,{fmt_dim(head_in_now)}] --CriticMLP--> V(s_t)")

        # Section: BYOL path (informational)
        lines.append("")
        lines.append("BYOL PATH (algorithm side; provides c_t)")
        lines.append("  Window W of x_{t−W+1..t} (and optional a_{t−W..t−1})")
        lines.append(f"    └─> ObsEncoder (shared or separate) -> frames [B,W,{fmt_dim(feat_dim)}]")
        lines.append(f"         └─> GRU -> z_seq [B,{fmt_dim(ctx_dim if ctx_dim > 0 else 'z_dim')}] = c_t")
        lines.append("         ├─ online:  g_online -> q_online")
        lines.append("         └─ target:  f_targ (EMA), g_targ (EMA)")
        lines.append("     Loss:  BYOL(q_online, stopgrad(q_target))")
        lines.append("     Note: c_t is fed into policy (FiLM/concat) each step.")

        print("\n".join(lines))

    def _features(self, x: torch.Tensor, a_prev: torch.Tensor | None, c: torch.Tensor | None) -> torch.Tensor:
        """ shared feature extractor with optional prev-action and context """
        z = self.encoder(x)  # [B,F]
        if self.use_prev_action and a_prev is not None:
            z = z + self.aenc(a_prev)  # additive into the same feature space
        if self.film is not None:
            z = self.film(z, c)
        if self.ctx_mode == "concat" and self.ctx_dim > 0 and c is not None:
            z = torch.cat([z, c], dim=-1)
        return z

    def get_actor_obs_raw(self, obs):
        base = super().get_actor_obs(obs) # get base obs
        return base

    def get_critic_obs_raw(self, obs):
        base = super().get_critic_obs(obs) # get base obs
        return base

    # --- rsl-rl observation hooks ---
    def get_actor_obs(self, obs):
        base = self.get_actor_obs_raw(obs)

        a_prev = self._prev_action
        c = self._ctx
        if a_prev is not None and a_prev.shape[0] != base.shape[0]:
            if a_prev.shape[0] == 1:
                a_prev = a_prev.expand(base.shape[0], -1)
            else:
                a_prev = None
        if c is not None and c.shape[0] != base.shape[0]:
            if c.shape[0] == 1:
                c = c.expand(base.shape[0], -1)
            else:
                c = None
        return self._features(base, a_prev, c)

    def get_critic_obs(self, obs):
        base = self.get_critic_obs_raw(obs)
        a_prev = self._prev_action
        c = self._ctx
        if a_prev is not None and a_prev.shape[0] != base.shape[0]:
            if a_prev.shape[0] == 1:
                a_prev = a_prev.expand(base.shape[0], -1)
            else:
                a_prev = None
        if c is not None and c.shape[0] != base.shape[0]:
            if c.shape[0] == 1:
                c = c.expand(base.shape[0], -1)
            else:
                c = None
        return self._features(base, a_prev, c)

    # --- normalization updates over features ---
    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            z_a = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(z_a)
        if self.critic_obs_normalization:
            z_c = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(z_c)

    # --- rollout-time hooks used by PPOWithBYOL ---
    def set_belief(self, c_t: Optional[torch.Tensor]):
        self._ctx = c_t

    def set_prev_action(self, a_prev: Optional[torch.Tensor]):
        self._prev_action = a_prev

    def _ensure_built(self, obs_actor: torch.Tensor):
        # present for compatibility; this class builds everything at init
        return None


# # --- policy with shared encoder & belief injection ---
# class BYOLActorCritic(ActorCritic):
#     """
#     ActorCritic with:
#       - shared ObsEncoder feeding both heads
#       - optional previous-action additive features
#       - optional context 'c_t' via FiLM (concat avoided to keep head dims stable)
#     The BYOL GRU/belief is not here — the algorithm calls .set_belief(c_t) each step.
#     """
#     def __init__(self, *base_args,
#                  use_prev_action: bool = False,
#                  ctx_mode: str = "film",      # 'none'|'concat'|'film'
#                  ctx_dim: int = 0,
#                  **kwargs):
#         # Forward all base args/kwargs to ActorCritic to match RSL-RL API exactly
#         super().__init__(*base_args, **kwargs)

#         # policy-side options
#         self.use_prev_action = bool(use_prev_action)
#         # NOTE: 'concat' silently mapped to 'film' before; keep that for now but make it explicit.
#         self.ctx_mode = "film" if str(ctx_mode) == "concat" else str(ctx_mode)
#         self.ctx_dim = int(ctx_dim)
#         # components are built lazily once we know actor input dim
#         self._obs_dim: Optional[int] = None
#         self.encoder: Optional[nn.Module] = None
#         self.aenc: Optional[nn.Module] = None
#         self.film: Optional[nn.Module] = None
#         # slot for external belief c_t (set by algorithm each step)
#         self._ctx: Optional[torch.Tensor] = None
#         self._prev_action: Optional[torch.Tensor] = None

#     def _infer_actor_input_dim(self) -> int:
#         if hasattr(self, "actor_mlp"):
#             for m in self.actor_mlp.modules():
#                 if isinstance(m, nn.Linear):
#                     return int(m.in_features)
#         raise RuntimeError("Could not infer actor input dim from model; ensure obs tensor is provided.")

#     def _ensure_built(self, obs_actor: torch.Tensor):
#         if self.encoder is not None:
#             return
#         obs_dim = int(obs_actor.shape[-1]) if hasattr(obs_actor, 'shape') else None
#         if obs_dim is None:
#             obs_dim = self._infer_actor_input_dim()
#         self._obs_dim = obs_dim
#         # Keep output dim equal to actor input dim to preserve head shapes
#         self.encoder = ObsEncoder(obs_dim, feat_dim=obs_dim)
#         # pick device from obs or any parameter in the base model
#         if hasattr(obs_actor, "device"):
#             dev = obs_actor.device
#         else:
#             try:
#                 dev = next(self.parameters()).device
#             except Exception:
#                 dev = torch.device("cpu")
#         self.encoder.to(dev)
#         if self.use_prev_action:
#             # infer action dimension from base attribute if available
#             act_dim = None
#             if hasattr(self, "num_actions"):
#                 try:
#                     act_dim = int(getattr(self, "num_actions"))
#                 except Exception:
#                     act_dim = None
#             if act_dim is None and hasattr(self, "actor_mlp"):
#                 for m in self.actor_mlp.modules():
#                     if isinstance(m, nn.Linear):
#                         act_dim = int(m.out_features)
#             if act_dim is None:
#                 raise RuntimeError("Could not infer action dimension; set use_prev_action=False or expose num_actions.")
#             self.aenc = ActionEncoder(act_dim, feat_dim=obs_dim)
#             self.aenc.to(dev)
#         if self.ctx_mode == "film" and self.ctx_dim > 0:
#             self.film = FiLM(self.ctx_dim, feat_dim=obs_dim)
#             self.film.to(dev)

#     # called by algorithm each step (before act)
#     def set_belief(self, c_t: Optional[torch.Tensor]): self._ctx = c_t
#     def set_prev_action(self, a_prev: Optional[torch.Tensor]): self._prev_action = a_prev

#     # feature path shared by actor & critic
#     def _features(self, obs_actor: torch.Tensor):
#         """Return BYOL-conditioned features from normalized actor observations.

#         Computes the backbone encoder under no_grad (trained by BYOL), and
#         applies FiLM (trained by PPO). Handles batch-size/device alignment
#         between rollout-time (N) and update-time (B) shapes gracefully.
#         """
#         self._ensure_built(obs_actor)
#         # 1) Encoder is frozen for PPO gradients:
#         with torch.no_grad():
#             z = self.encoder(obs_actor)
#         # 2) Prev-action conditioning MUST remain learnable by PPO:
#         if self.use_prev_action and (self._prev_action is not None):
#             pa = self._prev_action
#             try:
#                 pa = pa.to(z.device)
#             except Exception:
#                 pass
#             if pa.shape[0] == z.shape[0]:
#                 z = z + self.aenc(pa)
#             elif pa.shape[0] == 1:
#                 z = z + self.aenc(pa.expand(z.shape[0], -1))
#             else:
#                 pass
#         if self.film is not None and (self._ctx is not None):
#             c = self._ctx
#             try:
#                 c = c.to(z.device)
#             except Exception:
#                 pass
#             if c.shape[0] == z.shape[0]:
#                 z = self.film(z, c)
#             elif c.shape[0] == 1:
#                 z = self.film(z, c.expand(z.shape[0], -1))
#             else:
#                 # shape mismatch during minibatch evaluation: skip FiLM
#                 pass
#         return z

#     # Integrate BYOL features by overriding observation extraction hooks
#     def get_actor_obs(self, obs):
#         """Return actor observations passed through BYOL features."""
#         base_obs = ActorCritic.get_actor_obs(self, obs)
#         return self._features(base_obs)

#     def get_critic_obs(self, obs):
#         if hasattr(ActorCritic, "get_critic_obs"):
#             base_obs = ActorCritic.get_critic_obs(self, obs)
#         else:
#             base_obs = ActorCritic.get_actor_obs(self, obs)
#         return self._features(base_obs)
