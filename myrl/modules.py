"""Actor-Critic and Augmented Actor-Critic models 
"""
from __future__ import annotations

from typing import Optional
import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization
from myrl.utils_core import ObsEncoder, FiLM

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
        # print(f"Actor MLP: {self.actor}")

        # critic
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        # critic observation normalization
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()
        # print(f"Critic MLP: {self.critic}")

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
      - optional context 'c_t' via FiLM or concat
    """
    def __init__(self, *base_args,
                 rpo_actor: bool = False,
                 rpo_alpha: float = 0.5,
                 ctx_mode: str = "concat",      # 'none'|'concat'|'film'
                 ctx_dim: int = 32,
                 feat_dim: Optional[int] = 128,  # ObsEncoder output dim; defaults to obs dim
                 **kwargs):
        """Augmented ActorCritic compatible with rsl-rl.

        Args:
            rpo_actor (bool): Whether to apply RPO-style mean perturbations during sampling.
            rpo_alpha (float): Magnitude of the uniform perturbation applied when `rpo_actor` is True.
            ctx_mode (str): Context mode to use ('none', 'concat', 'film').
            ctx_dim (int): Dimension of the context vector.
            feat_dim (Optional[int]): Feature dimension for the encoder. If None, defaults to observation dimension.
            actor_layer_norm (bool): use a tiny layer norm if it takes context input (default: False).

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
        init_noise_std = kwargs.get("init_noise_std", 1.0)
        super().__init__(*base_args, **kwargs)

        if isinstance(init_noise_std, torch.Tensor):
            init_noise_std = float(init_noise_std.detach().cpu().item())
        else:
            init_noise_std = float(init_noise_std)
        self._init_noise_std = init_noise_std

        self.rpo_actor = bool(rpo_actor)
        self.rpo_alpha = float(rpo_alpha)
        if self.rpo_alpha < 0.0:
            raise ValueError(f"rpo_alpha must be non-negative, got {self.rpo_alpha}.")

        self._feat_dim = feat_dim

        # Shared encoder and optional conditioners
        out_dim = int(self._obs_dim if feat_dim is None else feat_dim)
        self.encoder = ObsEncoder(self._obs_dim, feat_dim=out_dim)
        self.ctx_mode = str(ctx_mode)
        self.ctx_dim = int(ctx_dim)

        # optional modules
        self.film = FiLM(self.ctx_dim, self.encoder.out_dim) if (self.ctx_mode == "film" and self.ctx_dim > 0) else None

        # compute head input dim 
        head_in = self.encoder.out_dim + (self.ctx_dim if (self.ctx_mode == "concat" and self.ctx_dim > 0) else 0)

        # Overwrite actor and critic with new input dims
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

        # print workflow diagram
        print("=================== PPO_BYOL WORKFLOW ====================")
        self._draw_model_diagram()
        print("========================================================")

    def _actor_distribution_from_features(
        self, features: torch.Tensor, apply_rpo: bool
    ) -> Normal:
        """Builds the action distribution from encoded features."""
        mean = self.actor(features)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )

        if apply_rpo and self.rpo_actor and self.training:
            noise = torch.empty_like(mean).uniform_(-self.rpo_alpha, self.rpo_alpha)
            mean = mean + noise

        return Normal(mean, std)
    
    def _draw_model_diagram(self):
        """Prints a diagram of the model architecture.

        Includes:
        - Policy path (shared ObsEncoder -> optional context via FiLM/concat -> heads)
        - BYOL path (sequence encoder + GRU -> online/target projection; context c_t comes from algorithm)
        """
        obs_dim = getattr(self, "_obs_dim", None)
        feat_dim = getattr(self.encoder, "out_dim", None) if hasattr(self, "encoder") else None
        ctx_dim = int(getattr(self, "ctx_dim", 0))
        head_in = feat_dim + (ctx_dim if (self.ctx_mode == "concat" and ctx_dim > 0) else 0)

        def fmt_dim(x):
            if x is None:
                return "?"
            try:
                return str(int(x))
            except Exception:
                return str(x)

        lines: list[str] = []

        # Section: Policy path
        lines.append("POLICY PATH")
        lines.append(f"  x_t [N,{fmt_dim(obs_dim)}] --ObsEncoder--> z_t [N,{fmt_dim(feat_dim)}]")
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

    def _features(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        """ shared feature extractor with optional context """
        z = self.encoder(x)  # [B,F]
        if self.film is not None:
            z = self.film(z, c)
        if self.ctx_mode == "concat" and self.ctx_dim > 0 and c is not None:
            z = torch.cat([z, c], dim=-1)
        return z

    def get_actor_obs_raw(self, obs):
        base = super().get_actor_obs(obs) 
        return base

    def get_critic_obs_raw(self, obs):
        base = super().get_critic_obs(obs)
        return base

    # --- rsl-rl observation hooks ---
    def get_actor_obs(self, obs):
        base = self.get_actor_obs_raw(obs)
        c = self._ctx
        return self._features(base, c)

    def get_critic_obs(self, obs):
        base = self.get_critic_obs_raw(obs)
        c = self._ctx
        return self._features(base, c)
    
    def update_distribution(self, obs, apply_rpo: Optional[bool] = None):
        """Update action distribution using encoded (and normalized) observations."""
        if apply_rpo is None:
            apply_rpo = self.rpo_actor
        dist = self._actor_distribution_from_features(obs, apply_rpo)
        self.distribution = dist
        return dist

    def act(self, obs, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs)
        return self.distribution.sample()

    # --- normalization updates (only on raw obs) ---
    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            z_a = self.get_actor_obs_raw(obs)
            self.actor_obs_normalizer.update(z_a)
        if self.critic_obs_normalization:
            z_c = self.get_critic_obs_raw(obs)
            self.critic_obs_normalizer.update(z_c)

    # --- rollout-time hooks used by PPOWithBYOL ---
    def set_belief(self, c_t: Optional[torch.Tensor]):
        self._ctx = c_t

