"""Actor-Critic and Augmented Actor-Critic models 
"""
from __future__ import annotations

from typing import Optional
import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.networks import MLP, EmpiricalNormalization
from myrl.utils.core import ObsEncoder, FiLM

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
    A lightweight ActorCritic with shared encoder and optional context conditioning.
    """
    def __init__(self, *base_args,
                 ctx_mode: str = "concat",      # 'none'|'concat'|'film'
                 ctx_dim: int = 32,
                 feat_dim: Optional[int] = 128,  # ObsEncoder output dim; defaults to obs dim
                 **kwargs):
        """Augmented ActorCritic compatible with rsl-rl."""

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

    def _actor_distribution_from_features(
        self, features: torch.Tensor, apply_rpo: bool = False
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

        if apply_rpo:
            z = (2.0 * torch.rand_like(mean) - 1.0) * 0.1  # RPO noise
            mean = mean + z
        return Normal(mean, std)

    def _features(self, x: torch.Tensor, c: torch.Tensor | None) -> torch.Tensor:
        z = self.encoder(x)  # [B,F]
        if self.film is not None:
            z = self.film(z, c)
        if self.ctx_mode == "concat" and self.ctx_dim > 0 and c is not None:
            z = torch.cat([z, c], dim=-1)
        return z

    def get_actor_obs_raw(self, obs):
        return super().get_actor_obs(obs)

    def get_critic_obs_raw(self, obs):
        return super().get_critic_obs(obs)

    # --- rsl-rl observation hooks ---
    def get_actor_obs(self, obs):
        base = self.get_actor_obs_raw(obs)
        c = self._ctx
        return self._features(base, c)

    def get_critic_obs(self, obs):
        base = self.get_critic_obs_raw(obs)
        c = self._ctx
        return self._features(base, c)
    
    def update_distribution(self, obs, apply_rpo: bool = False):
        """Update action distribution using encoded (and normalized) observations."""
        dist = self._actor_distribution_from_features(obs, apply_rpo)
        self.distribution = dist
        return dist 

    def act(self, obs, apply_rpo: bool = False, **kwargs):
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self.update_distribution(obs, apply_rpo=apply_rpo)
        return self.distribution.sample()

    # --- normalization updates (only on raw obs) ---
    def update_normalization(self, obs):
        """Keep the running stats in sync with the encoder/ctx-expanded features."""
        if self.actor_obs_normalization:
            z_a = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(z_a)
        if self.critic_obs_normalization:
            z_c = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(z_c)

    # --- rollout-time hooks used by PPOWithBYOL ---
    def set_belief(self, c_t: Optional[torch.Tensor]):
        self._ctx = c_t
