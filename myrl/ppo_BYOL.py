from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms import PPO

from myrl.utils_core import BYOLSeq, sample_byol_windows_phy, ObsEncoder, extract_policy_obs_and_dones
from myrl.modules import ActorCriticAug

class PPOWithBYOL(PPO):
    """PPO with BYOL."""
    def __init__(self,
            policy: ActorCriticAug,

            # BYOL hyperparameters
            enable_byol: bool = True,
            share_byol_encoder: bool = True,
            use_ib: bool = False,
            byol_lambda: float = 0.2,           
            byol_window: int = 16,
            byol_batch: int = 512,                 # -1 -> auto: 2*minibatch_size capped at 512
            byol_tau_start: float = 0.99,
            byol_tau_end: float = 0.999,
            byol_z_dim: int = 64,
            byol_proj_dim: int = 128,
            byol_update_proportion: float = 0.5,

            # BYOL augmentations
            byol_delay: int = 2,
            byol_gaussian_jitter_std: float = 0.05,
            byol_causal_padding_proportion: float = 0.1,
            byol_frame_drop: float = 0.1,

            byol_ctx_agg: str = "last",         # 'last' | 'mean' | 'attn'

            # passthrough PPO kwargs (e.g., device, lr, clips)
            **ppo_kwargs,
        ):
        # forward standard PPO kwargs to base class
        super().__init__(policy, **ppo_kwargs)

        # BYOL enable/disable flag
        self.enable_byol = bool(enable_byol)
        print(f"[PPOWithBYOL] BYOL enabled: {self.enable_byol}")

        # Policy requirements (only strict when BYOL is enabled)
        if self.enable_byol and not isinstance(self.policy, ActorCriticAug):
            raise ValueError("PPOWithBYOL with BYOL enabled requires the policy to be ActorCriticAug")

        # BYOL hyperparameters
        self.share_byol_encoder = share_byol_encoder
        self.byol_lambda = byol_lambda
        self.byol_window = byol_window
        self.byol_batch = byol_batch
        self.byol_tau_start = byol_tau_start
        self.byol_tau_end = byol_tau_end
        self.byol_z_dim = byol_z_dim
        self.byol_proj_dim = byol_proj_dim
        self.byol_update_proportion = byol_update_proportion

        # BYOL augmentations
        self.byol_delay = byol_delay
        self.byol_gaussian_jitter_std = byol_gaussian_jitter_std
        self.byol_causal_padding_proportion = byol_causal_padding_proportion
        self.byol_frame_drop = byol_frame_drop

        self.byol_ctx_agg = str(byol_ctx_agg)

        if self.enable_byol:
            if self.share_byol_encoder:
                byol_obs_encoder = self.policy.encoder
                print("[PPOWithBYOL] sharing observation encoder between policy and BYOL")
            else:
                byol_obs_encoder = ObsEncoder(self.policy._obs_dim, self.policy._feat_dim)

            self.byol = BYOLSeq(
                byol_obs_encoder,
                z_dim=self.byol_z_dim,
                proj_dim=self.byol_proj_dim,
                tau=self.byol_tau_start,
                output_type=self.byol_ctx_agg,
                use_information_bottleneck=bool(use_ib),
            ).to(self.device)

            # ctx compatibility check
            if getattr(self.policy, "ctx_mode", None) in ("film", "concat"):
                if int(self.policy.ctx_dim) != int(self.byol_z_dim):
                    raise ValueError(
                        f"ctx_dim ({self.policy.ctx_dim}) must equal byol_z_dim ({self.byol_z_dim}) "
                        f"when ctx_mode='{self.policy.ctx_mode}'."
                    )

            # Freeze target networks
            self.byol.f_targ.eval()
            self.byol.g_targ.eval()

        """ Optimizer with parameter groups and LR multipliers."""
        enc = list(self.policy.encoder.parameters())         
        heads = list(self.policy.actor.parameters()) + list(self.policy.critic.parameters())
        if self.policy.noise_std_type == 'scalar': 
            heads += [self.policy.std]
        else: heads += [self.policy.log_std]
        policy_aux = []
        if getattr(self.policy, "film", None) is not None:
            policy_aux += list(self.policy.film.parameters())
        if getattr(self.policy, "aenc", None) is not None:
            policy_aux += list(self.policy.aenc.parameters())
        heads += policy_aux

        include_backbone = (not self.share_byol_encoder)
        if self.enable_byol:
            byol_params = self.byol.online_parameters(
                include_obs_encoder=include_backbone,
            )
        else:
            byol_params = []

        enc_mult  = ppo_kwargs.get("lr_mult_encoder", 0.5) 
        byol_mult = ppo_kwargs.get("lr_mult_byol",    1.5)

        param_groups = [
            {"params": enc,        "lr": self.learning_rate * enc_mult, "tag":"policy"},
            {"params": heads,      "lr": self.learning_rate, "tag":"policy"},]
        if byol_params:
            param_groups.append({"params": byol_params, "lr": self.learning_rate * byol_mult, "tag":"byol"})

        self.optimizer = optim.Adam(param_groups)
        self._base_group_lrs = [g['lr'] for g in self.optimizer.param_groups]
        self._base_policy_lr = self.learning_rate

        self.param_to_clip = enc + heads + byol_params
        
        # print optimizer info
        print(f"[PPOWithBYOL] Optimizer param groups:")
        for i, g in enumerate(self.optimizer.param_groups):
            print(f"  group {i}: lr={g['lr']}, num_params={len(g['params'])}")

        # lazy init
        self.debug_ = True
        self.ctx_h = None
        self.ctx_buf = None
        self.cnts = 0
        self.total_num_iters = 1000

    def _zero_ctx(self):
        if not self.enable_byol:
            return
        self.ctx_h = torch.zeros(1, self.storage.num_envs, self.byol_z_dim, device=self.device)
        # give the policy a [N, Z] context (while GRU keeps [1,N,Z])
        self.policy.set_belief(torch.zeros(self.storage.num_envs, self.byol_z_dim, device=self.device))
        # reset rolling windows
        if hasattr(self, "_obs_win") and self._obs_win is not None:
            self._obs_win.zero_()
        if hasattr(self, "_win_idx"):
            self._win_idx = 0

    def init_storage(self, *args, **kwargs):
        super(PPOWithBYOL, self).init_storage(*args, **kwargs)

        if not self.enable_byol:
            return
        
        self.ctx_buf = torch.zeros(
            self.storage.num_transitions_per_env,
            self.storage.num_envs,
            self.byol_z_dim,
            device=self.device,
        )

        if getattr(self.policy, "ctx_mode", None) in ["film", "concat"]:
            print("[PPO-BYOL] context mode:", self.policy.ctx_mode)
            self._zero_ctx()
        else:
            raise NotImplementedError(f"ctx_mode {self.policy.ctx_mode} not implemented")
        
        # Ring buffer for observation windows
        W = int(self.byol_window)
        N = int(self.storage.num_envs)
        self._obs_win = torch.zeros(W, N, self.policy._obs_dim, device=self.device)
        self._win_idx = 0

    def process_env_step(self, obs, rewards, dones, extras):
        super().process_env_step(obs, rewards, dones, extras)
        """ Overriden to reset BYOL GRU state on env done. """
        d = dones.squeeze(-1).to(torch.bool) # reset on env done
        to = extras.get("time_outs", None) # also reset on time_outs
        if to is not None:
            d |= to.squeeze(-1).to(torch.bool)
        if d.any():
            if self.ctx_h is not None:
                self.ctx_h[:, d, :] = 0.0
            # clear ring buffer for done envs
            if hasattr(self, "_obs_win") and self._obs_win is not None:
                self._obs_win[:, d, :] = 0.0

    def act(self, obs):
        # Overriden to use BYOL to infer context first when BYOL is enabled.
        if self.enable_byol:
            ct = self._infer(obs)            # [num_envs, z_dim]
            self.ctx_buf[self.storage.step] = ct
            self.policy.set_belief(ct)

        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs

        return self.transition.actions

    def _infer(self, obs):
        """ Infer latent context. Used during rollout to update GRU state.
        """
        base = self.policy.get_actor_obs_raw(obs)  # [N, obs_dim]
        N = base.shape[0]
        W = int(self.byol_window)
        with torch.no_grad():
            # write current frame into ring buffer
            self._obs_win[self._win_idx] = base
            # chronological order: oldest .. newest
            shift = -((self._win_idx + 1) % W)
            obs_seq = torch.roll(self._obs_win, shifts=shift, dims=0).permute(1, 0, 2)  # [N,W,D]
            z = self.byol.infer_ctx(obs_seq)  # [N, z_dim]
            # advance ring index
            self._win_idx = (self._win_idx + 1) % W
            c_t = z.detach()
        return c_t

    def compute_returns(self, obs):
        # overriden to infer context for last step before bootstrapping value
        if self.enable_byol:
            ct = self._infer(obs)            # infer belief before sampling. shape: [num_envs, z_dim]
            self.policy.set_belief(ct)
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):
        """Overriden to add BYOL loss when enabled; otherwise run PPO path here."""
        if self.rnd or self.symmetry or self.policy.is_recurrent:
            raise NotImplementedError("RND, symmetry, and recurrent policy not supported in PPOWithBYOL")
        
        self.policy.train()
        if self.enable_byol and self.byol is not None:
            self.byol.train() 
            if self.byol_tau_start != self.byol_tau_end:
                p = self.cnts / float(self.total_num_iters)
                self.byol.tau = self.byol_tau_start + p * (self.byol_tau_end - self.byol_tau_start)
            obs_raw, dones_raw = extract_policy_obs_and_dones(self.storage) 


        # prepare batches
        a_all = self.storage.actions  

        # flatten the (T, N, ...) arrays to (T * N, ...) for PPO
        b_obs = self.storage.observations.flatten(0, 1) # NOTE: rsl-rl PPO agent uses 'obs' key.
        b_logprobs = self.storage.actions_log_prob.flatten(0, 1)
        b_actions = a_all.flatten(0, 1)
        b_values = self.storage.values.flatten(0, 1)
        b_returns = self.storage.returns.flatten(0, 1)
        b_advantages = self.storage.advantages.flatten(0, 1)
        b_mu = self.storage.mu.flatten(0, 1)
        b_sigma = self.storage.sigma.flatten(0, 1)
        b_ctx = self.ctx_buf.reshape((-1, self.byol_z_dim)) if self.enable_byol else None

        # Get raw data from storage
        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        b_inds = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        clipfracs = []

        byol_mismatch = None # for logging and intrinsic reward (not implemented yet)
        if self.enable_byol:
            diag_B = min(128, batch_size)
            v1_d, v2_d, _, _ = sample_byol_windows_phy(
                obs=obs_raw, dones=dones_raw, W=self.byol_window, B=diag_B,
                delay=self.byol_delay,
                gaussian_jitter_std=self.byol_gaussian_jitter_std,
                causal_padding_proportion=self.byol_causal_padding_proportion,
                frame_drop=self.byol_frame_drop,
                device=self.device,
                )
            with torch.no_grad():
                byol_mismatch = self.byol.mismatch_per_window(v1_d, v2_d).mean().item()


        for _ in range(self.num_learning_epochs):
            cached_v1_v2 = None
            if self.enable_byol and (self.byol_lambda > 0):
                B = self.byol_batch if (self.byol_batch is not None and self.byol_batch > 0) else min(512, 2 * mini_batch_size)
                cached_v1_v2 = sample_byol_windows_phy(
                    obs=obs_raw, dones=dones_raw, W=self.byol_window, B=B,
                    delay=self.byol_delay,
                    gaussian_jitter_std=self.byol_gaussian_jitter_std,
                    causal_padding_proportion=self.byol_causal_padding_proportion,
                    frame_drop=self.byol_frame_drop,
                    device=self.device,
                )

            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                mb_inds = b_inds[start:end]
                original_batch_size = len(mb_inds)

                # Reset/set context only if BYOL is enabled
                if self.enable_byol:
                    self.policy.set_belief(b_ctx[mb_inds])
                self.policy.act(b_obs[mb_inds])
                newlogprobs = self.policy.get_actions_log_prob(b_actions[mb_inds])
                newvalue = self.policy.evaluate(b_obs[mb_inds])

                new_mu = self.policy.action_mean[:original_batch_size]
                new_sigma = self.policy.action_std[:original_batch_size]
                new_entropy = self.policy.entropy[:original_batch_size]

                logratio = newlogprobs - torch.squeeze(b_logprobs[mb_inds])
                ratio = logratio.exp()

                """ adaptive LR and BYOL loss lambda"""
                with torch.no_grad():
                    clipfracs.append((torch.abs(ratio - 1.0) > self.clip_param).float().mean().item())
                    kl = torch.sum(
                        torch.log(new_sigma / b_sigma[mb_inds] + 1.0e-5)
                        + (torch.square(b_sigma[mb_inds]) + torch.square(b_mu[mb_inds] - new_mu))
                        / (2.0 * torch.square(new_sigma))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.desired_kl is not None:
                        if self.schedule == "adaptive":
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        elif self.schedule == "linear":
                            self.learning_rate = self._base_policy_lr * (1 - self.cnts / float(self.total_num_iters))

                        # update optimizer learning rates (only for policy groups)
                        for pg, base in zip(self.optimizer.param_groups, self._base_group_lrs):
                            if pg.get("tag", "policy") == "policy":
                                pg["lr"] = self.learning_rate
                
                mb_adv = b_advantages[mb_inds]
                if self.normalize_advantage_per_mini_batch:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                # PPO losses
                pg_loss1 = - torch.squeeze(mb_adv) * ratio
                pg_loss2 = - torch.squeeze(mb_adv) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                if self.use_clipped_value_loss:
                    value_clipped = b_values[mb_inds] + (newvalue - b_values[mb_inds]).clamp(-self.clip_param, self.clip_param)
                    value_losses = (newvalue - b_returns[mb_inds]).pow(2)
                    value_losses_clipped = (value_clipped - b_returns[mb_inds]).pow(2)
                    v_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    v_loss = (b_returns[mb_inds] - newvalue).pow(2).mean()

                entropy_loss = new_entropy.mean()

                # BYOL loss
                byol_loss = torch.tensor(0.0, device=self.device)
                if self.enable_byol and (self.byol_lambda > 0) and cached_v1_v2 is not None:
                    v1, v2, _, _ = cached_v1_v2
                    byol_loss = self.byol.loss(v1, v2)
                    if self.byol_update_proportion:
                        _mask = torch.rand(original_batch_size, device=self.device) < self.byol_update_proportion
                        byol_loss = (byol_loss * _mask.float()).sum() / torch.clamp(_mask.float().sum(), min=1.0)

                loss = pg_loss + self.value_loss_coef * v_loss - self.entropy_coef * entropy_loss + self.byol_lambda * byol_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.param_to_clip, self.max_grad_norm)
                self.optimizer.step()
            
                # Update targets with EMA. TODO: every epoch or iteration?
                if self.enable_byol and (self.byol_lambda > 0):
                    self.byol.ema_update()

        # log
        loss_dict = {
            "value_function": v_loss.item(),
            "surrogate": pg_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": kl_mean.item(),
            "clipfrac": float(np.mean(clipfracs) if clipfracs else 0.0),
        }
        if self.enable_byol:
            loss_dict["byol"] = float(byol_loss.item())
            loss_dict["byol_tau"] = float(self.byol.tau)
            loss_dict["byol_lambda"] = float(self.byol_lambda)
            if byol_mismatch is not None:
                loss_dict["byol_mismatch"] = float(byol_mismatch)

        self.storage.clear()
        self.cnts += 1

        return loss_dict
