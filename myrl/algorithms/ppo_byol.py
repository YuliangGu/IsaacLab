from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms import PPO

from myrl.utils.core import (
    BYOLSeq,
    sample_byol_windows_phy,
    ObsEncoder,
    extract_policy_obs_and_dones,
    PrivInfoGroup,
    PrivInfoBuffer,
    extract_flat_obs_dict,
)
from myrl.models import ActorCriticAug

class PPOWithBYOL(PPO):
    """PPO with BYOL."""
    def __init__(self,
            policy: ActorCriticAug,
            # BYOL enable/disable
            enable_byol: bool = True,
            share_byol_encoder: bool = True,
            use_transformer: bool = False,
            
            # BYOL hyperparameters
            byol_lambda: float = 0.2,           
            byol_window: int = 16,
            byol_batch: int = 512,                 # -1 -> auto: 2 * minibatch (<= 512)
            byol_tau_start: float = 0.99,
            byol_tau_end: float = 0.999,
            byol_z_dim: int = 64,
            byol_proj_dim: int = 128,
            byol_update_proportion: float = 0.5,
            byol_intrinsic_coef: float = 0.01,

            # BYOL augmentations
            byol_delay: int = 2,
            byol_gaussian_jitter_std: float = 0.05,
            byol_frame_drop: float = 0.1,

            byol_ctx_agg: str = "mean",
            priv_info_group: PrivInfoGroup = PrivInfoGroup(),
            # PPO kwargs (e.g., device, lr, clips)
            **ppo_kwargs,
        ):
        total_iters_kwarg = ppo_kwargs.pop("total_num_iters", None)
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
        self.byol_intrinsic_coef = float(byol_intrinsic_coef)

        # BYOL augmentations
        self.byol_delay = byol_delay
        self.byol_gaussian_jitter_std = byol_gaussian_jitter_std
        self.byol_frame_drop = byol_frame_drop

        self.byol_ctx_agg = str(byol_ctx_agg)
        self._priv_info_cfg = priv_info_group
        self._priv_info_group: PrivInfoGroup | None = None
        self._priv_info_buffer: PrivInfoBuffer | None = None

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
                use_transformer=bool(use_transformer),
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
        else: 
            heads += [self.policy.log_std]
        policy_aux = []
        if getattr(self.policy, "film", None) is not None:
            policy_aux += list(self.policy.film.parameters())
        if getattr(self.policy, "aenc", None) is not None:
            policy_aux += list(self.policy.aenc.parameters())
        heads += policy_aux

        include_backbone = (not self.share_byol_encoder)
        if self.enable_byol:
            byol_params = self.byol.online_parameters(include_obs_encoder=include_backbone)
        else:
            byol_params = []

        enc_mult  = ppo_kwargs.get("lr_mult_encoder", 1.0) 
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
        

        # lazy init
        self.debug_ = True
        self.ctx_h = None
        self.ctx_buf = None
        self.cnts = 0
        self._last_intrinsic_bonus = 0.0
        self._last_logged_mismatch = None
        total_iters = total_iters_kwarg
        if total_iters is None or total_iters <= 0:
            total_iters = 1000
        self.total_num_iters = max(1, int(total_iters))

    def _zero_ctx(self):
        if not self.enable_byol:
            return
        self.ctx_h = torch.zeros(1, self.storage.num_envs, self.byol_z_dim, device=self.device)
        self.policy.set_belief(torch.zeros(self.storage.num_envs, self.byol_z_dim, device=self.device))
        if hasattr(self, "_obs_win") and self._obs_win is not None:
            self._obs_win.zero_()
        if hasattr(self, "_win_idx"):
            self._win_idx = 0

    def init_storage(self, *args, **kwargs):
        super(PPOWithBYOL, self).init_storage(*args, **kwargs)
        self._init_priv_info_buffer()

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
        
        # IMPORTANT: we use a ring buffer to have a sliding window of observations for BYOL
        W = int(self.byol_window)
        N = int(self.storage.num_envs)
        self._obs_win = torch.zeros(W, N, self.policy._obs_dim, device=self.device)
        self._win_idx = 0

    def process_env_step(self, obs, rewards, dones, extras):
        if self._priv_info_buffer is not None:
            self._priv_info_buffer.record(int(self.storage.step), extras)

        # original process_env_step
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)   

        # BYOL context reset on done
        d = dones.squeeze(-1).to(torch.bool)                            
        to = extras.get("time_outs", None)                              
        if to is not None:
            d |= to.squeeze(-1).to(torch.bool)
        if d.any():
            if self.ctx_h is not None:
                self.ctx_h[:, d, :] = 0.0
            if hasattr(self, "_obs_win") and self._obs_win is not None: # clear on dones
                self._obs_win[:, d, :] = 0.0

    def act(self, obs):
        if self.enable_byol:
            ct = self._infer(obs)            # [num_envs, z_dim]
            self.ctx_buf[self.storage.step] = ct
            self.policy.set_belief(ct)

        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        # self.transition.action_mean = self.policy.action_mean.detach()
        # self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.mu = self.policy.action_mean.detach()
        self.transition.sigma = self.policy.action_std.detach()
        self.transition.action_mean = self.transition.mu
        self.transition.action_sigma = self.transition.sigma
        self.transition.observations = obs

        return self.transition.actions

    def _infer(self, obs):
        """ Infer latent context. Used during rollout to update GRU state."""
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
        if self.enable_byol and self.byol_intrinsic_coef > 0.0:
            obs_raw, dones_raw = extract_policy_obs_and_dones(self.storage)
            self._last_intrinsic_bonus = self._apply_intrinsic_rewards(obs_raw, dones_raw) # in-place modify rewards
        else:
            self._last_intrinsic_bonus = 0.0

        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def update(self):
        if self.rnd or self.symmetry or self.policy.is_recurrent:
            raise NotImplementedError("RND, symmetry, and recurrent policy not supported in PPOWithBYOL")
        
        self.policy.train()
        curr_byol_lambda = self.byol_lambda
        if self.enable_byol and self.byol is not None:
            self.byol.train() 
            if self.byol_tau_start != self.byol_tau_end:
                p = self.cnts / float(self.total_num_iters)
                self.byol.tau = self.byol_tau_start + p * (self.byol_tau_end - self.byol_tau_start)
            obs_raw, dones_raw = extract_policy_obs_and_dones(self.storage) 
            if curr_byol_lambda > 0 and self.total_num_iters > 0:
                warmup_iters = max(1, int(self.total_num_iters * 0.1))
                ramp = min(1.0, self.cnts / float(warmup_iters))
                curr_byol_lambda = float(curr_byol_lambda * 0.5 * (1.0 - np.cos(np.pi * ramp)))

        # prepare batches
        a_all = self.storage.actions

        # Build a flat dict-of-groups for both policy and critic paths
        groups = list(set(self.policy.obs_groups["policy"] + self.policy.obs_groups["critic"]))
        obs_flat = extract_flat_obs_dict(self.storage, groups) # {group: [T*N, D_g]}

        # b_obs = self.storage.observations.flatten(0, 1) # NOTE: rsl-rl PPO agent uses 'obs' key.
        # keep vectorized obs structure
        b_logprobs = self.storage.actions_log_prob.flatten(0, 1)
        b_actions = a_all.flatten(0, 1)
        b_values = self.storage.values.flatten(0, 1)
        b_returns = self.storage.returns.flatten(0, 1)
        b_advantages = self.storage.advantages.flatten(0, 1)
        b_mu = self.storage.mu.flatten(0, 1)
        b_sigma = self.storage.sigma.flatten(0, 1)
        b_ctx = self.ctx_buf.reshape((-1, self.byol_z_dim)) if self.enable_byol else None # recontextualization(reconstruction)

        # Get raw data from storage
        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        b_inds = torch.randperm(batch_size, device=self.device) # use full batch size for shuffling
        clipfracs = []

        byol_mismatch = getattr(self, "_last_logged_mismatch", None)
        if byol_mismatch is None and self.enable_byol:
            diag_B = min(128, batch_size)
            v1_d, v2_d, _, _ = sample_byol_windows_phy(
                obs=obs_raw, dones=dones_raw, W=self.byol_window, B=diag_B,
                delay=self.byol_delay,
                gaussian_jitter_std=self.byol_gaussian_jitter_std,
                frame_drop=self.byol_frame_drop,
                device=self.device,
                priv=self._priv_info_buffer,
            )
            with torch.no_grad():
                byol_mismatch = float(self.byol.mismatch_per_window(v1_d, v2_d).mean().item())
            self._last_logged_mismatch = byol_mismatch


        for _ in range(self.num_learning_epochs):
            cached_v1_v2 = None
            if self.enable_byol and (curr_byol_lambda > 0):
                B = self.byol_batch if (self.byol_batch is not None and self.byol_batch > 0) else min(512, 2 * mini_batch_size)
                cached_v1_v2 = sample_byol_windows_phy(
                    obs=obs_raw, dones=dones_raw, W=self.byol_window, B=B,
                    delay=self.byol_delay,
                    gaussian_jitter_std=self.byol_gaussian_jitter_std,
                    frame_drop=self.byol_frame_drop,
                    device=self.device,
                    priv=self._priv_info_buffer,
                )

            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size if i < self.num_mini_batches - 1 else batch_size
                mb_inds = b_inds[start:end]
                original_batch_size = len(mb_inds)

                if self.enable_byol:
                    self.policy.set_belief(b_ctx[mb_inds])  # recontextualization
                mb_obs = {g: v[mb_inds] for g, v in obs_flat.items()}
                self.policy.act(mb_obs)  # just to set distribution
                newlogprobs = self.policy.get_actions_log_prob(b_actions[mb_inds])
                newvalue = self.policy.evaluate(mb_obs)

                new_mu = self.policy.action_mean[:original_batch_size]
                new_sigma = self.policy.action_std[:original_batch_size]
                new_entropy = self.policy.entropy[:original_batch_size]

                logratio = newlogprobs - torch.squeeze(b_logprobs[mb_inds])
                ratio = logratio.exp()

                """ adaptive LR and BYOL loss lambda"""
                with torch.no_grad():
                    clipfracs.append((torch.abs(ratio - 1.0) > self.clip_param).float().mean().item())
                    # use approx KL for adaptive LR
                    kl_approx = (ratio - 1.0 - torch.log(ratio))
                    kl_mean = kl_approx.mean()

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
                if self.enable_byol and (curr_byol_lambda > 0) and cached_v1_v2 is not None:
                    v1, v2, _, _ = cached_v1_v2
                    loss_vec = self.byol.loss_per_sample(v1, v2) 
                    weights = torch.ones_like(loss_vec)
                    if self.byol_update_proportion < 1.0:
                        B = loss_vec.shape[0]
                        mask_ = torch.randperm(B, device=self.device) < int(B * self.byol_update_proportion)
                        denom = torch.clamp(mask_.sum(), min=1.0)
                        byol_loss = ((loss_vec * weights) * mask_.float()).sum() / denom
                    else:
                        byol_loss = loss_vec.mean()

                loss = pg_loss + self.value_loss_coef * v_loss - self.entropy_coef * entropy_loss + curr_byol_lambda * byol_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.param_to_clip, self.max_grad_norm)
                self.optimizer.step()

                # Update targets with EMA. TODO: every update, epoch, or mini-batch?
                if self.enable_byol and (curr_byol_lambda > 0):
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
            loss_dict["byol_lambda"] = float(curr_byol_lambda)
            if byol_mismatch is not None:
                loss_dict["byol_mismatch"] = float(byol_mismatch)
            if self.byol_intrinsic_coef > 0.0:
                loss_dict["byol_intrinsic"] = float(self._last_intrinsic_bonus)

        self.storage.clear()
        self.cnts += 1
        if self._priv_info_buffer is not None:
            self._priv_info_buffer.zero_()

        return loss_dict

    @torch.no_grad()
    def _apply_intrinsic_rewards(self, obs_raw: torch.Tensor, dones_raw: torch.Tensor) -> float:
        if (not self.enable_byol) or (self.byol is None) or (self.byol_intrinsic_coef <= 0.0):
            self._last_logged_mismatch = None
            return 0.0

        total_frames = obs_raw.shape[0] * obs_raw.shape[1]
        if total_frames == 0:
            self._last_logged_mismatch = None
            return 0.0

        B = min(256, max(1, total_frames // max(1, self.byol_window // 2)))
        v1, v2, picks, _ = sample_byol_windows_phy(
            obs=obs_raw,
            dones=dones_raw,
            W=self.byol_window,
            B=B,
            delay=self.byol_delay,
            gaussian_jitter_std=self.byol_gaussian_jitter_std,
            frame_drop=self.byol_frame_drop,
            device=self.device,
            priv=self._priv_info_buffer,
        )
        if v1.numel() == 0:
            self._last_logged_mismatch = None
            return 0.0
        mismatch = self.byol.mismatch_per_window(v1, v2)
        if mismatch.numel() == 0:
            self._last_logged_mismatch = None
            return 0.0
        self._last_logged_mismatch = float(mismatch.mean().item())
        return self._inject_intrinsic_reward(mismatch, picks, v1.shape[1])

    @torch.no_grad()
    def _inject_intrinsic_reward(
        self,
        mismatch: torch.Tensor,
        picks: list[tuple[int, int]],
        window_size: int,
    ) -> float:
        if mismatch is None or mismatch.numel() == 0 or not picks:
            return 0.0
        rewards = getattr(self.storage, "rewards", None)
        if rewards is None:
            return 0.0
        if rewards.ndim == 3 and rewards.shape[-1] == 1:
            rew_view = rewards[..., 0]
        else:
            rew_view = rewards
        if rew_view.numel() == 0:
            return 0.0

        T, N = rew_view.shape[0], rew_view.shape[1]
        if T == 0 or N == 0:
            return 0.0
        count = min(len(picks), mismatch.shape[0])
        if count == 0:
            return 0.0

        device = rew_view.device
        idx_tensor = torch.tensor(picks[:count], dtype=torch.long, device=device)
        starts = idx_tensor[:, 0].clamp_(0, max(0, T - 1))
        envs = idx_tensor[:, 1].clamp_(0, max(0, N - 1))
        scale = -self.byol_intrinsic_coef / float(max(1, window_size))
        bonus = mismatch[:count].to(device=device, dtype=rew_view.dtype) * scale
        rew_view.index_put_((starts, envs), bonus, accumulate=True)
        return float(bonus.mean().item())

    def _resolve_priv_info_group(self) -> PrivInfoGroup | None:
        cfg = self._priv_info_cfg
        if cfg in (None, False):
            return None
        if isinstance(cfg, PrivInfoGroup):
            return cfg
        group = PrivInfoGroup()
        if isinstance(cfg, dict):
            for key, value in cfg.items():
                if hasattr(group, key):
                    setattr(group, key, value)
        return group

    def _init_priv_info_buffer(self) -> None:
        group = self._resolve_priv_info_group()
        self._priv_info_group = group
        if group is None or not hasattr(self, "storage"):
            self._priv_info_buffer = None
            return
        buf = PrivInfoBuffer(group, self.storage.num_transitions_per_env, self.storage.num_envs, self.device)
        if not buf.active:
            self._priv_info_buffer = None
            return
        buf.zero_()
        self._priv_info_buffer = buf
        setattr(self.storage, "priv_info", buf)
