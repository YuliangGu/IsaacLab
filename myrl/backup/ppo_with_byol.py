"""PPO augmented with BYOL context, trained per minibatch.

- During rollout, we maintain a lightweight GRU state over actor observations
  and expose it to the policy via ``set_belief(c_t)``. Optionally condition on
  the previous action.
- During update, PPO and the BYOL module train in the same minibatch loop so
  the encoder sees aligned gradients.
"""

import torch
from rsl_rl.algorithms.ppo import PPO
import torch.nn.functional as F
from typing import Optional
from myrl.byol_utils import BYOLSeq, extract_policy_obs_and_dones, sample_byol_windows
from myrl.storage_adapter import BYOLRolloutView

class PPOWithBYOL(PPO):
    def __init__(self, *args,
                 byol_enabled: bool = True,
                 byol_coef: float = 0.1,
                 byol_tau: float = 0.996,
                 byol_window: int = 16,
                 byol_batch: int = 512,
                 byol_use_actions: bool = False,
                 byol_train_backbone: bool = True,
                 byol_noise_std: float = 0.01,
                 byol_feat_drop: float = 0.05,
                 byol_frame_drop: float = 0.05,
                 byol_time_warp_scale: float = 0.0,
                 byol_max_shift: int = 0,
                 ctx_mode: str = "film",           # 'none'|'concat'|'film' (policy-side)
                 use_prev_action: bool = False,    # policy-side
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.byol_enabled = bool(byol_enabled)
        self.byol_coef = float(byol_coef)
        self.byol_tau = float(byol_tau)
        self.byol_window = int(byol_window)
        self.byol_batch = int(byol_batch)
        self.byol_use_actions = bool(byol_use_actions)
        self.augs = dict(noise_std=byol_noise_std, feat_drop=byol_feat_drop,
                         frame_drop=byol_frame_drop, time_warp_scale=byol_time_warp_scale,
                         max_shift=byol_max_shift)
        # BYOL will be built lazily on first policy call when encoder dims are known
        self.byol = None
        # If policy.ctx_dim is 0/absent, fall back to a sensible default
        _ctx = int(getattr(self.policy, "ctx_dim", 0) or 0)
        self._ctx_hidden_size = _ctx if _ctx > 0 else 128

        self.byol_train_backbone = bool(byol_train_backbone)
        self.ctx_mode = getattr(self.policy, "ctx_mode", ctx_mode)
        self.use_prev_action = getattr(self.policy, "use_prev_action", use_prev_action)
        self._prev_action: Optional[torch.Tensor] = None
        self._ctx_h: Optional[torch.Tensor] = None  
        try:
            print(
                "[BYOL][init] "
                f"enabled={int(self.byol_enabled)} coef={self.byol_coef} tau={self.byol_tau} "
                f"W={self.byol_window} B={self.byol_batch} use_actions={int(self.byol_use_actions)} "
                f"ctx_mode={self.ctx_mode} ctx_dim={getattr(self.policy,'ctx_dim',0)} "
                f"use_prev_action={int(getattr(self.policy,'use_prev_action', False))} "
                f"train_backbone={int(self.byol_train_backbone)}"
            )
            print(
                "[PPO][init] "
                f"lr={getattr(self,'learning_rate', None)} epochs={getattr(self, 'num_learning_epochs', None)} "
                f"minibatches={getattr(self, 'num_mini_batches', None)} clip={getattr(self, 'clip_param', None)} "
                f"entropy_coef={getattr(self, 'entropy_coef', None)} value_coef={getattr(self, 'value_loss_coef', None)} "
                f"gamma={getattr(self, 'gamma', None)} lam={getattr(self, 'lam', None)}"
            )
        except Exception:
            pass

    # ----------------- helpers -----------------
    def _ensure_byol(self):
        if self.byol is None:
            act_enc = getattr(self.policy, "aenc", None) if self.byol_use_actions else None
            with torch.inference_mode(False):
                self.byol = BYOLSeq(
                    self.policy.encoder,
                    z_dim=self._ctx_hidden_size,
                    proj_dim=self._ctx_hidden_size,
                    tau=self.byol_tau,
                    action_encoder=act_enc,
                    train_backbone=self.byol_train_backbone,
                ).to(self.device)
            # add BYOL params to PPO optimizer (once)
            try:
                if not hasattr(self, "_byol_param_groups_added"):
                    # Always add BYOL projection and prediction heads (lightweight)
                    self.optimizer.add_param_group({
                        "params": list(self.byol.g_online.parameters()) + list(self.byol.q_online.parameters())
                    })
                    # Optionally add backbone components so BYOL can train them too
                    if self.byol_train_backbone:
                        params = []
                        # shared obs encoder + GRU backbone
                        params += list(self.byol.f_online.obs_encoder.parameters())
                        params += list(self.byol.f_online.gru.parameters())
                        # optional action encoder
                        if getattr(self.byol.f_online, "action_encoder", None) is not None and self.byol_use_actions:
                            params += list(self.byol.f_online.action_encoder.parameters())
                        if params:
                            self.optimizer.add_param_group({"params": params})
                    self._byol_param_groups_added = True
            except Exception:
                pass
        return self.byol

    @torch.no_grad()
    def _precompute_beliefs_flat(self, storage) -> torch.Tensor:
        """Compute [T,N,Z] beliefs over the rollout, then flatten -> [TN,Z]."""
        obs, dones = extract_policy_obs_and_dones(storage)
        actions = storage.actions if (self.byol_use_actions and hasattr(storage, "actions")) else None
        self._ensure_byol()
        self.byol.eval()
        bel_TNZ = self.byol.beliefs_from_rollout(obs, dones, actions=actions)  # [T,N,Z]
        return bel_TNZ.reshape(-1, bel_TNZ.shape[-1]).detach()                 # [TN,Z]

    def _mb_indices_from_sample(self, sample) -> torch.Tensor | None:
        """Try common keys to retrieve flat indices of this minibatch."""
        for k in ("indices", "inds", "flat_inds", "mb_inds", "batch_inds"):
            if k in sample:
                v = sample[k]
                if isinstance(v, torch.Tensor): return v.to(self.device).long().reshape(-1)
                try:
                    import numpy as _np
                    if isinstance(v, _np.ndarray):
                        return torch.as_tensor(v, device=self.device).long().reshape(-1)
                except Exception:
                    pass
        return None

    def _iter_minibatches(self, storage):
        """Duck-typed adaptor over rsl-rl storage to yield training minibatches."""
        if not isinstance(storage, BYOLRolloutView) and hasattr(storage, "observations") and hasattr(storage, "mini_batch_generator"):
            storage = BYOLRolloutView(storage)
            try:
                self.storage = storage
            except Exception:
                pass
        # Preferred path: our wrapper provides dicts and indices; request 1 epoch here
        if hasattr(storage, "mini_batch_generator"):
            return storage.mini_batch_generator(self.num_mini_batches, num_epochs=1)
        # Fallbacks for older variants:
        if hasattr(storage, "get_training_minibatch"):
            return (storage.get_training_minibatch(i) for i in range(self.num_mini_batches))
        if hasattr(storage, "mini_batches"):
            return iter(storage.mini_batches)
        raise RuntimeError("Unknown storage API for minibatch generation.")

    def _extract_action(self, ret):
        if torch.is_tensor(ret):
            return ret
        act = getattr(ret, "actions", None)
        if act is None:
            act = getattr(ret, "action", None)
        if act is None and isinstance(ret, (tuple, list)) and len(ret) > 0:
            act = ret[0]
        if act is None:
            raise TypeError(f"Unsupported return from super().act: {type(ret)}")
        return act

    # ---- rollout-time hooks ----
    def _get_from_container(self, container, key, default=None):
        try:
            if isinstance(container, dict):
                return container.get(key, default)
            val = container.get(key)
            return val if val is not None else default
        except Exception:
            try:
                return container[key]
            except Exception:
                return default

    def _split_obs(self, obs):
        """Extract actor/critic tensors from a container for BYOL context update."""
        if torch.is_tensor(obs):
            return obs, None
        actor_obs = self._get_from_container(obs, "policy", None)
        critic_obs = self._get_from_container(obs, "critic", None)
        if actor_obs is None:
            actor_obs = self._get_from_container(obs, "obs", None)
        if actor_obs is None:
            raise TypeError(f"Unsupported obs type for PPOWithBYOL.act: {type(obs)}")
        return actor_obs, critic_obs

    def _policy_set_inputs(self, actor_obs: torch.Tensor):
        # Ensure policy’s BYOL-aware parts exist (build encoders lazily)
        if hasattr(self.policy, "_ensure_built"):
            try:
                self.policy._ensure_built(actor_obs)
            except Exception:
                pass

        # If not BYOL, do not build belief nor BYOL module
        if not self.byol_enabled:
            if hasattr(self.policy, "set_belief"):
                self.policy.set_belief(None)
            if hasattr(self.policy, "set_prev_action"):
                self.policy.set_prev_action(self._prev_action if self.use_prev_action else None)
            return

        # Build BYOL once and keep around
        self._ensure_byol()

        # --- Belief update can stay under no_grad (and even in inference mode) ---
        with torch.no_grad():
            z = self.policy.encoder(actor_obs).unsqueeze(1)  # [N,1,F]
            if self._ctx_h is None:
                self._ctx_h = torch.zeros(1, z.shape[0], self._ctx_hidden_size, device=self.device)
            _, self._ctx_h = self.byol.f_online.gru(z, self._ctx_h)
            c_t = self._ctx_h.squeeze(0).detach()

        if hasattr(self.policy, "set_belief"):
            self.policy.set_belief(c_t)
        if hasattr(self.policy, "set_prev_action"):
            self.policy.set_prev_action(self._prev_action if self.use_prev_action else None)

    def _policy_reset_on_dones(self, dones: torch.Tensor):
        if dones is None:
            return
        if dones.dim() == 3 and dones.shape[-1] == 1:
            mask = dones.squeeze(-1)
        else:
            mask = dones
        if self._ctx_h is not None and mask.any():
            self._ctx_h[:, mask, :] = 0.0
        if self._prev_action is not None and mask.any():
            self._prev_action[mask] = 0.0

    def prepare_bootstrap(self, storage=None) -> None:
        """Prepare belief (c_T) for bootstrapping.
        """
        storage = storage or getattr(self, "storage", None) or getattr(self, "rollout_storage", None)
        if storage is None:
            return
        if not self.byol_enabled:
            if hasattr(self.policy, "set_belief"):
                self.policy.set_belief(None)
            return
        try:
            obs_all, dones_all = extract_policy_obs_and_dones(storage)  # [T,N,D], [T,N]
        except Exception:
            return
        # Ensure policy built and BYOL exists
        if hasattr(self.policy, "_ensure_built"):
            try:
                self.policy._ensure_built(obs_all[-1])
            except Exception:
                pass
        self._ensure_byol()
        if self.byol is None or not hasattr(self.policy, "encoder"):
            return
        with torch.no_grad():
            last_obs = obs_all[-1]  # [N,D]
            N = last_obs.shape[0]
            if self._ctx_h is None or self._ctx_h.shape[1] != N:
                self._ctx_h = torch.zeros(1, N, self._ctx_hidden_size, device=last_obs.device)
            # zero hidden where last step was terminal
            mask = dones_all[-1]
            if mask.dim() == 2 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            if mask.any():
                self._ctx_h[:, mask, :] = 0.0
            z = self.policy.encoder(last_obs).unsqueeze(1)  # [N,1,F]
            _, self._ctx_h = self.byol.f_online.gru(z, self._ctx_h)
            c_T = self._ctx_h.squeeze(0).detach()
        if hasattr(self.policy, "set_belief"):
            self.policy.set_belief(c_T)

    def update(self, storage=None):
        """Joint PPO + BYOL update per minibatch (feedforward path).

        - Wraps storage to yield dict minibatches with flat indices.
        - Precomputes beliefs [T*N,Z] once and indexes per minibatch.
        - Recomputes PPO losses on current params and adds BYOL auxiliary.
        """

        storage = storage or getattr(self, "storage", None) or getattr(self, "rollout_storage", None)
        if storage is None:
            return super().update()

        # Use adapter so we get dict minibatches and flat indices
        mb_iter = self._iter_minibatches(storage)

        # Precompute beliefs once per update
        belief_flat = None
        if self.byol_enabled:
            self._ensure_byol()
            self.byol.train()
            belief_flat = self._precompute_beliefs_flat(storage)  # [T*N, Z]

        # Fetch rollout tensors ONCE; reuse across minibatches
        obs_all, dones_all = extract_policy_obs_and_dones(storage)
        actions_all = storage.actions if (self.byol_use_actions and hasattr(storage, "actions")) else None

        logs = {}
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_byol = 0.0
        n_updates = 0

        for _ in range(self.num_learning_epochs):
                for sample in self._iter_minibatches(storage):
                    obs_b = sample.get("observations")
                    critic_obs_b = sample.get("critic_observations", None)
                    actions_b = sample.get("actions")
                    old_logp_b = sample.get("old_action_log_probs")
                    adv_b = sample.get("advantages")
                    ret_b = sample.get("returns")
                    target_values_b = sample.get("target_values")
                    if obs_b is None or actions_b is None or old_logp_b is None or adv_b is None or ret_b is None:
                        raise RuntimeError("Minibatch missing required fields (obs/actions/logp/adv/ret).")

                    # Normalize advantages per minibatch if requested
                    if getattr(self, "normalize_advantage_per_mini_batch", False):
                        with torch.no_grad():
                            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                    # Inject belief for this minibatch
                    c_mb = None
                    if belief_flat is not None:
                        mb_inds = sample.get("indices", None)
                        if mb_inds is not None:
                            c_mb = belief_flat[mb_inds]
                        else:
                            c_mb = belief_flat.mean(0, keepdim=True)
                    if hasattr(self.policy, "set_belief"):
                        self.policy.set_belief(c_mb)

                    # Recompute distribution and value under current params
                    self.policy.act(obs_b)
                    new_logp = self.policy.get_actions_log_prob(actions_b)
                    entropy = self.policy.entropy
                    new_values = self.policy.evaluate(critic_obs_b if critic_obs_b is not None else obs_b)

                    logratio = new_logp - torch.squeeze(old_logp_b)
                    ratio =  logratio.exp()

                    if getattr(self, "desired_kl", None) is not None and getattr(self, "schedule", None) == "adaptive":
                        # use the KL esitimator:
                        with torch.no_grad():
                            approx_kl = ((ratio - 1) - logratio).mean()
                        if approx_kl > 2.0 * self.desired_kl:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif approx_kl < 0.5 * self.desired_kl and approx_kl > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                    # PPO losses
                    surr1 = -torch.squeeze(adv_b) * ratio
                    surr2 = -torch.squeeze(adv_b) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    surrogate_loss = torch.max(surr1, surr2).mean()

                    if getattr(self, "use_clipped_value_loss", False) and (target_values_b is not None):
                        value_clipped = target_values_b + (new_values - target_values_b).clamp(-self.clip_param, self.clip_param)
                        value_losses = (new_values - ret_b).pow(2)
                        value_losses_clipped = (value_clipped - ret_b).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (ret_b - new_values).pow(2).mean()

                    entropy_loss = -self.entropy_coef * entropy.mean()
                    loss = surrogate_loss + self.value_loss_coef * value_loss + entropy_loss

                    # BYOL auxiliary on shared encoder
                    byol_loss_val = torch.tensor(0.0, device=self.device)
                    if self.byol_enabled and (self.byol_coef > 0.0):
                        # obs_all, dones_all = extract_policy_obs_and_dones(storage)
                        # actions_all = storage.actions if (self.byol_use_actions and hasattr(storage, "actions")) else None
                        v1, v2, _, _ = sample_byol_windows(
                            obs=obs_all, dones=dones_all, W=self.byol_window, B=self.byol_batch,
                            device=self.device, actions=actions_all, **self.augs
                        )
                        with torch.inference_mode(False), torch.enable_grad():
                            byol_loss_val = self.byol.loss(v1, v2)
                        loss = loss + self.byol_coef * byol_loss_val

                    # Step
                    self.optimizer.zero_grad()
                    loss.backward()
                    try:
                        clip_params = list(self.policy.parameters())
                        if self.byol_enabled:
                            clip_params += self.byol.online_parameters(False, False)
                        torch.nn.utils.clip_grad_norm_(clip_params, self.max_grad_norm)
                    except Exception:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    if self.byol_enabled:
                        self.byol.ema_update()


        logs["byol/enabled"] = int(self.byol_enabled)
        logs["byol/loss"] = float(mean_byol / max(1, n_updates))

        logs["ppo/surrogate_loss"] = float(mean_surrogate_loss / max(1, n_updates))
        logs["ppo/value_loss"] = float(mean_value_loss / max(1, n_updates))
        logs["ppo/entropy"] = float(mean_entropy / max(1, n_updates))

        # Clear underlying storage like rsl-rl
        try:
            orig = storage.inner if isinstance(storage, BYOLRolloutView) else storage
            orig.clear()
        except Exception:
            pass

        return logs

    def act(self, obs, critic_obs=None):
        """ NOTE: double check. the policy should takes online inferred z_t from sequential BYOL"""
        actor_obs, critic_from_obs = self._split_obs(obs)
        if critic_obs is None:
            critic_obs = critic_from_obs
        self._policy_set_inputs(actor_obs)
        with torch.no_grad():
            ret = super().act(obs)  # keep base return type intact
            action = self._extract_action(ret)
        # remember last action for next step's conditioning
        self._prev_action = action.detach().clone() if self.use_prev_action else None
        return ret

    def on_env_step(self, dones: torch.Tensor):
        # call this from the runner after each step; zeros hidden states where done
        self._policy_reset_on_dones(dones)

    # ---- integrate with base runner without modifications ----
    def process_env_step(self, obs, rewards, dones, infos):
        # let base class handle storage, etc.
        super().process_env_step(obs, rewards, dones, infos)
        # then reset BYOL hidden states based on dones
        self._policy_reset_on_dones(dones)

    def compute_returns(self, *args, **kwargs):
        # ensure we have c_T for bootstrap value
        try:
            self.prepare_bootstrap()
        except Exception:
            pass
        return super().compute_returns(*args, **kwargs)

