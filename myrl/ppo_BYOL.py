""" PPO with BYOL-based auxiliary task."""

from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.algorithms import PPO

from myrl.utils_core import BYOLSeq, sample_byol_windows, ObsEncoder, ActionEncoder, extract_policy_obs_and_dones
from myrl.modules import ActorCriticAug

# class PPO:
#     """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

#     policy: ActorCritic
#     """The actor critic module."""

#     def __init__(
#         self,
#         policy,
#         num_learning_epochs=5,
#         num_mini_batches=4,
#         clip_param=0.2,
#         gamma=0.99,
#         lam=0.95,
#         value_loss_coef=1.0,
#         entropy_coef=0.01,
#         learning_rate=0.001,
#         max_grad_norm=1.0,
#         use_clipped_value_loss=True,
#         schedule="adaptive",
#         desired_kl=0.01,
#         device="cpu",
#         normalize_advantage_per_mini_batch=False,
#         # RND parameters
#         rnd_cfg: dict | None = None,
#         # Symmetry parameters
#         symmetry_cfg: dict | None = None,
#         # Distributed training parameters
#         multi_gpu_cfg: dict | None = None,
#     ):
#         # device-related parameters
#         self.device = device
#         self.is_multi_gpu = multi_gpu_cfg is not None
#         # Multi-GPU parameters
#         if multi_gpu_cfg is not None:
#             self.gpu_global_rank = multi_gpu_cfg["global_rank"]
#             self.gpu_world_size = multi_gpu_cfg["world_size"]
#         else:
#             self.gpu_global_rank = 0
#             self.gpu_world_size = 1

#         # RND components
#         if rnd_cfg is not None:
#             # Extract parameters used in ppo
#             rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
#             # Create RND module
#             self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
#             # Create RND optimizer
#             params = self.rnd.predictor.parameters()
#             self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
#         else:
#             self.rnd = None
#             self.rnd_optimizer = None

#         # Symmetry components
#         if symmetry_cfg is not None:
#             # Check if symmetry is enabled
#             use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
#             # Print that we are not using symmetry
#             if not use_symmetry:
#                 print("Symmetry not used for learning. We will use it for logging instead.")
#             # If function is a string then resolve it to a function
#             if isinstance(symmetry_cfg["data_augmentation_func"], str):
#                 symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
#             # Check valid configuration
#             if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
#                 raise ValueError(
#                     "Data augmentation enabled but the function is not callable:"
#                     f" {symmetry_cfg['data_augmentation_func']}"
#                 )
#             # Store symmetry configuration
#             self.symmetry = symmetry_cfg
#         else:
#             self.symmetry = None

#         # PPO components
#         self.policy = policy
#         self.policy.to(self.device)
#         # Create optimizer
#         self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
#         # Create rollout storage
#         self.storage: RolloutStorage = None  # type: ignore
#         self.transition = RolloutStorage.Transition()

#         # PPO parameters
#         self.clip_param = clip_param
#         self.num_learning_epochs = num_learning_epochs
#         self.num_mini_batches = num_mini_batches
#         self.value_loss_coef = value_loss_coef
#         self.entropy_coef = entropy_coef
#         self.gamma = gamma
#         self.lam = lam
#         self.max_grad_norm = max_grad_norm
#         self.use_clipped_value_loss = use_clipped_value_loss
#         self.desired_kl = desired_kl
#         self.schedule = schedule
#         self.learning_rate = learning_rate
#         self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

#     def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
#         # create rollout storage
#         self.storage = RolloutStorage(
#             training_type,
#             num_envs,
#             num_transitions_per_env,
#             obs,
#             actions_shape,
#             self.device,
#         )

#     def act(self, obs):
#         if self.policy.is_recurrent:
#             self.transition.hidden_states = self.policy.get_hidden_states()
#         # compute the actions and values
#         self.transition.actions = self.policy.act(obs).detach()
#         self.transition.values = self.policy.evaluate(obs).detach()
#         self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
#         self.transition.action_mean = self.policy.action_mean.detach()
#         self.transition.action_sigma = self.policy.action_std.detach()
#         # need to record obs before env.step()
#         self.transition.observations = obs
#         return self.transition.actions

#     def process_env_step(self, obs, rewards, dones, extras):
#         # update the normalizers
#         self.policy.update_normalization(obs)
#         if self.rnd:
#             self.rnd.update_normalization(obs)

#         # Record the rewards and dones
#         # Note: we clone here because later on we bootstrap the rewards based on timeouts
#         self.transition.rewards = rewards.clone()
#         self.transition.dones = dones

#         # Compute the intrinsic rewards and add to extrinsic rewards
#         if self.rnd:
#             # Compute the intrinsic rewards
#             self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
#             # Add intrinsic rewards to extrinsic rewards
#             self.transition.rewards += self.intrinsic_rewards

#         # Bootstrapping on time outs
#         if "time_outs" in extras:
#             self.transition.rewards += self.gamma * torch.squeeze(
#                 self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
#             )

#         # record the transition
#         self.storage.add_transitions(self.transition)
#         self.transition.clear()
#         self.policy.reset(dones)

#     def compute_returns(self, obs):
#         # compute value for the last step
#         last_values = self.policy.evaluate(obs).detach()
#         self.storage.compute_returns(
#             last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
#         )

#     def update(self):  # noqa: C901
#         mean_value_loss = 0
#         mean_surrogate_loss = 0
#         mean_entropy = 0
#         # -- RND loss
#         if self.rnd:
#             mean_rnd_loss = 0
#         else:
#             mean_rnd_loss = None
#         # -- Symmetry loss
#         if self.symmetry:
#             mean_symmetry_loss = 0
#         else:
#             mean_symmetry_loss = None

#         # generator for mini batches
#         if self.policy.is_recurrent:
#             generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
#         else:
#             generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

#         # generator for mini batches
#         if self.policy.is_recurrent:
#             generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
#         else:
#             generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

#         # iterate over batches
#         for (
#             obs_batch,
#             actions_batch,
#             target_values_batch,
#             advantages_batch,
#             returns_batch,
#             old_actions_log_prob_batch,
#             old_mu_batch,
#             old_sigma_batch,
#             hid_states_batch,
#             masks_batch,
#         ) in generator:

#             # number of augmentations per sample
#             # we start with 1 and increase it if we use symmetry augmentation
#             num_aug = 1
#             # original batch size
#             # we assume policy group is always there and needs augmentation
#             original_batch_size = obs_batch.batch_size[0]

#             # check if we should normalize advantages per mini batch
#             if self.normalize_advantage_per_mini_batch:
#                 with torch.no_grad():
#                     advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

#             # Perform symmetric augmentation
#             if self.symmetry and self.symmetry["use_data_augmentation"]:
#                 # augmentation using symmetry
#                 data_augmentation_func = self.symmetry["data_augmentation_func"]
#                 # returned shape: [batch_size * num_aug, ...]
#                 obs_batch, actions_batch = data_augmentation_func(
#                     obs=obs_batch,
#                     actions=actions_batch,
#                     env=self.symmetry["_env"],
#                 )
#                 # compute number of augmentations per sample
#                 # we assume policy group is always there and needs augmentation
#                 num_aug = int(obs_batch.batch_size[0] / original_batch_size)
#                 # repeat the rest of the batch
#                 # -- actor
#                 old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
#                 # -- critic
#                 target_values_batch = target_values_batch.repeat(num_aug, 1)
#                 advantages_batch = advantages_batch.repeat(num_aug, 1)
#                 returns_batch = returns_batch.repeat(num_aug, 1)

#             # Recompute actions log prob and entropy for current batch of transitions
#             # Note: we need to do this because we updated the policy with the new parameters
#             # -- actor
#             self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
#             actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
#             # -- critic
#             value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
#             # -- entropy
#             # we only keep the entropy of the first augmentation (the original one)
#             mu_batch = self.policy.action_mean[:original_batch_size]
#             sigma_batch = self.policy.action_std[:original_batch_size]
#             entropy_batch = self.policy.entropy[:original_batch_size]

#             # KL
#             if self.desired_kl is not None and self.schedule == "adaptive":
#                 with torch.inference_mode():
#                     kl = torch.sum(
#                         torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
#                         + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
#                         / (2.0 * torch.square(sigma_batch))
#                         - 0.5,
#                         axis=-1,
#                     )
#                     kl_mean = torch.mean(kl)

#                     # Reduce the KL divergence across all GPUs
#                     if self.is_multi_gpu:
#                         torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
#                         kl_mean /= self.gpu_world_size

#                     # Update the learning rate
#                     # Perform this adaptation only on the main process
#                     # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
#                     #       then the learning rate should be the same across all GPUs.
#                     if self.gpu_global_rank == 0:
#                         if kl_mean > self.desired_kl * 2.0:
#                             self.learning_rate = max(1e-5, self.learning_rate / 1.5)
#                         elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
#                             self.learning_rate = min(1e-2, self.learning_rate * 1.5)

#                     # Update the learning rate for all GPUs
#                     if self.is_multi_gpu:
#                         lr_tensor = torch.tensor(self.learning_rate, device=self.device)
#                         torch.distributed.broadcast(lr_tensor, src=0)
#                         self.learning_rate = lr_tensor.item()

#                     # Update the learning rate for all parameter groups
#                     for param_group in self.optimizer.param_groups:
#                         param_group["lr"] = self.learning_rate

#             # Surrogate loss
#             ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
#             surrogate = -torch.squeeze(advantages_batch) * ratio
#             surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
#                 ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
#             )
#             surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

#             # Value function loss
#             if self.use_clipped_value_loss:
#                 value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
#                     -self.clip_param, self.clip_param
#                 )
#                 value_losses = (value_batch - returns_batch).pow(2)
#                 value_losses_clipped = (value_clipped - returns_batch).pow(2)
#                 value_loss = torch.max(value_losses, value_losses_clipped).mean()
#             else:
#                 value_loss = (returns_batch - value_batch).pow(2).mean()

#             loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

#             # Symmetry loss
#             if self.symmetry:
#                 # obtain the symmetric actions
#                 # if we did augmentation before then we don't need to augment again
#                 if not self.symmetry["use_data_augmentation"]:
#                     data_augmentation_func = self.symmetry["data_augmentation_func"]
#                     obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
#                     # compute number of augmentations per sample
#                     num_aug = int(obs_batch.shape[0] / original_batch_size)

#                 # actions predicted by the actor for symmetrically-augmented observations
#                 mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

#                 # compute the symmetrically augmented actions
#                 # note: we are assuming the first augmentation is the original one.
#                 #   We do not use the action_batch from earlier since that action was sampled from the distribution.
#                 #   However, the symmetry loss is computed using the mean of the distribution.
#                 action_mean_orig = mean_actions_batch[:original_batch_size]
#                 _, actions_mean_symm_batch = data_augmentation_func(
#                     obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
#                 )

#                 # compute the loss (we skip the first augmentation as it is the original one)
#                 mse_loss = torch.nn.MSELoss()
#                 symmetry_loss = mse_loss(
#                     mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
#                 )
#                 # add the loss to the total loss
#                 if self.symmetry["use_mirror_loss"]:
#                     loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
#                 else:
#                     symmetry_loss = symmetry_loss.detach()

#             # Random Network Distillation loss
#             # TODO: Move this processing to inside RND module.
#             if self.rnd:
#                 # extract the rnd_state
#                 # TODO: Check if we still need torch no grad. It is just an affine transformation.
#                 with torch.no_grad():
#                     rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
#                     rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
#                 # predict the embedding and the target
#                 predicted_embedding = self.rnd.predictor(rnd_state_batch)
#                 target_embedding = self.rnd.target(rnd_state_batch).detach()
#                 # compute the loss as the mean squared error
#                 mseloss = torch.nn.MSELoss()
#                 rnd_loss = mseloss(predicted_embedding, target_embedding)

#             # Compute the gradients
#             # -- For PPO
#             self.optimizer.zero_grad()
#             loss.backward()
#             # -- For RND
#             if self.rnd:
#                 self.rnd_optimizer.zero_grad()  # type: ignore
#                 rnd_loss.backward()

#             # Collect gradients from all GPUs
#             if self.is_multi_gpu:
#                 self.reduce_parameters()

#             # Apply the gradients
#             # -- For PPO
#             nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
#             self.optimizer.step()
#             # -- For RND
#             if self.rnd_optimizer:
#                 self.rnd_optimizer.step()

#             # Store the losses
#             mean_value_loss += value_loss.item()
#             mean_surrogate_loss += surrogate_loss.item()
#             mean_entropy += entropy_batch.mean().item()
#             # -- RND loss
#             if mean_rnd_loss is not None:
#                 mean_rnd_loss += rnd_loss.item()
#             # -- Symmetry loss
#             if mean_symmetry_loss is not None:
#                 mean_symmetry_loss += symmetry_loss.item()

#         # -- For PPO
#         num_updates = self.num_learning_epochs * self.num_mini_batches
#         mean_value_loss /= num_updates
#         mean_surrogate_loss /= num_updates
#         mean_entropy /= num_updates
#         # -- For RND
#         if mean_rnd_loss is not None:
#             mean_rnd_loss /= num_updates
#         # -- For Symmetry
#         if mean_symmetry_loss is not None:
#             mean_symmetry_loss /= num_updates
#         # -- Clear the storage
#         self.storage.clear()

#         # construct the loss dictionary
#         loss_dict = {
#             "value_function": mean_value_loss,
#             "surrogate": mean_surrogate_loss,
#             "entropy": mean_entropy,
#         }
#         if self.rnd:
#             loss_dict["rnd"] = mean_rnd_loss
#         if self.symmetry:
#             loss_dict["symmetry"] = mean_symmetry_loss

#         return loss_dict

#     """
#     Helper functions
#     """

#     def broadcast_parameters(self):
#         """Broadcast model parameters to all GPUs."""
#         # obtain the model parameters on current GPU
#         model_params = [self.policy.state_dict()]
#         if self.rnd:
#             model_params.append(self.rnd.predictor.state_dict())
#         # broadcast the model parameters
#         torch.distributed.broadcast_object_list(model_params, src=0)
#         # load the model parameters on all GPUs from source GPU
#         self.policy.load_state_dict(model_params[0])
#         if self.rnd:
#             self.rnd.predictor.load_state_dict(model_params[1])

#     def reduce_parameters(self):
#         """Collect gradients from all GPUs and average them.

#         This function is called after the backward pass to synchronize the gradients across all GPUs.
#         """
#         # Create a tensor to store the gradients
#         grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
#         if self.rnd:
#             grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
#         all_grads = torch.cat(grads)

#         # Average the gradients across all GPUs
#         torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
#         all_grads /= self.gpu_world_size

#         # Get all parameters
#         all_params = self.policy.parameters()
#         if self.rnd:
#             all_params = chain(all_params, self.rnd.parameters())

#         # Update the gradients for all parameters with the reduced gradients
#         offset = 0
#         for param in all_params:
#             if param.grad is not None:
#                 numel = param.numel()
#                 # copy data back from shared buffer
#                 param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
#                 # update the offset for the next parameter
#                 offset += numel


class PPOWithBYOL(PPO):
    """PPO with BYOL-based auxiliary task."""

    def __init__(self,
            policy: ActorCriticAug,
            enable_byol: bool = True,
            share_byol_encoder: bool = True,
            byol_lambda: float = 0.2,           
            byol_window: int = 16,
            byol_batch: int = 512,                 # -1 -> auto: 2*minibatch_size capped at 512
            byol_tau_start: float = 0.99,
            byol_tau_end: float = 0.999,
            byol_z_dim: int = 64,
            byol_proj_dim: int = 128,
            byol_max_shift: int = 0,
            byol_noise_std: float = 0.02,
            byol_time_warp_scale: float = 0.01,
            byol_feat_drop: float = 0.1,
            byol_frame_drop: float = 0.1,
            byol_use_actions: bool = True,
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


        self.share_byol_encoder = share_byol_encoder
        self.byol_lambda = byol_lambda
        self.byol_window = byol_window
        self.byol_batch = byol_batch
        self.byol_tau_start = byol_tau_start
        self.byol_tau_end = byol_tau_end
        self.byol_z_dim = byol_z_dim
        self.byol_proj_dim = byol_proj_dim
        self.byol_max_shift = byol_max_shift
        self.byol_noise_std = byol_noise_std
        self.byol_time_warp_scale = byol_time_warp_scale
        self.byol_feat_drop = byol_feat_drop
        self.byol_frame_drop = byol_frame_drop
        self.byol_use_actions = byol_use_actions

        if self.enable_byol:
            # Build BYOL encoders
            if self.share_byol_encoder:
                byol_obs_encoder = self.policy.encoder
                print("[PPOWithBYOL] sharing observation encoder between policy and BYOL")
            else:
                print(f'[PPOWithBYOL] NOT sharing observation encoder between policy and BYOL')
                print(f'[debug] obs dim: {self.policy._obs_dim}, feat_dim: {self.policy._feat_dim}')
                byol_obs_encoder = ObsEncoder(self.policy._obs_dim, self.policy._feat_dim)

            byol_action_encoder = None
            if self.byol_use_actions:
                byol_action_encoder = ActionEncoder(self.policy._act_dim, self.policy._feat_dim)
                print("[PPOWithBYOL] using action encoder in BYOL")

            self.byol = BYOLSeq(
                byol_obs_encoder,
                z_dim=self.byol_z_dim,
                proj_dim=self.byol_proj_dim,
                tau=self.byol_tau_start,
                action_encoder=byol_action_encoder,
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

            if self.share_byol_encoder:
                joint_params = list(self.policy.parameters()) + self.byol.online_parameters(include_obs_encoder=False)
                if self.byol.f_online.action_encoder is not None:
                    joint_params += list(self.byol.f_online.action_encoder.parameters())
            else:
                joint_params = list(self.policy.parameters()) + self.byol.online_parameters(include_obs_encoder=True)

            self.param_to_clip = joint_params
            self.optimizer = optim.Adam(joint_params, lr=self.learning_rate)  # override
        else:
            # BYOL disabled: plain PPO
            self.byol = None
            self.param_to_clip = list(self.policy.parameters())
            self.optimizer = optim.Adam(self.param_to_clip, lr=self.learning_rate)

        # lazy init
        self.ctx_h = None
        self.a_prev = None
        self.ctx_buf = None
        self.cnts = 0
        self.total_num_iters = 1500 # NOTE: hardcoded for now

    def _zero_ctx(self):
        if not self.enable_byol:
            return
        self.ctx_h = torch.zeros(1, self.storage.num_envs, self.byol_z_dim, device=self.device)
        # give the policy a [N, Z] context (while GRU keeps [1,N,Z])
        self.policy.set_belief(torch.zeros(self.storage.num_envs, self.byol_z_dim, device=self.device))
        a_prev = torch.zeros(self.storage.num_envs, self.policy._act_dim, device=self.device)
        self.policy.set_prev_action(a_prev)

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
        

    def process_env_step(self, obs, rewards, dones, extras):
        super().process_env_step(obs, rewards, dones, extras)
        """ Overriden to reset BYOL GRU state on env done. """
        if self.ctx_h is not None:
            d = dones.squeeze(-1).to(torch.bool)  # [N]
            if d.any():
                self.ctx_h[:, d, :] = 0.0
                if self.a_prev is not None:
                    self.a_prev[d, :] = 0.0
                    self.policy.set_prev_action(self.a_prev)

    def act(self, obs):
        # Overriden to use BYOL to infer context first when BYOL is enabled.
        if self.enable_byol:
            ct = self._infer(obs)            # infer belief before sampling. shape: [num_envs, z_dim]
            self.ctx_buf[self.storage.step] = ct
            self.policy.set_belief(ct)

        # compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # need to record obs before env.step()
        self.transition.observations = obs

        # record prev action
        self.a_prev = self.transition.actions.clone()
        self.policy.set_prev_action(self.a_prev)
        return self.transition.actions

    # def act(self, obs):
    #     """Overriden to infer context first when BYOL is enabled."""
    #     if self.enable_byol:
    #         ct = self._infer(obs)            # infer belief before sampling. shape: [num_envs, z_dim]
    #         self.ctx_buf[self.storage.step] = ct
    #         self.policy.set_belief(ct)
    #     actions = super().act(obs)
    #     self.a_prev = actions.detach()
    #     self.policy.set_prev_action(self.a_prev)  # <-- feed prev action to policy if you set use_prev_action=True
    #     return actions

    def _infer(self, obs):
        """ Infer latent context. Used during rollout to update GRU state.
        """
        base = self.policy.get_actor_obs_raw(obs)
        with torch.no_grad():
            feats = self.byol.f_online.obs_encoder(base)               # [N, feat_dim]
            if self.byol.f_online.action_encoder is not None and self.a_prev is not None:
                feats = feats + self.byol.f_online.action_encoder(self.a_prev)
            feats = feats.unsqueeze(1)                                 #[N, 1, feat_dim]
            _, self.ctx_h = self.byol.f_online.gru(feats, self.ctx_h)  # h: [1, N, z_dim]
            c_t = self.ctx_h.squeeze(0).detach()
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
        
        # set BYOL to training mode (e.g., for BatchNorm)
        if self.enable_byol and self.byol is not None:
            self.byol.train() 

        if self.byol_tau_start != self.byol_tau_end:
            p = self.cnts / float(self.total_num_iters)
            self.byol.tau = self.byol_tau_start + p * (self.byol_tau_end - self.byol_tau_start)

        # Get raw data from storage for BYOL sampling (only if enabled)
        if self.enable_byol:
            obs_raw = self.storage.observations['policy'] # [T, N, obs_dim]
            dones_raw = self.storage.dones.squeeze(-1)  # [T, N]
        else:
            obs_raw, dones_raw = None, None

        # flatten the (T, N, ...) arrays to (T * N, ...) for PPO
        b_obs = self.storage.observations.flatten(0, 1) # NOTE: rsl-rl PPO agent uses 'obs' key. Others are 'normal'.
        b_logprobs = self.storage.actions_log_prob.flatten(0, 1)
        b_actions = self.storage.actions.flatten(0, 1)
        b_values = self.storage.values.flatten(0, 1)
        b_returns = self.storage.returns.flatten(0, 1)
        b_advantages = self.storage.advantages.flatten(0, 1)
        b_ctx = self.ctx_buf.reshape((-1, self.byol_z_dim)) if self.enable_byol else None

        # Get raw data from storage
        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        b_inds = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        clipfracs = []

        byol_mismatch = None # for logging and curriculum (not implemented yet)
        if self.enable_byol:
            try:
                diag_B = min(128, batch_size)
                if self.byol_use_actions:
                    v1_d, v2_d, _, _ = sample_byol_windows(
                        obs_raw, dones_raw, self.byol_window, diag_B,
                        max_shift=self.byol_max_shift, noise_std=self.byol_noise_std,
                        time_warp_scale=self.byol_time_warp_scale,
                        feat_drop=self.byol_feat_drop, frame_drop=self.byol_frame_drop,
                        device=self.device,
                        actions=self.storage.actions,  # <---- pass [T,N,A]
                    )
                else:
                    v1_d, v2_d, _, _ = sample_byol_windows(
                        obs_raw, dones_raw, self.byol_window, diag_B,
                        max_shift=self.byol_max_shift, noise_std=self.byol_noise_std,
                        time_warp_scale=self.byol_time_warp_scale,
                        feat_drop=self.byol_feat_drop, frame_drop=self.byol_frame_drop,
                        device=self.device,
                    )
                with torch.no_grad():
                    byol_mismatch = self.byol.mismatch_per_window(v1_d, v2_d).mean().item()
            except Exception:
                byol_mismatch = None

        for _ in range(self.num_learning_epochs):
            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                mb_inds = b_inds[start:end]

                # Reset/set context only if BYOL is enabled
                if self.enable_byol:
                    self.policy.set_belief(b_ctx[mb_inds])
                self.policy.act(b_obs[mb_inds])
                newlogprobs = self.policy.get_actions_log_prob(b_actions[mb_inds])

                newvalue = self.policy.evaluate(b_obs[mb_inds])
                entropy = self.policy.entropy

                logratio = newlogprobs - torch.squeeze(b_logprobs[mb_inds])
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_param).float().mean().item()]

                    if self.desired_kl is not None and self.schedule == "adaptive":
                        if approx_kl > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif approx_kl < self.desired_kl / 2.0 and approx_kl > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate
                
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

                entropy_loss = entropy.mean()

                aux_B = (2 * mini_batch_size) if self.byol_batch < 0 else self.byol_batch
                aux_B = min(aux_B, 1024)
                byol_loss_val = torch.tensor(0.0, device=self.device)
                if self.enable_byol and (self.byol_lambda > 0):
                    if self.byol_use_actions:
                        v1, v2, _, _ = sample_byol_windows(
                            obs_raw, dones_raw, self.byol_window, aux_B,
                            max_shift=self.byol_max_shift, noise_std=self.byol_noise_std,
                            time_warp_scale=self.byol_time_warp_scale,
                            feat_drop=self.byol_feat_drop, frame_drop=self.byol_frame_drop,
                            device=self.device,
                            actions=self.storage.actions,  # <---- pass [T,N,A]
                        )
                    else:
                        v1, v2, _, _ = sample_byol_windows(
                            obs_raw, dones_raw, self.byol_window, aux_B,
                            max_shift=self.byol_max_shift, noise_std=self.byol_noise_std,
                            time_warp_scale=self.byol_time_warp_scale,
                            feat_drop=self.byol_feat_drop, frame_drop=self.byol_frame_drop,
                            device=self.device
                        )
                    byol_loss_val = self.byol.loss(v1, v2)

                loss = pg_loss + self.value_loss_coef * v_loss - self.entropy_coef * entropy_loss + self.byol_lambda * byol_loss_val

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.param_to_clip, self.max_grad_norm)
                self.optimizer.step()
            
                # Update target networks with EMA. TODO: every epoch or iteration?
                if self.enable_byol and (self.byol_lambda > 0):
                    self.byol.ema_update()

        # log
        loss_dict = {
            "value_function": v_loss.item(),
            "surrogate": pg_loss.item(),
            "entropy": entropy_loss.item(),
            "byol": byol_loss_val.item(),
            "approx_kl": approx_kl.item(),
            "old_approx_kl": old_approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
            "byol_tau": self.byol.tau,
        }

        # # attach diagnostics if available
        if self.enable_byol:
            if byol_mismatch is not None:
                loss_dict["byol_mismatch"] = byol_mismatch

        # clear storage
        self._zero_ctx()
        self.storage.clear()
        self.cnts += 1

        return loss_dict
