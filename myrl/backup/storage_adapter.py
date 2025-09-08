from __future__ import annotations

import torch

from typing import Iterator, Dict, Any
from tensordict import TensorDict

from rsl_rl.utils import split_and_pad_trajectories

class BYOLRolloutView:
    """
    Read-only wrapper around an existing rollout storage.

    - Exposes the same attributes as the inner storage for compatibility.
    - Reimplements mini_batch_generator to yield dictionaries with keys:
      'observations', 'critic_observations', 'actions', 'old_action_log_probs',
      'advantages', 'returns', and 'indices' (flat TN indices).
    - Does not modify the underlying data; only views and indices are created.
    """

    def __init__(self, inner):
        self.inner = inner

    # passthrough common attributes
    def __getattr__(self, name):
        return getattr(self.inner, name)

    @property
    def device(self):
        return getattr(self.inner, "device", torch.device("cpu"))

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 1) -> Iterator[Dict[str, Any]]:
        """Yield dict minibatches plus flat indices.

        Mirrors the logic of rsl_rl RolloutStorage, but returns a normalized dict
        and exposes the flat indices used for the current mini-batch so that
        auxiliary modules can align precomputed tensors of shape [T*N, ...].
        """
        T = int(self.inner.num_transitions_per_env)
        N = int(self.inner.num_envs)

        # flatten core tensors [T,N,...] -> [T*N,...]
        # observations may be a TensorDict or dict; rely on flatten(0,1)
        observations = self.inner.observations.flatten(0, 1)
        actions = self.inner.actions.flatten(0, 1)
        values = getattr(self.inner, "values", None)
        if values is not None:
            values = values.flatten(0, 1)
        returns = getattr(self.inner, "returns", None)
        if returns is not None:
            returns = returns.flatten(0, 1)
        old_actions_log_prob = getattr(self.inner, "actions_log_prob", None)
        if old_actions_log_prob is not None:
            old_actions_log_prob = old_actions_log_prob.flatten(0, 1)
        advantages = getattr(self.inner, "advantages", None)
        if advantages is not None:
            advantages = advantages.flatten(0, 1)

        batch_size = T * N
        mini_batch_size = batch_size // int(num_mini_batches)

        for _ in range(int(num_epochs)):
            # sample fresh permutation per epoch
            full_perm = torch.randperm(batch_size, device=self.device)
            use_k = num_mini_batches * mini_batch_size
            indices = full_perm[:use_k]
            for i in range(int(num_mini_batches)):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                mb_inds = indices[start:end]
                # build sample dict
                sample = {
                    "observations": observations[mb_inds],
                    "critic_observations": observations[mb_inds],  # same key for simplicity
                    "actions": actions[mb_inds],
                    "old_action_log_probs": old_actions_log_prob[mb_inds] if old_actions_log_prob is not None else None,
                    "advantages": advantages[mb_inds] if advantages is not None else None,
                    "returns": returns[mb_inds] if returns is not None else None,
                    "target_values": values[mb_inds] if values is not None else None,
                    "indices": mb_inds,
                }
                yield sample

""" This is the original RSL-RL RolloutStorage (copied here for extension) """
class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
        device="cpu",
    ):
        # store inputs
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        # Core
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # for distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # for reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # for distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # for reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # if we are at the last step, bootstrap the return value
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    # for distillation
    def generator(self):
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield self.observations[i], self.actions[i], self.privileged_actions[i], self.dones[i]

    # for reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                # yield the mini-batch
                yield obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None

    # for reinfrocement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch

                first_traj = last_traj