"""Continue value learning after an independent actor KL stopping decision."""
import numpy as np
import torch
from datetime import datetime, timezone
from torch.nn import functional as F
from stable_baselines3.common.utils import explained_variance, FloatSchedule
from docs.research.tools.learned_control_asymmetric_v1.algorithm import AsymmetricPPO
from docs.research.tools.learned_control_optimizer_v3.policy import parameter_partition


class PreconditionedPPO(AsymmetricPPO):
    def __init__(self, *args, critic_learning_rate=1e-4, actor_max_grad_norm=.5,
                 critic_max_grad_norm=.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic_learning_rate = critic_learning_rate
        self.critic_lr_schedule = FloatSchedule(critic_learning_rate)
        self.actor_max_grad_norm = actor_max_grad_norm
        self.critic_max_grad_norm = critic_max_grad_norm
        self.actor_minibatches_this_run = 0
        self.critic_minibatches_this_run = 0
        self.optimizer_deadline_reached = False

    def collect_rollouts(self, *args, **kwargs):
        if self.optimizer_deadline_reached:
            return False
        return super().collect_rollouts(*args, **kwargs)

    def _optimizer_deadline_passed(self):
        deadline = getattr(self, "research_training_deadline_utc", None)
        return deadline is not None and datetime.now(timezone.utc) >= datetime.fromisoformat(deadline.replace("Z", "+00:00"))

    def _update_group_learning_rates(self):
        rates = {"actor": float(self.lr_schedule(self._current_progress_remaining)),
                 "critic": float(self.critic_lr_schedule(self._current_progress_remaining))}
        groups = self.policy.optimizer.param_groups
        if [group.get("name") for group in groups] != ["actor", "critic"]:
            raise ValueError("Grouped PPO requires the registered two named Adam groups")
        for group in groups:
            group["lr"] = rates[group["name"]]
            self.logger.record(f"train/{group['name']}_learning_rate", group["lr"])

    def optimizer_counters(self):
        result = {"actor_minibatches_this_run": self.actor_minibatches_this_run,
                  "critic_minibatches_this_run": self.critic_minibatches_this_run}
        for name, parameters in parameter_partition(self.policy).items():
            steps = [int(self.policy.optimizer.state.get(parameter, {}).get("step", 0)) for _, parameter in parameters]
            result[name + "_adam_step_min"] = min(steps)
            result[name + "_adam_step_max"] = max(steps)
        return result

    def train(self):
        self.policy.set_training_mode(True)
        self._update_group_learning_rates()
        partition = parameter_partition(self.policy)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None if self.clip_range_vf is None else self.clip_range_vf(self._current_progress_remaining)
        actor_active = True
        actor_updates, critic_updates, checked_batches = 0, 0, 0
        stopping_kl, stopping_epoch, stopping_batch = None, None, None
        entropy_losses, pg_losses, value_losses, clip_fractions, kls = [], [], [], [], []
        actor_norms, critic_norms = [], []
        preconditioning = {}
        loss, completed_critic_epochs = None, 0
        for epoch in range(self.n_epochs):
            epoch_started, epoch_complete = False, True
            for batch_index, data in enumerate(self.rollout_buffer.get(self.batch_size)):
                if self._optimizer_deadline_passed():
                    self.optimizer_deadline_reached = True
                    epoch_complete = False
                    break
                epoch_started = True
                # This graph reaches only the explicit critic branch.
                values = self.policy.predict_values_with_context(data.observations, data.critic_context).flatten()
                prediction = values if clip_range_vf is None else data.old_values + torch.clamp(values-data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(data.returns, prediction)
                value_losses.append(value_loss.item())
                actor_loss = None
                if actor_active:
                    distribution = self.policy.get_distribution(data.observations)
                    log_prob, entropy = distribution.log_prob(data.actions), distribution.entropy()
                    advantages = data.advantages
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    log_ratio = log_prob - data.old_log_prob
                    ratio = torch.exp(log_ratio)
                    policy_loss = -torch.min(advantages*ratio, advantages*torch.clamp(ratio, 1-clip_range, 1+clip_range)).mean()
                    entropy_loss = -entropy.mean()
                    with torch.no_grad():
                        kl = float(((torch.exp(log_ratio)-1)-log_ratio).mean())
                    if not np.isfinite(kl):
                        raise FloatingPointError("Nonfinite actor KL diagnostic")
                    kls.append(kl)
                    checked_batches += 1
                    pg_losses.append(policy_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    clip_fractions.append(float((torch.abs(ratio-1)>clip_range).float().mean()))
                    if self.target_kl is not None and kl > 1.5*self.target_kl:
                        actor_active = False
                        stopping_kl, stopping_epoch, stopping_batch = kl, epoch, batch_index
                    else:
                        actor_loss = policy_loss + self.ent_coef*entropy_loss
                self.policy.optimizer.zero_grad(set_to_none=True)
                loss = self.vf_coef*value_loss
                if actor_loss is not None:
                    loss = loss + actor_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite grouped PPO loss")
                loss.backward()
                actor_parameters = [parameter for _, parameter in partition["actor"]]
                critic_parameters = [parameter for _, parameter in partition["critic"]]
                if actor_loss is None:
                    if any(parameter.grad is not None for parameter in actor_parameters):
                        raise ValueError("Critic-only optimization leaked into actor gradients")
                else:
                    actor_norms.append(float(torch.nn.utils.clip_grad_norm_(actor_parameters, self.actor_max_grad_norm)))
                critic_norms.append(float(torch.nn.utils.clip_grad_norm_(critic_parameters, self.critic_max_grad_norm)))
                if not all(np.isfinite(value) for value in actor_norms[-1:] + critic_norms[-1:]):
                    raise FloatingPointError("Nonfinite grouped PPO gradients")
                self.policy.optimizer.step()
                for label, record in self.policy.optimizer.last_update_diagnostics.items():
                    combined = preconditioning.setdefault(label, {key: 0 for key in record})
                    for key, value in record.items():
                        combined[key] += value
                actor_updates += actor_loss is not None
                critic_updates += 1
            self._n_updates += int(epoch_started)
            completed_critic_epochs += int(epoch_complete)
            if self.optimizer_deadline_reached:
                break
        self.actor_minibatches_this_run += actor_updates
        self.critic_minibatches_this_run += critic_updates
        self.last_optimizer_rollout_timesteps = self.num_timesteps
        average = lambda values: float(np.mean(values)) if values else 0.
        metrics = {"entropy_loss": average(entropy_losses), "policy_gradient_loss": average(pg_losses),
            "value_loss": average(value_losses), "approx_kl": average(kls), "clip_fraction": average(clip_fractions),
            "loss": 0. if loss is None else loss.item(), "explained_variance": explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()),
            "n_updates": self._n_updates, "clip_range": clip_range,
            "actor_accepted_minibatches": actor_updates, "critic_accepted_minibatches": critic_updates,
            "actor_checked_minibatches": checked_batches, "actor_kl_stopped": int(not actor_active),
            "actor_stopping_kl": stopping_kl or 0., "actor_stopping_epoch": -1 if stopping_epoch is None else stopping_epoch,
            "actor_stopping_batch": -1 if stopping_batch is None else stopping_batch,
            "actor_preclip_gradient_norm": np.mean(actor_norms) if actor_norms else 0.,
            "critic_preclip_gradient_norm": average(critic_norms), "critic_completed_epochs": completed_critic_epochs,
            "optimizer_deadline_reached": int(self.optimizer_deadline_reached), **self.optimizer_counters()}
        for name, value in metrics.items():
            self.logger.record("train/"+name, value)
        for label, record in preconditioning.items():
            for key, value in record.items():
                self.logger.record("train/precondition_"+label+"_"+key, value)
            self.logger.record("train/precondition_"+label+"_rounding_zero_fraction",
                record["rounded_to_zero_elements"] / max(1, record["proposed_nonzero_elements"]))


__all__ = ["PreconditionedPPO"]
