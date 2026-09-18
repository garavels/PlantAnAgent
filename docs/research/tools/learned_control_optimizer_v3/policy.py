"""An unchanged hybrid actor with data-scaled Adam updates in two groups."""
import math

import torch

from docs.research.tools.learned_control_asymmetric_v1.policy import AsymmetricHybridPolicy
from docs.research.tools.learned_control_optimizer_v3.optimizer import UpdateScaledAdam


def parameter_partition(policy):
    actor, critic = [], []
    for name, parameter in policy.named_parameters():
        if name.startswith(("mlp_extractor.policy_net.", "action_net.")):
            actor.append((name, parameter))
        elif name.startswith(("mlp_extractor.value_net.", "value_net.", "private_value_head.")):
            critic.append((name, parameter))
        else:
            raise ValueError(f"Unclassified or shared trainable parameter: {name}")
    if not actor or not critic:
        raise ValueError("Expected two nonempty disjoint actor/critic branches")
    return {"actor": actor, "critic": critic}


class PreconditionedHybridPolicy(AsymmetricHybridPolicy):
    def __init__(self, *args, mean_update_scales, actor_body_update_scale=.01, **kwargs):
        self.mean_update_scales = tuple(float(value) for value in mean_update_scales)
        self.actor_body_update_scale = float(actor_body_update_scale)
        if (len(self.mean_update_scales) != 7
                or not all(math.isfinite(value) and value > 0 for value in self.mean_update_scales)
                or not math.isfinite(self.actor_body_update_scale) or self.actor_body_update_scale <= 0):
            raise ValueError("All seven head factors and the actor-body factor must be finite and positive")
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if self.optimizer_class is not torch.optim.Adam:
            raise ValueError("Registered update preconditioning requires ordinary Adam proposals")
        partition = parameter_partition(self)
        groups, scales, labels = [], [], []
        for group_name in ("actor", "critic"):
            named = partition[group_name]
            groups.append({"params": [parameter for _, parameter in named],
                           "name": group_name, "lr": lr_schedule(1)})
            for name, parameter in named:
                if name.startswith("mlp_extractor.policy_net."):
                    scale = self.actor_body_update_scale
                    label = "actor_body"
                elif name in ("action_net.weight", "action_net.bias"):
                    if parameter.shape[0] != 21:
                        raise ValueError("Expected the unchanged seven-gate/seven-mean/seven-scale head")
                    scale = torch.ones(21, dtype=parameter.dtype, device=parameter.device)
                    scale[7:14] = torch.as_tensor(self.mean_update_scales, dtype=parameter.dtype, device=parameter.device)
                    if parameter.ndim == 2:
                        scale = scale[:, None]
                    label = "actor_mean_head"
                else:
                    scale = 1.
                    label = "critic"
                scales.append(scale)
                labels.append(label)
        self.optimizer = UpdateScaledAdam(groups, update_scales=scales, update_labels=labels, **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(mean_update_scales=list(self.mean_update_scales),
                    actor_body_update_scale=self.actor_body_update_scale)
        return data


def optimizer_identity():
    return {"schema": "learned-control-preconditioned-optimizer-v3",
        "algorithm": "PPO with two named Adam groups and diagonal scaling of actual proposed actor parameter changes",
        "proposal": "Ordinary Adam computes moments, bias correction and the parameter proposal; p_new=p_old+positive_scale*(p_Adam-p_old)",
        "mean_head_scale": "Each of seven ON-mean weight rows and biases uses its authenticated training-teacher mean(action+1 | action>-1)",
        "body_scale": "Actor policy MLP updates use the declared positive factor, default0.01; gate and scale-logit rows use1; critic uses1",
        "actor": "All actor weights remain trainable; unchanged149-public-input/seven-action hybrid distribution; no runtime rule or new action bound",
        "partition": "Actor policy MLP and action head; critic value MLP, public value head and private value head; no shared trainable tensors",
        "gradient_clipping": "Independent actor and critic Euclidean gradient clipping precedes Adam; positive proposal scaling follows Adam",
        "kl_stopping": "Same v2 rule: stop actor for this rollout at a checked minibatch KL>1.5*target_KL; finish remaining critic minibatches",
        "critic_only_steps": "Actor gradients are None; no actor parameter rewrite or Adam moment/counter advance",
        "serialization": "Policy constructor reconstructs preconditioners from pinned policy kwargs; optimizer state retains the same two named groups",
        "numerics": "Actual floating-point Adam increments are scaled and rounded into parameter dtype; this is not a function-space trust region or a guarantee on realized output movement",
        "counter_semantics": "Separate accepted actor/critic minibatches and Adam step extrema; attempted and completed critic epochs are separate",
        "deadline": "Stop before the next optimizer minibatch after the registered deadline; preserve partial counters and take no further rollout",
        "environment_changes": False, "runtime_operating_rules": False}


__all__ = ["PreconditionedHybridPolicy", "parameter_partition", "optimizer_identity"]
