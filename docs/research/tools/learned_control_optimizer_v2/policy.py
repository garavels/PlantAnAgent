"""An identical hybrid actor with explicit disjoint Adam parameter groups."""
from docs.research.tools.learned_control_asymmetric_v1.policy import AsymmetricHybridPolicy


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


class GroupedHybridPolicy(AsymmetricHybridPolicy):
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        partition = parameter_partition(self)
        self.optimizer = self.optimizer_class([
            {"params": [parameter for _, parameter in partition[name]], "name": name, "lr": lr_schedule(1)}
            for name in ("actor", "critic")], **self.optimizer_kwargs)


def optimizer_identity():
    return {"schema": "learned-control-grouped-optimizer-v2",
        "algorithm": "PPO actor objective and value regression with independent Adam groups",
        "actor": "Unchanged hybrid149-observation/seven-action distribution and learned weights at transfer",
        "partition": "Actor policy MLP and action head; critic value MLP, public value head, and private value head. No shared trainable tensors.",
        "gradient_clipping": "Independent actor and critic Euclidean gradient norms",
        "kl_stopping": "Once a checked minibatch exceeds1.5*target_KL, stop actor updates for this rollout; continue all remaining critic minibatches/epochs",
        "critic_only_steps": "Actor gradients are None; Adam does not advance or change actor parameters or moments",
        "counter_semantics": "Separate cumulative accepted actor/critic minibatches and Adam step extrema; _n_updates counts attempted critic epochs, with completed epochs reported separately",
        "deadline": "Stop before the next optimization minibatch if the registered deadline has passed, retain partial counters, and perform no further rollout",
        "environment_changes": False, "runtime_operating_rules": False}
