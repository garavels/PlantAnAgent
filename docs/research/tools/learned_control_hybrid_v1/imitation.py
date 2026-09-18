"""Class-balanced OFF classification and conditional ON-action imitation."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from docs.research.tools.learned_control_v1 import imitation as original
from docs.research.tools.wholeplant_ppo_v1 import training as common

require_dataset = original.require_dataset
collect_dataset = original.collect_dataset


def balanced_gate_weights(off_labels):
    """Give each represented class half the loss within each coordinate."""
    labels = np.asarray(off_labels, dtype=bool)
    counts = labels.sum(axis=0)
    size = len(labels)
    off_weight = size / (2 * np.maximum(counts, 1))
    on_weight = size / (2 * np.maximum(size - counts, 1))
    return np.where(labels, off_weight, on_weight).astype(np.float32)


def fit_clone(model, normalizer, dataset, config, *, optimizer_seed, out, deadline):
    common.configure_torch()
    metadata = require_dataset(Path(dataset))
    with np.load(dataset, allow_pickle=False) as data:
        observations, labels, returns = (data[key].copy() for key in ("observations", "actions", "returns"))
    normalizer.obs_rms.update(observations)
    inputs = torch.as_tensor(normalizer.normalize_obs(observations), dtype=torch.float32)
    targets = torch.as_tensor(labels, dtype=torch.float32)
    off_np = labels == -1.
    off = torch.as_tensor(off_np, dtype=torch.float32)
    on = 1 - off
    weights = torch.as_tensor(balanced_gate_weights(off_np))
    gate_target = (2 * off - 1) * config["gate_logit_target"]
    on_frequency = on.mean(dim=0).clamp_min(1 / len(on))
    on_balance = on / on_frequency
    relative_floor = config["relative_magnitude_floor"]
    magnitude_scale = (targets + 1).clamp_min(2 * relative_floor) if relative_floor else torch.ones_like(targets)
    rng = np.random.default_rng(optimizer_seed)
    parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"])
    model.policy.set_training_mode(True)
    started = time.monotonic()

    for epoch in range(config["epochs"]):
        losses = []
        batches = np.array_split(rng.permutation(len(inputs)), max(1, int(np.ceil(len(inputs) / config["batch_size"]))))
        for batch in batches:
            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Deadline reached during hybrid behavior cloning")
            distribution = model.policy.get_distribution(inputs[batch])
            logits, means = distribution.gate_logits.float(), distribution.on_mean.float()
            classification = F.binary_cross_entropy_with_logits(logits, off[batch], reduction="none")
            confidence = ((logits - gate_target[batch]) / config["gate_logit_target"]).square()
            gate_loss = (weights[batch] * (classification + config["gate_confidence_weight"] * confidence)).mean()
            magnitude_loss = (on_balance[batch] * ((means - targets[batch]) / magnitude_scale[batch]).square()).mean()
            loss = config["gate_loss_weight"] * gate_loss + config["magnitude_loss_weight"] * magnitude_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            optimizer.step()
            losses.append([float(loss.detach()), float(gate_loss.detach()), float(magnitude_loss.detach())])
        with torch.no_grad():
            distribution = model.policy.get_distribution(inputs)
            predicted_off = distribution.gate_logits >= 0
            deterministic = distribution.mode()
            physically_off = deterministic == -1.
            abs_error = torch.abs(deterministic - targets)
            missed_on = (physically_off.float() * on).sum(dim=0) / on.sum(dim=0).clamp_min(1)
            leaked_off = ((~physically_off).float() * off).sum(dim=0) / off.sum(dim=0).clamp_min(1)
            row = {"epoch": epoch + 1, "total_loss": float(np.mean(losses, axis=0)[0]),
                   "gate_loss": float(np.mean(losses, axis=0)[1]),
                   "conditional_magnitude_loss": float(np.mean(losses, axis=0)[2]),
                   "deterministic_action_mae": float(abs_error.mean()),
                   "teacher_on_predicted_off_fraction_by_coordinate": missed_on.tolist(),
                   "teacher_off_predicted_on_fraction_by_coordinate": leaked_off.tolist(),
                   "teacher_on_gate_classified_off_fraction_by_coordinate": (predicted_off.float() * on).sum(dim=0).div(on.sum(dim=0).clamp_min(1)).tolist(),
                   "mean_off_probability_on_teacher_off": (torch.sigmoid(distribution.gate_logits) * off).sum(dim=0).div(off.sum(dim=0).clamp_min(1)).tolist(),
                   "on_magnitude_mae_by_coordinate": (torch.abs(distribution.on_mean.float().clamp(-1, 1) - targets) * on).sum(dim=0).div(on.sum(dim=0).clamp_min(1)).tolist(),
                   "elapsed_s": time.monotonic() - started}
        common.append_json(Path(out) / "cloning.jsonl", row)
        if (epoch + 1) % 10 == 0 or epoch + 1 == config["epochs"]:
            import json
            print(json.dumps(row), flush=True)
    critic_parameters = list(model.policy.mlp_extractor.value_net.parameters()) + list(model.policy.value_net.parameters())
    critic_optimizer = torch.optim.Adam(critic_parameters, lr=config["critic_learning_rate"])
    values = torch.as_tensor(returns, dtype=torch.float32)
    for epoch in range(config["critic_epochs"]):
        losses = []
        for batch in np.array_split(rng.permutation(len(inputs)), max(1, int(np.ceil(len(inputs) / config["batch_size"])))):
            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Deadline reached during hybrid critic initialization")
            prediction = model.policy.predict_values(inputs[batch]).flatten()
            loss = F.mse_loss(prediction, values[batch])
            critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_parameters, 5.)
            critic_optimizer.step()
            losses.append(float(loss.detach()))
        common.append_json(Path(out) / "critic-initialization.jsonl", {"epoch": epoch + 1, "mse": float(np.mean(losses))})
    result = {"schema": "learned-control-hybrid-cloning-v1", "config": config,
              "dataset": metadata["dataset"], "samples": len(inputs), "completed_at": common.now(),
              "elapsed_s": time.monotonic() - started,
              "teacher_off_fraction_by_coordinate": off.mean(dim=0).tolist(),
              "all_rare_on_rows_retained": True,
              "supervision": "Balanced learnedOFFclassification plus conditionalONmagnitude fitting; no runtime teacher or plant rule",
              "critic_target": "Actual undiscounted behavior-trajectory returns scaled0.001",
              "ppo_optimizer_state_initialized": False}
    common.publish_json(Path(out) / "cloning-result.json", result)
    return result
