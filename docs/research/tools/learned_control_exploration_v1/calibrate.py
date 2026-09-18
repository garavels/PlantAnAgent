"""Fit only a hybrid actor's scale head, preserving every deterministic action."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from docs.research.tools.learned_control_hybrid_v1 import training as hybrid
from docs.research.tools.learned_control_hybrid_v1.policy import distribution_identity
from docs.research.tools.learned_control_v1 import imitation, environment
from docs.research.tools.wholeplant_ppo_v1 import training as common

SCHEMA = "learned-control-exploration-calibration-v1"


def fit_scale_head(latent, labels, weight, bias, *, minimum, maximum, fraction,
                   epochs, batch_size, learning_rate, seed, deadline, output=None):
    """Train an independent linear head; no actor or critic parameter is mutable."""
    head = torch.nn.Linear(latent.shape[1], 7)
    with torch.no_grad():
        head.weight.copy_(weight)
        head.bias.copy_(bias)
    on = (labels > -1).float()
    balance = on / on.mean(0).clamp_min(1 / len(on))
    target = (fraction * (labels + 1)).clamp(minimum * 1.1, maximum * .999)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    initial_loss = None
    for epoch in range(epochs):
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("Exploration calibration deadline reached")
        losses = []
        for ids in np.array_split(rng.permutation(len(latent)), max(1, int(np.ceil(len(latent) / batch_size)))):
            scale = minimum + (maximum - minimum) * torch.sigmoid(head(latent[ids]))
            loss = (balance[ids] * (torch.log(scale) - torch.log(target[ids])).square()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        row = {"epoch": epoch + 1, "log_scale_mse": float(np.mean(losses))}
        if initial_loss is None:
            initial_loss = row["log_scale_mse"]
        if output:
            common.append_json(Path(output) / "calibration.jsonl", row)
        if (epoch + 1) % 20 == 0 or epoch + 1 == epochs:
            print(json.dumps(row), flush=True)
    with torch.no_grad():
        scale = minimum + (maximum - minimum) * torch.sigmoid(head(latent))
    stats = {"initial_log_scale_mse": initial_loss, "last_log_scale_mse": row["log_scale_mse"],
             "mean_on_std_by_coordinate": (scale * on).sum(0).div(on.sum(0).clamp_min(1)).tolist(),
             "target_on_std_by_coordinate": (target * on).sum(0).div(on.sum(0).clamp_min(1)).tolist()}
    return head.state_dict(), stats


def calibrate(root, parent, dataset, *, fraction=.03, epochs=120, seed=903,
              deadline_utc="2026-09-18T01:00:00Z"):
    common.configure_torch()
    root, parent, dataset = (Path(p).resolve() for p in (root, parent, dataset))
    metadata = imitation.require_dataset(dataset)
    policy = hybrid.load_checkpoint(parent)
    model = policy.model
    if model.num_timesteps or model.policy.optimizer.state:
        raise ValueError("Calibrate an initialized, untrained checkpoint with empty PPO optimizer")
    if not 0 < fraction < .5 or epochs < 1:
        raise ValueError("Invalid scale calibration recipe")
    root.mkdir(parents=True, exist_ok=False)
    recipe = {"parent_checkpoint": str(parent), "dataset": str(dataset), "fraction": fraction,
              "epochs": epochs, "seed": seed, "batch_size": 1024, "learning_rate": .003,
              "deadline_utc": deadline_utc}
    source_paths = {Path(row["path"]) for row in environment.source_records()}
    source_paths.update(Path(__file__).parent.glob("*.py"))
    source_paths.update(Path(hybrid.__file__).parent.glob("*.py"))
    source_paths.update((Path(imitation.__file__), Path(common.__file__)))
    plan = {"schema": SCHEMA, "recipe": recipe, "created_at": common.now(),
            "environment_identity": environment.environment_identity(),
            "source_files": [common.file_record(p) for p in sorted(source_paths)],
            "dependencies": [common.file_record(parent.with_suffix(s)) for s in (".zip", ".json", ".vecnormalize.pkl")]
                + [common.file_record(dataset), common.file_record(dataset.with_suffix(".json"))],
            "method": "Fit a neural scale head on training-only teacher ON actions; all gates, means, shared features, critic and observation statistics stay exact. No inference rule or action restriction added."}
    common.publish_json(root / "plan.json", plan)
    started = time.monotonic()
    with np.load(dataset, allow_pickle=False) as data:
        observations, actions = data["observations"].copy(), data["actions"].copy()
    normalized = torch.as_tensor(policy.normalize_observation(observations), dtype=torch.float32)
    labels = torch.as_tensor(actions, dtype=torch.float32)
    actor = model.policy
    before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    before_optimizer = common.optimizer_sha256(model)
    with torch.no_grad():
        before_actions = actor.get_distribution(normalized).mode().clone()
        features = actor.extract_features(normalized)
        latent = actor.mlp_extractor.forward_actor(features).detach()
    kw = model.policy_kwargs
    fitted, stats = fit_scale_head(latent, labels, actor.action_net.weight[14:].detach(),
        actor.action_net.bias[14:].detach(), minimum=kw["min_on_std"], maximum=kw["max_on_std"],
        fraction=fraction, epochs=epochs, batch_size=recipe["batch_size"],
        learning_rate=recipe["learning_rate"], seed=seed, deadline=common._deadline(deadline_utc), output=root)
    with torch.no_grad():
        actor.action_net.weight[14:].copy_(fitted["weight"])
        actor.action_net.bias[14:].copy_(fitted["bias"])
        after_actions = actor.get_distribution(normalized).mode()
    if not torch.equal(before_actions, after_actions):
        raise AssertionError("Exploration calibration changed deterministic policy")
    for name, tensor in actor.state_dict().items():
        if name in ("action_net.weight", "action_net.bias"):
            if not torch.equal(tensor[:14], before[name][:14]):
                raise AssertionError("Calibration changed gate or mean output weights")
        elif not torch.equal(tensor, before[name]):
            raise AssertionError(f"Calibration changed frozen parameter:{name}")
    if common.optimizer_sha256(model) != before_optimizer:
        raise AssertionError("Calibration changed PPO optimizer")
    model.research_plan_sha256 = common.sha(plan)
    out = root / "training"
    out.mkdir()
    checkpoint = out / "calibrated-policy.zip"
    model.save(checkpoint)
    normalizer = checkpoint.with_suffix(".vecnormalize.pkl")
    shutil.copyfile(parent.with_suffix(".vecnormalize.pkl"), normalizer)
    hashes = {"policy_tensor_sha256": common.policy_parameter_sha256(model),
              "actor_tensor_sha256": common.actor_parameter_sha256(model),
              "actor_mean_tensor_sha256": common.actor_mean_parameter_sha256(model),
              "optimizer_state_sha256": common.optimizer_sha256(model)}
    record = {"schema": SCHEMA, "kind": "exploration-calibrated-initialization", "created_at": common.now(),
              **hashes, "optimizer_seed": int(model.seed), "num_timesteps": 0, "additional_timesteps": 0,
              "ppo_update_count": 0, "plan_sha256": common.sha(plan),
              "environment_identity": plan["environment_identity"], "policy_distribution": distribution_identity(),
              "checkpoint": common.file_record(checkpoint), "normalizer": common.file_record(normalizer),
              "normalizer_observation_count": float(policy.normalizer.obs_rms.count),
              "entropy_coefficient": float(model.ent_coef), "gaussian_log_std": None,
              "deterministic_policy_structurally_unchanged": True,
              "exact_training_action_witness_count": len(observations),
              "actor_hash_scope": "All actor features and learned gates, ON means and ON scales"}
    common.publish_json(checkpoint.with_suffix(".json"), record)
    restored = hybrid.load_checkpoint(checkpoint)
    try:
        if not np.array_equal(restored.predict(observations, deterministic=True)[0], before_actions.numpy()):
            raise AssertionError("Reloaded deterministic actions changed")
        result = {"schema": SCHEMA, "completed_at": common.now(), "elapsed_s": time.monotonic()-started,
                  "checkpoint": common.file_record(checkpoint), "dataset": metadata["dataset"],
                  "exact_deterministic_action_witnesses": len(observations), "frozen_parameters_verified": True,
                  "normalizer_bytes_preserved": True, "checkpoint_reopen_verified": True, **stats}
        common.publish_json(root / "result.json", result)
        return result
    finally:
        restored.close()
        policy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=.03)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=903)
    parser.add_argument("--deadline-utc", default="2026-09-18T01:00:00Z")
    args = vars(parser.parse_args())
    print(json.dumps(calibrate(**args)), flush=True)


if __name__ == "__main__":
    main()
