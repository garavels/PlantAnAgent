"""Train-only demonstrations and actor cloning without a deployed teacher.

Optional DAgger collection labels states visited by an already learned actor;
the learner still receives only the original public149 observation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from docs.research.tools.wholeplant_ppo_v1 import training as common

SCHEMA = "learned-control-demonstrations-v1"


def transform_targets(actions, off_target_margin=0., on_target_margin=0.):
    """Boundary labels may overshoot; only the existing Box clipping deploys."""
    raw = np.asarray(actions, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 7 or not np.isfinite(raw).all():
        raise ValueError("Demonstration actions must have shape(N,7) and be finite")
    if np.any(raw < -1) or np.any(raw > 1):
        raise ValueError("Teacher actions must be in the original action Box")
    if min(off_target_margin, on_target_margin) < 0:
        raise ValueError("Boundary margins must be nonnegative")
    result = raw.copy()
    result[raw == -1.] -= off_target_margin
    result[raw == 1.] += on_target_margin
    return result


def undiscounted_returns(rewards):
    raw = np.asarray(rewards, dtype=np.float64)
    return np.cumsum(raw[::-1], dtype=np.float64)[::-1].copy()


def _source_pins():
    from docs.research.tools.learned_control_v1 import environment
    paths = {Path(record["path"]).resolve() for record in environment.source_records()}
    paths.add(Path(__file__).resolve())
    return [common.file_record(path) for path in sorted(paths)]


def collect_dataset(root: Path, cases: list[dict], *, teacher_parameters=None,
                    learner_checkpoint: Path | None = None, teacher_probability=1.,
                    sampling_seed=7301, deadline_utc="2026-09-18T05:50:00Z"):
    """Freeze a collection recipe before simulating any registered case."""
    from docs.research.tools.learned_control_v1.environment import FixedEnv
    from docs.research.tools.wholeplant_ppo_v1.baselines import BaselinePolicy, BaselineConfig
    from docs.research.tools.learned_control_v1.training import load_checkpoint
    common.configure_torch()
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    deadline = common._deadline(deadline_utc)
    if not 0 <= teacher_probability <= 1:
        raise ValueError("Teacher mixture probability must be in[0,1]")
    if learner_checkpoint is None and teacher_probability != 1:
        raise ValueError("Non-teacher collection requires a learner checkpoint")
    if not cases or len({(row["condition"], int(row["seed"])) for row in cases}) != len(cases):
        raise ValueError("Collection cases must be nonempty and unique")
    config = BaselineConfig(**(teacher_parameters or {}))
    teacher = BaselinePolicy("storage", config)
    learner = load_checkpoint(learner_checkpoint) if learner_checkpoint else None
    learner_pins = [] if learner_checkpoint is None else [common.file_record(Path(learner_checkpoint)),
        common.file_record(Path(learner_checkpoint).with_suffix(".json")),
        common.file_record(Path(learner_checkpoint).with_suffix(".vecnormalize.pkl"))]
    plan = {"schema": SCHEMA, "created_at": common.now(), "cases": cases,
            "teacher": teacher.metadata(), "teacher_parameters": asdict(config),
            "learner_dependencies": learner_pins, "teacher_probability": teacher_probability,
            "sampling_seed": sampling_seed, "deadline_utc": deadline_utc,
            "source_files": _source_pins(), "split": "train",
            "label": "Teacher action on actual public observation regardless of behavior policy",
            "returns": "Undiscounted actual behavior-trajectory rewards, scaled0.001; no invented reward"}
    common.publish_json(root / "collection-plan.json", plan)
    rng = np.random.default_rng(sampling_seed)
    observations, actions, rewards, returns, episode_indices, behavior_flags = [], [], [], [], [], []
    rows = []
    started = time.monotonic()
    try:
        for index, case in enumerate(cases):
            env = FixedEnv(condition=case["condition"], split="train", record_minutes=False)
            try:
                obs, _ = env.reset(seed=int(case["seed"]))
                episode_rewards = []
                done = False
                while not done:
                    if datetime.now(timezone.utc) >= deadline:
                        raise TimeoutError("Demonstration deadline reached; incomplete dataset is not published")
                    label = teacher.predict(obs, deterministic=True)[0]
                    teacher_step = learner is None or rng.random() < teacher_probability
                    action = label if teacher_step else learner.predict(obs, deterministic=True)[0]
                    observations.append(np.asarray(obs, dtype=np.float32).copy())
                    actions.append(np.asarray(label, dtype=np.float32).copy())
                    episode_indices.append(index)
                    behavior_flags.append(teacher_step)
                    obs, reward, terminated, truncated, info = env.step(action)
                    rewards.append(reward * .001)
                    episode_rewards.append(reward * .001)
                    done = terminated or truncated
                returns.extend(undiscounted_returns(episode_rewards).tolist())
                row = {"case": case, "decisions": len(episode_rewards),
                       "raw_reward_eur": float(sum(episode_rewards) / .001),
                       "economic_metrics": info.get("economic_metrics"), "metrics": info.get("metrics")}
                rows.append(row)
                common.append_json(root / "episodes.jsonl", row)
                print(json.dumps({"demonstration_episode": index + 1, "total_episodes": len(cases),
                                  "elapsed_s": time.monotonic() - started}), flush=True)
            finally:
                env.close()
        if _source_pins() != plan["source_files"]:
            raise ValueError("Demonstration sources changed during collection")
        path = root / "demonstrations.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, observations=np.asarray(observations, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.float32), rewards=np.asarray(rewards, dtype=np.float32),
                returns=np.asarray(returns, dtype=np.float32), episode_indices=np.asarray(episode_indices, dtype=np.int32),
                behavior_teacher=np.asarray(behavior_flags, dtype=np.bool_))
        metadata = {"schema": SCHEMA, "completed_at": common.now(), "dataset": common.file_record(path),
                    "plan": common.file_record(root / "collection-plan.json"), "plan_sha256": common.sha(plan),
                    "source_files": plan["source_files"], "samples": len(observations),
                    "episodes": len(rows), "observation_shape": [149], "action_shape": [7],
                    "teacher_parameters": asdict(config), "cases": cases,
                    "elapsed_s": time.monotonic() - started, "summary": rows}
        common.publish_json(path.with_suffix(".json"), metadata)
        return metadata
    except BaseException as exc:
        common.publish_json(root / "failure.json", {"at": common.now(), "type": type(exc).__name__,
            "error": str(exc), "collected_samples": len(observations), "completed_episodes": len(rows)})
        raise
    finally:
        if learner is not None:
            learner.close()


def require_dataset(path: Path):
    path = Path(path).resolve()
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["schema"] != SCHEMA:
        raise ValueError("Unknown demonstration schema")
    actual = common.file_record(path)
    if any(actual[key] != metadata["dataset"][key] for key in ("bytes", "sha256")):
        raise ValueError("Demonstration bytes changed")
    plan_path = Path(metadata["plan"]["path"])
    if common.file_record(plan_path) != metadata["plan"]:
        raise ValueError("Demonstration collection plan changed")
    plan = json.loads(plan_path.read_text())
    if common.sha(plan) != metadata["plan_sha256"] or plan["split"] != "train":
        raise ValueError("Demonstration plan is not the registered train-only source")
    with np.load(path, allow_pickle=False) as data:
        n = metadata["samples"]
        if data["observations"].shape != (n, 149) or data["actions"].shape != (n, 7):
            raise ValueError("Demonstration interface differs")
        if not all(np.isfinite(data[key]).all() for key in ("observations", "actions", "rewards", "returns")):
            raise ValueError("Demonstrations contain nonfinite values")
        transform_targets(data["actions"])
    return metadata


def fit_clone(model, normalizer, dataset: Path, config, *, optimizer_seed, out: Path, deadline):
    import torch
    common.configure_torch()
    metadata = require_dataset(dataset)
    with np.load(dataset, allow_pickle=False) as data:
        observations = data["observations"].copy()
        labels = data["actions"].copy()
        returns = data["returns"].copy()
    # Only this training dataset fits the scaler. No development observation
    # enters its moments, and no terminal or future information enters inputs.
    normalizer.obs_rms.update(observations)
    inputs = torch.as_tensor(normalizer.normalize_obs(observations), dtype=torch.float32)
    targets = torch.as_tensor(transform_targets(labels, config["off_target_margin"], config["on_target_margin"]))
    values = torch.as_tensor(returns, dtype=torch.float32)
    rng = np.random.default_rng(optimizer_seed)
    actor_parameters = list(model.policy.mlp_extractor.policy_net.parameters()) + list(model.policy.action_net.parameters())
    optimizer = torch.optim.Adam(actor_parameters, lr=config["learning_rate"])
    started = time.monotonic()
    model.policy.set_training_mode(True)

    def mean_actions(batch):
        return model.policy.get_distribution(batch).distribution.mean

    for epoch in range(config["epochs"]):
        losses = []
        for batch in np.array_split(rng.permutation(len(inputs)), max(1, int(np.ceil(len(inputs) / config["batch_size"])))):
            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Deadline reached during behavior cloning")
            prediction = mean_actions(inputs[batch])
            loss = torch.nn.functional.mse_loss(prediction, targets[batch])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            mean = mean_actions(inputs)
            error = torch.abs(torch.clamp(mean, -1, 1) - torch.as_tensor(labels))
            off = torch.as_tensor(labels == -1)
            row = {"epoch": epoch + 1, "actor_mse": float(np.mean(losses)),
                   "clipped_action_mae": float(error.mean()), "clipped_action_p99_error": float(torch.quantile(error, .99)),
                   "teacher_off_leak_fraction": float((mean[off] > -1).float().mean()) if off.any() else 0.,
                   "elapsed_s": time.monotonic() - started}
        common.append_json(out / "cloning.jsonl", row)
        if (epoch + 1) % 10 == 0 or epoch + 1 == config["epochs"]:
            print(json.dumps(row), flush=True)
    critic_parameters = list(model.policy.mlp_extractor.value_net.parameters()) + list(model.policy.value_net.parameters())
    critic_optimizer = torch.optim.Adam(critic_parameters, lr=config["critic_learning_rate"])
    for epoch in range(config["critic_epochs"]):
        losses = []
        for batch in np.array_split(rng.permutation(len(inputs)), max(1, int(np.ceil(len(inputs) / config["batch_size"])))):
            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Deadline reached during critic initialization")
            prediction = model.policy.predict_values(inputs[batch]).flatten()
            loss = torch.nn.functional.mse_loss(prediction, values[batch])
            critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_parameters, 5.)
            critic_optimizer.step()
            losses.append(float(loss.detach()))
        common.append_json(out / "critic-initialization.jsonl", {"epoch": epoch + 1, "mse": float(np.mean(losses))})
    result = {"schema": "learned-control-cloning-v1", "config": config,
              "dataset": metadata["dataset"], "samples": len(inputs), "completed_at": common.now(),
              "elapsed_s": time.monotonic() - started,
              "supervision": "Teacher actions, with declared boundary-target transform; no runtime teacher or snapping",
              "critic_target": "Actual undiscounted behavior-trajectory returns scaled0.001",
              "ppo_optimizer_state_initialized": False}
    common.publish_json(out / "cloning-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True,
                        help="JSON with cases, optionalteacher_parameters, learner_checkpoint and collection settings")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    if recipe.get("learner_checkpoint"):
        recipe["learner_checkpoint"] = Path(recipe["learner_checkpoint"])
    print(json.dumps(collect_dataset(args.root, **recipe), indent=2))


if __name__ == "__main__":
    main()
