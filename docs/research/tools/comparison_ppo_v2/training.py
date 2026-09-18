"""Fresh 30 kg/h comparison PPO with fixed endpoints and a bounded local runtime.

Only the scalar environment reward enters PPO. No demonstrations, controller
prior, constraint learner or historical policy parameters are imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np

SCHEMA = "comparison-direct-scalar-ppo-v2"
TRAIN_BASE = 20_000_000
TRAIN_END = 20_100_000
DEFAULT_SEEDS = (101, 211, 307)
DEFAULT_DEADLINE = "2026-09-17T17:27:48Z"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def publish_json(path: Path, value: Any) -> None:
    """Publish an immutable JSON record; retained attempts are never overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def append_json(path: Path, value: Any) -> None:
    with path.open("a") as stream:
        stream.write(canonical_bytes(value).decode() + "\n")


def _deadline(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Deadline must include a timezone")
    return result.astimezone(timezone.utc)


def _environment():
    from docs.research.tools.comparison_ppo_v2 import environment
    return environment


def source_records() -> list[dict[str, Any]]:
    records = _environment().source_records()
    paths = {Path(item["path"]).resolve() for item in records}
    paths.add(Path(__file__).resolve())
    paths.add(ROOT / "uv.lock")
    protocol = ROOT / "docs/research/COMPARISON_PPO_V2_PROTOCOL.md"
    if protocol.exists():
        paths.add(protocol)
    return [file_record(path) for path in sorted(paths)]


def runtime_record() -> dict[str, Any]:
    return {
        "python": sys.version, "executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "gymnasium", "torch", "stable-baselines3", "scipy")},
        "torch_intraop_threads": 1, "torch_interop_threads": 1,
    }


def build_plan(*, total_timesteps: int, optimizer_seeds=DEFAULT_SEEDS,
               deadline_utc=DEFAULT_DEADLINE, environments=4, n_steps=2048,
               checkpoint_steps=65536, ent_coef=0.01) -> dict[str, Any]:
    _deadline(deadline_utc)
    block = environments * n_steps
    if environments < 1 or n_steps < 2 or total_timesteps <= 0 or total_timesteps % block:
        raise ValueError("The fixed transition budget must contain whole vector rollouts")
    if checkpoint_steps <= 0 or checkpoint_steps % block:
        raise ValueError("Diagnostic checkpoint interval must contain whole rollouts")
    if not optimizer_seeds or len(set(optimizer_seeds)) != len(optimizer_seeds):
        raise ValueError("Optimizer seeds must be nonempty and unique")
    return {
        "schema": SCHEMA, "algorithm": "stable_baselines3.PPO",
        "training_variant": {"delivery_rate_kg_h": 30.0, "opening_battery": "sampled", "battery_mode": "enabled"},
        "optimizer_seeds": list(optimizer_seeds), "total_timesteps": total_timesteps,
        "environments": environments, "n_steps": n_steps,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_timesteps": list(range(checkpoint_steps, total_timesteps, checkpoint_steps)),
        "evaluation_checkpoint_timesteps": sorted(set([0, total_timesteps] +
                                                     [step for step in (65536, total_timesteps // 2)
                                                      if 0 < step < total_timesteps and step % block == 0])),
        "deadline_utc": deadline_utc,
        "deadline_policy": "Stop at next vector decision; preserve incomplete checkpoint, no final endpoint or success label.",
        "initialization": "Fresh standard SB3 MlpPolicy random weights and default Gaussian log_std=0; no imported tensors or behavior cloning.",
        "reward_scale": 0.001,
        "observation_normalization": {"norm_obs": True, "norm_reward": False,
                                      "clip_obs": 10.0, "epsilon": 1e-8},
        "objective": "Only the actual scalar interval economic reward; daily costs are diagnostics and do not enter PPO.",
        "ppo": {"learning_rate": 0.0003, "batch_size": 256, "n_epochs": 10,
                "gamma": 1.0, "gae_lambda": 0.95, "clip_range": 0.2,
                "ent_coef": float(ent_coef), "vf_coef": 0.5, "max_grad_norm": 0.5,
                "device": "cpu", "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}},
        "training_seeds": {"first": TRAIN_BASE, "exclusive_last": TRAIN_END,
                           "mapping": "20000000 + environments * local_episode_ordinal + environment_index",
                           "same_across_optimizer_seeds": True},
        "selection": "All fixed final endpoints, no best-policy or development-score selection. Intermediate checkpoints are diagnostics only.",
        "untrained_comparator": {"optimizer_seed": 101, "checkpoint": "initial-policy.zip"},
        "environment_identity": _environment().environment_identity(),
        "environment_identity_sha256": sha(_environment().environment_identity()),
        "source_identity_sha256": (_environment().source_identity_sha256()
                                   if hasattr(_environment(), "source_identity_sha256")
                                   else sha(_environment().source_records())),
        "runtime": runtime_record(), "source_files": source_records(),
        "registered_evaluation_eligible": False,
    }


def freeze_plan(root: Path, **kwargs) -> dict[str, Any]:
    root = root.resolve()
    record = build_plan(**kwargs)
    root.mkdir(parents=True, exist_ok=True)
    publish_json(root / "plan.json", record)
    return record


def require_plan(root: Path) -> dict[str, Any]:
    record = json.loads((Path(root) / "plan.json").read_text())
    expected = build_plan(total_timesteps=record["total_timesteps"],
                          optimizer_seeds=record["optimizer_seeds"],
                          deadline_utc=record["deadline_utc"],
                          environments=record["environments"], n_steps=record["n_steps"],
                          checkpoint_steps=record["checkpoint_steps"],
                          ent_coef=record["ppo"]["ent_coef"])
    if canonical_bytes(record) != canonical_bytes(expected):
        raise ValueError("Frozen training settings, environment, runtime or source changed")
    return record


class TrainingEpisodes(gym.Wrapper):
    """Optimizer-independent scenario allocation, with compact outcome records."""
    def __init__(self, env, *, index: int, count: int):
        super().__init__(env)
        self.index, self.count = index, count
        self.resets = self.decisions = self.episode_decisions = 0
        self.episode_reward = 0.0
        self.current_seed: int | None = None
        self.completed: list[dict[str, Any]] = []

    def reset(self, *, seed=None, options=None):
        actual = TRAIN_BASE + self.count * self.resets + self.index
        if actual >= TRAIN_END:
            raise ValueError("Exhausted frozen training seed domain")
        observation, info = self.env.reset(seed=actual, options=options)
        self.resets += 1
        self.current_seed = actual
        self.episode_reward = 0.0
        self.episode_decisions = 0
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.decisions += 1
        self.episode_decisions += 1
        self.episode_reward += float(reward)
        if terminated or truncated:
            row = {"seed": self.current_seed, "decisions": self.episode_decisions,
                   "raw_reward_eur": self.episode_reward,
                   "terminated": bool(terminated), "truncated": bool(truncated)}
            for key in ("economic_metrics", "metrics"):
                if key in info:
                    row[key] = info[key]
            metrics = info.get("metrics", {})
            row["delivered_fraction"] = metrics.get("delivery_fill_rate")
            row["delivered_kg"] = metrics.get("delivered_kg")
            row["unmet_kg"] = metrics.get("unmet_kg")
            row["forced_stops"] = metrics.get("electrolyser_forced_stops")
            costs = info.get("cycle_constraint", {})
            row["all_settled_cycles_feasible"] = costs.get("all_settled_cycles_feasible")
            row["settled_cycle_count"] = costs.get("settled_cycle_count")
            self.completed.append(row)
            info["training_episode"] = row
        return observation, reward, terminated, truncated, info

    def snapshot(self):
        return {"environment_index": self.index, "resets": self.resets,
                "decisions": self.decisions, "current_seed": self.current_seed,
                "current_episode_decisions": self.episode_decisions,
                "completed_episodes": len(self.completed)}


class ScaleReward(gym.Wrapper):
    def __init__(self, env, scale=0.001):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, float(reward) * self.scale, terminated, truncated, info


def configure_torch() -> None:
    import torch
    torch.set_num_threads(1)
    # A repeated call in the same test process may follow initialized interop.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)


def make_learner(plan: dict[str, Any], optimizer_seed: int):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    configure_torch()
    if optimizer_seed not in plan["optimizer_seeds"]:
        raise ValueError("Optimizer seed is outside frozen plan")
    wrappers = []

    def factory(index):
        def create():
            wrapped = TrainingEpisodes(_environment().ComparisonEnv(delivery_rate_kg_h=30.0, split="train"),
                                       index=index, count=plan["environments"])
            wrappers.append(wrapped)
            return ScaleReward(Monitor(wrapped), plan["reward_scale"])
        return create

    vector = VecNormalize(DummyVecEnv([factory(i) for i in range(plan["environments"])]),
                          **plan["observation_normalization"])
    try:
        model = PPO("MlpPolicy", vector, seed=optimizer_seed, n_steps=plan["n_steps"],
                    verbose=0, **plan["ppo"])
        model.research_identity = plan["environment_identity"]
        model.research_plan_sha256 = sha(plan)
        return model, vector, wrappers
    except BaseException:
        vector.close()
        raise


def _tensor_hash(items) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(items):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def policy_parameter_sha256(model) -> str:
    return _tensor_hash(model.policy.state_dict().items())


def actor_parameter_sha256(model) -> str:
    return _tensor_hash((name, tensor) for name, tensor in model.policy.state_dict().items()
                        if name == "log_std" or name.startswith(("action_net.", "mlp_extractor.policy_net.", "features_extractor.")))


def actor_mean_parameter_sha256(model) -> str:
    return _tensor_hash((name, tensor) for name, tensor in model.policy.state_dict().items()
                        if name.startswith(("action_net.", "mlp_extractor.policy_net.", "features_extractor.")))


def optimizer_sha256(model) -> str:
    def convert(value):
        if hasattr(value, "detach"):
            a = value.detach().cpu().contiguous().numpy()
            return {"dtype": str(a.dtype), "shape": list(a.shape),
                    "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value
    return sha(convert(model.policy.optimizer.state_dict()))


def save_checkpoint(model, path: Path, *, plan, wrappers, kind, probe_observations=None):
    from stable_baselines3 import PPO
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".writing.zip")
    if temporary.exists():
        raise FileExistsError(temporary)
    model.save(temporary)
    temporary.rename(path)
    normalizer_path = path.with_suffix(".vecnormalize.pkl")
    normalizer = model.get_vec_normalize_env()
    if normalizer is None or normalizer_path.exists():
        raise ValueError("Checkpoint requires an unpublished observation normalizer")
    normalizer.save(normalizer_path)
    restored_normalizer = _load_normalizer(normalizer_path)
    try:
        for attribute in ("mean", "var", "count"):
            if not np.array_equal(getattr(normalizer.obs_rms, attribute),
                                  getattr(restored_normalizer.obs_rms, attribute)):
                raise ValueError("Checkpoint observation statistics did not reopen exactly")
    finally:
        restored_normalizer.close()
    reopened = PPO.load(path, device="cpu")
    policy_hash = policy_parameter_sha256(model)
    actor_hash = actor_parameter_sha256(model)
    optim_hash = optimizer_sha256(model)
    if (policy_parameter_sha256(reopened) != policy_hash or
            actor_parameter_sha256(reopened) != actor_hash or
            optimizer_sha256(reopened) != optim_hash or
            reopened.num_timesteps != model.num_timesteps or
            reopened.research_identity != plan["environment_identity"] or
            reopened.research_plan_sha256 != sha(plan)):
        raise ValueError("Checkpoint did not reopen exactly")
    prediction_verified = False
    if probe_observations is not None:
        before, _ = model.predict(probe_observations, deterministic=True)
        after, _ = reopened.predict(probe_observations, deterministic=True)
        if not np.array_equal(before, after):
            raise ValueError("Reopened checkpoint changed deterministic predictions")
        prediction_verified = True
    record = {"schema": SCHEMA, "kind": kind, "created_at": now(),
              "optimizer_seed": int(model.seed), "num_timesteps": model.num_timesteps,
              "ppo_update_count": model._n_updates,
              "optimizer_minibatch_steps": model._n_updates * ((plan["environments"] * plan["n_steps"] + plan["ppo"]["batch_size"] - 1) // plan["ppo"]["batch_size"]),
              "plan_sha256": sha(plan), "environment_identity": plan["environment_identity"],
              "checkpoint": file_record(path), "policy_tensor_sha256": policy_hash,
              "normalizer": file_record(normalizer_path), "normalizer_reopen_verified": True,
              "normalizer_observation_count": float(normalizer.obs_rms.count),
              "actor_tensor_sha256": actor_hash, "optimizer_state_sha256": optim_hash,
              "actor_mean_tensor_sha256": actor_mean_parameter_sha256(model),
              "gaussian_log_std": model.policy.log_std.detach().cpu().tolist(),
              "checkpoint_reopen_verified": True,
              "deterministic_prediction_reopen_verified": prediction_verified,
              "allocator": [wrapper.snapshot() for wrapper in wrappers]}
    publish_json(path.with_suffix(".json"), record)
    return record


def _load_normalizer(path: Path):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    vector = DummyVecEnv([lambda: _environment().ComparisonEnv(delivery_rate_kg_h=30.0, split="train")])
    try:
        normalizer = VecNormalize.load(path, vector)
        normalizer.training = False
        normalizer.norm_reward = False
        return normalizer
    except BaseException:
        vector.close()
        raise


class LoadedPolicy:
    """Inference on raw public observations using immutable checkpoint statistics."""
    def __init__(self, model, normalizer):
        self.model, self.normalizer = model, normalizer

    def __getattr__(self, name):
        return getattr(self.model, name)

    def normalize_observation(self, observation):
        return self.normalizer.normalize_obs(np.asarray(observation, dtype=np.float32))

    def predict(self, observation, *args, **kwargs):
        return self.model.predict(self.normalize_observation(observation), *args, **kwargs)

    def deterministic_details(self, observation):
        import torch
        normalized = self.normalize_observation(observation)
        tensor, _ = self.model.policy.obs_to_tensor(normalized)
        with torch.no_grad():
            raw = self.model.policy.get_distribution(tensor).get_actions(deterministic=True).cpu().numpy()
        if np.asarray(observation).ndim == 1:
            raw = raw[0]
        clipped = np.clip(raw, self.model.action_space.low, self.model.action_space.high)
        return {"normalized_input": normalized, "unclipped_mean": raw, "clipped_action": clipped}

    def close(self):
        self.normalizer.close()


def load_checkpoint(path: Path, plan_path: Path | None = None):
    from stable_baselines3 import PPO
    path = Path(path).resolve()
    metadata = json.loads(path.with_suffix(".json").read_text())
    actual_checkpoint = file_record(path)
    if any(metadata["checkpoint"][key] != actual_checkpoint[key] for key in ("bytes", "sha256")):
        raise ValueError("Checkpoint bytes differ from sidecar")
    normalizer_path = path.with_suffix(".vecnormalize.pkl")
    actual_normalizer = file_record(normalizer_path)
    if any(metadata["normalizer"][key] != actual_normalizer[key] for key in ("bytes", "sha256")):
        raise ValueError("Checkpoint observation statistics differ from sidecar")
    if plan_path is not None:
        plan = require_plan(Path(plan_path).parent)
        if metadata["plan_sha256"] != sha(plan):
            raise ValueError("Checkpoint belongs to another frozen plan")
    model = PPO.load(path, device="cpu")
    if (policy_parameter_sha256(model) != metadata["policy_tensor_sha256"] or
            actor_parameter_sha256(model) != metadata["actor_tensor_sha256"] or
            actor_mean_parameter_sha256(model) != metadata["actor_mean_tensor_sha256"] or
            optimizer_sha256(model) != metadata["optimizer_state_sha256"] or
            model.num_timesteps != metadata["num_timesteps"] or
            model.research_plan_sha256 != metadata["plan_sha256"] or
            model.research_identity != metadata["environment_identity"]):
        raise ValueError("Checkpoint metadata or reopened tensors differ")
    return LoadedPolicy(model, _load_normalizer(normalizer_path))


def train(root: Path, optimizer_seed: int) -> dict[str, Any]:
    from stable_baselines3.common.callbacks import BaseCallback
    root = root.resolve()
    plan = require_plan(root)
    deadline = _deadline(plan["deadline_utc"])
    if datetime.now(timezone.utc) >= deadline:
        raise ValueError("Training deadline has already passed")
    out = root / f"seed-{optimizer_seed}"
    out.mkdir(exist_ok=False)
    publish_json(out / "attempt.json", {"started_at": now(), "optimizer_seed": optimizer_seed,
                                       "plan_sha256": sha(plan), "pid": os.getpid()})
    model = vector = None
    wrappers = []
    started = time.monotonic()
    deadline_reached = False

    class Progress(BaseCallback):
        saved = set()
        initial_observations = None

        def _on_training_start(self):
            self.initial_observations = self.model._last_obs.copy()

        def _on_rollout_start(self):
            steps = self.model.num_timesteps
            if steps in plan["checkpoint_timesteps"] and steps not in self.saved:
                require_plan(root)
                record = save_checkpoint(self.model, out / "checkpoints" / f"step-{steps:06d}.zip",
                                         plan=plan, wrappers=wrappers, kind="diagnostic",
                                         probe_observations=self.initial_observations)
                self.saved.add(steps)
                append_json(out / "checkpoint-events.jsonl", record)
            if steps:
                metrics = {key: float(value) for key, value in self.model.logger.name_to_value.items()
                           if isinstance(value, (int, float, np.number)) and np.isfinite(value)}
                append_json(out / "optimization.jsonl", {"num_timesteps": steps,
                            "ppo_update_count": self.model._n_updates, "metrics": metrics})

        def _on_step(self):
            nonlocal deadline_reached
            for info in self.locals.get("infos", []):
                if "training_episode" in info:
                    append_json(out / "episodes.jsonl", info["training_episode"])
            if self.num_timesteps % 4096 == 0:
                elapsed = time.monotonic() - started
                recent = [row for wrapper in wrappers for row in wrapper.completed[-4:]]
                row = {"optimizer_seed": optimizer_seed, "num_timesteps": self.num_timesteps,
                       "elapsed_s": round(elapsed, 3), "transitions_per_second": round(self.num_timesteps / elapsed, 3),
                       "remaining_estimated_s": round((plan["total_timesteps"] - self.num_timesteps) * elapsed / self.num_timesteps, 3),
                       "at": now()}
                row["recent_completed_episode_count"] = len(recent)
                for key in ("raw_reward_eur", "delivered_fraction", "forced_stops"):
                    values = [item[key] for item in recent if isinstance(item.get(key), (int, float))]
                    row[f"recent_mean_{key}"] = float(np.mean(values)) if values else None
                append_json(out / "progress.jsonl", row)
                print(json.dumps(row), flush=True)
            if datetime.now(timezone.utc) >= deadline:
                deadline_reached = True
                return False
            return True

    try:
        model, vector, wrappers = make_learner(plan, optimizer_seed)
        initial = save_checkpoint(model, out / "initial-policy.zip", plan=plan,
                                  wrappers=wrappers, kind="initial")
        callback = Progress()
        model.learn(total_timesteps=plan["total_timesteps"], callback=callback)
        require_plan(root)
        snapshots = [wrapper.snapshot() for wrapper in wrappers]
        if sum(item["decisions"] for item in snapshots) != model.num_timesteps:
            raise ValueError("Physical transitions differ from SB3 transition ledger")
        complete = (not deadline_reached and model.num_timesteps == plan["total_timesteps"])
        artifact = out / ("final-policy.zip" if complete else "incomplete-policy.zip")
        endpoint = save_checkpoint(model, artifact, plan=plan, wrappers=wrappers,
                                   kind="final" if complete else "incomplete",
                                   probe_observations=callback.initial_observations)
        changed = endpoint["actor_tensor_sha256"] != initial["actor_tensor_sha256"]
        if complete and not changed:
            raise ValueError("Completed learner did not change actor parameters")
        result = {"schema": SCHEMA, "status": "completed" if complete else "deadline_incomplete",
                  "optimizer_seed": optimizer_seed, "completed_at": now(),
                  "elapsed_s": time.monotonic() - started,
                  "num_timesteps": model.num_timesteps, "ppo_update_count": model._n_updates,
                  "requested_timesteps": plan["total_timesteps"], "plan_sha256": sha(plan),
                  "initial_checkpoint": initial, "endpoint": endpoint,
                  "actor_parameters_changed": changed, "allocator": snapshots,
                  "actor_mean_parameters_changed": endpoint["actor_mean_tensor_sha256"] != initial["actor_mean_tensor_sha256"],
                  "completed_episodes": sum(item["completed_episodes"] for item in snapshots)}
        publish_json(out / "result.json", result)
        return result
    except BaseException as exc:
        if not (out / "failure.json").exists():
            publish_json(out / "failure.json", {"at": now(), "type": type(exc).__name__,
                                               "error": str(exc), "elapsed_s": time.monotonic() - started,
                                               "num_timesteps": getattr(model, "num_timesteps", 0)})
        raise
    finally:
        if vector is not None:
            vector.close()


def campaign(root: Path) -> dict[str, Any]:
    """Run the fixed optimizer seeds concurrently, retaining every child attempt."""
    root = root.resolve()
    plan = require_plan(root)
    if any((root / f"seed-{seed}").exists() for seed in plan["optimizer_seeds"]):
        raise FileExistsError("Campaign contains an existing attempt; never restart or overwrite it")
    logs = root / "logs"
    logs.mkdir(exist_ok=False)
    children = []
    streams = []
    started = time.monotonic()
    launch = {"started_at": now(), "plan_sha256": sha(plan), "children": []}
    try:
        for seed in plan["optimizer_seeds"]:
            path = logs / f"seed-{seed}.log"
            stream = path.open("x")
            streams.append(stream)
            command = [sys.executable, str(Path(__file__).resolve()), "train", "--root", str(root),
                       "--optimizer-seed", str(seed)]
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            children.append((seed, process))
            launch["children"].append({"optimizer_seed": seed, "pid": process.pid,
                                        "command": command, "log": str(path)})
        publish_json(root / "campaign-launch.json", launch)
        hard_stop = _deadline(plan["deadline_utc"]).timestamp() + 60
        while any(process.poll() is None for _, process in children):
            if time.time() >= hard_stop:
                for _, process in children:
                    if process.poll() is None:
                        process.terminate()
                break
            time.sleep(1)
        results = []
        for seed, process in children:
            try:
                exit_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = process.wait(timeout=10)
            result_path = root / f"seed-{seed}" / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            results.append({"optimizer_seed": seed, "exit_code": exit_code,
                            "status": result["status"] if result else "failed_without_result",
                            "num_timesteps": result["num_timesteps"] if result else None,
                            "result": file_record(result_path) if result else None})
        require_plan(root)
        closure = {"schema": SCHEMA, "completed_at": now(), "plan_sha256": sha(plan),
                   "elapsed_s": time.monotonic() - started, "runs": results,
                   "all_fixed_endpoints_completed": all(row["exit_code"] == 0 and row["status"] == "completed" for row in results)}
        publish_json(root / "campaign-closure.json", closure)
        return closure
    except BaseException:
        for _, process in children:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--total-timesteps", type=int, required=True)
    freeze.add_argument("--optimizer-seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    freeze.add_argument("--deadline-utc", default=DEFAULT_DEADLINE)
    freeze.add_argument("--environments", type=int, default=4)
    freeze.add_argument("--n-steps", type=int, default=2048)
    freeze.add_argument("--checkpoint-steps", type=int, default=65536)
    freeze.add_argument("--ent-coef", type=float, default=0.01)
    execute = sub.add_parser("train")
    execute.add_argument("--root", type=Path, required=True)
    execute.add_argument("--optimizer-seed", type=int, required=True)
    batch = sub.add_parser("campaign")
    batch.add_argument("--root", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        payload = vars(args).copy()
        payload.pop("command")
        print(json.dumps(freeze_plan(**payload), indent=2))
    elif args.command == "train":
        print(json.dumps(train(args.root, args.optimizer_seed), indent=2))
    elif args.command == "campaign":
        result = campaign(args.root)
        print(json.dumps(result, indent=2))
        if not result["all_fixed_endpoints_completed"]:
            raise SystemExit(1)
    else:
        record = require_plan(args.root)
        if args.checkpoint:
            load_checkpoint(args.checkpoint, args.root / "plan.json")
        print(json.dumps({"verified": True, "plan_sha256": sha(record)}))


if __name__ == "__main__":
    main()
