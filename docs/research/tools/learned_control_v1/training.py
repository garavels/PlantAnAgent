"""Prospectively registered PPO recipes over the unchanged whole-plant task.

Each directory is one immutable attempt.  Observation statistics, optimizer
state, deterministic actor outputs, and policy tensors are saved together.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
from docs.research.tools.wholeplant_ppo_v1 import training as old

SCHEMA = "learned-control-training-v1"
now, sha, canonical_bytes = old.now, old.sha, old.canonical_bytes
file_record, publish_json, append_json = old.file_record, old.publish_json, old.append_json
configure_torch = old.configure_torch
policy_parameter_sha256 = old.policy_parameter_sha256
actor_parameter_sha256 = old.actor_parameter_sha256
actor_mean_parameter_sha256 = old.actor_mean_parameter_sha256
optimizer_sha256 = old.optimizer_sha256
LoadedPolicy = old.LoadedPolicy


def _environment():
    from docs.research.tools.learned_control_v1 import environment
    return environment


def default_recipe(**changes):
    """Return visible defaults; freeze serializes every effective setting."""
    recipe = {
        "optimizer_seed": 1101, "scenario_sampler_seed": 8100,
        "initialization": "fresh", "initial_checkpoint": None, "dataset": None,
        "total_timesteps": 262144, "environments": 4, "n_steps": 2048,
        "checkpoint_steps": 65536, "deadline_utc": "2026-09-18T05:50:00Z",
        "mixture_weights": {"ordinary": .5, "pv6": .1, "pv18": .1,
                            "pv36": .1, "meth24": .1, "meth48": .1},
        "reward_scale": .001,
        "normalization": {"norm_obs": True, "norm_reward": False,
                          "clip_obs": 10., "epsilon": 1e-8},
        "freeze_observation_statistics": False,
        "learning_rate_start": .0001, "learning_rate_end": .00001,
        "entropy_start": .002, "entropy_end": 0.,
        "reset_log_std": None,
        "ppo": {"batch_size": 256, "n_epochs": 10, "gamma": 1.,
                "gae_lambda": .99, "clip_range": .2, "vf_coef": .5,
                "max_grad_norm": .5, "target_kl": None,
                "policy_kwargs": {"net_arch": {"pi": [128, 128], "vf": [128, 128]},
                                  "log_std_init": -1.5}},
        "behavior_clone": {"epochs": 40, "batch_size": 512,
                           "learning_rate": .001, "off_target_margin": .1,
                           "on_target_margin": 0., "critic_epochs": 0,
                           "critic_learning_rate": .0003},
        "extra_source_paths": [],
    }
    unknown = set(changes) - set(recipe)
    if unknown:
        raise ValueError(f"Unknown recipe fields: {sorted(unknown)}")
    recipe.update(changes)
    return recipe


def _validate(recipe):
    required = set(default_recipe())
    if set(recipe) != required:
        raise ValueError(f"Recipe fields differ: {sorted(set(recipe) ^ required)}")
    old._deadline(recipe["deadline_utc"])
    block = recipe["environments"] * recipe["n_steps"]
    if recipe["environments"] < 1 or recipe["n_steps"] < 2:
        raise ValueError("Invalid rollout geometry")
    for key in ("total_timesteps", "checkpoint_steps"):
        if recipe[key] <= 0 or recipe[key] % block:
            raise ValueError(f"{key} must contain whole vector rollouts")
    if recipe["ppo"]["gamma"] != 1 or recipe["reward_scale"] != .001:
        raise ValueError("Preserve the registered undiscounted scalar economic objective")
    if recipe["initialization"] not in ("fresh", "continue", "behavior_clone"):
        raise ValueError("Unknown initialization")
    if bool(recipe["initial_checkpoint"]) != (recipe["initialization"] == "continue"):
        raise ValueError("Only continuation requires an initial checkpoint")
    if bool(recipe["dataset"]) != (recipe["initialization"] == "behavior_clone"):
        raise ValueError("Only behavior cloning requires a dataset")
    if recipe["normalization"]["norm_reward"]:
        raise ValueError("Reward normalization is not admitted")
    for key in ("learning_rate_start", "learning_rate_end", "entropy_start", "entropy_end"):
        if not np.isfinite(recipe[key]) or recipe[key] < 0:
            raise ValueError(f"Invalid schedule: {key}")


def _pins(recipe):
    records = _environment().source_records()
    paths = {Path(row["path"]).resolve() for row in records}
    paths.update((Path(__file__).resolve(), Path(__file__).with_name("imitation.py"), ROOT / "uv.lock"))
    paths.update(Path(path).resolve() for path in recipe["extra_source_paths"])
    return [file_record(path) for path in sorted(paths)]


def freeze_plan(root: Path, recipe: dict[str, Any]):
    root = Path(root).resolve()
    _validate(recipe)
    dependencies = []
    if recipe["initial_checkpoint"]:
        path = Path(recipe["initial_checkpoint"]).resolve()
        checked = load_checkpoint(path)
        checked.close()
        dependencies += [file_record(path), file_record(path.with_suffix(".json")),
                         file_record(path.with_suffix(".vecnormalize.pkl"))]
    if recipe["dataset"]:
        from docs.research.tools.learned_control_v1.imitation import require_dataset
        path = Path(recipe["dataset"]).resolve()
        require_dataset(path)
        dependencies += [file_record(path), file_record(path.with_suffix(".json"))]
    plan = {"schema": SCHEMA, "recipe": recipe, "created_at": now(),
            "environment_identity": _environment().environment_identity(),
            "source_files": _pins(recipe), "dependencies": dependencies,
            "runtime": old.runtime_record(),
            "objective": "Unchanged undiscounted economic reward scaled by0.001; no auxiliary reward or deployed fallback",
            "training_allocation": "Per-vector sampler seeds independent of optimizer seed; train-only scenario domain",
            "endpoint_policy": "Full requested additional transitions or distinctly named incomplete checkpoint at deadline"}
    publish_json(root / "plan.json", plan)
    return plan


def require_plan(root: Path):
    plan = json.loads((Path(root) / "plan.json").read_text())
    if plan["schema"] != SCHEMA:
        raise ValueError("Unknown frozen training schema")
    _validate(plan["recipe"])
    if canonical_bytes(_pins(plan["recipe"])) != canonical_bytes(plan["source_files"]):
        raise ValueError("Frozen training source changed")
    if plan["environment_identity"] != _environment().environment_identity():
        raise ValueError("Frozen environment identity changed")
    for row in plan["dependencies"]:
        if file_record(Path(row["path"])) != row:
            raise ValueError("Frozen training dependency changed")
    if plan["runtime"] != old.runtime_record():
        raise ValueError("Frozen training runtime changed")
    return plan


@dataclass(frozen=True)
class RunLinearSchedule:
    start: float
    end: float
    prior_steps: int
    additional_steps: int

    def __call__(self, progress_remaining):
        steps = (1 - float(progress_remaining)) * (self.prior_steps + self.additional_steps)
        fraction = np.clip((steps - self.prior_steps) / self.additional_steps, 0., 1.)
        return float(self.start + fraction * (self.end - self.start))


class TrainingEpisodes(gym.Wrapper):
    def __init__(self, env, index):
        super().__init__(env)
        self.index, self.decisions, self.resets = index, 0, 0
        self.episode_decisions, self.episode_reward = 0, 0.
        self.current_scenario = None
        self.completed = []

    def reset(self, *, seed=None, options=None):
        # SB3 seeds depend on the optimizer seed; scenario allocation must not.
        observation, info = self.env.reset(options=options)
        self.resets += 1
        self.episode_decisions, self.episode_reward = 0, 0.
        self.current_scenario = info.get("training_scenario", {})
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.decisions += 1
        self.episode_decisions += 1
        self.episode_reward += float(reward)
        if terminated or truncated:
            row = {"scenario": self.current_scenario, "decisions": self.episode_decisions,
                   "raw_reward_eur": self.episode_reward, "terminated": bool(terminated),
                   "truncated": bool(truncated), "economic_metrics": info.get("economic_metrics"),
                   "metrics": info.get("metrics"), "vector_index": self.index}
            self.completed.append(row)
            info["training_episode"] = row
        return observation, reward, terminated, truncated, info

    def snapshot(self):
        return {"environment_index": self.index, "resets": self.resets,
                "decisions": self.decisions, "current_scenario": self.current_scenario,
                "current_episode_decisions": self.episode_decisions,
                "completed_episodes": len(self.completed)}


def make_learner(plan):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed
    configure_torch()
    recipe, wrappers = plan["recipe"], []
    set_random_seed(recipe["optimizer_seed"])

    def factory(index):
        def create():
            wrapped = TrainingEpisodes(_environment().MixtureEnv(
                weights=recipe["mixture_weights"], sampler_seed=recipe["scenario_sampler_seed"] + index), index)
            wrappers.append(wrapped)
            return old.ScaleReward(wrapped, recipe["reward_scale"])
        return create

    raw_vector = DummyVecEnv([factory(index) for index in range(recipe["environments"])])
    try:
        if recipe["initialization"] == "continue":
            path = Path(recipe["initial_checkpoint"])
            vector = VecNormalize.load(path.with_suffix(".vecnormalize.pkl"), raw_vector)
            model = PPO.load(path, env=vector, device="cpu")
            # Retain the exact actor/critic/optimizer and observation statistics;
            # expose every deliberate new-run hyperparameter in the recipe.
            if model.policy_kwargs != recipe["ppo"]["policy_kwargs"]:
                raise ValueError("Continuation must preserve checkpoint policy architecture and initialization metadata")
            model.seed = recipe["optimizer_seed"]
            model.n_steps = recipe["n_steps"]
            for key, value in recipe["ppo"].items():
                if key not in ("policy_kwargs", "clip_range"):
                    setattr(model, key, value)
            from stable_baselines3.common.utils import FloatSchedule
            model.clip_range = FloatSchedule(recipe["ppo"]["clip_range"])
            model.rollout_buffer = model.rollout_buffer_class(model.n_steps,
                model.observation_space, model.action_space, device=model.device,
                gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=model.n_envs,
                **model.rollout_buffer_kwargs)
        else:
            vector = VecNormalize(raw_vector, **recipe["normalization"])
            model = PPO("MlpPolicy", vector, seed=recipe["optimizer_seed"], n_steps=recipe["n_steps"],
                        learning_rate=recipe["learning_rate_start"], ent_coef=recipe["entropy_start"],
                        device="cpu", verbose=0, **recipe["ppo"])
        # Loading a checkpoint restores its seed during _setup_model; establish
        # this registered optimizer seed after that restoration, too.
        set_random_seed(recipe["optimizer_seed"])
        for key, value in recipe["normalization"].items():
            if getattr(vector, key) != value:
                raise ValueError(f"Continuation observation normalization differs: {key}")
        vector.norm_reward = False
        vector.training = not recipe["freeze_observation_statistics"]
        if recipe["reset_log_std"] is not None:
            import torch
            with torch.no_grad():
                model.policy.log_std.fill_(float(recipe["reset_log_std"]))
        prior = model.num_timesteps
        model.learning_rate = RunLinearSchedule(recipe["learning_rate_start"], recipe["learning_rate_end"],
                                                prior, recipe["total_timesteps"])
        model.lr_schedule = model.learning_rate
        model.research_identity = plan["environment_identity"]
        model.research_plan_sha256 = sha(plan)
        model.research_initial_num_timesteps = prior
        return model, vector, wrappers
    except BaseException:
        raw_vector.close()
        raise


def save_checkpoint(model, path, *, plan, wrappers, kind, probe_observations=None):
    """Use the proven exact reopen checks, with an explicit new-run ledger."""
    from stable_baselines3 import PPO
    path = Path(path)
    if any(p.exists() for p in (path, path.with_suffix(".json"), path.with_suffix(".vecnormalize.pkl"))):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".writing.zip")
    model.save(temporary)
    temporary.rename(path)
    normalizer_path = path.with_suffix(".vecnormalize.pkl")
    normalizer = model.get_vec_normalize_env()
    normalizer.save(normalizer_path)
    restored = old._load_normalizer(normalizer_path)
    try:
        for attribute in ("mean", "var", "count"):
            if not np.array_equal(getattr(normalizer.obs_rms, attribute), getattr(restored.obs_rms, attribute)):
                raise ValueError("Observation normalizer did not reopen exactly")
    finally:
        restored.close()
    reopened = PPO.load(path, device="cpu")
    hashes = {"policy_tensor_sha256": policy_parameter_sha256(model),
              "actor_tensor_sha256": actor_parameter_sha256(model),
              "actor_mean_tensor_sha256": actor_mean_parameter_sha256(model),
              "optimizer_state_sha256": optimizer_sha256(model)}
    if (policy_parameter_sha256(reopened) != hashes["policy_tensor_sha256"]
            or optimizer_sha256(reopened) != hashes["optimizer_state_sha256"]
            or reopened.num_timesteps != model.num_timesteps
            or reopened.research_plan_sha256 != sha(plan)
            or reopened.research_identity != plan["environment_identity"]):
        raise ValueError("Saved checkpoint did not reopen exactly")
    if probe_observations is not None:
        before = model.predict(probe_observations, deterministic=True)[0]
        after = reopened.predict(probe_observations, deterministic=True)[0]
        if not np.array_equal(before, after):
            raise ValueError("Saved checkpoint changed deterministic predictions")
    record = {"schema": SCHEMA, "kind": kind, "created_at": now(), **hashes,
              "optimizer_seed": int(model.seed), "num_timesteps": model.num_timesteps,
              "additional_timesteps": model.num_timesteps - model.research_initial_num_timesteps,
              "ppo_update_count": model._n_updates, "plan_sha256": sha(plan),
              "environment_identity": plan["environment_identity"],
              "checkpoint": file_record(path), "normalizer": file_record(normalizer_path),
              "normalizer_observation_count": float(normalizer.obs_rms.count),
              "gaussian_log_std": model.policy.log_std.detach().cpu().tolist(),
              "entropy_coefficient": float(model.ent_coef),
              "checkpoint_reopen_verified": True, "normalizer_reopen_verified": True,
              "deterministic_prediction_reopen_verified": probe_observations is not None,
              "allocator": [wrapper.snapshot() for wrapper in wrappers]}
    publish_json(path.with_suffix(".json"), record)
    return record


def load_checkpoint(path: Path, plan_path: Path | None = None):
    """Authenticate old or new full PPO checkpoints; inference uses raw inputs."""
    path = Path(path).resolve()
    if plan_path is not None:
        plan = require_plan(Path(plan_path).parent)
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["plan_sha256"] != sha(plan):
            raise ValueError("Checkpoint belongs to a different run")
    return old.load_checkpoint(path)


def train(root: Path):
    from stable_baselines3.common.callbacks import BaseCallback
    root = Path(root).resolve()
    plan, started = require_plan(root), time.monotonic()
    recipe = plan["recipe"]
    deadline = old._deadline(recipe["deadline_utc"])
    if datetime.now(timezone.utc) >= deadline:
        raise ValueError("Training deadline has passed")
    out = root / "training"
    out.mkdir(exist_ok=False)
    publish_json(out / "attempt.json", {"started_at": now(), "pid": os.getpid(), "plan_sha256": sha(plan)})
    model = vector = None
    wrappers = []
    deadline_reached = False

    class Progress(BaseCallback):
        saved = set()
        initial_observations = None

        def _on_training_start(self):
            self.initial_observations = self.model._last_obs.copy()

        def _on_rollout_start(self):
            steps = self.model.num_timesteps - self.model.research_initial_num_timesteps
            fraction = np.clip(steps / recipe["total_timesteps"], 0., 1.)
            self.model.ent_coef = float(recipe["entropy_start"] + fraction *
                                        (recipe["entropy_end"] - recipe["entropy_start"]))
            if steps and steps % recipe["checkpoint_steps"] == 0 and steps not in self.saved:
                require_plan(root)
                save_checkpoint(self.model, out / "checkpoints" / f"step-{steps:07d}.zip",
                    plan=plan, wrappers=wrappers, kind="development-intermediate",
                    probe_observations=self.initial_observations)
                self.saved.add(steps)
            if steps:
                metrics = {key: float(value) for key, value in self.model.logger.name_to_value.items()
                           if isinstance(value, (int, float, np.number)) and np.isfinite(value)}
                append_json(out / "optimization.jsonl", {"additional_timesteps": steps,
                    "ppo_update_count": self.model._n_updates, "entropy_coefficient": self.model.ent_coef,
                    "metrics": metrics})

        def _on_step(self):
            nonlocal deadline_reached
            for info in self.locals.get("infos", []):
                if "training_episode" in info:
                    append_json(out / "episodes.jsonl", info["training_episode"])
            steps = self.model.num_timesteps - self.model.research_initial_num_timesteps
            if steps % 8192 == 0:
                elapsed = time.monotonic() - started
                row = {"at": now(), "additional_timesteps": steps, "elapsed_s": elapsed,
                       "transitions_per_second": steps / elapsed,
                       "remaining_estimated_s": (recipe["total_timesteps"] - steps) * elapsed / max(1, steps)}
                append_json(out / "progress.jsonl", row)
                print(json.dumps(row), flush=True)
            if datetime.now(timezone.utc) >= deadline:
                deadline_reached = True
                return False
            return True

        def _on_rollout_end(self):
            # PPO.train follows this callback, so the final optimization update
            # really uses the declared endpoint rather than the preceding block.
            steps = self.model.num_timesteps - self.model.research_initial_num_timesteps
            fraction = np.clip(steps / recipe["total_timesteps"], 0., 1.)
            self.model.ent_coef = float(recipe["entropy_start"] + fraction *
                                        (recipe["entropy_end"] - recipe["entropy_start"]))

    try:
        model, vector, wrappers = make_learner(plan)
        random_initial = save_checkpoint(model, out / "pre-initialization-policy.zip", plan=plan,
                                         wrappers=wrappers, kind="before-initialization")
        if recipe["initialization"] == "behavior_clone":
            from docs.research.tools.learned_control_v1.imitation import fit_clone
            fit_clone(model, vector, Path(recipe["dataset"]), recipe["behavior_clone"],
                      optimizer_seed=recipe["optimizer_seed"], out=out, deadline=deadline)
        initial = save_checkpoint(model, out / "initial-policy.zip", plan=plan,
                                  wrappers=wrappers, kind="initialized")
        callback = Progress()
        model.learn(total_timesteps=recipe["total_timesteps"], callback=callback, reset_num_timesteps=False)
        require_plan(root)
        additional = model.num_timesteps - model.research_initial_num_timesteps
        if sum(wrapper.decisions for wrapper in wrappers) != additional:
            raise ValueError("Physical transition ledger differs from PPO")
        complete = not deadline_reached and additional == recipe["total_timesteps"]
        endpoint = save_checkpoint(model, out / ("final-policy.zip" if complete else "incomplete-policy.zip"),
            plan=plan, wrappers=wrappers, kind="final" if complete else "incomplete",
            probe_observations=callback.initial_observations)
        result = {"schema": SCHEMA, "status": "completed" if complete else "deadline_incomplete",
                  "completed_at": now(), "elapsed_s": time.monotonic() - started,
                  "requested_additional_timesteps": recipe["total_timesteps"], "additional_timesteps": additional,
                  "num_timesteps": model.num_timesteps, "ppo_update_count": model._n_updates,
                  "plan_sha256": sha(plan), "before_initialization": random_initial,
                  "initial_checkpoint": initial, "endpoint": endpoint,
                  "actor_mean_parameters_changed": initial["actor_mean_tensor_sha256"] != endpoint["actor_mean_tensor_sha256"],
                  "allocator": [wrapper.snapshot() for wrapper in wrappers]}
        publish_json(out / "result.json", result)
        return result
    except BaseException as exc:
        publish_json(out / "failure.json", {"at": now(), "type": type(exc).__name__, "error": str(exc),
            "elapsed_s": time.monotonic() - started, "num_timesteps": getattr(model, "num_timesteps", 0)})
        raise
    finally:
        if vector is not None:
            vector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "train", "verify"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        if args.recipe is None:
            parser.error("freeze requires --recipe")
        result = freeze_plan(args.root, json.loads(args.recipe.read_text()))
    elif args.command == "train":
        result = train(args.root)
    else:
        result = {"verified": True, "plan_sha256": sha(require_plan(args.root))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
