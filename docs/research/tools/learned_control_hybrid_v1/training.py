"""Registered PPO/BC runs with learned OFF atoms and censored-Gaussian ONs."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from docs.research.tools.learned_control_v1 import training as base
from docs.research.tools.wholeplant_ppo_v1 import training as common
from docs.research.tools.learned_control_hybrid_v1.policy import HybridActorCriticPolicy, distribution_identity

SCHEMA = "learned-control-hybrid-training-v1"
configure_torch = common.configure_torch
file_record, publish_json, append_json, sha, now = common.file_record, common.publish_json, common.append_json, common.sha, common.now


def default_recipe(**changes):
    recipe = base.default_recipe()
    recipe["ppo"]["policy_kwargs"] = {"net_arch": {"pi": [128, 128], "vf": [128, 128]},
        "min_on_std": .0001, "max_on_std": .1, "initial_on_std": .001,
        "initial_off_probability": .5}
    recipe["behavior_clone"] = {"epochs": 100, "batch_size": 512, "learning_rate": .001,
        "gate_logit_target": 6., "gate_confidence_weight": .25, "gate_loss_weight": 1.,
        "magnitude_loss_weight": 1., "relative_magnitude_floor": .05,
        "critic_epochs": 0, "critic_learning_rate": .0003}
    recipe["entropy_start"], recipe["entropy_end"] = 0., 0.
    recipe["freeze_observation_statistics"] = True
    unknown = set(changes) - set(recipe)
    if unknown:
        raise ValueError(f"Unknown recipe fields:{sorted(unknown)}")
    recipe.update(changes)
    return recipe


def _validate(recipe):
    base._validate(recipe)
    if recipe["reset_log_std"] is not None:
        raise ValueError("Hybrid distribution has state-dependent ONscale and noGaussianlog_std")
    if "log_std_init" in recipe["ppo"]["policy_kwargs"]:
        raise ValueError("Use initial_on_std for the hybrid policy")
    if recipe["initialization"] == "fresh" and recipe["freeze_observation_statistics"]:
        raise ValueError("Fresh policy must collect normalization statistics or initialize them byBC")
    c = recipe["behavior_clone"]
    if c["gate_logit_target"] <= 0 or not 0 <= c["relative_magnitude_floor"] <= 1:
        raise ValueError("Invalid hybrid imitation configuration")


def _pins(recipe):
    paths = {Path(row["path"]).resolve() for row in base._pins(recipe)}
    paths.update(Path(__file__).parent.glob("*.py"))
    return [file_record(path) for path in sorted(paths)]


def freeze_plan(root, recipe):
    recipe = copy.deepcopy(recipe)
    _validate(recipe)
    dependencies = []
    if recipe["initial_checkpoint"]:
        path = Path(recipe["initial_checkpoint"]).resolve()
        policy = load_checkpoint(path)
        policy.close()
        dependencies.extend(file_record(path.with_suffix(suffix)) for suffix in (".zip", ".json", ".vecnormalize.pkl"))
    if recipe["dataset"]:
        from docs.research.tools.learned_control_hybrid_v1.imitation import require_dataset
        path = Path(recipe["dataset"]).resolve()
        require_dataset(path)
        dependencies.extend((file_record(path), file_record(path.with_suffix(".json"))))
    plan = {"schema": SCHEMA, "recipe": recipe, "created_at": now(),
            "environment_identity": base._environment().environment_identity(),
            "policy_distribution": distribution_identity(), "source_files": _pins(recipe),
            "dependencies": dependencies, "runtime": common.runtime_record(),
            "objective": "Unchanged undiscounted economic reward scaled0.001; no auxiliary reward or deployed fallback",
            "training_allocation": "Per-vector sampler seeds independent of optimizer seed; train-only scenario domain",
            "endpoint_policy": "Full additional transitions or distinctly named incomplete checkpoint at deadline"}
    publish_json(Path(root) / "plan.json", plan)
    return plan


def require_plan(root):
    plan = json.loads((Path(root) / "plan.json").read_text())
    _validate(plan["recipe"])
    if (plan["schema"] != SCHEMA or plan["source_files"] != _pins(plan["recipe"])
            or plan["policy_distribution"] != distribution_identity()
            or plan["environment_identity"] != base._environment().environment_identity()
            or plan["runtime"] != common.runtime_record()):
        raise ValueError("Frozen hybrid source,configuration,or runtime changed")
    for row in plan["dependencies"]:
        if file_record(Path(row["path"])) != row:
            raise ValueError("Frozen hybrid training dependency changed")
    return plan


def make_learner(plan):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed, FloatSchedule
    configure_torch()
    recipe, wrappers = plan["recipe"], []
    set_random_seed(recipe["optimizer_seed"])

    def factory(index):
        def create():
            wrapped = base.TrainingEpisodes(base._environment().MixtureEnv(weights=recipe["mixture_weights"],
                sampler_seed=recipe["scenario_sampler_seed"] + index), index)
            wrappers.append(wrapped)
            return common.ScaleReward(wrapped, recipe["reward_scale"])
        return create

    raw_vector = DummyVecEnv([factory(i) for i in range(recipe["environments"])])
    try:
        if recipe["initialization"] == "continue":
            path = Path(recipe["initial_checkpoint"])
            vector = VecNormalize.load(path.with_suffix(".vecnormalize.pkl"), raw_vector)
            model = PPO.load(path, env=vector, device="cpu")
            if not isinstance(model.policy, HybridActorCriticPolicy) or model.policy_kwargs != recipe["ppo"]["policy_kwargs"]:
                raise ValueError("Continuation must preserve the hybrid policy architecture")
            model.seed, model.n_steps = recipe["optimizer_seed"], recipe["n_steps"]
            for key, value in recipe["ppo"].items():
                if key not in ("policy_kwargs", "clip_range"):
                    setattr(model, key, value)
            model.clip_range = FloatSchedule(recipe["ppo"]["clip_range"])
            model.rollout_buffer = model.rollout_buffer_class(model.n_steps, model.observation_space, model.action_space,
                device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=model.n_envs,
                **model.rollout_buffer_kwargs)
        else:
            vector = VecNormalize(raw_vector, **recipe["normalization"])
            model = PPO(HybridActorCriticPolicy, vector, seed=recipe["optimizer_seed"], n_steps=recipe["n_steps"],
                learning_rate=recipe["learning_rate_start"], ent_coef=recipe["entropy_start"],
                device="cpu", verbose=0, **recipe["ppo"])
        set_random_seed(recipe["optimizer_seed"])
        for key, value in recipe["normalization"].items():
            if getattr(vector, key) != value:
                raise ValueError(f"Normalization differs:{key}")
        vector.training = not recipe["freeze_observation_statistics"]
        vector.norm_reward = False
        prior = model.num_timesteps
        model.learning_rate = base.RunLinearSchedule(recipe["learning_rate_start"], recipe["learning_rate_end"], prior, recipe["total_timesteps"])
        model.lr_schedule = model.learning_rate
        model.research_identity = plan["environment_identity"]
        model.research_plan_sha256 = sha(plan)
        model.research_initial_num_timesteps = prior
        model.research_policy_distribution = distribution_identity()
        return model, vector, wrappers
    except BaseException:
        raw_vector.close()
        raise


def save_checkpoint(model, path, *, plan, wrappers, kind, probe_observations=None):
    from stable_baselines3 import PPO
    import random
    import torch
    path = Path(path)
    if any(path.with_suffix(suffix).exists() for suffix in (".zip", ".json", ".vecnormalize.pkl")):
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".writing.zip")
    model.save(temporary)
    temporary.rename(path)
    normalizer_path = path.with_suffix(".vecnormalize.pkl")
    normalizer = model.get_vec_normalize_env()
    normalizer.save(normalizer_path)
    restored = common._load_normalizer(normalizer_path)
    try:
        for attribute in ("mean", "var", "count"):
            if not np.array_equal(getattr(normalizer.obs_rms, attribute), getattr(restored.obs_rms, attribute)):
                raise ValueError("Observation statistics did not reopen exactly")
    finally:
        restored.close()
    # SB3 load calls set_random_seed. A verification read must not restart the
    # live learner's exploration or minibatch-permutation streams.
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        reopened = PPO.load(path, device="cpu")
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
    hashes = {"policy_tensor_sha256": common.policy_parameter_sha256(model),
              "actor_tensor_sha256": common.actor_parameter_sha256(model),
              "actor_mean_tensor_sha256": common.actor_mean_parameter_sha256(model),
              "optimizer_state_sha256": common.optimizer_sha256(model)}
    if (common.policy_parameter_sha256(reopened) != hashes["policy_tensor_sha256"]
            or common.optimizer_sha256(reopened) != hashes["optimizer_state_sha256"]
            or reopened.num_timesteps != model.num_timesteps or reopened.research_plan_sha256 != sha(plan)
            or reopened.research_policy_distribution != distribution_identity()):
        raise ValueError("Hybrid checkpoint did not reopen exactly")
    if probe_observations is not None:
        if not np.array_equal(model.predict(probe_observations, deterministic=True)[0], reopened.predict(probe_observations, deterministic=True)[0]):
            raise ValueError("Hybrid deterministic predictions changed on reopening")
    record = {"schema": SCHEMA, "kind": kind, "created_at": now(), **hashes,
        "optimizer_seed": int(model.seed), "num_timesteps": model.num_timesteps,
        "additional_timesteps": model.num_timesteps - model.research_initial_num_timesteps,
        "ppo_update_count": model._n_updates, "plan_sha256": sha(plan),
        "environment_identity": plan["environment_identity"], "policy_distribution": distribution_identity(),
        "checkpoint": file_record(path), "normalizer": file_record(normalizer_path),
        "normalizer_observation_count": float(normalizer.obs_rms.count),
        "entropy_coefficient": float(model.ent_coef), "gaussian_log_std": None,
        "actor_hash_scope": "All shared actor features and21learnedoutputparameters; includes gates,ONmeans andONscales",
        "checkpoint_reopen_verified": True, "normalizer_reopen_verified": True,
        "checkpoint_verification_preserves_training_rng": True,
        "deterministic_prediction_reopen_verified": probe_observations is not None,
        "allocator": [wrapper.snapshot() for wrapper in wrappers]}
    publish_json(path.with_suffix(".json"), record)
    return record


def load_checkpoint(path, plan_path=None):
    policy = common.load_checkpoint(Path(path))
    if not isinstance(policy.model.policy, HybridActorCriticPolicy):
        policy.close()
        raise ValueError("Expected a hybrid actor checkpoint")
    metadata = json.loads(Path(path).with_suffix(".json").read_text())
    if metadata["policy_distribution"] != distribution_identity() or policy.model.research_policy_distribution != distribution_identity():
        policy.close()
        raise ValueError("Hybrid distribution identity changed")
    if plan_path is not None and metadata["plan_sha256"] != sha(require_plan(Path(plan_path).parent)):
        policy.close()
        raise ValueError("Hybrid checkpoint belongs to a different run")
    return policy


def train(root):
    from stable_baselines3.common.callbacks import BaseCallback
    root = Path(root).resolve()
    plan, started = require_plan(root), time.monotonic()
    recipe = plan["recipe"]
    deadline = common._deadline(recipe["deadline_utc"])
    if datetime.now(timezone.utc) >= deadline:
        raise ValueError("Training deadline passed")
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
            if steps and steps % recipe["checkpoint_steps"] == 0 and steps not in self.saved:
                require_plan(root)
                save_checkpoint(self.model, out / "checkpoints" / f"step-{steps:07d}.zip", plan=plan,
                    wrappers=wrappers, kind="development-intermediate", probe_observations=self.initial_observations)
                self.saved.add(steps)
            if steps:
                metrics = {key: float(value) for key, value in self.model.logger.name_to_value.items()
                           if isinstance(value, (int, float, np.number)) and np.isfinite(value)}
                append_json(out / "optimization.jsonl", {"additional_timesteps": steps,
                    "ppo_update_count": self.model._n_updates, "entropy_coefficient": self.model.ent_coef,
                    "metrics": metrics})

        def _on_rollout_end(self):
            steps = self.model.num_timesteps - self.model.research_initial_num_timesteps
            fraction = np.clip(steps / recipe["total_timesteps"], 0., 1.)
            self.model.ent_coef = float(recipe["entropy_start"] + fraction * (recipe["entropy_end"] - recipe["entropy_start"]))

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

    try:
        model, vector, wrappers = make_learner(plan)
        before = save_checkpoint(model, out / "pre-initialization-policy.zip", plan=plan, wrappers=wrappers, kind="before-initialization")
        if recipe["initialization"] == "behavior_clone":
            from docs.research.tools.learned_control_hybrid_v1.imitation import fit_clone
            fit_clone(model, vector, Path(recipe["dataset"]), recipe["behavior_clone"],
                optimizer_seed=recipe["optimizer_seed"], out=out, deadline=deadline)
        initial = save_checkpoint(model, out / "initial-policy.zip", plan=plan, wrappers=wrappers, kind="initialized")
        callback = Progress()
        model.learn(total_timesteps=recipe["total_timesteps"], callback=callback, reset_num_timesteps=False)
        require_plan(root)
        additional = model.num_timesteps - model.research_initial_num_timesteps
        if sum(wrapper.decisions for wrapper in wrappers) != additional:
            raise ValueError("Physical transition ledger differs from PPO")
        complete = not deadline_reached and additional == recipe["total_timesteps"]
        endpoint = save_checkpoint(model, out / ("final-policy.zip" if complete else "incomplete-policy.zip"),
            plan=plan, wrappers=wrappers, kind="final" if complete else "incomplete", probe_observations=callback.initial_observations)
        result = {"schema": SCHEMA, "status": "completed" if complete else "deadline_incomplete",
            "completed_at": now(), "elapsed_s": time.monotonic() - started,
            "requested_additional_timesteps": recipe["total_timesteps"], "additional_timesteps": additional,
            "num_timesteps": model.num_timesteps, "ppo_update_count": model._n_updates,
            "plan_sha256": sha(plan), "before_initialization": before, "initial_checkpoint": initial, "endpoint": endpoint,
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
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_plan(args.root, json.loads(args.recipe.read_text()))
    elif args.command == "train":
        result = train(args.root)
    else:
        result = {"verified": True, "plan_sha256": sha(require_plan(args.root))}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
