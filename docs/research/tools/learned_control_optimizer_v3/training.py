"""Independent actor/critic PPO optimization with an unchanged hybrid actor."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from docs.research.tools.learned_control_v1 import training as base
from docs.research.tools.learned_control_v1 import imitation
from docs.research.tools.learned_control_hybrid_v1 import training as hybrid
from docs.research.tools.learned_control_hybrid_v1.policy import distribution_identity
from docs.research.tools.learned_control_asymmetric_v1.policy import AsymmetricHybridPolicy, critic_identity
from docs.research.tools.learned_control_asymmetric_v1 import training as asymmetric
from docs.research.tools.learned_control_optimizer_v3.policy import PreconditionedHybridPolicy, optimizer_identity, parameter_partition
from docs.research.tools.learned_control_optimizer_v3.algorithm import PreconditionedPPO
from docs.research.tools.wholeplant_ppo_v1 import training as common

SCHEMA = "learned-control-preconditioned-optimizer-v3"
configure_torch = common.configure_torch
file_record, publish_json, append_json, sha, now = common.file_record, common.publish_json, common.append_json, common.sha, common.now


def default_recipe(**changes):
    recipe = asymmetric.default_recipe()
    recipe["ppo"]["policy_kwargs"]["critic_context_mode"] = "constant"
    recipe["learning_rate_start"], recipe["learning_rate_end"] = 3e-6, 1e-6
    recipe.update(critic_learning_rate_start=1e-4, critic_learning_rate_end=3e-5,
                  actor_max_grad_norm=.5, critic_max_grad_norm=.5,
                  preconditioning_dataset=None)
    recipe["ppo"]["policy_kwargs"].update(mean_update_scales=None, actor_body_update_scale=.01)
    recipe.update(changes)
    return recipe


def _validate(recipe):
    prior_recipe = copy.deepcopy(recipe)
    if not prior_recipe.pop("preconditioning_dataset"):
        raise ValueError("Authenticated training labels are required for update preconditioning")
    for key in ("critic_learning_rate_start", "critic_learning_rate_end", "actor_max_grad_norm", "critic_max_grad_norm"):
        if not np.isfinite(prior_recipe.pop(key)) or recipe[key] <= 0:
            raise ValueError(f"Invalid grouped optimizer setting: {key}")
    asymmetric._validate(prior_recipe)
    if recipe["initialization"] == "behavior_clone":
        raise ValueError("Initialize by transferring a registered hybridBCcheckpoint; no new directBC path in this candidate")
    if recipe["ppo"]["policy_kwargs"].get("critic_context_hidden", 64) < 1:
        raise ValueError("Private critic hidden dimension must be positive")
    if recipe["ppo"]["policy_kwargs"].get("critic_context_mode", "condition") not in ("condition", "constant"):
        raise ValueError("Unknown critic conditioning mode")
    scales = recipe["ppo"]["policy_kwargs"].get("mean_update_scales")
    body = recipe["ppo"]["policy_kwargs"].get("actor_body_update_scale")
    if (not isinstance(scales, list) or len(scales) != 7
            or not all(np.isfinite(x) and x > 0 for x in scales)
            or body is None or not np.isfinite(body) or body <= 0):
        raise ValueError("Update preconditioners must be seven positive head factors and a positive body factor")


def derive_preconditioning(dataset):
    if not dataset:
        raise ValueError("An authenticated training dataset is required")
    dataset = Path(dataset).resolve()
    metadata = imitation.require_dataset(dataset)
    if any(isinstance(c["seed"], bool) or not isinstance(c["seed"], int)
           or not 50_000_000 <= c["seed"] < 51_000_000 for c in metadata["cases"]):
        raise ValueError("Preconditioning labels must come only from the current training seed domain")
    with np.load(dataset, allow_pickle=False) as data:
        actions = data["actions"].astype(np.float64)
    counts, scales = [], []
    for index in range(7):
        on = actions[:, index] > -1.
        if not on.any():
            raise ValueError("Every action coordinate needs observed teacher-ON training labels")
        counts.append(int(on.sum()))
        scales.append(float(np.mean(actions[on, index] + 1)))
    if not all(np.isfinite(value) and value > 0 for value in scales):
        raise ValueError("Derived update scales must be finite and strictly positive")
    return {"dataset": file_record(dataset), "metadata": file_record(dataset.with_suffix(".json")),
        "collection_plan": file_record(Path(metadata["plan"]["path"])),
        "samples": len(actions), "teacher_on_counts": counts, "mean_update_scales": scales,
        "derivation": "For each original action coordinate, mean(action+1) over authenticated training teacher labels strictly greater than-1; no floor, cap, or action bound"}


def _pins(recipe):
    paths = {Path(row["path"]).resolve() for row in asymmetric._pins(recipe)}
    paths.update(Path(__file__).parent.glob("*.py"))
    return [file_record(path) for path in sorted(paths)]


def freeze_plan(root, recipe):
    recipe = copy.deepcopy(recipe)
    preconditioning = derive_preconditioning(recipe["preconditioning_dataset"])
    recipe["preconditioning_dataset"] = preconditioning["dataset"]["path"]
    requested = recipe["ppo"]["policy_kwargs"].get("mean_update_scales")
    if requested is not None and requested != preconditioning["mean_update_scales"]:
        raise ValueError("Requested scales differ from authenticated training label derivation")
    recipe["ppo"]["policy_kwargs"]["mean_update_scales"] = preconditioning["mean_update_scales"]
    _validate(recipe)
    dependencies = [preconditioning[key] for key in ("dataset", "metadata", "collection_plan")]
    if recipe["initial_checkpoint"]:
        path = Path(recipe["initial_checkpoint"]).resolve()
        policy = hybrid.load_checkpoint(path)
        policy.close()
        dependencies.extend(file_record(path.with_suffix(suffix)) for suffix in (".zip", ".json", ".vecnormalize.pkl"))
    plan = {"schema": SCHEMA, "recipe": recipe, "created_at": now(),
        "environment_identity": base._environment().environment_identity(),
        "policy_distribution": distribution_identity(), "private_critic": critic_identity(),
        "optimization": optimizer_identity(),
        "preconditioning": preconditioning,
        "critic_context_mode": recipe["ppo"]["policy_kwargs"].get("critic_context_mode", "condition"),
        "source_files": _pins(recipe), "dependencies": dependencies, "runtime": common.runtime_record(),
        "objective": "Unchanged undiscounted economic reward scaled0.001; private condition is an action-independent training baseline,not an actor input or extra reward",
        "training_allocation": "Existing per-vector scenario streams; actor observation remains149publiccoordinates",
        "endpoint_policy": "Full additional transitions or distinctly named incomplete checkpoint at deadline"}
    publish_json(Path(root) / "plan.json", plan)
    return plan


def require_plan(root):
    plan = json.loads((Path(root) / "plan.json").read_text())
    _validate(plan["recipe"])
    if (plan["schema"] != SCHEMA or plan["source_files"] != _pins(plan["recipe"])
            or plan["policy_distribution"] != distribution_identity() or plan["private_critic"] != critic_identity()
            or plan["optimization"] != optimizer_identity()
            or plan["critic_context_mode"] != plan["recipe"]["ppo"]["policy_kwargs"].get("critic_context_mode", "condition")
            or plan["environment_identity"] != base._environment().environment_identity()
            or plan["runtime"] != common.runtime_record()):
        raise ValueError("Frozen asymmetric source,configuration,or runtime changed")
    for row in plan["dependencies"]:
        if file_record(Path(row["path"])) != row:
            raise ValueError("Frozen asymmetric dependency changed")
    if (derive_preconditioning(plan["recipe"]["preconditioning_dataset"]) != plan["preconditioning"]
            or plan["recipe"]["ppo"]["policy_kwargs"]["mean_update_scales"] != plan["preconditioning"]["mean_update_scales"]):
        raise ValueError("Pinned training-label preconditioning changed")
    return plan


def _tensor_state(value):
    if hasattr(value, "detach"):
        array = value.detach().cpu().contiguous().numpy()
        return {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(key): _tensor_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_state(item) for item in value]
    return value


def optimizer_named_state(policy, names=None):
    wanted = None if names is None else set(names)
    return {name: _tensor_state(policy.optimizer.state.get(parameter, {}))
            for name, parameter in policy.named_parameters() if wanted is None or name in wanted}


def transfer_hybrid(source, target):
    """Copy actor/old critic and their named Adam states, adding only critic slots."""
    extra_keys = {"critic_context_hidden", "critic_context_mode", "mean_update_scales", "actor_body_update_scale"}
    expected_old = {key: value for key, value in target.policy_kwargs.items() if key not in extra_keys}
    actual_old = {key: value for key, value in source.policy_kwargs.items() if key not in extra_keys}
    if expected_old != actual_old:
        raise ValueError("Asymmetric transfer cannot silently alter actor architecture or distribution settings")
    if isinstance(source.policy, AsymmetricHybridPolicy) and source.policy.critic_context_hidden != target.policy.critic_context_hidden:
        raise ValueError("Asymmetric continuation must preserve private critic architecture")
    source_names = [name for name, _ in source.policy.named_parameters()]
    missing, unexpected = target.policy.load_state_dict(source.policy.state_dict(), strict=False)
    if unexpected or any(not name.startswith("private_value_head.") for name in missing):
        raise ValueError("Actor or public critic parameters were not transferred exactly")
    source_parameters = {id(parameter): name for name, parameter in source.policy.named_parameters()}
    target_parameters = {id(parameter): name for name, parameter in target.policy.named_parameters()}
    source_state, target_state = source.policy.optimizer.state_dict(), target.policy.optimizer.state_dict()
    source_ids = {source_parameters[id(parameter)]: state_id
        for live, stored in zip(source.policy.optimizer.param_groups, source_state["param_groups"])
        for parameter, state_id in zip(live["params"], stored["params"])}
    target_ids = {target_parameters[id(parameter)]: state_id
        for live, stored in zip(target.policy.optimizer.param_groups, target_state["param_groups"])
        for parameter, state_id in zip(live["params"], stored["params"])}
    target_state["state"] = {target_ids[name]: copy.deepcopy(source_state["state"][state_id])
        for name, state_id in source_ids.items() if state_id in source_state["state"]}
    # Preserve each old tensor's moments. The two explicitly registered learning
    # rates/group partition are the intervention, not inherited source settings.
    source_groups_by_name = {source_parameters[id(parameter)]: group
        for group in source.policy.optimizer.param_groups for parameter in group["params"]}
    for group in target_state["param_groups"]:
        for name, parameter in parameter_partition(target.policy)[group["name"]]:
            if name in source_groups_by_name:
                prior = source_groups_by_name[name]
                for key in set(prior) - {"params", "lr", "name"}:
                    if group.get(key) != prior[key]:
                        raise ValueError(f"Named Adam transfer changes optimizer setting {key}")
    target.policy.optimizer.load_state_dict(target_state)
    actor_before, actor_after = common.actor_parameter_sha256(source), common.actor_parameter_sha256(target)
    old_optimizer = optimizer_named_state(source.policy)
    new_optimizer = optimizer_named_state(target.policy, source_names)
    if actor_before != actor_after or old_optimizer != new_optimizer:
        raise ValueError("Actor tensors or existing optimizer states changed during private critic transfer")
    probe = torch.as_tensor(np.random.default_rng(718).normal(size=(32, 149)).astype(np.float32))
    with torch.no_grad():
        before_distribution = source.policy.get_distribution(probe)
        before = before_distribution.get_actions(deterministic=True)
        before_logprob = before_distribution.log_prob(before)
        after_distribution = target.policy.get_distribution(probe)
        after = after_distribution.get_actions(deterministic=True)
        after_logprob = after_distribution.log_prob(before)
    if not torch.equal(before, after):
        raise ValueError("Actor predictions changed while adding the private critic")
    if not torch.equal(before_logprob, after_logprob):
        raise ValueError("Actor log-probabilities changed during grouped optimizer transfer")
    target.num_timesteps, target._n_updates = source.num_timesteps, source._n_updates
    return {"source_actor_sha256": actor_before, "transferred_actor_sha256": actor_after,
        "actor_tensors_exact": True, "deterministic_probe_exact": True, "actor_logprob_probe_exact": True,
        "optimizer_groups": [group["name"] for group in target.policy.optimizer.param_groups],
        "existing_optimizer_named_states_exact": True, "existing_optimizer_states_sha256": sha(old_optimizer),
        "target_mean_update_scales": list(target.policy.mean_update_scales),
        "target_actor_body_update_scale": target.policy.actor_body_update_scale,
        "added_parameter_names": missing, "private_correction_initially_zero": not isinstance(source.policy, AsymmetricHybridPolicy),
        "source_num_timesteps": source.num_timesteps, "source_ppo_update_count": source._n_updates}


def make_learner(plan):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.utils import set_random_seed
    configure_torch()
    recipe, wrappers = plan["recipe"], []

    def factory(index):
        def create():
            wrapped = base.TrainingEpisodes(base._environment().MixtureEnv(weights=recipe["mixture_weights"],
                sampler_seed=recipe["scenario_sampler_seed"] + index), index)
            wrappers.append(wrapped)
            return common.ScaleReward(wrapped, recipe["reward_scale"])
        return create

    raw_vector = DummyVecEnv([factory(i) for i in range(recipe["environments"])])
    source = None
    try:
        if recipe["initialization"] == "continue":
            path = Path(recipe["initial_checkpoint"])
            source = hybrid.load_checkpoint(path)
            vector = VecNormalize.load(path.with_suffix(".vecnormalize.pkl"), raw_vector)
        else:
            vector = VecNormalize(raw_vector, **recipe["normalization"])
        set_random_seed(recipe["optimizer_seed"])
        model = PreconditionedPPO(PreconditionedHybridPolicy, vector, seed=recipe["optimizer_seed"], n_steps=recipe["n_steps"],
            learning_rate=recipe["learning_rate_start"], ent_coef=recipe["entropy_start"],
            critic_learning_rate=recipe["critic_learning_rate_start"], actor_max_grad_norm=recipe["actor_max_grad_norm"],
            critic_max_grad_norm=recipe["critic_max_grad_norm"],
            device="cpu", verbose=0, **recipe["ppo"])
        model.research_transfer = transfer_hybrid(source.model, model) if source else {"fresh_initialization": True}
        if source:
            source.close()
            source = None
        set_random_seed(recipe["optimizer_seed"])
        for key, value in recipe["normalization"].items():
            if getattr(vector, key) != value:
                raise ValueError(f"Normalization differs:{key}")
        vector.training, vector.norm_reward = not recipe["freeze_observation_statistics"], False
        prior = model.num_timesteps
        model.learning_rate = base.RunLinearSchedule(recipe["learning_rate_start"], recipe["learning_rate_end"], prior, recipe["total_timesteps"])
        model.lr_schedule = model.learning_rate
        model.critic_learning_rate = base.RunLinearSchedule(recipe["critic_learning_rate_start"], recipe["critic_learning_rate_end"], prior, recipe["total_timesteps"])
        model.critic_lr_schedule = model.critic_learning_rate
        for group in model.policy.optimizer.param_groups:
            group["lr"] = recipe["learning_rate_start"] if group["name"] == "actor" else recipe["critic_learning_rate_start"]
        model.research_identity, model.research_plan_sha256 = plan["environment_identity"], sha(plan)
        model.research_initial_num_timesteps = prior
        model.research_policy_distribution, model.research_private_critic = distribution_identity(), critic_identity()
        model.research_private_critic_mode = model.policy.critic_context_mode
        model.research_optimizer_identity = optimizer_identity()
        model.research_training_deadline_utc = recipe["deadline_utc"]
        return model, vector, wrappers
    except BaseException:
        if source:
            source.close()
        raw_vector.close()
        raise


def save_checkpoint(model, path, *, plan, wrappers, kind, probe_observations=None):
    from stable_baselines3 import PPO
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
            or reopened.research_private_critic_mode != model.research_private_critic_mode
            or reopened.research_private_critic != critic_identity()
            or reopened.research_optimizer_identity != optimizer_identity()
            or reopened.actor_minibatches_this_run != model.actor_minibatches_this_run
            or reopened.critic_minibatches_this_run != model.critic_minibatches_this_run
            or reopened.policy.mean_update_scales != model.policy.mean_update_scales
            or reopened.policy.actor_body_update_scale != model.policy.actor_body_update_scale
            or [group.get("name") for group in reopened.policy.optimizer.param_groups] != ["actor", "critic"]):
        raise ValueError("Asymmetric checkpoint did not reopen exactly through the existingPPOloader")
    if probe_observations is not None:
        if not np.array_equal(model.predict(probe_observations, deterministic=True)[0], reopened.predict(probe_observations, deterministic=True)[0]):
            raise ValueError("Public actor predictions changed on reopening")
    record = {"schema": SCHEMA, "kind": kind, "created_at": now(), **hashes,
        "optimizer_seed": int(model.seed), "num_timesteps": model.num_timesteps,
        "additional_timesteps": model.num_timesteps - model.research_initial_num_timesteps,
        "ppo_update_count": model._n_updates, "plan_sha256": sha(plan), "environment_identity": plan["environment_identity"],
        "policy_distribution": distribution_identity(), "private_critic": critic_identity(), "initialization_transfer": model.research_transfer,
        "optimization": optimizer_identity(), "optimizer_counters": model.optimizer_counters(),
        "preconditioning": plan["preconditioning"],
        "optimizer_preconditioners": {"actor_mean_head": list(model.policy.mean_update_scales),
            "actor_body": model.policy.actor_body_update_scale, "gate_and_scale_heads": 1., "critic": 1.},
        "optimizer_learning_rates": {group["name"]: float(group["lr"]) for group in model.policy.optimizer.param_groups},
        "private_critic_conditioning_mode": model.research_private_critic_mode,
        "checkpoint": file_record(path), "normalizer": file_record(normalizer_path),
        "normalizer_observation_count": float(normalizer.obs_rms.count), "entropy_coefficient": float(model.ent_coef),
        "gaussian_log_std": None, "checkpoint_reopen_verified": True, "normalizer_reopen_verified": True,
        "checkpoint_verification_preserves_training_rng": True,
        "deterministic_prediction_reopen_verified": probe_observations is not None,
        "allocator": [wrapper.snapshot() for wrapper in wrappers]}
    publish_json(path.with_suffix(".json"), record)
    return record


def load_checkpoint(path, plan_path=None):
    policy = common.load_checkpoint(Path(path))
    if not isinstance(policy.model.policy, PreconditionedHybridPolicy):
        policy.close()
        raise ValueError("Expected a grouped-optimizer hybrid checkpoint")
    metadata = json.loads(Path(path).with_suffix(".json").read_text())
    if metadata["optimization"] != optimizer_identity() or policy.model.research_optimizer_identity != optimizer_identity():
        policy.close()
        raise ValueError("Grouped optimizer identity changed")
    if metadata["optimizer_counters"] != PreconditionedPPO.optimizer_counters(policy.model):
        policy.close()
        raise ValueError("Grouped optimizer counters changed")
    actual_preconditioners = {"actor_mean_head": list(policy.model.policy.mean_update_scales),
        "actor_body": policy.model.policy.actor_body_update_scale, "gate_and_scale_heads": 1., "critic": 1.}
    if (metadata["optimizer_preconditioners"] != actual_preconditioners
            or metadata["preconditioning"]["mean_update_scales"] != actual_preconditioners["actor_mean_head"]):
        policy.close()
        raise ValueError("Restored optimizer preconditioning differs from the pinned factors")
    if metadata["private_critic"] != critic_identity() or policy.model.research_private_critic != critic_identity():
        policy.close()
        raise ValueError("Private critic identity changed")
    if metadata["private_critic_conditioning_mode"] != policy.model.research_private_critic_mode or policy.model.policy.critic_context_mode != policy.model.research_private_critic_mode:
        policy.close()
        raise ValueError("Private critic conditioning mode changed")
    if plan_path is not None and metadata["plan_sha256"] != sha(require_plan(Path(plan_path).parent)):
        policy.close()
        raise ValueError("Asymmetric checkpoint belongs to another run")
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
    wrappers, deadline_reached = [], False

    class Progress(BaseCallback):
        saved = set()
        logged_optimizer_steps = set()
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
            self.record_optimization()

        def record_optimization(self):
            last_step = getattr(self.model, "last_optimizer_rollout_timesteps", self.model.research_initial_num_timesteps)
            steps = last_step - self.model.research_initial_num_timesteps
            if steps and steps not in self.logged_optimizer_steps:
                metrics = {key: float(value) for key, value in self.model.logger.name_to_value.items()
                    if isinstance(value, (int, float, np.number)) and np.isfinite(value)}
                append_json(out / "optimization.jsonl", {"additional_timesteps": steps,
                    "ppo_update_count": self.model._n_updates, "entropy_coefficient": self.model.ent_coef,
                    "optimizer_counters": self.model.optimizer_counters(), "metrics": metrics})
                self.logged_optimizer_steps.add(steps)

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
        initial = save_checkpoint(model, out / "initial-policy.zip", plan=plan, wrappers=wrappers, kind="initialized-asymmetric-transfer")
        publish_json(out / "transfer.json", model.research_transfer)
        callback = Progress()
        model.learn(total_timesteps=recipe["total_timesteps"], callback=callback, reset_num_timesteps=False)
        callback.record_optimization()
        require_plan(root)
        additional = model.num_timesteps - model.research_initial_num_timesteps
        if sum(wrapper.decisions for wrapper in wrappers) != additional:
            raise ValueError("Physical transition ledger differs from PPO")
        complete = not deadline_reached and not model.optimizer_deadline_reached and additional == recipe["total_timesteps"]
        endpoint = save_checkpoint(model, out / ("final-policy.zip" if complete else "incomplete-policy.zip"),
            plan=plan, wrappers=wrappers, kind="final" if complete else "incomplete", probe_observations=callback.initial_observations)
        result = {"schema": SCHEMA, "status": "completed" if complete else "deadline_incomplete",
            "completed_at": now(), "elapsed_s": time.monotonic() - started,
            "requested_additional_timesteps": recipe["total_timesteps"], "additional_timesteps": additional,
            "num_timesteps": model.num_timesteps, "ppo_update_count": model._n_updates, "plan_sha256": sha(plan),
            "initial_checkpoint": initial, "endpoint": endpoint, "private_critic": critic_identity(),
            "optimization": optimizer_identity(), "optimizer_counters": model.optimizer_counters(),
            "critic_context_mode": model.research_private_critic_mode,
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
