"""Discarded training-only runtime fixture; no production or frozen-source edits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.research.tools.free_ppo_v1 import train


def profile(training_root: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    plan = train.require_plan(training_root)
    stages = {name: {"seconds": 0.0, "calls": 0} for name in
              ("vector_step_wait", "vector_reset", "rollout_policy_forward", "ppo_optimization")}
    model = vector = None

    def timed(name, original):
        def invoke(*args, **kwargs):
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                stages[name]["seconds"] += time.perf_counter() - started
                stages[name]["calls"] += 1
        return invoke

    construction_start = time.perf_counter()
    try:
        model, vector, wrappers = train.make_learner(plan, 101)
        construction_seconds = time.perf_counter() - construction_start
        initial_actor = train.actor_mean_parameter_sha256(model)
        # These closures replace methods only on this discarded fixture's
        # instances. No class, imported module or production process is patched.
        vector.step_wait = timed("vector_step_wait", vector.step_wait)
        vector.reset = timed("vector_reset", vector.reset)
        model.policy.forward = timed("rollout_policy_forward", model.policy.forward)
        model.train = timed("ppo_optimization", model.train)
        started_at = datetime.now(timezone.utc).isoformat()
        learn_start = time.perf_counter()
        model.learn(total_timesteps=8192)
        learn_seconds = time.perf_counter() - learn_start
        completed_at = datetime.now(timezone.utc).isoformat()
        if model.num_timesteps != 8192:
            raise ValueError("Profile did not execute the exact fixture transition count")
        if sum(wrapper.decisions for wrapper in wrappers) != 8192:
            raise ValueError("Profile physical-transition ledger differs")
        changed = initial_actor != train.actor_mean_parameter_sha256(model)
        if not changed:
            raise ValueError("Fixture did not execute an actor-mean update")
        measured = sum(item["seconds"] for item in stages.values())
        if measured > learn_seconds + 1e-6:
            raise ValueError("Stage timers overlap; fractions would be misleading")
        stages["other_rollout_and_framework_overhead"] = {
            "seconds": learn_seconds - measured, "calls": None}
        for item in stages.values():
            item["fraction_of_learn"] = item["seconds"] / learn_seconds
        environment_seconds = stages["vector_step_wait"]["seconds"] + stages["vector_reset"]["seconds"]
        optimization_seconds = stages["ppo_optimization"]["seconds"]
        inference_seconds = stages["rollout_policy_forward"]["seconds"]
        record = {
            "schema": "free-ppo-runtime-profile-v1", "started_at": started_at,
            "completed_at": completed_at, "scope": "Discarded engineering fixture on training scenarios only; no policy weights retained.",
            "production_plan": train.file_record(training_root / "plan.json"),
            "production_plan_sha256": train.sha(plan), "profiler_source": train.file_record(Path(__file__)),
            "runtime": train.runtime_record(), "optimizer_seed": 101,
            "physical_transitions": model.num_timesteps, "environments": plan["environments"],
            "ppo_rollouts": 1, "ppo_epochs": model._n_updates,
            "optimizer_minibatches": model._n_updates * (8192 // plan["ppo"]["batch_size"]),
            "actor_mean_parameters_changed": changed,
            "construction_seconds": construction_seconds, "learn_seconds": learn_seconds,
            "learn_transitions_per_second": 8192 / learn_seconds,
            "stages": stages, "allocator": [wrapper.snapshot() for wrapper in wrappers],
            "environment_fraction": environment_seconds / learn_seconds,
            "neural_inference_and_optimization_fraction": (inference_seconds + optimization_seconds) / learn_seconds,
            "ideal_speedup_if_optimization_were_free": learn_seconds / (learn_seconds - optimization_seconds),
            "ideal_speedup_if_inference_and_optimization_were_free": learn_seconds / (learn_seconds - optimization_seconds - inference_seconds),
            "limitations": [
                "One 8192-transition rollout, not a hardware benchmark or a long-run scaling study.",
                "The production learners continue concurrently; evaluation may also share this local host.",
                "Vector step_wait includes plant simulation, observation construction, automatic episode resets and observation normalization.",
                "Explicit vector reset is timed separately; automatic resets occur inside vector step_wait.",
                "Policy forward times rollout neural inference; PPO train times all optimization, including its internal evaluation/gradients.",
                "Instance-level perf_counter wrappers add small overhead; no cProfile or global monkeypatch is used.",
                "Production progress-file writes, diagnostic checkpoint I/O and checkpoint verification are excluded from learn timing.",
                "Ideal speedup figures are arithmetic upper bounds assuming unchanged remaining work and zero transfer overhead, not predictions for a GPU or Bocconi node.",
            ],
        }
        train.require_plan(training_root)
        train.publish_json(output, record)
        return record
    finally:
        if vector is not None:
            vector.close()


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = profile(args.training_root.resolve(), args.output.resolve())
    print(json.dumps({key: result[key] for key in
                      ("physical_transitions", "learn_seconds", "learn_transitions_per_second", "stages",
                       "ideal_speedup_if_inference_and_optimization_were_free")}, indent=2))


if __name__ == "__main__":
    main()
