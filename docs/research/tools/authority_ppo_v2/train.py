"""Durable six-endpoint PPO campaign with frozen inputs and deadline enforcement."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
for folder in (ROOT, ROOT / "backend/src", ROOT / "simulator/src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import gymnasium as gym
import numpy as np
from docs.research.tools.authority_ppo_v2 import benchmark, policy
from docs.research.tools import coverage_ppo_v1 as legacy
from plant_backend.rl.v2.training.config import BETA_PI_LAGRANGIAN_ALGORITHM_ID
from plant_backend.rl.v2.training.custody_io import mkdir_exclusive, publish_bytes, publish_json, read_regular
from plant_backend.rl.v2.training.custody_json import canonical_bytes

SCHEMA = policy.SCHEMA
ARMS = policy.ARMS
OPTIMIZER_SEEDS = (101, 211, 307)
TRAIN_BASE = 9_940_000
TOTAL_STEPS = 262_144
ENVIRONMENTS = 4
EPISODE_DECISIONS = 672
CHECKPOINT_STEPS = 65_536
REWARD_SCALE = legacy.REWARD_SCALE
INITIAL_STD = legacy.INITIAL_STD
DEFAULT_DEADLINE = "2026-09-17T05:26:17Z"
file_record = legacy.file_record
now = legacy.now
actor_parameter_sha256 = policy.actor_parameter_sha256
ppo_settings = legacy.ppo_settings


def actor_mean_parameter_sha256(model: Any) -> str:
    digest = hashlib.sha256(b"authority-ppo-actor-mean-parameters-v2\x00")
    count = 0
    for name, tensor in sorted(model.policy.state_dict().items()):
        if not name.startswith(("action_net.", "mlp_extractor.policy_net.", "features_extractor.")):
            continue
        count += 1
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if not count:
        raise ValueError("No deterministic actor parameters recognized")
    return digest.hexdigest()


def source_records() -> list[dict[str, Any]]:
    # Explicit new campaign inputs only: evaluator/report/helper files can evolve
    # without changing a training task. Broad backend/simulator coverage includes
    # every imported physics/training dependency and protects lazy imports.
    paths = {
        Path(__file__), Path(policy.__file__), Path(benchmark.__file__),
        benchmark.CONTRACT_PATH,
        ROOT / "docs/research/benchmarks/feasible25_offgrid_v1.json",
        ROOT / "docs/research/AUTHORITY_PPO_V2_PROTOCOL.md",
        ROOT / "docs/research/tools/coverage_ppo_v1.py",
        ROOT / "docs/research/tools/feasible_benchmark_v1.py",
        ROOT / "docs/research/tools/feasible_mission_replay.py",
        ROOT / "docs/research/tools/feasible_mission_controller.py",
        ROOT / "docs/research/tools/offgrid_coverage_v1.py",
        ROOT / "uv.lock",
    }
    for directory in (ROOT / "backend/src/plant_backend", ROOT / "simulator/src/plant_sim"):
        paths.update(directory.rglob("*.py"))
    paths.discard(legacy.EXCLUDED_UNRELATED_SOURCE)
    return [file_record(path.resolve()) for path in sorted(paths)]


class _ScaleReward(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, float(reward) * REWARD_SCALE, terminated, truncated, info


def make_learner(optimizer_seed: int, arm: str, *, fixture_steps: int | None = None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPOSettingsV1, build_constrained_model
    from plant_backend.rl.v2.training.episode_allocator import EpisodeSeedAllocatorV1

    policy._check_arm(arm)
    if optimizer_seed not in OPTIMIZER_SEEDS:
        raise ValueError("Optimizer seed is outside frozen plan")
    if fixture_steps is not None and not 1 <= fixture_steps <= 2048:
        raise ValueError("Fixture is limited to one 8192-transition rollout")
    allocator = EpisodeSeedAllocatorV1.fixture(base_seed=TRAIN_BASE, episode_decisions=EPISODE_DECISIONS)
    def factory(index):
        return lambda: _ScaleReward(Monitor(allocator.wrap(
            policy.AuthorityResidualEnv(benchmark.BenchmarkEnv(split="train"), arm),
            environment_index=index,
        )))
    vector = DummyVecEnv([factory(i) for i in range(ENVIRONMENTS)])
    try:
        settings = asdict(ppo_settings())
        if fixture_steps is not None:
            settings.update(n_steps=fixture_steps, batch_size=ENVIRONMENTS * fixture_steps, n_epochs=1)
        model = build_constrained_model(
            vector,
            settings=ConstrainedPPOSettingsV1.for_algorithm(BETA_PI_LAGRANGIAN_ALGORITHM_ID, optimizer_seed=optimizer_seed),
            ppo_kwargs=settings, verbose=1,
        )
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.log_std.fill_(math.log(INITIAL_STD))
        base = benchmark.BenchmarkEnv(split="train")
        try:
            model.research_identity = policy.interface_identity(base, arm)
        finally:
            base.close()
        return model, allocator, vector
    except BaseException:
        vector.close()
        raise


def _deadline(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Deadline must include a UTC offset")
    return result.astimezone(timezone.utc)


def plan(deadline_utc: str = DEFAULT_DEADLINE) -> dict[str, Any]:
    _deadline(deadline_utc)
    contract = benchmark.load_benchmark_contract()
    return {
        "schema": SCHEMA, "benchmark_contract_sha256": contract["contract_sha256"],
        "arms": list(ARMS), "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "physical_training_seeds": [TRAIN_BASE, TRAIN_BASE + 391],
        "same_training_scenarios_across_arms_and_optimizer_seeds": True,
        "total_transitions_per_policy": TOTAL_STEPS, "environments": ENVIRONMENTS,
        "resets_per_policy": 392, "completed_episodes_per_policy": 388,
        "partial_final_episodes_per_policy": 4, "partial_final_episode_decisions": 352,
        "ppo": asdict(ppo_settings()), "algorithm": BETA_PI_LAGRANGIAN_ALGORITHM_ID,
        "reward_scale": REWARD_SCALE, "initial_gaussian_std": INITIAL_STD,
        "costs": "Unchanged four normalized daily stock deficits, zero limits; gamma=1, GAE lambda=.95.",
        "initialization": "Fresh optimizer-seeded network with exactly zero actor mean head; shared initial tensors for paired arms.",
        "narrow_fraction": legacy.RESIDUAL_FRACTION, "broad_fraction": policy.BROAD_FRACTION,
        "exploration": "Same action-space Gaussian; physical exploration differs by interface, so results compare the complete interfaces.",
        "prior": policy.make_coverage_prior().to_dict(),
        "selection": "All six fixed final endpoints retained and evaluated; intermediate checkpoints diagnostic only, no selection.",
        "intermediate_checkpoint_transitions": [65536, 131072, 196608],
        "held_out_gate": "All six final checkpoint identities frozen before opening held-out scenarios.",
        "deadline_utc": deadline_utc,
        "deadline_policy": "Callback aborts at next decision if wall deadline reached. An incomplete run has failure.json and cannot pass the endpoint gate.",
        "limitation": "Constraint costs and projected commands do not guarantee full-mission success or safety.",
        "threads": {"torch_intraop": 1, "torch_interop": 1},
        "runtime": {"python": sys.version, "executable": sys.executable, "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "gymnasium", "scipy", "torch", "stable-baselines3")
        }},
        "excluded_source": {"path": str(legacy.EXCLUDED_UNRELATED_SOURCE), "reason": "Independent MPC not imported by policy training; evaluator freezes its own source."},
        "source_files": source_records(), "registered_evaluation_eligible": False,
    }


def freeze_plan(root: Path, deadline_utc: str = DEFAULT_DEADLINE) -> dict[str, Any]:
    record = plan(deadline_utc)
    mkdir_exclusive(root)
    publish_json(root / "plan.json", record)
    return record


def require_plan(root: Path) -> dict[str, Any]:
    recorded = json.loads(read_regular(root / "plan.json"))
    if canonical_bytes(recorded) != canonical_bytes(plan(recorded["deadline_utc"])):
        raise ValueError("Training task, settings, runtime or source changed after plan freeze")
    return recorded


_require_plan = require_plan


def endpoint_directory(root: Path, arm: str, optimizer_seed: int) -> Path:
    policy._check_arm(arm)
    if optimizer_seed not in OPTIMIZER_SEEDS:
        raise ValueError("Unknown optimizer seed")
    return root / f"{arm}-seed-{optimizer_seed}"


def _save_checkpoint(model: Any, allocator: Any, artifact: Path) -> dict[str, Any]:
    from plant_backend.rl.v2.training.research_effort_batch import preflight_checkpoint
    snapshot = allocator.snapshot().to_dict()
    model.pin_external_episode_state(snapshot)
    # Save into memory, then publish exclusively/durably; model.save(path) alone
    # can silently overwrite a previously admitted checkpoint.
    buffer = io.BytesIO()
    model.save(buffer)
    publish_bytes(artifact, buffer.getvalue())
    state = model.embedded_constraint_state
    preflight_checkpoint(artifact, model.research_identity, state)
    return {"checkpoint": file_record(artifact), "constraint_state": state, "allocator": snapshot}


def train(root: Path, arm: str, optimizer_seed: int) -> dict[str, Any]:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPO, policy_parameter_sha256
    from plant_backend.rl.v2.training.research_batch import training_callback

    frozen = require_plan(root)
    deadline = _deadline(frozen["deadline_utc"])
    if datetime.now(timezone.utc) >= deadline:
        raise TimeoutError("Frozen training deadline has already passed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = endpoint_directory(root, arm, optimizer_seed)
    mkdir_exclusive(out)
    mkdir_exclusive(out / "diagnostics")
    mkdir_exclusive(out / "checkpoints")
    publish_json(out / "attempt.json", {
        "started_at": now(), "arm": arm, "optimizer_seed": optimizer_seed,
        "plan_sha256": hashlib.sha256(canonical_bytes(frozen)).hexdigest(),
    })
    model = vector = None
    started = time.monotonic()
    intermediates = []

    class Progress(BaseCallback):
        def _on_rollout_start(self):
            if self.num_timesteps and self.num_timesteps % CHECKPOINT_STEPS == 0:
                artifact = out / "checkpoints" / f"policy-{self.num_timesteps:06d}.zip"
                saved = _save_checkpoint(self.model, allocator, artifact)
                record = {"num_timesteps": self.num_timesteps, "checkpoint": saved["checkpoint"],
                          "actor_mean_sha256": actor_mean_parameter_sha256(self.model),
                          "use": "diagnostic only; never selected for final evaluation"}
                publish_json(artifact.with_suffix(".json"), record)
                intermediates.append(record)

        def _on_step(self):
            if self.num_timesteps % 4096 == 0:
                item = {"arm": arm, "optimizer_seed": optimizer_seed,
                        "transitions": self.num_timesteps, "elapsed_s": round(time.monotonic() - started, 3)}
                with (out / "progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(item) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                print(json.dumps(item), flush=True)
            return datetime.now(timezone.utc) < deadline

    try:
        model, allocator, vector = make_learner(optimizer_seed, arm)
        buffer = io.BytesIO()
        torch.save(model.policy.state_dict(), buffer)
        publish_bytes(out / "initial-policy.pt", buffer.getvalue())
        initial_tensors = policy_parameter_sha256(model)
        initial_actor = actor_parameter_sha256(model)
        initial_mean = actor_mean_parameter_sha256(model)
        model.learn(total_timesteps=TOTAL_STEPS,
                    callback=CallbackList([training_callback(out / "diagnostics"), Progress()]))
        if model.num_timesteps != TOTAL_STEPS:
            raise TimeoutError(f"Exact endpoint not reached: {model.num_timesteps}/{TOTAL_STEPS}")
        if allocator.decisions_consumed_by_environment != (TOTAL_STEPS // ENVIRONMENTS,) * ENVIRONMENTS:
            raise ValueError("Allocator did not reach exact endpoint")
        if model.episode_decision_indices != tuple(cursor.decision_offset for cursor in allocator.cursors):
            raise ValueError("Learner and allocator decision cursors differ")
        ledgers = model.rollout_ledgers
        if len(ledgers) != 32 or any(row.transition_count != 8192 or row.optimizer_steps != 320 for row in ledgers):
            raise ValueError("Incomplete rollout or optimizer ledger")
        if [item["num_timesteps"] for item in intermediates] != [65536, 131072, 196608]:
            raise ValueError("Missing prospective intermediate checkpoints")
        artifact = out / "final-policy.zip"
        saved = _save_checkpoint(model, allocator, artifact)
        state = saved["constraint_state"]
        final_tensors = policy_parameter_sha256(model)
        final_actor = actor_parameter_sha256(model)
        final_mean = actor_mean_parameter_sha256(model)
        reopened = ConstrainedPPO.load(artifact, device="cpu")
        attestation = reopened.loaded_constraint_state_attestation()
        deterministic_actions, _ = reopened.predict(model._last_obs.copy(), deterministic=True)
        if (attestation.artifact_sha256 != saved["checkpoint"]["sha256"]
            or attestation.num_timesteps != TOTAL_STEPS or attestation.constraint_state != state):
            raise ValueError("Saved endpoint failed exact reopen attestation")
        if (policy_parameter_sha256(reopened) != final_tensors or initial_tensors == final_tensors
            or actor_parameter_sha256(reopened) != final_actor or initial_actor == final_actor
            or actor_mean_parameter_sha256(reopened) != final_mean or initial_mean == final_mean
            or not np.any(deterministic_actions != 0)):
            raise ValueError("Reopened tensors differ or deterministic actor did not change")
        require_plan(root)
        result = {
            "schema": SCHEMA, "completed_at": now(), "arm": arm, "optimizer_seed": optimizer_seed,
            "elapsed_s": time.monotonic() - started, "num_timesteps": model.num_timesteps,
            "minibatch_updates": sum(row.optimizer_steps for row in ledgers),
            "initial_checkpoint": file_record(out / "initial-policy.pt"),
            "final_checkpoint": saved["checkpoint"], "intermediate_checkpoints": intermediates,
            "initial_tensor_sha256": initial_tensors, "final_tensor_sha256": final_tensors,
            "initial_actor_sha256": initial_actor, "final_actor_sha256": final_actor,
            "initial_actor_mean_sha256": initial_mean, "final_actor_mean_sha256": final_mean,
            "actor_parameters_changed": True, "actor_mean_parameters_changed": True,
            "deterministic_action_at_final_train_observations": deterministic_actions.tolist(),
            "policy_identity": model.research_identity, "constraint_state": state,
            "allocator": saved["allocator"], "checkpoint_reopen_verified": True,
            "source_files": source_records(), "benchmark_contract_sha256": frozen["benchmark_contract_sha256"],
            "claim": "Actual PPO endpoint completed and deterministic actor changed; performance requires independent evaluation.",
        }
        publish_json(out / "result.json", result)
        return result
    except BaseException as exc:
        publish_json(out / "failure.json", {"at": now(), "type": type(exc).__name__, "message": str(exc),
                                            "num_timesteps": None if model is None else model.num_timesteps})
        raise
    finally:
        if vector is not None:
            vector.close()


def load_final(root: Path, arm: str, optimizer_seed: int):
    from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPO, policy_parameter_sha256
    from plant_backend.rl.v2.training.research_effort_batch import preflight_checkpoint
    require_plan(root)
    out = endpoint_directory(root, arm, optimizer_seed)
    result = json.loads(read_regular(out / "result.json"))
    artifact = out / "final-policy.zip"
    if (result["schema"] != SCHEMA or result["arm"] != arm or result["optimizer_seed"] != optimizer_seed
        or file_record(artifact) != result["final_checkpoint"]):
        raise ValueError("Final checkpoint provenance differs")
    preflight_checkpoint(artifact, result["policy_identity"], result["constraint_state"])
    model = ConstrainedPPO.load(artifact, device="cpu")
    if (model.num_timesteps != TOTAL_STEPS or model.research_identity != result["policy_identity"]
        or model.embedded_constraint_state != result["constraint_state"]
        or policy_parameter_sha256(model) != result["final_tensor_sha256"]
        or actor_parameter_sha256(model) != result["final_actor_sha256"]
        or actor_mean_parameter_sha256(model) != result["final_actor_mean_sha256"]):
        raise ValueError("Loaded policy identity, endpoint or tensors differ")
    return model


def load_initial(root: Path, arm: str, optimizer_seed: int):
    import torch
    from plant_backend.rl.v2.training.constrained_ppo import policy_parameter_sha256
    require_plan(root)
    out = endpoint_directory(root, arm, optimizer_seed)
    result = json.loads(read_regular(out / "result.json"))
    artifact = out / "initial-policy.pt"
    if file_record(artifact) != result["initial_checkpoint"]:
        raise ValueError("Initial checkpoint bytes changed")
    model, _, vector = make_learner(optimizer_seed, arm)
    try:
        model.policy.load_state_dict(torch.load(io.BytesIO(read_regular(artifact)), map_location="cpu", weights_only=True), strict=True)
        if (policy_parameter_sha256(model) != result["initial_tensor_sha256"]
            or actor_mean_parameter_sha256(model) != result["initial_actor_mean_sha256"]
            or torch.count_nonzero(model.policy.action_net.weight).item()
            or torch.count_nonzero(model.policy.action_net.bias).item()):
            raise ValueError("Initial tensors differ or initial mean is no longer exactly zero")
        return model
    finally:
        vector.close()


def freeze_endpoints(root: Path) -> dict[str, Any]:
    frozen = require_plan(root)
    records = []
    for arm in ARMS:
        for seed in OPTIMIZER_SEEDS:
            load_final(root, arm, seed)
            out = endpoint_directory(root, arm, seed)
            records.append({"arm": arm, "optimizer_seed": seed, "result": file_record(out / "result.json"),
                            "checkpoint": file_record(out / "final-policy.zip")})
    body = {"schema": SCHEMA, "frozen_at": now(), "plan_sha256": hashlib.sha256(canonical_bytes(frozen)).hexdigest(),
            "endpoints": records, "selection": "All six fixed final endpoints; no exclusions."}
    publish_json(root / "endpoints-frozen.json", body)
    return body


def require_frozen_endpoints(root: Path) -> dict[str, Any]:
    frozen = require_plan(root)
    body = json.loads(read_regular(root / "endpoints-frozen.json"))
    if (body["schema"] != SCHEMA or body["plan_sha256"] != hashlib.sha256(canonical_bytes(frozen)).hexdigest()
        or [(row["arm"], row["optimizer_seed"]) for row in body["endpoints"]]
        != [(arm, seed) for arm in ARMS for seed in OPTIMIZER_SEEDS]):
        raise ValueError("Held-out gate lacks all six fixed final endpoints")
    for row in body["endpoints"]:
        for field in ("result", "checkpoint"):
            if file_record(Path(row[field]["path"])) != row[field]:
                raise ValueError("Final endpoint changed after freeze")
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-plan", "train", "freeze-endpoints"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--optimizer-seed", type=int, choices=OPTIMIZER_SEEDS)
    parser.add_argument("--deadline-utc", default=DEFAULT_DEADLINE)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "freeze-plan":
        result = freeze_plan(root, args.deadline_utc)
    elif args.command == "freeze-endpoints":
        result = freeze_endpoints(root)
    else:
        if args.optimizer_seed is None or args.arm is None:
            parser.error("train requires --arm and --optimizer-seed")
        result = train(root, args.arm, args.optimizer_seed)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
