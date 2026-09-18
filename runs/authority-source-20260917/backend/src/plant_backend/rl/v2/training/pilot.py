"""Single exploratory PI policy; deliberately separate from registered custody.

No registered configuration, checkpoint metadata, authorization or public export
is produced. The immutable pilot plan describes the learner independently of
the controller-neutral environment template used to construct physical tasks.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, cast

import numpy as np

from . import ppo
from .config import BETA_PI_LAGRANGIAN_ALGORITHM_ID, ControllerNeutralTrainingConfigV1
from .custody_io import mkdir_exclusive, publish_bytes, publish_json, read_regular
from .custody_json import canonical_bytes
from .custody_process import capture_process
from .episode_allocator import EpisodeSeedAllocatorV1
from .runtime_provenance import capture, verify_unchanged
from .. import beta_spaces
from ..beta_config import CyclicMarketTaskConfigV1

ROOT = Path("/Volumes/Lexar/PlantAnAgentResearch/pilot-pi-seed101-v1")
MODULE = "plant_backend.rl.v2.training.pilot"
SCHEMA = "plant-beta-single-policy-pilot-v1"
STEPS = 1_048_576
SEED = 101
TRAIN_BASE = 9_500_000
EVALUATION_SEEDS = tuple(range(9_800_000, 9_800_004))
ACTORS = ("initial", "final", "H")
TRAIN_TIMEOUT = 36_000
EVALUATION_TIMEOUT = 1_200
LOG_LIMIT = 64 * 1024**2
MINIMUM_FREE = 5 * 1024**3


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def record(path: Path) -> dict[str, Any]:
    raw = read_regular(path)
    return {"path": str(path), "bytes": len(raw), "sha256": digest(raw)}


def environment_template(root: Path) -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        seed=SEED,
        total_timesteps=STEPS,
        checkpoint_every=0,
        output_dir=str(root / "artifacts"),
        beta_task=CyclicMarketTaskConfigV1(),
        notes="Environment template only; learner authority is the separate pilot plan.",
    ).validate()


def specification(root: Path) -> dict[str, Any]:
    from .constrained_ppo import ConstrainedPPOSettingsV1

    return {
        "schema": SCHEMA,
        "root": str(root),
        "run_kind": "exploratory_pilot",
        "algorithm": BETA_PI_LAGRANGIAN_ALGORITHM_ID,
        "optimizer_seed": SEED,
        "total_timesteps": STEPS,
        "environment_template": environment_template(root).to_dict(),
        "learner_settings": ConstrainedPPOSettingsV1.for_algorithm(
            BETA_PI_LAGRANGIAN_ALGORITHM_ID, optimizer_seed=SEED
        ).to_dict(),
        "training_episode_seed_base": TRAIN_BASE,
        "training_episode_seed_last": TRAIN_BASE + 1563,
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "evaluation_actors": list(ACTORS),
        "training_timeout_seconds": TRAIN_TIMEOUT,
        "evaluation_timeout_seconds": EVALUATION_TIMEOUT,
        "stream_limit_bytes": LOG_LIMIT,
        "minimum_free_bytes": MINIMUM_FREE,
        "registered_evaluation_eligible": False,
        "resume_or_replacement_allowed": False,
        "authorization": "User: Let's start by training one policy on one seed and see what happens.",
        "claim": "Exploratory learning and descriptive fixture scenarios only.",
    }


def configure_host() -> None:
    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(12)


def build_learner(
    config: ControllerNeutralTrainingConfigV1, *, base_seed: int
) -> tuple[Any, EpisodeSeedAllocatorV1, Any]:
    """Compose existing production components; never weaken their run validators."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from .constrained_ppo import ConstrainedPPOSettingsV1, build_constrained_model

    config.validate()
    if config.beta_rl is not None or config.n_envs != 4 or config.resume_from:
        raise ValueError("pilot requires a fresh four-environment task template")
    allocator = EpisodeSeedAllocatorV1.fixture(
        base_seed=base_seed,
        episode_decisions=config.episode.decision_steps(config.plant),
    )

    def factory(index: int) -> Any:
        return lambda: Monitor(
            allocator.wrap(ppo.make_env(config), environment_index=index)
        )

    vector = DummyVecEnv([factory(index) for index in range(config.n_envs)])
    try:
        model = build_constrained_model(
            vector,
            settings=ConstrainedPPOSettingsV1.for_algorithm(
                BETA_PI_LAGRANGIAN_ALGORITHM_ID, optimizer_seed=config.seed
            ),
            ppo_kwargs=asdict(config.ppo),
            verbose=1,
        )
        return model, allocator, vector
    except BaseException:
        vector.close()
        raise


def rollout_callback(directory: Path) -> Any:
    from stable_baselines3.common.callbacks import BaseCallback
    from .constrained_ppo import ConstrainedPPO

    class Progress(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.written = 0

        def flush(self) -> None:
            for ledger in cast(ConstrainedPPO, self.model).rollout_ledgers[
                self.written :
            ]:
                publish_json(
                    directory / f"{ledger.rollout_index:03d}.json", ledger.to_dict()
                )
                self.written += 1

        def _on_rollout_start(self) -> None:
            self.flush()

        def _on_training_end(self) -> None:
            self.flush()

        def _on_step(self) -> bool:
            return True

    return Progress()


def validate_endpoint(
    model: Any,
    allocator: EpisodeSeedAllocatorV1,
    config: ControllerNeutralTrainingConfigV1,
) -> dict[str, Any]:
    rollout_size = config.n_envs * config.ppo.n_steps
    if (
        config.total_timesteps % rollout_size
        or model.num_timesteps != config.total_timesteps
    ):
        raise ValueError("pilot did not reach its exact complete-rollout endpoint")
    snapshot = allocator.snapshot().to_dict()
    expected = config.total_timesteps // config.n_envs
    if allocator.decisions_consumed_by_environment != (expected,) * config.n_envs:
        raise ValueError("pilot allocator progress differs from model progress")
    if model.episode_decision_indices != tuple(
        c.decision_offset for c in allocator.cursors
    ):
        raise ValueError("pilot constraint and allocator cursors differ")
    ledgers = model.rollout_ledgers
    if len(ledgers) != config.total_timesteps // rollout_size or any(
        ledger.rollout_index != index + 1
        or ledger.transition_count != rollout_size
        or ledger.optimizer_steps
        != config.ppo.n_epochs * rollout_size // config.ppo.batch_size
        for index, ledger in enumerate(ledgers)
    ):
        raise ValueError("pilot rollout or optimizer budget differs")
    model.pin_external_episode_state(snapshot)
    return snapshot


def train_model(
    root: Path, config: ControllerNeutralTrainingConfigV1, *, base_seed: int
) -> dict[str, Any]:
    """Core seam also tested with tiny, separately seeded real-learning fixtures."""
    from .constrained_ppo import ConstrainedPPO
    import torch

    model, allocator, vector = build_learner(config, base_seed=base_seed)
    try:
        initial = root / "artifacts" / "initial-policy.pt"
        # Save just the untrained inference policy; never call it a trained checkpoint.
        torch.save(model.policy.state_dict(), initial)
        initial_record = record(initial)
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=rollout_callback(root / "rollouts"),
        )
        snapshot = validate_endpoint(model, allocator, config)
        artifact = root / "artifacts" / "final-policy.zip"
        model.save(artifact)
        state = model.embedded_constraint_state
        artifact_record = record(artifact)
        reopened = ConstrainedPPO.load(artifact, device="cpu")
        attestation = reopened.loaded_constraint_state_attestation()
        if (
            attestation.artifact_sha256 != artifact_record["sha256"]
            or attestation.num_timesteps != config.total_timesteps
            or attestation.constraint_state != state
        ):
            raise ValueError(
                "reopened pilot artifact differs from saved learner evidence"
            )
        from ...checkpoints import verify_policy_model_spaces

        verify_policy_model_spaces(
            reopened,
            observation_space=vector.observation_space,
            action_space=vector.action_space,
            path=artifact,
        )
        result = {
            "schema": SCHEMA,
            "status": "completed",
            "completed_at": timestamp(),
            "registered_evaluation_eligible": False,
            "environment_template": config.to_dict(),
            "checkpoint_contract_digest": ppo.policy_contract(
                config
            ).checkpoint_contract_digest,
            "initial_policy": initial_record,
            "final_policy": artifact_record,
            "allocator": snapshot,
            "constraint_state": state,
        }
        publish_json(root / "training-result.json", result)
        return result
    finally:
        vector.close()


def admit_plan(root: Path) -> dict[str, Any]:
    from .runtime_provenance import BetaTrainingRuntimeProvenanceV1
    from .capacity_benchmark import _host_identity

    if root != ROOT or root != root.resolve():
        raise ValueError("pilot root differs from the prospective specification")
    plan = json.loads(read_regular(root / "plan.json"))
    if canonical_bytes(plan["specification"]) != canonical_bytes(specification(root)):
        raise ValueError("pilot plan differs from the prospective specification")
    verify_unchanged(
        BetaTrainingRuntimeProvenanceV1.from_dict(plan["runtime"]),
        capture(require_clean=True),
    )
    if plan["host"] != _host_identity():
        raise ValueError("pilot host identity changed")
    if record(Path(plan["gate"]["path"])) != plan["gate"]:
        raise ValueError("pilot gate evidence changed")
    return cast(dict[str, Any], plan)


def evaluate_episode(
    config: ControllerNeutralTrainingConfigV1, *, actor: str, policy: Any, seed: int
) -> dict[str, Any]:
    from ..setpoint_controllers import LegacyHeuristicSetpointControllerV1
    from ..beta_env import PlantCyclicMarketEnvV2Beta

    if actor not in ACTORS or seed not in EVALUATION_SEEDS:
        raise ValueError("evaluation actor or seed is outside the pilot plan")
    env = ppo.make_env(config)
    if not isinstance(env, PlantCyclicMarketEnvV2Beta):
        env.close()
        raise ValueError("pilot evaluation requires the actual cyclic-market beta task")
    history: list[dict[str, Any]] = []
    failure = None
    terminated = truncated = False
    try:
        observation, _ = env.reset(seed=seed)
        heuristic = LegacyHeuristicSetpointControllerV1()
        for _ in range(env.episode_decisions):
            before = np.asarray(observation).tolist()
            if actor == "H":
                decision = heuristic.compute_command(
                    env.controller_observation,
                    constraints=env.constraints,
                    config=env.config,
                )
                action = decision.to_dict()
                observation, reward, terminated, truncated, info = (
                    env.step_process_setpoints(
                        decision.command,
                        controller_invalid=decision.invalid,
                    )
                )
            else:
                raw_action, _ = policy.predict(observation, deterministic=True)
                action = raw_action.tolist()
                observation, reward, terminated, truncated, info = env.step(raw_action)
            history.append(
                {
                    "observation": before,
                    "action": action,
                    "reward_eur": float(reward),
                    "info": info,
                }
            )
            if terminated or truncated:
                break
    except Exception as reason:
        failure = {"type": type(reason).__name__, "message": str(reason)}
    try:
        complete = (
            failure is None
            and terminated
            and not truncated
            and len(history) == env.episode_decisions
        )
        result = {
            "schema": SCHEMA,
            "actor": actor,
            "seed": seed,
            "failure": failure,
            "execution_complete": complete,
            "decisions": len(history),
            "terminated": terminated,
            "truncated": truncated,
            "physical_metrics": env.metrics.to_dict(),
            "economic_metrics": env.economic_metrics.to_dict(),
            "cycle_settlements": [item.to_dict() for item in env.cycle_settlements],
            "all_cycles_feasible": complete
            and len(env.cycle_settlements) == 7
            and all(item.feasible for item in env.cycle_settlements),
            "registered_hard_safety_gate_performed": False,
            "manifest": env.audit_manifest() if history else None,
            "history": history,
        }
        return result
    finally:
        env.close()


def evaluate(root: Path) -> dict[str, Any]:
    from .constrained_ppo import ConstrainedPPO

    training = json.loads(read_regular(root / "training-result.json"))
    config = environment_template(root)
    if canonical_bytes(training["environment_template"]) != canonical_bytes(
        config.to_dict()
    ):
        raise ValueError("pilot training task differs from the evaluation task")
    for name in ("initial_policy", "final_policy"):
        expected = training[name]
        if record(Path(expected["path"])) != expected:
            raise ValueError("pilot policy artifact changed before evaluation")
    final = ConstrainedPPO.load(root / "artifacts" / "final-policy.zip", device="cpu")
    if (
        final.embedded_constraint_state != training["constraint_state"]
        or final.num_timesteps != STEPS
    ):
        raise ValueError("pilot final learner state differs from its saved endpoint")
    initial = load_initial_policy(root / "artifacts" / "initial-policy.pt", final)
    rows = []
    for seed in EVALUATION_SEEDS:
        for actor in ACTORS:
            result = evaluate_episode(
                config,
                actor=actor,
                policy=initial if actor == "initial" else final,
                seed=seed,
            )
            raw = canonical_bytes(result)
            encoded = gzip.compress(raw, compresslevel=6, mtime=0)
            if len(encoded) > LOG_LIMIT:
                raise ValueError("pilot trajectory exceeded its stored size cap")
            path = root / "evaluation" / f"{seed}-{actor}.json.gz"
            publish_bytes(path, encoded)
            rows.append(
                {
                    **{
                        key: value
                        for key, value in result.items()
                        if key not in ("history", "manifest")
                    },
                    "trajectory": record(path),
                    "decoded_sha256": digest(raw),
                }
            )
    report = {
        "schema": SCHEMA,
        "completed_at": timestamp(),
        "cells": rows,
        "claim": "Descriptive exploratory scenarios; no registered selection or held-out claim.",
    }
    publish_json(root / "evaluation-result.json", report)
    return report


def load_initial_policy(path: Path, final: Any) -> Any:
    """Restore all initial weights into the same admitted policy architecture."""
    import torch

    initial = deepcopy(final.policy)
    initial.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=True
    )
    return initial


def run(root: Path, gate: Path, *, authorized: bool) -> dict[str, Any]:
    if not authorized:
        raise ValueError(
            "pilot execution requires the explicit authorization assertion"
        )
    if root != ROOT or root != root.resolve() or root.exists():
        raise ValueError("pilot requires its exact, absent external root; no retry")
    runtime = capture(require_clean=True)
    gate_raw = read_regular(gate)
    gate_report = json.loads(gate_raw)
    if (
        gate_report.get("source_commit") != runtime.git_head
        or gate_report.get("passed") is not True
    ):
        raise ValueError("pilot gate must pass at this exact clean source commit")
    if shutil.disk_usage(root.parent).free < MINIMUM_FREE:
        raise ValueError("pilot needs 5 GiB free before claiming an attempt")
    from .capacity_benchmark import _host_identity

    plan = {
        "specification": specification(root),
        "runtime": runtime.to_dict(),
        "host": _host_identity(),
        "gate": record(gate),
        "claimed_at": timestamp(),
        "pid": os.getpid(),
    }
    mkdir_exclusive(root)
    publish_json(root / "plan.json", plan)
    for name in ("logs", "artifacts", "rollouts", "evaluation"):
        mkdir_exclusive(root / name)
    stages: list[dict[str, Any]] = []
    failure = None
    try:
        for stage, timeout in (
            ("_train", TRAIN_TIMEOUT),
            ("_evaluate", EVALUATION_TIMEOUT),
        ):
            admit_plan(root)
            process, codes = capture_process(
                [sys.executable, "-m", MODULE, stage],
                stdout=root / "logs" / f"{stage}.stdout.log",
                stderr=root / "logs" / f"{stage}.stderr.log",
                timeout_seconds=timeout,
                log_limit_bytes=LOG_LIMIT,
            )
            stages.append({"stage": stage, "process": process, "codes": sorted(codes)})
            publish_json(root / f"{stage}-process.json", stages[-1])
            admit_plan(root)
            if codes:
                raise RuntimeError(f"pilot {stage} failed: {sorted(codes)}")
        result = json.loads(read_regular(root / "training-result.json"))
        if result["status"] != "completed":
            raise ValueError("pilot training did not complete")
    except BaseException as reason:
        failure = {"type": type(reason).__name__, "message": str(reason)}
    report = {
        "schema": SCHEMA,
        "closed_at": timestamp(),
        "status": "failed" if failure else "completed",
        "failure": failure,
        "plan": record(root / "plan.json"),
        "stages": stages,
        "results": [
            record(root / name)
            for name in ("training-result.json", "evaluation-result.json")
            if (root / name).is_file()
        ],
        "registered_evaluation_eligible": False,
    }
    publish_json(root / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "_train", "_evaluate"))
    parser.add_argument("--assert-authorized", action="store_true")
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args(argv)
    configure_host()
    if args.command == "run":
        if args.gate is None:
            parser.error("run requires --gate at the exact source revision")
        report = run(ROOT, args.gate, authorized=args.assert_authorized)
        print(
            json.dumps(
                {"status": report["status"], "report": str(ROOT / "report.json")}
            )
        )
        return 0 if report["status"] == "completed" else 1
    admit_plan(ROOT)
    publish_json(
        ROOT / f"{args.command}-attempt.json",
        {"started_at": timestamp(), "pid": os.getpid()},
    )
    if args.command == "_train":
        train_model(ROOT, environment_template(ROOT), base_seed=TRAIN_BASE)
    else:
        evaluate(ROOT)
    admit_plan(ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
