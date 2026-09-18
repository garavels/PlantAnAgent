"""Prospectively fixed adaptive PPO development batch, outside registered custody."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, cast

import gymnasium as gym
from .. import beta_spaces
from ..beta_config import CyclicMarketTaskConfigV1
from . import ppo
from .config import (
    BETA_PI_LAGRANGIAN_ALGORITHM_ID as PI,
    BETA_SCALAR_PPO_ALGORITHM_ID as SCALAR,
    ControllerNeutralTrainingConfigV1,
)
from .custody_io import mkdir_exclusive, publish_bytes, publish_json, read_regular
from .custody_json import canonical_bytes
from .custody_process import capture_process
from .episode_allocator import EpisodeSeedAllocatorV1
from .runtime_provenance import (
    BetaTrainingRuntimeProvenanceV1,
    capture,
    verify_unchanged,
)

MODULE = "plant_backend.rl.v2.training.research_batch"
ROOT = Path("/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-001")
SCHEMA = "plant-adaptive-rl-development-batch-001"
TOTAL_STEPS = 131_072
OPTIMIZER_SEED = 101
TRAIN_SEED_BASE = 9_600_000
DEV_SEEDS = tuple(range(9_700_000, 9_700_008))
BASELINES = ("H", "OperatingMPC", "M24", "E-beta")
LOG_LIMIT = 64 * 1024**2
ARCHIVE_LIMIT = 256 * 1024**2
TRAIN_TIMEOUT = 3 * 3600
EVAL_TIMEOUT = 600
MINIMUM_FREE = 10 * 1024**3
INTEGRITY_EXIT = 86


class TrainingUnavailable(ValueError):
    """A retained failed training cell cannot supply an evaluation policy."""


@dataclass(frozen=True)
class Arm:
    name: str
    algorithm: str
    reward_scale: float
    initial_std: float


ARMS = (
    Arm("A-pi-raw", PI, 1.0, 1.0),
    Arm("B-pi-scaled", PI, 0.001, 1.0),
    Arm("C-scalar-raw", SCALAR, 1.0, 1.0),
    Arm("D-scalar-scaled", SCALAR, 0.001, 1.0),
    Arm("E-pi-scaled-lowstd", PI, 0.001, 0.2),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_record(path: Path) -> dict[str, Any]:
    data = read_regular(path)
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def template(root: Path) -> ControllerNeutralTrainingConfigV1:
    return ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        seed=OPTIMIZER_SEED,
        total_timesteps=TOTAL_STEPS,
        checkpoint_every=0,
        output_dir=str(root / "artifacts"),
        beta_task=CyclicMarketTaskConfigV1(),
        notes="Adaptive research environment template; learner and optimizer reward units in batch plan.",
    ).validate()


def specification(root: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "root": str(root),
        "arms": [asdict(a) for a in ARMS],
        "environment_template": template(root).to_dict(),
        "training_seed_base": TRAIN_SEED_BASE,
        "development_seeds": list(DEV_SEEDS),
        "baselines": list(BASELINES),
        "initial_baseline_source": ARMS[0].name,
        "checkpoint_selection": "fixed final endpoint only",
        "torch_intraop_threads": 1,
        "torch_interop_threads": 1,
        "train_timeout_seconds": TRAIN_TIMEOUT,
        "evaluation_timeout_seconds": EVAL_TIMEOUT,
        "stream_limit_bytes": LOG_LIMIT,
        "archive_limit_bytes": ARCHIVE_LIMIT,
        "minimum_free_bytes": MINIMUM_FREE,
        "registered_evaluation_eligible": False,
        "authorization": "User granted full research autonomy to diagnose learning failure and iterate experiments toward verified operational benefit.",
    }


class OptimizerRewardScale(gym.Wrapper):  # type: ignore[type-arg]
    """Scale the learner signal without touching physical or constraint evidence."""

    def __init__(self, env: gym.Env, factor: float) -> None:  # type: ignore[type-arg]
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("optimizer reward scale must be positive and finite")
        super().__init__(env)
        self.factor = factor

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, float(reward) * self.factor, terminated, truncated, info


def make_learner(
    config: ControllerNeutralTrainingConfigV1, arm: Arm, *, seed_base: int
) -> tuple[Any, EpisodeSeedAllocatorV1, Any]:
    import torch
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from .constrained_ppo import ConstrainedPPOSettingsV1, build_constrained_model

    config.validate()
    if config.beta_rl is not None or config.n_envs != 4 or config.resume_from:
        raise ValueError("adaptive research requires a fresh four-environment template")
    allocator = EpisodeSeedAllocatorV1.fixture(
        base_seed=seed_base,
        episode_decisions=config.episode.decision_steps(config.plant),
    )

    def factory(index: int) -> Any:
        return lambda: OptimizerRewardScale(
            Monitor(allocator.wrap(ppo.make_env(config), environment_index=index)),
            arm.reward_scale,
        )

    vector = DummyVecEnv([factory(index) for index in range(4)])
    try:
        model = build_constrained_model(
            vector,
            settings=ConstrainedPPOSettingsV1.for_algorithm(
                arm.algorithm, optimizer_seed=config.seed
            ),
            ppo_kwargs=asdict(config.ppo),
            verbose=1,
        )
        # The new experiment records this explicit pre-learning change. Frozen
        # settings and constructors remain unchanged; no registered artifact is made.
        if arm.initial_std != 1.0:
            with torch.no_grad():
                model.policy.log_std.fill_(math.log(arm.initial_std))
        return model, allocator, vector
    except BaseException:
        vector.close()
        raise


def training_callback(directory: Path) -> Any:
    from stable_baselines3.common.callbacks import BaseCallback
    from .constrained_ppo import ConstrainedPPO
    from .research_diagnostics import capture_rollout_diagnostics

    class Recorder(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.written = 0

        def flush(self) -> None:
            model = cast(ConstrainedPPO, self.model)
            for ledger in model.rollout_ledgers[self.written :]:
                publish_json(
                    directory / f"rollout-{ledger.rollout_index:03d}.json",
                    ledger.to_dict(),
                )
                self.written += 1

        def _on_rollout_start(self) -> None:
            self.flush()

        def _on_rollout_end(self) -> None:
            model = cast(ConstrainedPPO, self.model)
            diagnostic = capture_rollout_diagnostics(model)
            publish_json(
                directory / f"diagnostic-{len(model.rollout_ledgers)+1:03d}.json",
                diagnostic,
            )

        def _on_training_end(self) -> None:
            self.flush()

        def _on_step(self) -> bool:
            return True

    return Recorder()


def train_cell(
    directory: Path,
    config: ControllerNeutralTrainingConfigV1,
    arm: Arm,
    *,
    seed_base: int,
) -> dict[str, Any]:
    import torch
    from .constrained_ppo import ConstrainedPPO
    from .pilot import validate_endpoint

    publish_json(
        directory / "attempt.json",
        {
            "started_at": now(),
            "arm": asdict(arm),
            "template": config.to_dict(),
            "seed_base": seed_base,
        },
    )
    mkdir_exclusive(directory / "diagnostics")
    model, allocator, vector = make_learner(config, arm, seed_base=seed_base)
    try:
        buffer = io.BytesIO()
        torch.save(model.policy.state_dict(), buffer)
        publish_bytes(directory / "initial-policy.pt", buffer.getvalue())
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=training_callback(directory / "diagnostics"),
        )
        snapshot = validate_endpoint(model, allocator, config)
        artifact = directory / "final-policy.zip"
        model.save(artifact)
        state = model.embedded_constraint_state
        identity = file_record(artifact)
        reopened = ConstrainedPPO.load(artifact, device="cpu")
        attestation = reopened.loaded_constraint_state_attestation()
        if (
            attestation.artifact_sha256 != identity["sha256"]
            or attestation.constraint_state != state
            or attestation.num_timesteps != config.total_timesteps
        ):
            raise ValueError(
                "adaptive final model failed exact reopen/endpoint verification"
            )
        result = {
            "schema": SCHEMA,
            "completed_at": now(),
            "arm": asdict(arm),
            "environment_template": config.to_dict(),
            "total_timesteps": model.num_timesteps,
            "initial_policy": file_record(directory / "initial-policy.pt"),
            "final_policy": identity,
            "checkpoint_contract_digest": ppo.policy_contract(
                config
            ).checkpoint_contract_digest,
            "constraint_state": state,
            "allocator": snapshot,
            "training_reward_units": f"EUR multiplied by {arm.reward_scale:g}; evaluation always raw EUR",
            "registered_evaluation_eligible": False,
        }
        publish_json(directory / "result.json", result)
        return result
    finally:
        vector.close()


def read_training_result(
    directory: Path, config: ControllerNeutralTrainingConfigV1, arm: Arm
) -> dict[str, Any]:
    result = json.loads(read_regular(directory / "result.json"))
    if canonical_bytes(result["environment_template"]) != canonical_bytes(
        config.to_dict()
    ) or result["arm"] != asdict(arm):
        raise ValueError("adaptive checkpoint task or learner differs from frozen cell")
    return cast(dict[str, Any], result)


def load_final(
    directory: Path, config: ControllerNeutralTrainingConfigV1, arm: Arm
) -> Any:
    from .constrained_ppo import ConstrainedPPO

    result = read_training_result(directory, config, arm)
    if file_record(directory / "final-policy.zip") != result["final_policy"]:
        raise ValueError("adaptive policy bytes changed")
    model = ConstrainedPPO.load(directory / "final-policy.zip", device="cpu")
    if (
        model.num_timesteps != config.total_timesteps
        or model.embedded_constraint_state != result["constraint_state"]
    ):
        raise ValueError("adaptive policy state differs from completed cell")
    return model


def load_initial(
    directory: Path, config: ControllerNeutralTrainingConfigV1, arm: Arm
) -> Any:
    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy

    result = read_training_result(directory, config, arm)
    if file_record(directory / "initial-policy.pt") != result["initial_policy"]:
        raise ValueError("adaptive initial policy bytes changed")
    env = ppo.make_env(config)
    try:
        policy = ActorCriticPolicy(
            env.observation_space,
            env.action_space,
            lr_schedule=lambda _: 0.0,
            net_arch={"pi": [64, 64], "vf": [64, 64]},
        )
        weights = torch.load(
            io.BytesIO(read_regular(directory / "initial-policy.pt")),
            map_location="cpu",
            weights_only=True,
        )
        policy.load_state_dict(
            {
                name: value
                for name, value in weights.items()
                if not name.startswith("cost_critic.")
            },
            strict=True,
        )
        return policy
    finally:
        env.close()


def require_plan() -> dict[str, Any]:
    from .capacity_benchmark import _host_identity

    plan = json.loads(read_regular(ROOT / "plan.json"))
    if canonical_bytes(plan["specification"]) != canonical_bytes(specification(ROOT)):
        raise ValueError("adaptive batch specification changed")
    verify_unchanged(
        BetaTrainingRuntimeProvenanceV1.from_dict(plan["runtime"]),
        capture(require_clean=True),
    )
    if (
        _host_identity() != plan["host"]
        or file_record(Path(plan["gate"]["path"])) != plan["gate"]
    ):
        raise ValueError("adaptive batch host or gate changed")
    return cast(dict[str, Any], plan)


def require_successful_training(arm: Arm) -> None:
    receipt = json.loads(read_regular(ROOT / "receipts" / f"train-{arm.name}.json"))
    if (
        receipt["stage"] != "_train"
        or receipt["name"] != f"train-{arm.name}"
        or not isinstance(receipt["codes"], list)
    ):
        raise ValueError("adaptive training receipt identity is inconsistent")
    if receipt["codes"]:
        raise TrainingUnavailable(
            "adaptive policy has no successfully closed training child"
        )
    if receipt["process"]["exit_code"] != 0:
        raise ValueError("adaptive training receipt omits a child failure")


def evaluate_cell(actor: str, seed: int) -> None:
    from .research_evaluation import evaluate_episode

    directory = ROOT / "evaluations" / f"{seed}-{actor}"
    publish_json(
        directory / "attempt.json", {"actor": actor, "seed": seed, "started_at": now()}
    )
    config = template(ROOT)
    policy = None
    model_actor = actor
    if actor == "initial":
        require_successful_training(ARMS[0])
        policy = load_initial(ROOT / "training" / ARMS[0].name, config, ARMS[0])
        model_actor = "PPO"
    elif actor not in BASELINES:
        arm = next(a for a in ARMS if a.name == actor)
        require_successful_training(arm)
        policy = load_final(ROOT / "training" / actor, config, arm)
        model_actor = "PPO"
    episode = evaluate_episode(
        config, actor=model_actor, policy=policy, seed=seed, policy_id=actor
    )
    encoded = gzip.compress(canonical_bytes(episode), compresslevel=6, mtime=0)
    if len(encoded) > ARCHIVE_LIMIT:
        raise ValueError("adaptive evaluation archive exceeded stored-size cap")
    publish_bytes(directory / "episode.json.gz", encoded)
    publish_json(
        directory / "summary.json",
        {
            "actor": actor,
            "seed": seed,
            "summary": episode["summary"],
            "archive": file_record(directory / "episode.json.gz"),
        },
    )


def replay_cell(actor: str, seed: int) -> None:
    from .research_evaluation import verify_episode_replay

    directory = ROOT / "evaluations" / f"{seed}-{actor}"
    publish_json(
        directory / "replay-attempt.json",
        {"actor": actor, "seed": seed, "started_at": now()},
    )
    summary = json.loads(read_regular(directory / "summary.json"))
    archive = directory / "episode.json.gz"
    if file_record(archive) != summary["archive"]:
        raise ValueError("adaptive trajectory changed before replay")
    episode = json.loads(gzip.decompress(read_regular(archive)))
    result = verify_episode_replay(template(ROOT), episode)
    publish_json(directory / "replay.json", result)


def closed_child(
    stage: str, name: str, timeout: int, extra: list[str]
) -> dict[str, Any]:
    require_plan()
    process, codes = capture_process(
        [sys.executable, "-m", MODULE, stage, *extra],
        stdout=ROOT / "logs" / f"{name}.stdout.log",
        stderr=ROOT / "logs" / f"{name}.stderr.log",
        timeout_seconds=timeout,
        log_limit_bytes=LOG_LIMIT,
    )
    receipt = {
        "stage": stage,
        "name": name,
        "process": process,
        "codes": sorted(codes),
        "closed_at": now(),
    }
    publish_json(ROOT / "receipts" / f"{name}.json", receipt)
    if "root_custody_lost" in codes or process["exit_code"] == INTEGRITY_EXIT:
        raise RuntimeError(
            "operator interruption, root custody loss or evidence integrity failure; batch aborted"
        )
    require_plan()
    return receipt


def run(gate: Path, *, authorized: bool) -> dict[str, Any]:
    from .capacity_benchmark import _host_identity

    if not authorized:
        raise ValueError("adaptive batch requires research authorization assertion")
    if ROOT.exists() or ROOT != ROOT.resolve():
        raise ValueError("adaptive root must be absent and canonical; no retry")
    runtime = capture(require_clean=True)
    checked = json.loads(read_regular(gate))
    if (
        checked.get("passed") is not True
        or checked.get("source_commit") != runtime.git_head
    ):
        raise ValueError("adaptive gate must pass at the exact clean source")
    if shutil.disk_usage(ROOT.parent).free < MINIMUM_FREE:
        raise ValueError("adaptive batch needs 10 GiB free")
    plan = {
        "specification": specification(ROOT),
        "runtime": runtime.to_dict(),
        "host": _host_identity(),
        "gate": file_record(gate),
        "started_at": now(),
    }
    mkdir_exclusive(ROOT)
    publish_json(ROOT / "plan.json", plan)
    for name in ("logs", "receipts", "training", "evaluations"):
        mkdir_exclusive(ROOT / name)
    receipts = []
    failure = None
    try:
        for arm in ARMS:
            mkdir_exclusive(ROOT / "training" / arm.name)
            receipts.append(
                closed_child(
                    "_train", f"train-{arm.name}", TRAIN_TIMEOUT, ["--cell", arm.name]
                )
            )
        actors = (*BASELINES, "initial", *(a.name for a in ARMS))
        for seed in DEV_SEEDS:
            for actor in actors:
                name = f"{seed}-{actor}"
                mkdir_exclusive(ROOT / "evaluations" / name)
                receipt = closed_child(
                    "_evaluate",
                    f"evaluate-{name}",
                    EVAL_TIMEOUT,
                    ["--actor", actor, "--seed", str(seed)],
                )
                receipts.append(receipt)
                if not receipt["codes"]:
                    receipts.append(
                        closed_child(
                            "_replay",
                            f"replay-{name}",
                            EVAL_TIMEOUT,
                            ["--actor", actor, "--seed", str(seed)],
                        )
                    )
    except BaseException as reason:
        failure = {"type": type(reason).__name__, "message": str(reason)}
    report = {
        "schema": SCHEMA,
        "status": "aborted" if failure else "closed",
        "closed_at": now(),
        "failure": failure,
        "plan": file_record(ROOT / "plan.json"),
        "receipts": receipts,
        "claim": "Adaptive diagnostic development; no selected policy or confirmation claim.",
        "registered_evaluation_eligible": False,
    }
    publish_json(ROOT / "report.json", report)
    return report


def execute_child(action: Callable[[], Any], *, artifact_phase: bool) -> int:
    """Report a reserved exit for integrity failures without hiding local failures."""

    def failed_integrity(exc: Exception) -> int:
        print(
            json.dumps({"integrity_failure": type(exc).__name__, "message": str(exc)}),
            file=sys.stderr,
        )
        return INTEGRITY_EXIT

    try:
        require_plan()
    except Exception as exc:
        return failed_integrity(exc)
    try:
        action()
    except TrainingUnavailable:
        raise
    except Exception as exc:
        if not artifact_phase:
            raise
        # Expected controller failures are already preserved inside the evaluator.
        # Any unhandled publication/load/replay exception prevents trusted evidence.
        return failed_integrity(exc)
    try:
        require_plan()
    except Exception as exc:
        return failed_integrity(exc)
    return 0


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "_train", "_evaluate", "_replay"))
    parser.add_argument("--assert-authorized", action="store_true")
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--cell", choices=[a.name for a in ARMS])
    parser.add_argument(
        "--actor", choices=[*BASELINES, "initial", *(a.name for a in ARMS)]
    )
    parser.add_argument("--seed", type=int, choices=DEV_SEEDS)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.command == "run":
        if args.gate is None:
            parser.error("run requires --gate")
        result = run(args.gate, authorized=args.assert_authorized)
        print(
            json.dumps(
                {"status": result["status"], "report": str(ROOT / "report.json")}
            )
        )
        return 1 if result["failure"] else 0
    if args.command == "_train":
        if args.cell is None:
            parser.error("_train requires --cell")
        arm = next(a for a in ARMS if a.name == args.cell)
        return execute_child(
            lambda: train_cell(
                ROOT / "training" / arm.name,
                template(ROOT),
                arm,
                seed_base=TRAIN_SEED_BASE,
            ),
            artifact_phase=False,
        )
    else:
        if args.actor is None or args.seed is None:
            parser.error("evaluation/replay requires --actor and --seed")
        return execute_child(
            lambda: (evaluate_cell if args.command == "_evaluate" else replay_cell)(
                args.actor, args.seed
            ),
            artifact_phase=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
