"""Fixed four-rollout scalar PPO screen around the causal OperatingMPC prior."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import io
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, cast

from .. import beta_spaces
from ..beta_config import CyclicMarketTaskConfigV1
from ..beta_env import PlantCyclicMarketEnvV2Beta
from . import ppo
from .research_batch import OptimizerRewardScale, file_record, now, training_callback
from .research_effort_batch import (
    Arm,
    TrainingIntegrityError,
    TrainingUnavailable,
    preflight_checkpoint,
)
from .research_mpc_residual import MPCResidualIntegrityError
from .config import (
    BETA_SCALAR_PPO_ALGORITHM_ID as SCALAR,
    ControllerNeutralTrainingConfigV1,
)
from .custody_io import mkdir_exclusive, publish_bytes, publish_json, read_regular
from .custody_json import canonical_bytes
from .custody_process import capture_process
from .episode_allocator import EpisodeSeedAllocatorV1, EpisodeSeedCustodyError
from .runtime_provenance import (
    BetaTrainingRuntimeProvenanceV1,
    capture,
    verify_unchanged,
)

MODULE = "plant_backend.rl.v2.training.research_mpc_residual_batch"
ROOT = Path("/Volumes/Lexar/PlantAnAgentResearch/adaptive-rl-batch-003")
SCHEMA = "plant-adaptive-rl-mpc-residual-batch-003"
TOTAL_STEPS = 32_768
OPTIMIZER_SEED = 101
TRAIN_SEED_BASE = 9_620_000
TRAIN_SEED_LAST = 9_620_051
DEV_SEEDS = tuple(range(9_700_200, 9_700_208))
BASELINES = ("H", "OperatingMPC", "M24", "E-beta")
LOG_LIMIT = 64 * 1024**2
ARCHIVE_LIMIT = 256 * 1024**2
TRAIN_TIMEOUT = 6 * 3600
EVAL_TIMEOUT = 600
MINIMUM_FREE = 10 * 1024**3
INTEGRITY_EXIT = 86


ARMS = (Arm("A-mpc-residual-scalar", SCALAR, 0.001, 0.2, True),)
INITIAL_SOURCES = {"initial-mpc-residual": ARMS[0]}

ACTORS = (*BASELINES, *INITIAL_SOURCES, *(a.name for a in ARMS))


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
        "training_seed_last": TRAIN_SEED_LAST,
        "training_rollouts": 4,
        "minibatch_optimizer_steps": 1280,
        "development_seeds": list(DEV_SEEDS),
        "baselines": list(BASELINES),
        "initial_baseline_sources": {
            name: arm.name for name, arm in INITIAL_SOURCES.items()
        },
        "actors": list(ACTORS),
        "policy_identities": {
            arm.name: policy_identity(template(root), arm) for arm in ARMS
        },
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


def control_environment(config: ControllerNeutralTrainingConfigV1, arm: Arm) -> Any:
    """Construct the actual policy interface without consuming a reset."""
    from .research_mpc_residual import ResearchMPCResidualWrapper

    env = ppo.make_env(config)
    if not isinstance(env, PlantCyclicMarketEnvV2Beta):
        env.close()
        raise ValueError("adaptive effort screen requires the beta task")
    if not arm.residual or arm.algorithm != SCALAR:
        env.close()
        raise ValueError("MPC residual screen requires its scalar residual interface")
    try:
        return ResearchMPCResidualWrapper(env)
    except BaseException:
        env.close()
        raise


def policy_identity(
    config: ControllerNeutralTrainingConfigV1, arm: Arm
) -> dict[str, Any]:
    from .constrained_ppo import ConstrainedPPO

    env = control_environment(config, arm)
    try:
        checkpoint = env.checkpoint_contract
        return {
            "schema": "plant-research-mpc-residual-policy-identity-v1",
            "representation": "residual-mpc",
            "checkpoint_contract": checkpoint,
            "checkpoint_contract_digest": env.checkpoint_contract_digest,
            "interface_contract": checkpoint["interface_contract"],
            "interface_contract_digest": checkpoint["interface_contract_digest"],
            "observation_size": int(env.observation_space.shape[0]),
            "action_size": int(env.action_space.shape[0]),
            "observation_space": ConstrainedPPO._box_space_record(
                "PPO observation space", env.observation_space
            ),
            "action_space": ConstrainedPPO._box_space_record(
                "PPO action space", env.action_space
            ),
            "initialization": {
                "optimizer_seed": config.seed,
                "actor_output_head_weight": "exactly zero",
                "actor_output_head_bias": "exactly zero",
                "initial_gaussian_std": arm.initial_std,
                "deterministic_initial_command": "exact cached OperatingMPC",
                "matching_scope": "single fresh 139-input MPC residual policy",
            },
            "registered_evaluation_eligible": False,
        }
    finally:
        env.close()


def verify_model_spaces(
    model: Any, config: ControllerNeutralTrainingConfigV1, arm: Arm
) -> None:
    from ...checkpoints import verify_policy_model_spaces

    env = control_environment(config, arm)
    try:
        verify_policy_model_spaces(
            model,
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
    finally:
        env.close()


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
            Monitor(
                allocator.wrap(
                    control_environment(config, arm), environment_index=index
                )
            ),
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
        with torch.no_grad():
            model.policy.log_std.fill_(math.log(arm.initial_std))
            action_head = cast(torch.nn.Linear, model.policy.action_net)
            action_head.weight.zero_()
            action_head.bias.zero_()
        setattr(model, "research_identity", policy_identity(config, arm))
        verify_model_spaces(model, config, arm)
        return model, allocator, vector
    except BaseException:
        vector.close()
        raise


def allocator_prefix(allocator: EpisodeSeedAllocatorV1) -> dict[str, Any]:
    """Retain observed custody state without asserting a successful endpoint."""
    return {
        "schema": "plant-research-mpc-residual-allocator-prefix-v1",
        "allocator_identity": allocator.identity,
        "ledger": [entry.to_dict() for entry in allocator.ledger],
        "cursors": [cursor.to_dict() for cursor in allocator.cursors],
        "decisions_consumed_by_environment": list(
            allocator.decisions_consumed_by_environment
        ),
        "successful_endpoint": False,
    }


def cell_callback(directory: Path, allocator: EpisodeSeedAllocatorV1) -> Any:
    """Reuse immutable optimizer diagnostics and retain reached allocator prefixes."""
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    class CustodyRecorder(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.written_resets = 0
            self.rollouts = 0

        def _on_step(self) -> bool:
            for entry in allocator.ledger[self.written_resets :]:
                publish_json(
                    directory / "allocator" / f"reset-{entry.ledger_index:03d}.json",
                    entry.to_dict(),
                )
                self.written_resets += 1
            return True

        def _on_rollout_end(self) -> None:
            self.rollouts += 1
            publish_json(
                directory / "allocator" / f"rollout-{self.rollouts:03d}.json",
                allocator_prefix(allocator),
            )

    class EvidenceCallbacks(CallbackList):
        def checked(self, action: Callable[[], Any]) -> Any:
            try:
                return action()
            except Exception as exc:
                raise TrainingIntegrityError(
                    "adaptive training evidence callback failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        def _on_step(self) -> bool:
            return bool(self.checked(super()._on_step))

        def _on_rollout_start(self) -> None:
            self.checked(super()._on_rollout_start)

        def _on_rollout_end(self) -> None:
            self.checked(super()._on_rollout_end)

        def _on_training_end(self) -> None:
            self.checked(super()._on_training_end)

    return EvidenceCallbacks(
        [training_callback(directory / "diagnostics"), CustodyRecorder()]
    )


def retain_training_failure(
    directory: Path,
    model: Any,
    allocator: EpisodeSeedAllocatorV1,
    reason: BaseException,
) -> None:
    """Keep failures descriptive; never authorize loading an unfinished policy."""
    try:
        publish_json(
            directory / "training-failure.json",
            {
                "schema": SCHEMA,
                "failed_at": now(),
                "failure": {"type": type(reason).__name__, "message": str(reason)},
                "num_timesteps": model.num_timesteps,
                "allocator_prefix": allocator_prefix(allocator),
                "rollout_ledgers": [
                    ledger.to_dict() for ledger in model.rollout_ledgers
                ],
                "successful_endpoint": False,
                "registered_evaluation_eligible": False,
            },
        )
    except Exception as exc:
        raise TrainingIntegrityError(
            "adaptive failed-training evidence publication failed: "
            f"{type(exc).__name__}: {exc}; original {type(reason).__name__}: {reason}"
        ) from exc


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

    try:
        publish_json(
            directory / "attempt.json",
            {
                "started_at": now(),
                "arm": asdict(arm),
                "template": config.to_dict(),
                "seed_base": seed_base,
                "policy_identity": policy_identity(config, arm),
            },
        )
        mkdir_exclusive(directory / "diagnostics")
        mkdir_exclusive(directory / "allocator")
        model, allocator, vector = make_learner(config, arm, seed_base=seed_base)
    except Exception as exc:
        raise TrainingIntegrityError(
            "adaptive pre-learning evidence or interface admission failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        try:
            buffer = io.BytesIO()
            torch.save(model.policy.state_dict(), buffer)
            publish_bytes(directory / "initial-policy.pt", buffer.getvalue())
        except Exception as exc:
            raise TrainingIntegrityError(
                "adaptive initial-policy publication failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=cell_callback(directory, allocator),
        )
        try:
            snapshot = validate_endpoint(model, allocator, config)
            artifact = directory / "final-policy.zip"
            model.save(artifact)
            state = model.embedded_constraint_state
            identity = file_record(artifact)
            preflight_checkpoint(artifact, model.research_identity, state)
            reopened = ConstrainedPPO.load(artifact, device="cpu")
            verify_model_spaces(reopened, config, arm)
            attestation = reopened.loaded_constraint_state_attestation()
            if (
                attestation.artifact_sha256 != identity["sha256"]
                or attestation.constraint_state != state
                or attestation.num_timesteps != config.total_timesteps
                or canonical_bytes(getattr(reopened, "research_identity", None))
                != canonical_bytes(model.research_identity)
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
                "policy_identity": model.research_identity,
                "checkpoint_contract_digest": model.research_identity[
                    "checkpoint_contract_digest"
                ],
                "constraint_state": state,
                "allocator": snapshot,
                "training_reward_units": f"EUR multiplied by {arm.reward_scale:g}; evaluation always raw EUR",
                "registered_evaluation_eligible": False,
            }
            publish_json(directory / "result.json", result)
            return result
        except Exception as exc:
            raise TrainingIntegrityError(
                "adaptive post-learning endpoint or publication failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    except BaseException as exc:
        retain_training_failure(directory, model, allocator, exc)
        raise
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
    identity = policy_identity(config, arm)
    if result.get("schema") != SCHEMA or canonical_bytes(
        result.get("policy_identity")
    ) != canonical_bytes(identity):
        raise ValueError(
            "adaptive policy interface or initialization differs from frozen cell"
        )
    if (
        result.get("checkpoint_contract_digest")
        != identity["checkpoint_contract_digest"]
    ):
        raise ValueError("adaptive checkpoint identity digest differs")
    if result.get("total_timesteps") != config.total_timesteps:
        raise ValueError("adaptive completed training budget differs")
    return cast(dict[str, Any], result)


def load_final(
    directory: Path, config: ControllerNeutralTrainingConfigV1, arm: Arm
) -> Any:
    from .constrained_ppo import ConstrainedPPO

    result = read_training_result(directory, config, arm)
    if file_record(directory / "final-policy.zip") != result["final_policy"]:
        raise ValueError("adaptive policy bytes changed")
    preflight_checkpoint(
        directory / "final-policy.zip",
        result["policy_identity"],
        result["constraint_state"],
    )
    model = ConstrainedPPO.load(directory / "final-policy.zip", device="cpu")
    verify_model_spaces(model, config, arm)
    if (
        model.algorithm_id != arm.algorithm
        or model.num_timesteps != config.total_timesteps
        or model.embedded_constraint_state != result["constraint_state"]
        or canonical_bytes(getattr(model, "research_identity", None))
        != canonical_bytes(result["policy_identity"])
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
    from .constrained_ppo import LearnerRNGStateV1

    caller_rng = LearnerRNGStateV1.capture()
    env = control_environment(config, arm)
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
        action_head = cast(torch.nn.Linear, policy.action_net)
        if not torch.equal(
            action_head.weight, torch.zeros_like(action_head.weight)
        ) or not torch.equal(action_head.bias, torch.zeros_like(action_head.bias)):
            raise ValueError("adaptive initial actor output head is not exactly zero")
        if not torch.equal(
            policy.log_std, torch.full_like(policy.log_std, math.log(arm.initial_std))
        ):
            raise ValueError("adaptive initial exploration differs")
        verify_model_spaces(policy, config, arm)
        return policy
    finally:
        env.close()
        caller_rng.restore()


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
    if (
        receipt["process"]["exit_code"] != 0
        or receipt["process"].get("started") is not True
        or not receipt.get("closed_at")
        or receipt["process"].get("command")
        != [sys.executable, "-m", MODULE, "_train", "--cell", arm.name]
    ):
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
    arm = None
    if actor in INITIAL_SOURCES:
        arm = INITIAL_SOURCES[actor]
        require_successful_training(arm)
        policy = load_initial(ROOT / "training" / arm.name, config, arm)
        model_actor = "PPO"
    elif actor not in BASELINES:
        arm = next(a for a in ARMS if a.name == actor)
        require_successful_training(arm)
        policy = load_final(ROOT / "training" / actor, config, arm)
        model_actor = "PPO"
    if arm is not None and arm.residual:
        from .research_mpc_residual_evaluation import evaluate_residual_episode

        episode = evaluate_residual_episode(
            config, policy=policy, seed=seed, policy_id=actor
        )
    else:
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
    from .research_mpc_residual_evaluation import ACTOR as residual_actor

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
    if (
        summary.get("actor") != actor
        or canonical_bytes(summary.get("seed")) != canonical_bytes(seed)
        or canonical_bytes(episode.get("seed")) != canonical_bytes(seed)
        or canonical_bytes(summary.get("summary"))
        != canonical_bytes(episode.get("summary"))
        or episode["summary"].get("policy_id") != actor
        or episode["summary"].get("actor")
        != (actor if actor in BASELINES else residual_actor)
    ):
        raise ValueError("adaptive replay slot identity or summary differs")
    if actor == "initial-mpc-residual" or any(
        a.name == actor and a.residual for a in ARMS
    ):
        from .research_mpc_residual_evaluation import verify_residual_episode_replay

        result = verify_residual_episode_replay(template(ROOT), episode)
    else:
        result = verify_episode_replay(template(ROOT), episode)
    publish_json(directory / "replay.json", result)


def closed_child(
    stage: str, name: str, timeout: int, extra: list[str]
) -> dict[str, Any]:
    require_plan()
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE:
        raise RuntimeError("adaptive batch needs 10 GiB free before child admission")
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
        for seed in DEV_SEEDS:
            for actor in ACTORS:
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
    except (
        TrainingIntegrityError,
        EpisodeSeedCustodyError,
        MPCResidualIntegrityError,
    ) as exc:
        return failed_integrity(exc)
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
    parser.add_argument("--actor", choices=ACTORS)
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
