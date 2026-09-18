"""Standalone physical evidence admission, initially restricted to synthetic arms.

No production entry point is provided here. Fixture scope is checked before the
legacy semantic verifier can regenerate an exogenous manifest. The old four-arm
format keeps its historical eligibility semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from .. import beta_controller_benchmark as benchmark
from .. import beta_controller_benchmark_verifier as verifier
from .. import beta_spaces
from ..mpc_schema import ServiceMPCSetpointSettingsV1, domain_sha256
from ..training import custody_io as io
from ..training.config import (
    BETA_FIXTURE_EPISODE_SEED_MAX,
    BETA_FIXTURE_EPISODE_SEED_MIN,
    BETA_REGISTERED_RUN_KIND,
    ControllerNeutralTrainingConfigV1,
)
from ..training.custody_authorization import repository_root
from ..training.custody_json import (
    authenticate,
    canonical_bytes,
    domain_digest,
    require_envelope,
)
from .contract import PREFIX, PROTOCOL_ID, PROTOCOL_SHA256, analysis_contract
from .operational import episode_scorecard

SCHEMA = PREFIX + "synthetic-arm-evidence-v1"
FILENAME = "synthetic-arm.json"
CLAIM_BOUNDARY = "synthetic physical evidence only; no registered policy admission or study authorization"
FIELDS = frozenset(
    {
        "schema",
        "protocol",
        "analysis_contract_sha256",
        "stage",
        "seed",
        "actor",
        "training_task",
        "checkpoint",
        "source_provenance",
        "service_mpc_settings",
        "manifest_identity",
        "audit_manifest",
        "run",
        "claim_boundary",
        "result_claim",
        "sha256",
    }
)


def _fixture_seed(seed: object) -> int:
    if type(seed) is not int or not (
        BETA_FIXTURE_EPISODE_SEED_MIN <= seed <= BETA_FIXTURE_EPISODE_SEED_MAX
    ):
        raise ValueError("synthetic arm seed must be in 9,000,000..9,999,999")
    return seed


def _fixture_config(raw: object) -> ControllerNeutralTrainingConfigV1:
    payload = verifier._mapping("synthetic training task", raw)
    # Close the scope before config interpretation or manifest reconstruction.
    declared = payload.get("beta_rl")
    if (
        isinstance(declared, Mapping)
        and declared.get("run_kind") == BETA_REGISTERED_RUN_KIND
    ):
        raise ValueError("synthetic evidence cannot admit a registered configuration")
    config = ControllerNeutralTrainingConfigV1.from_dict(payload)
    verifier._exact("synthetic training task", payload, config.to_dict())
    if config.environment_id != beta_spaces.ENVIRONMENT_ID or config.beta_task is None:
        raise ValueError("synthetic arm requires the cyclic-market beta environment")
    return config


def _actor(arm: str, config: ControllerNeutralTrainingConfigV1) -> dict[str, Any]:
    if arm not in benchmark.BETA_CONTROLLER_ARM_ORDER:
        raise ValueError("unknown study arm")
    policy = arm == benchmark.PPO_ARM
    return {
        "arm": arm,
        "algorithm": config.beta_rl.algorithm if policy and config.beta_rl else None,
        "optimizer_seed": config.seed if policy else None,
    }


def _settings(
    config: ControllerNeutralTrainingConfigV1,
) -> ServiceMPCSetpointSettingsV1:
    return ServiceMPCSetpointSettingsV1(
        task_horizon_h=config.episode.horizon_h,
        planning_horizon_h=min(24.0, config.episode.horizon_h),
        interval_s=config.episode.decision_interval_s,
    )


def _standalone_result(raw: Mapping[str, Any]) -> dict[str, Any]:
    run = deepcopy(dict(raw))
    run["format"] = verifier.STUDY_ARM_RESULT_FORMAT
    del run["value_comparison_eligible"]
    run["exclusion_reasons"] = [
        reason
        for reason in run["exclusion_reasons"]
        if reason != "paired_cell_ineligible"
    ]
    body = cast(dict[str, Any], benchmark._timing_free(run))
    body.pop("deterministic_evidence_sha256", None)
    run["deterministic_evidence_sha256"] = domain_sha256(
        verifier.STUDY_ARM_RESULT_FORMAT, body
    )
    return run


def synthetic_arms_from_benchmark(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert a fully verified fixture benchmark; never convert protected seeds."""
    report = deepcopy(dict(raw))
    seeds = verifier._array("synthetic benchmark seeds", report.get("seeds"))
    if not seeds:
        raise ValueError("synthetic benchmark seeds are empty")
    for seed in seeds:
        _fixture_seed(seed)
    config = _fixture_config(report.get("training_task"))
    verifier._exact(
        "canonical study M24 settings",
        report.get("service_mpc_settings"),
        _settings(config).to_dict(),
    )
    verifier.verify_beta_controller_benchmark(report)
    result = []
    for cell in report["cells"]:
        for arm in benchmark.BETA_CONTROLLER_ARM_ORDER:
            result.append(
                authenticate(
                    {
                        "schema": SCHEMA,
                        "protocol": {"id": PROTOCOL_ID, "raw_sha256": PROTOCOL_SHA256},
                        "analysis_contract_sha256": analysis_contract()["sha256"],
                        "stage": "fixture",
                        "seed": cell["seed"],
                        "actor": _actor(arm, config),
                        "training_task": deepcopy(report["training_task"]),
                        "checkpoint": (
                            deepcopy(report["checkpoint"])
                            if arm == benchmark.PPO_ARM
                            else None
                        ),
                        "source_provenance": deepcopy(report["source_provenance"]),
                        "service_mpc_settings": deepcopy(
                            report["service_mpc_settings"]
                        ),
                        "manifest_identity": deepcopy(cell["manifest_identity"]),
                        "audit_manifest": deepcopy(cell["audit_manifest"]),
                        "run": _standalone_result(cell["runs"][arm]),
                        "claim_boundary": CLAIM_BOUNDARY,
                        "result_claim": None,
                    }
                )
            )
    return result


def require_synthetic_path(path: Path | str, *, filename: str) -> Path:
    """Reject path aliases and in-repository evidence before reading any file."""
    target = Path(path)
    if (
        not target.is_absolute()
        or str(target) != str(path)
        or target != target.resolve()
        or target.name != filename
        or target.is_relative_to(repository_root())
    ):
        raise ValueError(
            "synthetic evidence needs its exact filename at a canonical external path"
        )
    io.require_directory(target.parent)
    return target


def file_stamp(path: Path) -> tuple[int, ...]:
    status = path.lstat()
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _public_issue(boundary: Mapping[str, Any]) -> dict[str, Any]:
    observation = boundary["typed_observation"]
    plant = observation["plant"]
    return {
        "market": observation["market"],
        "weather": plant["forecast"],
        "delivery_schedule": {
            key: plant["delivery"][key]
            for key in (
                "steps_to_next",
                "delivery_period_steps",
                "next_requested_kg",
                "cumulative_requested_kg",
            )
        },
    }


def _verify_fixture_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    if envelope["stage"] != "fixture":
        raise ValueError("synthetic arm stage differs")
    seed = _fixture_seed(envelope["seed"])
    config = _fixture_config(envelope["training_task"])
    verifier._exact(
        "study protocol",
        envelope["protocol"],
        {"id": PROTOCOL_ID, "raw_sha256": PROTOCOL_SHA256},
    )
    if (
        envelope["analysis_contract_sha256"] != analysis_contract()["sha256"]
        or envelope["claim_boundary"] != CLAIM_BOUNDARY
        or envelope["result_claim"] is not None
    ):
        raise ValueError("synthetic arm claim/contract differs")
    return _verify_arm_body(envelope, config=config, seed=seed)


def _verify_arm_body(
    envelope: dict[str, Any], *, config: ControllerNeutralTrainingConfigV1, seed: int
) -> dict[str, Any]:
    """Shared physics reconstruction after the caller admits stage/task/seed scope."""
    actor = verifier._mapping("study actor", envelope["actor"])
    arm = verifier._text("study arm", actor.get("arm"))
    verifier._exact("study actor", actor, _actor(arm, config))
    settings = _settings(config)
    verifier._exact(
        "canonical study M24 settings",
        envelope["service_mpc_settings"],
        settings.to_dict(),
    )
    source = verifier._verify_source_provenance(envelope["source_provenance"])
    checkpoint: dict[str, Any] = {}
    if arm == benchmark.PPO_ARM:
        checkpoint, _, _ = verifier._verify_checkpoint_and_policy_descriptor(
            envelope["checkpoint"], config=config
        )
        if checkpoint["plant_sim_version"] != source["runtime"]["plant_sim"]["version"]:
            raise ValueError("checkpoint and source simulator versions differ")
    elif envelope["checkpoint"] is not None:
        raise ValueError("baseline evidence must not carry a policy checkpoint")
    raw_run = verifier._mapping("study arm result", envelope["run"])
    highs_versions: set[str] = set()
    boundaries: list[dict[str, Any]] = []
    identity = None
    manifest = None
    if raw_run.get("status") == "not_run_preflight_failed":
        if (
            envelope["manifest_identity"] is not None
            or envelope["audit_manifest"] is not None
        ):
            raise ValueError("preflight failure retains a manifest")
        run = verifier._verify_preflight_arm(raw_run, arm=arm, standalone=True)
        if run["failure"] is None:
            raise ValueError("preflight arm failure is missing")
    else:
        identity = verifier._verify_manifest_identity(
            envelope["manifest_identity"], name="study manifest identity", config=config
        )
        if (
            checkpoint
            and identity["checkpoint_contract_sha256"]
            != checkpoint["checkpoint_contract_sha256"]
        ):
            raise ValueError("study task and checkpoint contract identities differ")
        manifest = verifier._verify_audit_manifest(
            envelope["audit_manifest"], identity=identity, config=config, seed=seed
        )
        run, boundaries = verifier._verify_normal_arm(
            raw_run,
            arm=arm,
            config=config,
            settings=settings,
            manifest_identity=identity,
            manifest=manifest,
            checkpoint=checkpoint,
            paired_eligible=None,
            expected_scipy_version=str(source["runtime"]["scipy"]),
            highs_versions=highs_versions,
            standalone=True,
        )
    if len(highs_versions) > 1:
        raise ValueError("study arm solver stages report inconsistent HiGHS versions")
    return {
        "verification": "synthetic_arm_reconstructed",
        "source_evidence_verified": True,
        "registered_checkpoint_verified": False,
        "result_claim": None,
        "stage": envelope["stage"],
        "seed": seed,
        "actor": actor,
        "source_provenance": source,
        "manifest_identity": identity,
        "envelope_sha256": envelope["sha256"],
        "arm_evidence_sha256": run["deterministic_evidence_sha256"],
        "highs_versions": sorted(highs_versions),
        "initial_public_state_sha256": (
            domain_digest(PREFIX + "initial-public-state-v1", boundaries[0])
            if boundaries
            else None
        ),
        "public_issue_sha256": [
            domain_digest(PREFIX + "public-issue-v1", _public_issue(item))
            for item in boundaries
        ],
        "classification": {
            key: deepcopy(run.get(key))
            for key in (
                "status",
                "failure",
                "execution_complete",
                "cycle_feasible",
                "hard_safety_passed",
                "value_comparison_candidate",
                "exclusion_reasons",
                "expected_decisions",
                "expected_cycle_count",
                "completed_decisions",
                "physical_steps",
                "hard_safety_metrics",
                "cycle_settlements",
                "cumulative_cycle_constraint_cost",
            )
        },
        "economic_metrics": deepcopy(run["economic_metrics"]),
        "physical_metrics": deepcopy(run["physical_metrics"]),
        "operational": episode_scorecard(
            seed=seed, actor=actor, config=config, manifest=manifest, run=run
        ),
    }


def verify_synthetic_arm_file(path: Path | str) -> dict[str, Any]:
    """Reopen a sealed fixture and reconstruct its physical evidence, without a run."""
    target = require_synthetic_path(path, filename=FILENAME)
    before = file_stamp(target)
    raw = io.read_regular(target)
    envelope = require_envelope(raw, schema=SCHEMA, fields=FIELDS)
    result = _verify_fixture_envelope(envelope)
    record = io.file_record(target)
    if io.read_regular(target) != raw or file_stamp(target) != before:
        raise ValueError("synthetic arm changed during verification")
    return {**result, "file": record}
