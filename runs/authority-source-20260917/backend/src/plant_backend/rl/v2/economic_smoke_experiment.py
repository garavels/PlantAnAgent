"""One-shot runner for the preregistered economic task-expression smoke cell."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from plant_sim import PlantConfig

from ...controllers.base import Constraints
from .config import EpisodeConfigV2
from .economic_env import PlantEconomicSetpointEnvV2
from .economics import EconomicAssumptionsV1
from .evaluation import collect_provenance
from .mpc_execution import public_observation_sha256
from .mpc_experiment import find_repository_root, validate_artifact_destination
from .mpc_schema import (
    SERVICE_MPC_SETPOINT_CONTROLLER_ID,
    ServiceMPCSetpointSettingsV1,
    canonical_json,
    domain_sha256,
)
from .mpc_setpoint_controller import ServiceMPCSetpointControllerV1
from .mpc_setpoint_run import (
    build_mpc_setpoint_run_evidence,
    verify_mpc_setpoint_run_digests,
)
from .reward import RewardWeightsV2
from .setpoint_controllers import (
    LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID,
    LegacyHeuristicSetpointControllerV1,
)

ECONOMIC_SMOKE_FORMAT = "economic-task-mechanism-smoke-v1"
ECONOMIC_SMOKE_DOMAIN = ECONOMIC_SMOKE_FORMAT
ECONOMIC_SMOKE_PROTOCOL_PATH = "docs/research/ECONOMIC_MECHANISM_SMOKE_PROTOCOL.md"
ECONOMIC_SMOKE_PROTOCOL_SHA256 = (
    "41c0ce9a1d49bcbfca34d88c0fb20b7fc7eb0ab91d15ed405aded512135cd795"
)
ECONOMIC_SMOKE_LOCK_SHA256 = (
    "080d96275581ebbec1b2c27b354ea54e3e66419bfaadb034896da1fd22b228dd"
)
ECONOMIC_SMOKE_SEED = 0
ECONOMIC_SMOKE_HORIZON_H = 6.0
ECONOMIC_SMOKE_DELIVERY_RATE_KG_H = 35.0
ECONOMIC_SMOKE_EXPECTED_DECISIONS = 24
ECONOMIC_SMOKE_EXPECTED_PHYSICAL_STEPS = 360
ECONOMIC_SMOKE_EXPECTED_MANIFEST_SHA256 = (
    "e54b1ddbfba2e22aa38761b137105b74063f18192fe0fa55f6760d5c5f6c0272"
)
ECONOMIC_SMOKE_EXPECTED_PHYSICAL_EPISODE_SHA256 = (
    "ea124b10b48a42231e1ae45f6647427630bd23f89e0df29a301e04077b56b5d7"
)
ECONOMIC_SMOKE_EXPECTED_EPISODE_SHA256 = (
    "b81b392a36ea4cb08c1cc3ca5e4942b23d4557959dabfa3b7f379cff3a52c0b7"
)
ECONOMIC_SMOKE_EXPECTED_CHECKPOINT_SHA256 = (
    "c94450ff315736c2a8db045152b3ca3bfdee2d6c3a47bfd97b2ee4f09587be62"
)
ECONOMIC_SMOKE_EXPECTED_ASSUMPTIONS_SHA256 = (
    "dce6ba3082e0e2a16c1e341589afb25d8f1d0b2da9124522499dafd705fe6ee2"
)
ECONOMIC_SMOKE_MPC_SETTINGS_SHA256 = (
    "7f89b35d23ee9313f9a8723050a2dcbc94f9303e18e96cab81e86fa9f2a0261e"
)

ECONOMIC_SMOKE_CONTRACT = {
    "format": ECONOMIC_SMOKE_FORMAT,
    "protocol": {
        "path": ECONOMIC_SMOKE_PROTOCOL_PATH,
        "raw_file_sha256": ECONOMIC_SMOKE_PROTOCOL_SHA256,
    },
    "dependency_lock": {"path": "uv.lock", "sha256": ECONOMIC_SMOKE_LOCK_SHA256},
    "cell": {
        "seed": ECONOMIC_SMOKE_SEED,
        "horizon_h": ECONOMIC_SMOKE_HORIZON_H,
        "delivery_rate_kg_h": ECONOMIC_SMOKE_DELIVERY_RATE_KG_H,
        "decision_interval_s": 900.0,
        "expected_decisions_per_arm": ECONOMIC_SMOKE_EXPECTED_DECISIONS,
        "expected_physical_steps_per_arm": ECONOMIC_SMOKE_EXPECTED_PHYSICAL_STEPS,
        "manifest_sha256": ECONOMIC_SMOKE_EXPECTED_MANIFEST_SHA256,
        "physical_exogenous_episode_sha256": (
            ECONOMIC_SMOKE_EXPECTED_PHYSICAL_EPISODE_SHA256
        ),
        "economic_episode_sha256": ECONOMIC_SMOKE_EXPECTED_EPISODE_SHA256,
        "checkpoint_contract_sha256": ECONOMIC_SMOKE_EXPECTED_CHECKPOINT_SHA256,
        "economic_assumptions_sha256": (ECONOMIC_SMOKE_EXPECTED_ASSUMPTIONS_SHA256),
    },
    "arms": {
        "H": {
            "controller_id": LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID,
            "uses_tariff": False,
        },
        "M": {
            "controller_id": SERVICE_MPC_SETPOINT_CONTROLLER_ID,
            "settings_sha256": ECONOMIC_SMOKE_MPC_SETTINGS_SHA256,
            "uses_tariff": False,
            "economic_mpc": False,
        },
    },
    "gates": {
        "E1": "24 decisions and 360 physical steps per arm; normal termination",
        "E2": "typed execution, zero shutdown/out-of-range/forced-stop events",
        "E3": "delivered and unmet methane equal within 1e-6 kg",
        "E4": "absolute value gap >1 EUR and at least one material raw driver",
        "E5": "interval, component, liability, and identity accounting closure",
        "E6": "one-cell descriptive interpretation only",
    },
    "execution_order": ["H", "M"],
    "retry_policy": "one execution per committed revision; retain failure",
}
ECONOMIC_SMOKE_CONTRACT_SHA256 = domain_sha256(
    "economic-task-mechanism-smoke-contract-v1",
    ECONOMIC_SMOKE_CONTRACT,
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def _close(left: float, right: float, *, tolerance: float = 1e-8) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def fixed_episode() -> EpisodeConfigV2:
    return EpisodeConfigV2(
        horizon_h=ECONOMIC_SMOKE_HORIZON_H,
        delivery_rate_kg_h=ECONOMIC_SMOKE_DELIVERY_RATE_KG_H,
    )


def fixed_mpc_settings() -> ServiceMPCSetpointSettingsV1:
    settings = ServiceMPCSetpointSettingsV1(
        task_horizon_h=ECONOMIC_SMOKE_HORIZON_H,
        planning_horizon_h=ECONOMIC_SMOKE_HORIZON_H,
    )
    if settings.sha256 != ECONOMIC_SMOKE_MPC_SETTINGS_SHA256:
        raise RuntimeError("fixed economic-smoke MPC settings identity drifted")
    return settings


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, str]:
    assumptions = dict(
        _mapping("economic assumptions", manifest["economic_assumptions"])
    )
    return {
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "physical_exogenous_episode_sha256": str(manifest["exogenous_episode_sha256"]),
        "economic_episode_sha256": str(manifest["economic_episode_sha256"]),
        "checkpoint_contract_sha256": str(
            _mapping("checkpoint", manifest["checkpoint_contract"])[
                "checkpoint_contract_digest"
            ]
        ),
        "economic_assumptions_sha256": hashlib.sha256(
            canonical_json(assumptions).encode("utf-8")
        ).hexdigest(),
    }


def _expected_identity() -> dict[str, str]:
    return {
        "manifest_sha256": ECONOMIC_SMOKE_EXPECTED_MANIFEST_SHA256,
        "physical_exogenous_episode_sha256": (
            ECONOMIC_SMOKE_EXPECTED_PHYSICAL_EPISODE_SHA256
        ),
        "economic_episode_sha256": ECONOMIC_SMOKE_EXPECTED_EPISODE_SHA256,
        "checkpoint_contract_sha256": ECONOMIC_SMOKE_EXPECTED_CHECKPOINT_SHA256,
        "economic_assumptions_sha256": ECONOMIC_SMOKE_EXPECTED_ASSUMPTIONS_SHA256,
    }


def _new_environment(
    *,
    episode: EpisodeConfigV2,
    plant: PlantConfig,
    constraints: Constraints,
    weights: RewardWeightsV2,
    assumptions: EconomicAssumptionsV1,
) -> PlantEconomicSetpointEnvV2:
    return PlantEconomicSetpointEnvV2(
        config=plant,
        episode=episode,
        constraints=constraints,
        weights=weights,
        economic_assumptions=assumptions,
    )


def _reset_environment(
    environment: PlantEconomicSetpointEnvV2,
    *,
    seed: int,
) -> dict[str, str]:
    environment.reset(seed=seed)
    identity = _manifest_identity(environment.audit_manifest())
    if (
        environment.economic_assumptions.sha256
        != identity["economic_assumptions_sha256"]
    ):
        raise RuntimeError("economic assumptions identity drifted")
    return identity


def _prepared_environment_identity(
    environment: PlantEconomicSetpointEnvV2,
) -> dict[str, str]:
    if environment.decisions_taken != 0 or environment.physics_steps_taken != 0:
        raise ValueError("prepared economic-smoke environment has already moved")
    identity = _manifest_identity(environment.audit_manifest())
    if (
        environment.economic_assumptions.sha256
        != identity["economic_assumptions_sha256"]
    ):
        raise RuntimeError("economic assumptions identity drifted")
    return identity


def _validate_preflight_identities(
    heuristic: Mapping[str, str],
    mpc: Mapping[str, str],
) -> None:
    expected = _expected_identity()
    if dict(heuristic) != expected:
        raise RuntimeError("heuristic reset identity differs from the frozen cell")
    if dict(mpc) != expected:
        raise RuntimeError("MPC reset identity differs from the frozen cell")
    if dict(heuristic) != dict(mpc):
        raise RuntimeError("paired economic-smoke reset identities differ")


def _interval_totals() -> dict[str, float]:
    return {
        "delivered_kg": 0.0,
        "unmet_kg": 0.0,
        "accepted_solar_kwh": 0.0,
        "battery_throughput_kwh": 0.0,
        "electrolyser_equivalent_operating_h": 0.0,
    }


def _record_interval(
    totals: dict[str, float],
    info: Mapping[str, Any],
) -> None:
    interval = _mapping("economic interval", info["economic_interval"])
    for name in totals:
        totals[name] += _finite(f"economic interval {name}", interval[name])


def _run_payload(
    *,
    arm: str,
    status: str,
    environment: PlantEconomicSetpointEnvV2,
    manifest_identity: Mapping[str, str],
    reward_trace: list[float],
    interval_totals: Mapping[str, float],
    execution_count: int,
    application_count: int,
    terminated: bool,
    truncated: bool,
    elapsed_ns: int,
    controller: Mapping[str, Any],
    evidence: Mapping[str, Any],
    failure: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "status": status,
        "controller": dict(controller),
        "manifest_identity": dict(manifest_identity),
        "completed_decisions": environment.decisions_taken,
        "physical_steps": environment.physics_steps_taken,
        "terminated": terminated,
        "truncated": truncated,
        "reward_trace_eur": list(reward_trace),
        "reward_trace_sum_eur": float(math.fsum(reward_trace)),
        "interval_totals": dict(interval_totals),
        "execution_count": execution_count,
        "application_count": application_count,
        "physical_metrics": environment.metrics.to_dict(),
        "economic_metrics": environment.economic_metrics.to_dict(),
        "evidence": dict(evidence),
        "failure": None if failure is None else dict(failure),
        "elapsed_ns": elapsed_ns,
    }


def execute_heuristic_arm(
    *,
    episode: EpisodeConfigV2,
    plant: PlantConfig,
    constraints: Constraints,
    weights: RewardWeightsV2,
    assumptions: EconomicAssumptionsV1,
    seed: int,
    environment: PlantEconomicSetpointEnvV2 | None = None,
) -> dict[str, Any]:
    """Execute H with complete generic command/application evidence."""

    if environment is None:
        environment = _new_environment(
            episode=episode,
            plant=plant,
            constraints=constraints,
            weights=weights,
            assumptions=assumptions,
        )
        manifest_identity = _reset_environment(environment, seed=seed)
    else:
        manifest_identity = _prepared_environment_identity(environment)
    controller = LegacyHeuristicSetpointControllerV1()
    controller.reset()
    reward_trace: list[float] = []
    totals = _interval_totals()
    decisions: list[dict[str, Any]] = []
    executions: list[dict[str, Any]] = []
    controller_elapsed_ns: list[int] = []
    dispatcher_elapsed_ns: list[list[int]] = []
    terminated = truncated = False
    failure: dict[str, Any] | None = None
    started_ns = time.perf_counter_ns()
    try:
        while not terminated and not truncated:
            decision_started_ns = time.perf_counter_ns()
            decision = controller.compute_command(
                environment.controller_observation,
                constraints=constraints,
                config=plant,
                commitment_profile=episode.electrolyser_commitment,
            )
            controller_elapsed_ns.append(time.perf_counter_ns() - decision_started_ns)
            _, reward, terminated, truncated, info = environment.step_process_setpoints(
                decision.command,
                controller_invalid=decision.invalid,
            )
            execution = environment.last_process_setpoint_execution
            if execution is None:
                raise RuntimeError("heuristic step omitted generic execution evidence")
            decisions.append(decision.to_dict())
            executions.append(execution.to_dict())
            elapsed = info.get("dispatcher_elapsed_ns")
            if not isinstance(elapsed, list) or any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in elapsed
            ):
                raise RuntimeError("heuristic dispatcher timing evidence is malformed")
            dispatcher_elapsed_ns.append([int(value) for value in elapsed])
            reward_trace.append(float(reward))
            _record_interval(totals, info)
    except Exception as exc:  # retained as experimental evidence
        failure = {
            "phase": "heuristic_execution",
            "exception_type": type(exc).__name__,
            "message": " ".join(str(exc).split())[:500],
        }
    elapsed_ns = time.perf_counter_ns() - started_ns
    status = (
        "complete" if failure is None and terminated and not truncated else "failed"
    )
    evidence_body = {
        "decision_count": len(decisions),
        "execution_count": len(executions),
        "application_count": sum(
            _integer("heuristic application count", item["application_count"])
            for item in executions
        ),
        "decision_sha256": [str(item["evidence"]["sha256"]) for item in decisions],
        "command_sha256": [str(item["command"]["sha256"]) for item in decisions],
        "execution_sha256": [str(item["sha256"]) for item in executions],
        "decisions": decisions,
        "executions": executions,
        "controller_elapsed_ns": controller_elapsed_ns,
        "dispatcher_elapsed_ns": dispatcher_elapsed_ns,
    }
    evidence_body["sha256"] = domain_sha256(
        "economic-smoke-heuristic-evidence-v1", evidence_body
    )
    return _run_payload(
        arm="H",
        status=status,
        environment=environment,
        manifest_identity=manifest_identity,
        reward_trace=reward_trace,
        interval_totals=totals,
        execution_count=len(executions),
        application_count=_integer(
            "heuristic application count",
            evidence_body["application_count"],
        ),
        terminated=terminated,
        truncated=truncated,
        elapsed_ns=elapsed_ns,
        controller=controller.to_dict(),
        evidence=evidence_body,
        failure=failure,
    )


def execute_mpc_arm(
    *,
    episode: EpisodeConfigV2,
    plant: PlantConfig,
    constraints: Constraints,
    weights: RewardWeightsV2,
    assumptions: EconomicAssumptionsV1,
    settings: ServiceMPCSetpointSettingsV1,
    seed: int,
    environment: PlantEconomicSetpointEnvV2 | None = None,
) -> dict[str, Any]:
    """Execute M while retaining and deeply validating its rich evidence graph."""

    if environment is None:
        environment = _new_environment(
            episode=episode,
            plant=plant,
            constraints=constraints,
            weights=weights,
            assumptions=assumptions,
        )
        manifest_identity = _reset_environment(environment, seed=seed)
    else:
        manifest_identity = _prepared_environment_identity(environment)
    controller = ServiceMPCSetpointControllerV1(settings)
    controller.reset()
    reward_trace: list[float] = []
    totals = _interval_totals()
    dispatcher_elapsed_ns: list[tuple[int, ...]] = []
    terminated = truncated = False
    failure: dict[str, Any] | None = None
    started_ns = time.perf_counter_ns()
    try:
        while not terminated and not truncated:
            decision = controller.compute_setpoints(
                environment.controller_observation,
                constraints=constraints,
                config=plant,
                commitment_profile=episode.electrolyser_commitment,
            )
            _, reward, terminated, truncated, info = environment.step_setpoints(
                decision.setpoints
            )
            execution = environment.last_setpoint_execution
            if execution is None:
                raise RuntimeError("MPC step omitted rich execution evidence")
            elapsed = info.get("dispatcher_elapsed_ns")
            if not isinstance(elapsed, list) or any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in elapsed
            ):
                raise RuntimeError("MPC dispatcher timing evidence is malformed")
            dispatcher_elapsed_ns.append(tuple(int(value) for value in elapsed))
            reward_trace.append(float(reward))
            _record_interval(totals, info)
            controller.observe_transition(
                environment.controller_observation,
                execution=execution,
            )
    except Exception as exc:  # retained as experimental evidence
        failure = {
            "phase": "mpc_execution",
            "exception_type": type(exc).__name__,
            "message": " ".join(str(exc).split())[:500],
        }
    elapsed_ns = time.perf_counter_ns() - started_ns
    status = (
        "complete"
        if failure is None
        and terminated
        and not truncated
        and not controller.has_pending_transition
        else "failed"
    )
    try:
        run_evidence = build_mpc_setpoint_run_evidence(
            controller.decision_events,
            tuple(dispatcher_elapsed_ns),
            settings=settings,
            completed_decisions=environment.decisions_taken,
            run_succeeded=status == "complete",
            pending_transition=controller.has_pending_transition,
            runner_failure_phase=None if failure is None else str(failure["phase"]),
            final_observation_sha256=public_observation_sha256(
                environment.controller_observation
            ),
        )
        verify_mpc_setpoint_run_digests(run_evidence)
    except Exception as exc:
        run_evidence = {
            "validation_failure": {
                "exception_type": type(exc).__name__,
                "message": " ".join(str(exc).split())[:500],
            }
        }
        if failure is None:
            failure = {
                "phase": "mpc_evidence_validation",
                **run_evidence["validation_failure"],
            }
            status = "failed"
    counts = _mapping("MPC evidence counts", run_evidence.get("counts", {}))
    return _run_payload(
        arm="M",
        status=status,
        environment=environment,
        manifest_identity=manifest_identity,
        reward_trace=reward_trace,
        interval_totals=totals,
        execution_count=_integer(
            "MPC execution count", counts.get("interval_execution_count", 0)
        ),
        application_count=_integer(
            "MPC application count", counts.get("physical_minute_count", 0)
        ),
        terminated=terminated,
        truncated=truncated,
        elapsed_ns=elapsed_ns,
        controller=controller.to_dict(),
        evidence=run_evidence,
        failure=failure,
    )


def _gate(
    identifier: str, passed: bool, observation: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "id": identifier,
        "status": "passed" if passed else "failed",
        "observation": dict(observation),
    }


def _arm_accounting_checks(
    run: Mapping[str, Any],
    *,
    assumptions: EconomicAssumptionsV1,
) -> dict[str, bool]:
    economic = _mapping("economic metrics", run["economic_metrics"])
    components = _mapping("economic components", economic["components_eur"])
    totals = _mapping("interval totals", run["interval_totals"])
    checks = {
        "reward_trace": _close(
            _finite("reward trace sum", run["reward_trace_sum_eur"]),
            _finite("net operating value", economic["net_operating_value_eur"]),
        ),
        "component_sum": _close(
            math.fsum(
                _finite(f"component {key}", value) for key, value in components.items()
            ),
            _finite("net operating value", economic["net_operating_value_eur"]),
        ),
    }
    for name in (
        "delivered_kg",
        "unmet_kg",
        "accepted_solar_kwh",
        "battery_throughput_kwh",
        "electrolyser_equivalent_operating_h",
    ):
        checks[f"interval_{name}"] = _close(
            _finite(f"interval {name}", totals[name]),
            _finite(f"episode {name}", economic[name]),
        )
    checks["methane_liability"] = _close(
        _finite(
            "methane liability component",
            components["methane_endowment_liability_change"],
        ),
        -_finite(
            "terminal methane deficit",
            economic["terminal_ch4_equivalent_deficit_kg"],
        )
        * assumptions.methane_endowment_value_eur_per_kg_ch4_equivalent,
    )
    checks["battery_liability"] = _close(
        _finite(
            "battery liability component",
            components["battery_endowment_liability_change"],
        ),
        -_finite("terminal battery deficit", economic["terminal_battery_deficit_kwh"])
        * assumptions.battery_endowment_value_eur_per_kwh,
    )
    checks["identity"] = dict(_mapping("run identity", run["manifest_identity"])) == (
        _expected_identity()
    )
    return checks


def _safe_arm_accounting_checks(
    run: Mapping[str, Any],
    *,
    assumptions: EconomicAssumptionsV1,
) -> tuple[dict[str, bool], dict[str, str] | None]:
    try:
        return _arm_accounting_checks(run, assumptions=assumptions), None
    except Exception as reason:
        return {"evidence_complete": False}, {
            "exception_type": type(reason).__name__,
            "message": " ".join(str(reason).split())[:500],
        }


def evaluate_smoke_gates(
    heuristic: Mapping[str, Any],
    mpc: Mapping[str, Any],
    *,
    assumptions: EconomicAssumptionsV1,
) -> tuple[dict[str, Any], ...]:
    """Apply the frozen E1--E6 decision rule to detached arm payloads."""

    runs = (heuristic, mpc)
    e1_observation = {
        str(run["arm"]): {
            "status": run["status"],
            "completed_decisions": run["completed_decisions"],
            "physical_steps": run["physical_steps"],
            "terminated": run["terminated"],
            "truncated": run["truncated"],
        }
        for run in runs
    }
    e1_pass = all(
        run["status"] == "complete"
        and _integer("completed decisions", run["completed_decisions"])
        == ECONOMIC_SMOKE_EXPECTED_DECISIONS
        and _integer("physical steps", run["physical_steps"])
        == ECONOMIC_SMOKE_EXPECTED_PHYSICAL_STEPS
        and run["terminated"] is True
        and run["truncated"] is False
        for run in runs
    )
    e1 = _gate("E1", e1_pass, e1_observation)

    e2_observation: dict[str, Any] = {}
    e2_pass = e1_pass
    for run in runs:
        metrics = _mapping("physical metrics", run["physical_metrics"])
        arm = str(run["arm"])
        values = {
            "shutdown_steps": _integer("shutdown steps", metrics["shutdown_steps"]),
            "out_of_range_decisions": _integer(
                "out-of-range decisions", metrics["out_of_range_decisions"]
            ),
            "forced_stops": _integer(
                "forced stops", metrics["electrolyser_forced_stops"]
            ),
            "execution_count": _integer("execution count", run["execution_count"]),
            "application_count": _integer(
                "application count", run["application_count"]
            ),
        }
        e2_observation[arm] = values
        e2_pass = e2_pass and values == {
            "shutdown_steps": 0,
            "out_of_range_decisions": 0,
            "forced_stops": 0,
            "execution_count": ECONOMIC_SMOKE_EXPECTED_DECISIONS,
            "application_count": ECONOMIC_SMOKE_EXPECTED_PHYSICAL_STEPS,
        }
    e2 = _gate("E2", e2_pass, e2_observation)

    h_metrics = _mapping("H physical metrics", heuristic["physical_metrics"])
    m_metrics = _mapping("M physical metrics", mpc["physical_metrics"])
    delivery_delta = _finite("M delivered", m_metrics["delivered_kg"]) - _finite(
        "H delivered", h_metrics["delivered_kg"]
    )
    unmet_delta = _finite("M unmet", m_metrics["unmet_kg"]) - _finite(
        "H unmet", h_metrics["unmet_kg"]
    )
    e3_pass = e2_pass and abs(delivery_delta) <= 1e-6 and abs(unmet_delta) <= 1e-6
    e3 = _gate(
        "E3",
        e3_pass,
        {"M_minus_H_delivered_kg": delivery_delta, "M_minus_H_unmet_kg": unmet_delta},
    )

    h_economic = _mapping("H economic metrics", heuristic["economic_metrics"])
    m_economic = _mapping("M economic metrics", mpc["economic_metrics"])
    value_delta = _finite(
        "M net value", m_economic["net_operating_value_eur"]
    ) - _finite("H net value", h_economic["net_operating_value_eur"])
    driver_names = (
        ("accepted_solar_kwh", 1.0),
        ("battery_throughput_kwh", 1.0),
        ("electrolyser_equivalent_operating_h", 0.01),
        ("terminal_ch4_equivalent_deficit_kg", 1e-3),
        ("terminal_battery_deficit_kwh", 0.1),
    )
    driver_deltas = {
        name: _finite(f"M {name}", m_economic[name])
        - _finite(f"H {name}", h_economic[name])
        for name, _ in driver_names
    }
    material_driver = any(
        abs(driver_deltas[name]) > threshold for name, threshold in driver_names
    )
    e4_pass = e3_pass and abs(value_delta) > 1.0 and material_driver
    e4 = _gate(
        "E4",
        e4_pass,
        {
            "M_minus_H_net_operating_value_eur": value_delta,
            "absolute_value_separation_eur": abs(value_delta),
            "material_driver": material_driver,
            "M_minus_H_drivers": driver_deltas,
        },
    )

    h_checks, h_accounting_failure = _safe_arm_accounting_checks(
        heuristic,
        assumptions=assumptions,
    )
    m_checks, m_accounting_failure = _safe_arm_accounting_checks(
        mpc,
        assumptions=assumptions,
    )
    paired_identity = heuristic["manifest_identity"] == mpc["manifest_identity"]
    e5_pass = all(h_checks.values()) and all(m_checks.values()) and paired_identity
    e5 = _gate(
        "E5",
        e5_pass,
        {
            "H": h_checks,
            "M": m_checks,
            "paired_identity": paired_identity,
            "accounting_evidence_failure": {
                "H": h_accounting_failure,
                "M": m_accounting_failure,
            },
        },
    )

    prerequisites = all(gate["status"] == "passed" for gate in (e1, e2, e3, e4, e5))
    winner = "M" if value_delta > 0.0 else "H" if value_delta < 0.0 else "tie"
    e6 = {
        "id": "E6",
        "status": "reported" if prerequisites else "not_evaluated",
        "observation": {
            "descriptive_higher_value_arm": winner if prerequisites else None,
            "claim_boundary": (
                "one development-cell task-expression result; no controller, RL, "
                "commercial, or commission superiority claim"
            ),
        },
    }
    return e1, e2, e3, e4, e5, e6


def _source_snapshot(
    *,
    repository: Path,
    episode: EpisodeConfigV2,
    plant: PlantConfig,
    constraints: Constraints,
    weights: RewardWeightsV2,
) -> dict[str, Any]:
    provenance = collect_provenance(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        root=repository,
    )
    return {
        "source": provenance["source"],
        "dependency_lock": provenance["dependency_lock"],
        "runtime": provenance["runtime"],
    }


def _validate_start_source(snapshot: Mapping[str, Any], repository: Path) -> None:
    source = _mapping("source", snapshot["source"])
    lock = _mapping("dependency lock", snapshot["dependency_lock"])
    if (
        source.get("git_dirty") is not False
        or source.get("identity_complete") is not True
    ):
        raise RuntimeError(
            "economic smoke requires a clean, completely identified source"
        )
    if lock.get("sha256") != ECONOMIC_SMOKE_LOCK_SHA256:
        raise RuntimeError("economic smoke dependency lock identity drifted")
    protocol = repository / ECONOMIC_SMOKE_PROTOCOL_PATH
    if not protocol.is_file() or hashlib.sha256(protocol.read_bytes()).hexdigest() != (
        ECONOMIC_SMOKE_PROTOCOL_SHA256
    ):
        raise RuntimeError("economic smoke protocol identity drifted")


def _source_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    first = _mapping("start source", before["source"])
    second = _mapping("end source", after["source"])
    return bool(
        first.get("git_dirty") is False
        and second.get("git_dirty") is False
        and first.get("identity_complete") is True
        and second.get("identity_complete") is True
        and first.get("git_revision") == second.get("git_revision")
        and first.get("worktree_identity_sha256")
        == second.get("worktree_identity_sha256")
    )


def _publish_exclusive(target: Path, payload: Mapping[str, Any]) -> str:
    encoded = (canonical_json(dict(payload)) + "\n").encode("utf-8")
    raw_sha256 = hashlib.sha256(encoded).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return raw_sha256


def run_registered_economic_smoke(output: Path | str) -> dict[str, Any]:
    """Execute and exclusively publish the single preregistered H-then-M cell."""

    repository = find_repository_root()
    target = validate_artifact_destination(output, repository_root=repository)
    plant = PlantConfig()
    episode = fixed_episode()
    constraints = Constraints()
    weights = RewardWeightsV2()
    assumptions = EconomicAssumptionsV1()
    settings = fixed_mpc_settings()
    source_start = _source_snapshot(
        repository=repository,
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
    )
    _validate_start_source(source_start, repository)
    heuristic_environment = _new_environment(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        assumptions=assumptions,
    )
    mpc_environment = _new_environment(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        assumptions=assumptions,
    )
    heuristic_identity = _reset_environment(
        heuristic_environment,
        seed=ECONOMIC_SMOKE_SEED,
    )
    mpc_identity = _reset_environment(
        mpc_environment,
        seed=ECONOMIC_SMOKE_SEED,
    )
    _validate_preflight_identities(heuristic_identity, mpc_identity)
    started_at = _utc_now()
    started_ns = time.perf_counter_ns()
    heuristic = execute_heuristic_arm(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        assumptions=assumptions,
        seed=ECONOMIC_SMOKE_SEED,
        environment=heuristic_environment,
    )
    mpc = execute_mpc_arm(
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
        assumptions=assumptions,
        settings=settings,
        seed=ECONOMIC_SMOKE_SEED,
        environment=mpc_environment,
    )
    elapsed_ns = time.perf_counter_ns() - started_ns
    completed_at = _utc_now()
    source_end = _source_snapshot(
        repository=repository,
        episode=episode,
        plant=plant,
        constraints=constraints,
        weights=weights,
    )
    source_valid = _source_unchanged(source_start, source_end)
    gates = evaluate_smoke_gates(heuristic, mpc, assumptions=assumptions)
    outcome_gates_pass = all(
        gate["status"] == "passed" for gate in gates if gate["id"] != "E6"
    )
    status = (
        "complete_all_gates_passed"
        if source_valid and outcome_gates_pass
        else "complete_gate_failed"
    )
    body: dict[str, Any] = {
        "format": ECONOMIC_SMOKE_FORMAT,
        "contract": ECONOMIC_SMOKE_CONTRACT,
        "contract_sha256": ECONOMIC_SMOKE_CONTRACT_SHA256,
        "status": status,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "elapsed_ns": elapsed_ns,
        "source_gate": {
            "status": "passed" if source_valid else "failed",
            "start": source_start,
            "end": source_end,
        },
        "arms": {"H": heuristic, "M": mpc},
        "gates": list(gates),
    }
    body["sha256"] = domain_sha256(ECONOMIC_SMOKE_DOMAIN, body)
    raw_sha256 = _publish_exclusive(target, body)
    return {
        "path": str(target),
        "raw_file_sha256": raw_sha256,
        "report_sha256": body["sha256"],
        "status": status,
        "elapsed_ns": elapsed_ns,
        "gates": [{"id": gate["id"], "status": gate["status"]} for gate in gates],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the frozen economic task-expression smoke cell once."
    )
    parser.add_argument("output", type=Path, help="new absolute JSON path outside Git")
    arguments = parser.parse_args(argv)
    published = run_registered_economic_smoke(arguments.output)
    print(json.dumps(published, allow_nan=False, sort_keys=True))
    return 0 if published["status"] == "complete_all_gates_passed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ECONOMIC_SMOKE_CONTRACT",
    "ECONOMIC_SMOKE_CONTRACT_SHA256",
    "ECONOMIC_SMOKE_FORMAT",
    "evaluate_smoke_gates",
    "execute_heuristic_arm",
    "execute_mpc_arm",
    "fixed_episode",
    "fixed_mpc_settings",
    "run_registered_economic_smoke",
]
