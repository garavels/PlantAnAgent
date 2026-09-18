#!/usr/bin/env python3
"""Exact simulator harness for delivery-only variants of retained scenarios.

Original exogenous objects are reconstructed under the ORIGINAL task and checked
against historical evidence. Each changed-rate environment reuses those objects;
only delivery amounts and the declared delivery-rate configuration change.
Controllers receive public observations, never the frozen future tape.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
for folder in ("backend/src", "simulator/src"):
    sys.path.insert(0, str(REPO / folder))

from plant_sim import SolarArray
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.demo_projection import forecast_view
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.mpc_execution import public_observation_sha256
from plant_backend.rl.v2.setpoint_control import ProcessSetpointTargetsV1
from plant_backend.rl.v2.setpoint_command import issue_process_setpoint_command

RESEARCH_ROOT = (
    REPO.parents[1] / "PlantAnAgentResearch/adaptive-rl-batch-008/evaluations"
)
DEMO_PATH = REPO / "runs/commission-demo/5a25ba80-6227-4c07-8790-adc38f02a3a1.json"
ALLOWED_SEEDS = (*range(9700300, 9700308), 9000034)
SCHEMA = "physical-feasible-mission-replay-v1"


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_pin(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def state_differences(saved: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Allow only the explicitly irrelevant night resource-factor display change."""
    differences = {
        k: {"saved": saved.get(k), "current": current.get(k)}
        for k in saved.keys() | current.keys()
        if saved.get(k) != current.get(k)
    }
    physical = dict(differences)
    if all(
        x.get("solar_available_kw") == 0 and x.get("solar_potential_kw") == 0
        for x in [saved, current]
    ):
        physical.pop("solar_output_factor", None)
    if physical:
        raise ValueError(f"Retained physical state differs: {physical}")
    return differences


@dataclass
class FrozenScenario:
    seed: int
    original_config: ControllerNeutralTrainingConfigV1
    episode: Any
    market: Any
    disturbances: Any
    original_manifest: dict[str, Any]
    source_pin: dict[str, Any]
    source_payload: dict[str, Any]
    expected_solar_kw: tuple[float, ...]
    validation: dict[str, Any]


@lru_cache(maxsize=1)
def load_frozen_scenario(seed: int) -> FrozenScenario:
    """Reconstruct original source once; validate before changing any contract."""
    if seed not in ALLOWED_SEEDS:
        raise ValueError(
            "Only the eight development seeds and public demo are admitted"
        )
    path = (
        DEMO_PATH
        if seed == 9000034
        else RESEARCH_ROOT / f"{seed}-OperatingMPC/episode.json.gz"
    )
    data = path.read_bytes()
    payload = json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)
    if seed == 9000034:
        if (
            canonical_sha({k: v for k, v in payload.items() if k != "sha256"})
            != payload["sha256"]
        ):
            raise ValueError("Demo recording checksum does not match")
        # The original public fixture explicitly constructs the beta defaults.
        original = PlantCyclicMarketEnvV2Beta()
        config = ControllerNeutralTrainingConfigV1(
            environment_id="plant-env-v2-cyclic-market-beta-v1",
            plant=original.config,
            episode=original.episode_config,
            constraints=original.constraints,
            diagnostic_reward=original.weights,
            economic_assumptions=original.economic_assumptions,
            beta_task=original.beta_task,
        )
    else:
        config = ControllerNeutralTrainingConfigV1.from_dict(payload["config"])
        original = PlantCyclicMarketEnvV2Beta(
            config=config.plant,
            episode=config.episode,
            constraints=config.constraints,
            weights=config.diagnostic_reward,
            economic_assumptions=config.economic_assumptions,
            beta_task=config.beta_task,
        )
    try:
        original.reset(seed=seed)
        episode = original._episode
        manifest = original.audit_manifest()
        if episode is None:
            raise RuntimeError("Reset omitted episode")
        solar = SolarArray(config.plant.solar)
        start_s = episode.scenario.start_hour * 3600
        expected = tuple(
            solar.clear_sky_kw(start_s + i * config.plant.dt_s)
            * episode.weather.tape.for_interval(i)
            for i in range(episode.physics_steps)
        )
        if seed == 9000034:
            frames = payload["frames"]
            if len(frames) != 673:
                raise ValueError("Original demo frame count differs")
            opening_differences = state_differences(
                frames[0]["state"], original.simulator.state.to_dict()
            )
            demo_plant = {
                **config.plant.to_dict(),
                "dt_h": config.plant.dt_h,
                "steps_per_day": config.plant.steps_per_day,
                "electrolyser_min_load_kw": config.plant.electrolyser.rated_kw
                * config.plant.electrolyser.min_load_frac,
            }
            if payload["snapshot"]["config"] != demo_plant:
                raise ValueError("Original demo plant configuration differs")
            if (
                frames[0]["commitment_profile"]
                != config.episode.electrolyser_commitment.to_dict()
            ):
                raise ValueError("Original demo equipment commitment profile differs")
            residuals = []
            for i in range(672):
                retained = (
                    frames[i + 1]["state"]["energy_produced_kwh"]
                    - frames[i]["state"]["energy_produced_kwh"]
                )
                residuals.append(
                    abs(
                        math.fsum(expected[i * 15 : (i + 1) * 15]) * config.plant.dt_h
                        - retained
                    )
                )
                if (
                    abs(
                        frames[i + 1]["state"]["solar_available_kw"]
                        - expected[(i + 1) * 15 - 1]
                    )
                    > 1e-9
                ):
                    raise ValueError("Original demo endpoint solar differs")
            if max(residuals) > 1e-6:
                raise ValueError("Original demo integrated interval solar differs")
            for index, frame in enumerate(frames):
                market = original.market_episode.issue_at_decision(
                    min(index, len(original.market_episode.issues) - 1)
                )
                weather = episode.weather.current_issue(
                    index * episode.decision_stride
                ).public()
                regenerated = forecast_view(
                    SimpleNamespace(
                        market=market, plant=SimpleNamespace(forecast=weather)
                    ),
                    config.plant.solar.peak_kw,
                )
                if regenerated != frame["forecasts"]:
                    raise ValueError(
                        f"Original demo public market/weather issue differs at decision {index}"
                    )
                if frame["assumptions"] != config.economic_assumptions.to_dict():
                    raise ValueError(
                        f"Original demo economic assumptions differ at decision {index}"
                    )
                elapsed_s = (
                    max(0, index * episode.decision_stride - 1) * config.plant.dt_s
                )
                scheduled = any(
                    f["start_s"] <= elapsed_s < f["start_s"] + f["duration_s"]
                    for f in manifest["disturbances"]["schedule"]
                )
                expected_health = {
                    unit: dict(health)
                    for unit, health in frames[0]["state"]["health"].items()
                }
                expected_health["dac"] = {
                    "status": "offline" if scheduled else "nominal",
                    "degradation": 0.0,
                    "offline": scheduled,
                }
                if frame["state"]["health"] != expected_health:
                    raise ValueError(
                        f"Original demo endpoint component health differs at decision {index}"
                    )
            validation = {
                "kind": "public_recording",
                "initial_physical_state_exact": True,
                "opening_metadata_differences": opening_differences,
                "equipment_exact": True,
                "solar_intervals_checked": 672,
                "maximum_solar_interval_residual_kwh": max(residuals),
                "endpoint_health_frames_checked": 673,
                "public_market_weather_issue_frames_exact": 673,
                "economic_assumption_frames_exact": 673,
                "limitation": "Public recording retains exact decision forecasts and endpoint health, not historical within-interval minute forcing or disturbance hashes.",
            }
        else:
            saved = payload["manifest"]
            for key in [
                "alpha_physical_exogenous_episode_sha256",
                "beta_exogenous_episode_sha256",
                "manifest_sha256",
            ]:
                if manifest[key] != saved[key]:
                    raise ValueError(f"Original {key} differs from retained episode")
            states = payload["minute_truth_states"]
            if len(states) != episode.physics_steps + 1:
                raise ValueError("Original minute evidence length differs")
            opening_differences = state_differences(
                states[0], original.simulator.state.to_dict()
            )
            residual = max(
                abs(a - b["solar_available_kw"])
                for a, b in zip(expected, states[1:], strict=True)
            )
            if residual > 1e-9:
                raise ValueError("Original minute solar forcing differs")
            validation = {
                "kind": "research_episode",
                "manifest_sha256_exact": True,
                "initial_physical_state_exact": True,
                "opening_metadata_differences": opening_differences,
                "solar_minutes_checked": len(expected),
                "maximum_solar_kw_residual": residual,
            }
        return FrozenScenario(
            seed,
            config,
            episode,
            original.market_episode,
            original.disturbance_episode,
            manifest,
            {
                "path": str(path),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
            payload,
            expected,
            validation,
        )
    finally:
        original.close()


class FixedMissionEnv(PlantCyclicMarketEnvV2Beta):
    """A new manifest and frozen schedule; no mutation of historical environments."""

    def __init__(self, frozen: FrozenScenario, delivery_rate_kg_h: float):
        if not math.isfinite(delivery_rate_kg_h) or delivery_rate_kg_h <= 0:
            raise ValueError("Delivery rate must be finite and positive")
        self.frozen_source = frozen
        cfg = frozen.original_config
        self.variant_rate_kg_h = float(delivery_rate_kg_h)
        commitments = tuple(
            replace(c, requested_kg=delivery_rate_kg_h * cfg.episode.delivery_period_h)
            for c in frozen.episode.commitments
        )
        self.fixed_episode = replace(frozen.episode, commitments=commitments)
        super().__init__(
            config=cfg.plant,
            episode=replace(cfg.episode, delivery_rate_kg_h=delivery_rate_kg_h),
            constraints=cfg.constraints,
            weights=cfg.diagnostic_reward,
            economic_assumptions=cfg.economic_assumptions,
            beta_task=cfg.beta_task,
        )

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self.current_policy_observation = result[0].copy()
        return result

    def step_process_setpoints(self, command, *, controller_invalid=False):
        result = super().step_process_setpoints(
            command, controller_invalid=controller_invalid
        )
        self.current_policy_observation = result[0].copy()
        return result

    def _sample_episode(self, seed: int):
        if seed != self.frozen_source.seed:
            raise ValueError("Fixed scenario cannot reset to another seed")
        self._market_episode = self.frozen_source.market
        self._disturbance_episode = self.frozen_source.disturbances
        return self.fixed_episode


def make_environment(seed: int, delivery_rate_kg_h: float) -> FixedMissionEnv:
    """Return reset beta environment at the exact retained opening inventory."""
    env = FixedMissionEnv(load_frozen_scenario(seed), delivery_rate_kg_h)
    env.reset(seed=seed)
    original = env.frozen_source.original_manifest
    changed = env.audit_manifest()
    for key in ["initial", "weather"]:
        if changed["episode"][key] != original["episode"][key]:
            raise ValueError(f"Delivery variant changed {key}")
    for key in [
        "market",
        "disturbances",
        "plant_config",
        "constraints",
        "cycle_reference",
        "beta_task",
    ]:
        if changed[key] != original[key]:
            raise ValueError(f"Delivery variant changed {key}")
    return env


def command_for_targets(
    env: FixedMissionEnv,
    targets: ProcessSetpointTargetsV1,
    *,
    issuer_id: str = "physical-feasibility-controller-v1",
    evidence: Any = None,
):
    """Bind four absolute kW targets to the current public plant observation."""
    return issue_process_setpoint_command(
        targets,
        decision_index=env.decisions_taken,
        observation_sha256=public_observation_sha256(env.controller_observation),
        hold_duration_s=env.episode_config.decision_interval_s,
        issuer_id=issuer_id,
        source_schema="physical-feasibility-decision-v1",
        source_sha256=canonical_sha(
            evidence if evidence is not None else targets.to_dict()
        ),
        constraints=env.constraints,
        config=env.config,
    )


def run_controller(
    env: FixedMissionEnv,
    controller: Any,
    *,
    max_decisions: int | None = None,
    output_path: Path | str | None = None,
    retain_minutes: bool = True,
    controller_id: str | None = None,
) -> dict[str, Any]:
    """Run standard public-observation compute_command controller on exact physics.

    A callable may implement the same observation and keyword signature and return
    ProcessSetpointTargetsV1. No controller receives env or frozen source objects.
    A partial result never passes complete mission, delivery or return checks.
    """
    if env.decisions_taken != 0:
        raise ValueError("Runner requires a reset environment")
    if max_decisions is not None and (
        isinstance(max_decisions, bool) or max_decisions < 1
    ):
        raise ValueError("max_decisions must be positive or None")
    if hasattr(controller, "reset"):
        controller.reset()
    identity = controller_id or getattr(controller, "id", type(controller).__name__)
    limit = min(max_decisions or env.episode_decisions, env.episode_decisions)
    minutes = [env.simulator.state.to_dict()]
    frames = [
        {
            "decision_index": 0,
            "state": minutes[0],
            "commitment": env.simulator.electrolyser_commitment.to_dict(),
        }
    ]
    commands = []
    terminated = truncated = False
    failure = None
    try:
        for index in range(limit):
            if hasattr(controller, "set_observation"):
                controller.set_observation(env.current_policy_observation.copy())
            observation = env.cyclic_market_controller_observation
            compute = getattr(controller, "compute_command", controller)
            decision = compute(
                observation,
                constraints=env.constraints,
                config=env.config,
                commitment_profile=env.episode_config.electrolyser_commitment,
            )
            if isinstance(decision, ProcessSetpointTargetsV1):
                command = command_for_targets(env, decision, issuer_id=identity)
                invalid = False
                evidence = decision.to_dict()
            else:
                command = decision.command
                invalid = getattr(decision, "invalid", False)
                evidence = getattr(decision, "evidence", None)
                if hasattr(evidence, "to_dict"):
                    evidence = evidence.to_dict()
            _, reward, terminated, truncated, info = env.step_process_setpoints(
                command, controller_invalid=invalid
            )
            trace = env.last_minute_truth_states
            if trace is None or len(trace) != 16:
                raise RuntimeError("Missing exact fifteen-minute execution trace")
            minutes.extend(state.to_dict() for state in trace[1:])
            commands.append(
                {
                    "decision_index": index,
                    "command": command.to_dict(),
                    "controller_invalid": invalid,
                    "evidence": evidence,
                    "reward_eur": reward,
                }
            )
            frames.append(
                {
                    "decision_index": index + 1,
                    "state": minutes[-1],
                    "commitment": env.simulator.electrolyser_commitment.to_dict(),
                    "cycle_constraint": info.get("cycle_constraint"),
                    "cycle_constraint_transition": info.get(
                        "cycle_constraint_transition"
                    ),
                }
            )
            if terminated or truncated:
                break
    except Exception as exc:
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "decision": env.decisions_taken,
            "physics_steps": env.physics_steps_taken,
        }
    physical = env.metrics.to_dict()
    manifest = env.audit_manifest()
    checks = verify_minute_trace(env, minutes)
    settlements = [s.to_dict() for s in env.cycle_settlements]
    delivery = env.simulator.delivery_audit()
    complete = (
        failure is None
        and terminated
        and not truncated
        and env.decisions_taken == env.episode_decisions
        and env.physics_steps_taken == env.episode_physics_steps
        and len(minutes) == env.episode_physics_steps + 1
    )
    daily = (
        complete and len(settlements) == 7 and all(s["feasible"] for s in settlements)
    )
    delivery_pass = (
        complete
        and len(delivery["settlements"]) == 28
        and all(x["unmet_kg"] <= 1e-9 for x in delivery["settlements"])
    )
    hard_safety = (
        complete
        and checks["exogenous_forcing_exact"]
        and checks["resource_bounds_passed"]
        and checks["operator_reserve_respected"]
        and checks["component_power_limits_respected"]
        and checks["unexpected_critical_or_offline_steps"] == 0
        and checks["minimum_battery_soc"] > env.config.battery.min_soc + 1e-9
        and physical["out_of_range_decisions"] == 0
        and physical["electrolyser_forced_stops"] == 0
    )
    body = {
        "schema": SCHEMA,
        "seed": env.frozen_source.seed,
        "controller": identity,
        "delivery_rate_kg_h": env.variant_rate_kg_h,
        "source": env.frozen_source.source_pin,
        "source_reconstruction": env.frozen_source.validation,
        "source_code": file_pin(Path(__file__)),
        "change": "Delivery amounts only; original initial inventory, weather, forecasts, market, DAC faults, equipment, hard safety and daily targets preserved.",
        "registered_evaluation_eligible": False,
        "execution_complete": complete,
        "failure": failure,
        "hard_safety_passed": hard_safety,
        "every_delivery_passed": delivery_pass,
        "delivery_unmet_tolerance_kg": 1e-9,
        "all_daily_returns_passed": daily,
        "mission_passed": hard_safety and delivery_pass and daily,
        "decisions": env.decisions_taken,
        "physical_steps": env.physics_steps_taken,
        "physical_metrics": physical,
        "economic_metrics": env.economic_metrics.to_dict(),
        "minute_checks": checks,
        "cycle_settlements": settlements,
        "delivery": delivery,
        "manifest": manifest,
        "commands": commands,
        "decision_frames": frames,
        "minute_truth_states": minutes if retain_minutes else None,
        "minute_truth_sha256": canonical_sha(minutes),
        "terminated": terminated,
        "truncated": truncated,
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
        path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == ".gz" else raw)
    return body


def verify_minute_trace(
    env: FixedMissionEnv, minutes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Independently check minute availability, bounds, outages and battery safety."""
    faults = env.frozen_source.original_manifest["disturbances"]["schedule"]
    solar_errors = []
    health_mismatches = []
    bounds_failures = []
    unexpected = 0
    declared_steps = 0
    critical_steps = 0
    minimum_soc = min(s["battery_soc"] for s in minutes)
    for index, state in enumerate(minutes):
        if (
            state["step"] != index
            or state["time_s"]
            != env.episode_config.start_hour * 3600 + index * env.config.dt_s
        ):
            raise ValueError("Minute clock is discontinuous")
        for resource, level, capacity in [
            ("battery", "battery_energy_kwh", "battery_capacity_kwh"),
            ("h2", "h2_level_kg", "h2_capacity_kg"),
            ("co2", "co2_level_kg", "co2_capacity_kg"),
            ("ch4", "ch4_level_kg", "ch4_capacity_kg"),
        ]:
            if (
                not math.isfinite(state[level])
                or not -1e-9 <= state[level] <= state[capacity] + 1e-9
            ):
                bounds_failures.append(
                    {"step": index, "resource": resource, "value": state[level]}
                )
        if index == 0:
            continue
        start_s = (index - 1) * env.config.dt_s
        outage = any(
            f["start_s"] <= start_s < f["start_s"] + f["duration_s"] for f in faults
        )
        declared_steps += int(outage)
        non_dac_offline = False
        for unit, health in state["health"].items():
            expected = outage if unit == "dac" else False
            if health["offline"] != expected or health["degradation"] != 0:
                health_mismatches.append(
                    {
                        "step": index,
                        "unit": unit,
                        "health": health,
                        "expected_offline": expected,
                    }
                )
            non_dac_offline |= unit != "dac" and health["offline"]
        solar_errors.append(
            abs(
                state["solar_available_kw"]
                - env.frozen_source.expected_solar_kw[index - 1]
            )
        )
        critical = state["plant_status"] in {"critical", "offline"}
        critical_steps += int(critical)
        explained = (
            outage
            and state["health"]["dac"]["offline"]
            and not non_dac_offline
            and state["battery_soc"] > env.config.battery.min_soc + 1e-9
        )
        unexpected += int(critical and not explained)
    power_limits = {
        "electrolyser_power_kw": env.constraints.electrolyser_limit_kw(env.config),
        "dac_power_kw": env.constraints.dac_limit_kw(env.config),
        "methanator_power_kw": env.constraints.methanator_limit_kw(env.config),
        "solar_to_battery_kw": env.config.battery.max_charge_kw,
        "battery_to_electrolyser_kw": env.constraints.support_limit_kw(env.config),
    }
    maximum_powers = {field: max(s[field] for s in minutes) for field in power_limits}
    power_limits_respected = all(
        -1e-9 <= s[field] <= limit + 1e-7
        for s in minutes
        for field, limit in power_limits.items()
    )
    for state in minutes:
        for unit, field in [
            ("dac", "dac_power_kw"),
            ("electrolyser", "electrolyser_power_kw"),
            ("methanator", "methanator_power_kw"),
        ]:
            if state["health"][unit]["offline"] and abs(state[field]) > 1e-7:
                power_limits_respected = False
    cycle_checks = []
    reference = env.cycle_reference.targets(env.beta_task.cycle)
    for settlement in env.cycle_settlements:
        x = settlement.to_dict()
        step = x["boundary_decision"] * 15
        if step >= len(minutes):
            raise ValueError("Daily boundary missing from minute trace")
        state = minutes[step]
        actual = {
            "battery": state["battery_energy_kwh"],
            "h2": state["h2_level_kg"],
            "co2": state["co2_level_kg"],
            "ch4": state["ch4_level_kg"],
        }
        if x["targets"] != reference or x["actual"] != actual:
            raise ValueError(
                "Daily settlement differs from exact minute state or opening reference"
            )
        recomputed = all(
            max(0.0, reference[k] - actual[k]) <= x["tolerances"][k] + 1e-12
            for k in reference
        )
        if x["feasible"] != recomputed:
            raise ValueError("Daily feasibility differs from independent recomputation")
        cycle_checks.append({"cycle_index": x["cycle_index"], "feasible": recomputed})
    return {
        "minute_count": len(minutes),
        "maximum_solar_kw_residual": max(solar_errors, default=0),
        "exogenous_forcing_exact": max(solar_errors, default=0) <= 1e-9
        and not health_mismatches,
        "health_mismatches": health_mismatches,
        "resource_bounds_passed": not bounds_failures,
        "maximum_accepted_powers_kw": maximum_powers,
        "component_power_limits_kw": power_limits,
        "component_power_limits_respected": power_limits_respected,
        "resource_bounds_failures": bounds_failures,
        "minimum_battery_soc": minimum_soc,
        "physical_minimum_soc": env.config.battery.min_soc,
        "operator_reserve_soc": env.constraints.reserve_soc(env.config),
        "operator_reserve_respected": minimum_soc
        >= env.constraints.reserve_soc(env.config) - 1e-9,
        "declared_dac_outage_steps": declared_steps,
        "critical_or_offline_steps": critical_steps,
        "unexpected_critical_or_offline_steps": unexpected,
        "daily_settlements_independently_checked": cycle_checks,
    }


class SavedTargetsController:
    """Identity smoke-test adapter: retained targets, freshly bound observation."""

    def __init__(self, source: FrozenScenario):
        if source.seed == 9000034:
            raise ValueError(
                "Identity command test uses full research command evidence"
            )
        self.source = source
        self.index = 0

    def reset(self):
        self.index = 0

    def compute_command(self, observation, **kwargs):
        saved = self.source.source_payload["history"][self.index]["command"]["targets"][
            "setpoints_kw"
        ]
        self.index += 1
        return ProcessSetpointTargetsV1(
            saved["electrolyser"],
            saved["dac"],
            saved["methanator"],
            saved["battery_charge_ceiling"],
        )


def identity_smoke(seed: int = 9700300, decisions: int = 8) -> dict[str, Any]:
    env = make_environment(seed, 40.0)
    try:
        result = run_controller(
            env, SavedTargetsController(env.frozen_source), max_decisions=decisions
        )
        if result["failure"] is not None:
            raise ValueError(result["failure"])
        metadata_differences = 0
        for current, saved in zip(
            result["minute_truth_states"],
            env.frozen_source.source_payload["minute_truth_states"],
        ):
            metadata_differences += bool(state_differences(saved, current))
        return {
            "seed": seed,
            "decisions": result["decisions"],
            "physical_steps": result["physical_steps"],
            "all_physical_states_match": True,
            "night_factor_metadata_differences": metadata_differences,
            "minute_checks": result["minute_checks"],
            "source": result["source"],
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=9700300, choices=ALLOWED_SEEDS)
    parser.add_argument("--decisions", type=int, default=8)
    args = parser.parse_args()
    if not args.identity_smoke:
        parser.error("Use --identity-smoke, or import make_environment/run_controller")
    print(json.dumps(identity_smoke(args.seed, args.decisions), indent=2))


if __name__ == "__main__":
    main()
