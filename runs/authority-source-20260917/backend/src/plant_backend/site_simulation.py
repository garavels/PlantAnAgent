"""Opt-in ten-day, forecast-driven site projections on the existing plant physics.

The downloaded snapshot is fixed at acquisition. Internal 15-minute forecast
views are slices of that snapshot, never newly issued weather forecasts. The
nominal projection assumes that snapshot is realized; a stress multiplier only
changes the physical realization. No synthetic climate or daylight envelope is
substituted for a site's downloaded irradiance.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
import math
from numbers import Real
import time
from typing import Any, Callable

from plant_sim import DeliveryCommitment, InitialState, PlantConfig, SolarArray, SolarFactorTape
from plant_sim.faults import FaultSpec

from .controllers.base import Constraints
from .operational_mpc import (
    OperatingDecision,
    OperatingEvidence,
    OperationalMPCController,
)
from .rl.v2.beta_config import CyclicMarketTaskConfigV1
from .rl.v2.beta_disturbances import DisturbanceConfigV1
from .rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from .rl.v2.config import EpisodeConfigV2, EpisodeV2
from .rl.v2.economic_mpc_model import (
    EconomicMPCModelV1,
    _build_economic_mpc_model_from_prices,
)
from .rl.v2.economics import DayAheadTariffV1, EconomicAssumptionsV1
from .rl.v2.market import PriceForecastBinV1, PublicPriceForecastIssueV1
from .rl.v2.mpc_forecast import (
    MPCForecastBandV1,
    MPCForecastTrajectoryV1,
    public_forecast_issue_sha256,
)
from .rl.v2.mpc_model import MPCSparseRowV1
from .rl.v2.mpc_execution import public_observation_sha256
from .rl.v2.mpc_schema import MPCDeliveryTargetV1, ServiceMPCInputError, canonical_json
from .rl.v2.setpoint_command import issue_process_setpoint_command
from .rl.v2.setpoint_control import ProcessSetpointTargetsV1
from .rl.v2.setpoint_controllers import LegacyHeuristicSetpointControllerV1
from .rl.v2.weather import (
    FORECAST_WINDOWS_H,
    ForecastIssueV2,
    ForecastWindowV2,
    PublicForecastIssueV2,
    WeatherEpisodeV2,
)

HORIZON_HOURS = 240
PERFORMANCE_RATIO = 0.85
FORECAST_PROJECTION_VERSION = "site-fixed-snapshot-hourly-pv-v1"
FAULT_START_HOURS = 108
FAULT_DURATION_HOURS = 6


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _number(name: str, value: Any, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{name} must be between {lower:g} and {upper:g}")
    return number


def _scenario_horizon(scenario: dict[str, Any]) -> int:
    days = _number("duration_days", scenario.get("duration_days", 7), 1, 10)
    if not days.is_integer():
        raise ValueError("duration_days must be a whole number of days")
    return int(days) * 24


def scenario_solar_scale(
    scenario: dict[str, Any] | None, elapsed_s: float, *, forecast_only: bool = False
) -> float:
    """Scale relative to downloaded sunlight at an elapsed scenario time."""
    return math.prod(
        item["scale"] for item in (scenario or {}).get("sunlight", ())
        if item["start_hour"] <= elapsed_s / 3600 < item["end_hour"]
        and (not forecast_only or item["forecast_known"])
    )


def _utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Forecast timestamps must be ISO dates with a timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "Forecast timestamps must be ISO dates with a timezone"
        ) from exc
    if parsed.tzinfo is None:
        raise ValueError("Forecast timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SitePVProfile:
    """Hourly interval-average PV fractions, held constant within each hour."""

    forecast_id: str
    start_time: datetime
    fractions: tuple[float, ...]
    source_json: str
    site_json: str
    sunlight: tuple[tuple[float, float, float], ...] = ()

    @classmethod
    def from_forecast(
        cls, forecast: dict[str, Any], *, horizon_hours: int = HORIZON_HOURS
    ) -> SitePVProfile:
        hours = forecast.get("hours")
        if not isinstance(hours, list) or len(hours) < horizon_hours + 24:
            raise ValueError(
                f"The site forecast must contain at least {horizon_hours + 24} hourly intervals"
            )
        start = _utc(forecast.get("start_time"))
        if start.minute or start.second or start.microsecond:
            raise ValueError("The site forecast must start on a UTC hour boundary")
        fractions = []
        for index, hour in enumerate(hours):
            if not isinstance(hour, dict):
                raise ValueError("Each forecast hour must be an object")
            if _utc(hour.get("time")) != start + timedelta(hours=index):
                raise ValueError(
                    "Forecast intervals must be contiguous hourly UTC timestamps"
                )
            radiation = _number("shortwave_w_m2", hour.get("shortwave_w_m2"), 0, 2000)
            fractions.append(min(1.0, PERFORMANCE_RATIO * radiation / 1000.0))
        source = forecast.get("source")
        site = forecast.get("site")
        if not isinstance(source, dict) or not isinstance(site, dict):
            raise ValueError("Forecast source and site metadata are required")
        identifier = forecast.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("A forecast snapshot id is required")
        return cls(
            identifier,
            start,
            tuple(fractions),
            canonical_json(source),
            canonical_json(site),
        )

    def at(self, elapsed_s: float) -> float:
        index = int(math.floor(elapsed_s / 3600.0))
        if not 0 <= index < len(self.fractions):
            raise ValueError("Site solar profile does not cover this simulation time")
        scale = math.prod(
            scale for start, end, scale in self.sunlight if start <= elapsed_s / 3600 < end
        )
        return self.fractions[index] * scale

    def intervals(
        self, elapsed_s: float, interval_s: float = 900.0
    ) -> tuple[float, ...]:
        return tuple(
            self.at(elapsed_s + i * interval_s) for i in range(int(86400 / interval_s))
        )

    @property
    def source(self) -> dict[str, Any]:
        return dict(json.loads(self.source_json))


class SiteSolarArray(SolarArray):
    """PV forecast envelope replacing the inherited half-sine only for this run."""

    def __init__(self, config: Any, profile: SitePVProfile, origin_s: float) -> None:
        super().__init__(config)
        self.profile = profile
        self.origin_s = origin_s

    def clear_sky_kw(self, time_s: float) -> float:
        # PlantSimulator.reset briefly visits clock zero before scenario load.
        elapsed_s = max(0.0, time_s - self.origin_s)
        return self.config.peak_kw * self.profile.at(elapsed_s)


@dataclass(frozen=True)
class SiteForecastView(PublicForecastIssueV2):
    """Compatibility view whose provenance distinguishes slicing from issuance."""

    snapshot_id: str
    provider_issued_at: str | None
    retrieved_at: str | None
    valid_from: str

    def to_dict(self) -> dict[str, object]:
        result = super().to_dict()
        result.update(
            {
                "kind": "fixed_snapshot_projection_view",
                "snapshot_id": self.snapshot_id,
                "provider_forecast_issued_at": self.provider_issued_at,
                "retrieved_at": self.retrieved_at,
                "view_valid_from": self.valid_from,
                "issue_time_basis": "internal view boundary; provider issue time remains fixed",
                "uncertainty": "deterministic scenario; internal p10 equals mean and is not a calibrated quantile",
            }
        )
        return result


@dataclass(frozen=True)
class SiteForecastIssue(ForecastIssueV2):
    view: SiteForecastView

    def public(self) -> SiteForecastView:
        return self.view

    def to_dict(self) -> dict[str, object]:
        return self.view.to_dict()


@dataclass(frozen=True)
class SiteWeatherEpisode(WeatherEpisodeV2):
    snapshot_id: str
    source_json: str

    def private_audit_manifest(self) -> dict[str, object]:
        return {
            "generator": FORECAST_PROJECTION_VERSION,
            "snapshot_id": self.snapshot_id,
            "source": json.loads(self.source_json),
            "physics_timestep_s": self.dt_s,
            "physics_steps": self.physics_steps,
            "truth_sha256": self.truth_sha256,
            "forecasts_sha256": self.forecasts_sha256,
            "note": "Fixed downloaded snapshot; deterministic slices, no new weather issuances or random forecast innovations.",
        }


def _view(profile: SitePVProfile, boundary_step: int, dt_s: float) -> SiteForecastView:
    offset = boundary_step * dt_s
    values = profile.intervals(offset)
    windows = []
    for start, stop in FORECAST_WINDOWS_H:
        samples = values[int(start * 4) : int(stop * 4)]
        mean = math.fsum(samples) / len(samples)
        windows.append(
            ForecastWindowV2(
                f"{start:g}-{stop:g}h", start * 3600, stop * 3600, mean, mean
            )
        )
    source = profile.source
    return SiteForecastView(
        boundary_step,
        offset,
        tuple(windows),
        profile.forecast_id,
        source.get("forecast_issued_at"),
        source.get("retrieved_at"),
        _iso(profile.start_time + timedelta(seconds=offset)),
    )


class SiteEnvironment(PlantCyclicMarketEnvV2Beta):
    """A separate site scenario; synthetic benchmark construction is untouched."""

    def __init__(
        self,
        profile: SitePVProfile,
        *,
        solar_kw: float,
        battery_kwh: float,
        solar_scale: float,
        inject_fault: bool,
        scenario: dict[str, Any] | None = None,
    ) -> None:
        self.scenario_spec = deepcopy(scenario)
        spec = self.scenario_spec or {}
        self.horizon_hours = _scenario_horizon(spec) if scenario is not None else HORIZON_HOURS
        self.scenario_seed = int(_number("seed", spec.get("seed", 20260915), 0, 2147483647))
        if self.scenario_seed != spec.get("seed", self.scenario_seed):
            raise ValueError("seed must be an integer")
        self.sunlight = tuple(spec.get("sunlight", ()))
        for item in self.sunlight:
            start = _number("sunlight.start_hour", item.get("start_hour"), 0, self.horizon_hours)
            end = _number("sunlight.end_hour", item.get("end_hour"), 0, self.horizon_hours)
            _number("sunlight.scale", item.get("scale"), 0, 1)
            if start >= end or start % 1 or end % 1 or type(item.get("forecast_known")) is not bool:
                raise ValueError("Sunlight periods need increasing whole-hour boundaries and forecast_known")
        self.profile = replace(
            profile,
            sunlight=tuple(
                (item["start_hour"], item["end_hour"], item["scale"])
                for item in self.sunlight if item["forecast_known"]
            ),
        )
        self.source_profile = profile
        self.solar_scale = solar_scale
        self.inject_fault = inject_fault if scenario is None else bool(spec.get("faults"))
        base = PlantConfig()
        solar_kw = _number("solar_kw", spec.get("solar_kw", solar_kw), 100, 100000)
        battery_kwh = _number("battery_kwh", spec.get("battery_kwh", battery_kwh), 100, 200000)
        config = replace(
            base,
            solar=replace(base.solar, peak_kw=solar_kw),
            battery=replace(base.battery, capacity_kwh=battery_kwh),
        )
        episode = EpisodeConfigV2(
            horizon_h=self.horizon_hours,
            start_hour=float(profile.start_time.hour),
            delivery_rate_kg_h=25.0,
            battery_soc=(0.5, 0.5),
            h2_fill=(0.15, 0.15),
            co2_fill=(0.25, 0.25),
            ch4_coverage_h=(6.0, 6.0),
        )
        super().__init__(
            config=config,
            episode=episode,
            economic_assumptions=EconomicAssumptionsV1(
                tariff=DayAheadTariffV1(
                    id="site-offgrid-no-electricity-trade-v1",
                    hourly_eur_per_mwh=(0.0,) * 24,
                ),
                **spec.get("economics", {}),
            ),
            beta_task=CyclicMarketTaskConfigV1(
                disturbances=DisturbanceConfigV1(enabled=False)
            ),
        )
        solar = SiteSolarArray(config.solar, profile, episode.start_hour * 3600)
        self.simulator.solar = solar
        self.simulator.components["solar"] = solar

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        if seed is None and self.scenario_spec is not None:
            seed = self.scenario_seed
        return super().reset(seed=seed, options=options)

    @property
    def current_market_issue(self) -> PublicPriceForecastIssueV1:
        """Off-grid bookkeeping; no fabricated geographic power market."""
        index = min(self.decisions_taken, self.episode_decisions - 1)
        return PublicPriceForecastIssueV1(
            index,
            index * self.episode_config.decision_interval_s,
            0.0,
            tuple(PriceForecastBinV1(hour, 0.0, 0.0, 0.0) for hour in range(24)),
            "site-offgrid-no-electricity-trade-v1",
        )

    def _opportunity_price_eur_per_mwh(
        self, *, time_h: float, decision_index: int
    ) -> float:
        return 0.0

    def _economic_price_context(
        self, *, time_h: float, decision_index: int
    ) -> dict[str, Any]:
        return self.current_market_issue.to_dict()

    def _sample_episode(self, seed: int) -> EpisodeV2:
        base = super()._sample_episode(seed)
        tape = SolarFactorTape(
            self.config.dt_s,
            tuple(
                # Tape boundary k+1 drives physical interval k, whose PV
                # envelope samples its opening boundary rather than its end.
                self.solar_scale * scenario_solar_scale(
                    self.scenario_spec, max(0, step - 1) * self.config.dt_s
                )
                for step in range(base.physics_steps + 1)
            ),
        )
        views = tuple(
            _view(self.profile, step, self.config.dt_s)
            for step in range(0, base.physics_steps, base.decision_stride)
        )
        issues = tuple(
            SiteForecastIssue(
                view.issued_at_physics_boundary_step,
                view.issued_at_offset_s,
                view.windows,
                _digest(view.to_dict()),
                view,
            )
            for view in views
        )
        truth_sha = _digest(
            {
                "profile": self.profile.fractions,
                "scale": self.solar_scale,
                **({"sunlight": self.sunlight} if self.scenario_spec is not None else {}),
            }
        )
        forecasts_sha = _digest([view.to_dict() for view in views])
        values = {
            item.name: getattr(base.weather, item.name)
            for item in fields(WeatherEpisodeV2)
        }
        values.update(
            tape=tape,
            issues=issues,
            truth_origin_kind="external_tape",
            clear_sky_envelope_id=FORECAST_PROJECTION_VERSION,
            clear_sky_envelope_sha256=_digest(self.profile.fractions),
            truth_sha256=truth_sha,
            forecasts_sha256=forecasts_sha,
            episode_sha256=_digest(
                [self.profile.forecast_id, truth_sha, forecasts_sha]
            ),
        )
        weather = SiteWeatherEpisode(
            **values,
            snapshot_id=self.profile.forecast_id,
            source_json=self.profile.source_json,
        )
        faults = (
            (
                FaultSpec(
                    type="dac_outage",
                    start_s=FAULT_START_HOURS * 3600,
                    duration_s=FAULT_DURATION_HOURS * 3600,
                ),
            )
            if self.inject_fault
            else ()
        )
        scenario = replace(
            base.scenario,
            id="site_forecast_projection_v1",
            name="Ten-day site forecast projection",
            description="Downloaded hourly radiation, explicit PV conversion and optional DAC outage.",
            faults=faults,
        )
        commitments = base.commitments
        if self.scenario_spec is not None:
            spec = self.scenario_spec
            initial = dict(base.scenario.initial.to_dict())
            for name, maximum in (
                ("battery_soc", 1), ("h2_kg", self.config.h2_tank.capacity_kg),
                ("co2_kg", self.config.co2_tank.capacity_kg), ("ch4_kg", self.config.ch4_tank.capacity_kg),
            ):
                initial[name] = _number(name, spec.get("initial", {}).get(name, initial[name]), 0, maximum)
            explicit_faults = []
            for item in spec.get("faults", []):
                start = _number("fault.start_hour", item.get("start_hour"), 0, self.horizon_hours)
                duration = item.get("duration_hours")
                if duration is not None:
                    duration = _number("fault.duration_hours", duration, 0.25, self.horizon_hours)
                if start >= self.horizon_hours or start * 4 % 1 or (duration is not None and duration * 4 % 1):
                    raise ValueError("Fault timing must align to 15-minute boundaries within the scenario")
                explicit_faults.append(FaultSpec.parse({
                    "type": item.get("type"), "severity": item.get("severity"),
                    "start_s": start * 3600,
                    "duration_s": None if duration is None else duration * 3600,
                    "channel": item.get("channel"),
                }))
            commitments_list = []
            for index, item in enumerate(spec.get("deliveries", [])):
                due = _number("delivery.due_hour", item.get("due_hour"), 0.25, self.horizon_hours)
                # A contract can exceed storage capacity: that is an explicit
                # stress scenario whose deficit is settled as unmet demand.
                amount = _number("delivery.kg", item.get("kg"), 0, 100000)
                if amount <= 0:
                    raise ValueError("delivery.kg must be positive")
                if due * 4 % 1:
                    raise ValueError("Delivery times must align to 15-minute boundaries")
                commitments_list.append(DeliveryCommitment(
                    id=f"scenario-delivery-{index + 1:04d}",
                    due_step=round(due * 3600 / self.config.dt_s), requested_kg=amount,
                ))
            commitments = tuple(sorted(commitments_list, key=lambda item: item.due_step))
            if len({item.due_step for item in commitments}) != len(commitments):
                raise ValueError("Scenario delivery times must be distinct")
            scenario = replace(
                scenario, id="authored_site_scenario_v1", name=spec.get("name", "Authored scenario"),
                description=spec.get("description", ""), initial=InitialState(**initial),
                faults=tuple(explicit_faults),
            )
        return replace(base, scenario=scenario, weather=weather, commitments=commitments)


@dataclass(frozen=True, slots=True)
class SiteForecastTrajectory(MPCForecastTrajectoryV1):
    """Exact held-hourly forecast shape; envelope slots mean nominal scenario PV."""

    def _validate_projection_version(self) -> None:
        if self.projection_version != FORECAST_PROJECTION_VERSION:
            raise ValueError("Unsupported site forecast projection")

    def body(self) -> dict[str, Any]:
        return {
            **MPCForecastTrajectoryV1.body(self),
            "envelope_semantics": "nominal downloaded PV projection, not physical clear-sky geometry",
        }


def _trajectory(
    profile: SitePVProfile,
    forecast: PublicForecastIssueV2,
    config: PlantConfig,
    boundary_time_s: float,
) -> SiteForecastTrajectory:
    values = profile.intervals(forecast.issued_at_offset_s)
    bands = []
    names = []
    for start, stop in FORECAST_WINDOWS_H:
        first, end = int(start * 4), int(stop * 4)
        mean = math.fsum(values[first:end]) / (end - first)
        name = f"{start:g}-{stop:g}h"
        bands.append(
            MPCForecastBandV1(
                name,
                start * 3600,
                stop * 3600,
                first,
                end - first,
                mean,
                mean,
                1.0,
                mean,
                0.0,
            )
        )
        names.extend([name] * (end - first))
    return SiteForecastTrajectory(
        FORECAST_PROJECTION_VERSION,
        public_forecast_issue_sha256(forecast),
        forecast.issued_at_physics_boundary_step,
        forecast.issued_at_offset_s,
        boundary_time_s,
        900.0,
        config.solar.peak_kw,
        values,
        values,
        tuple(value * config.solar.peak_kw for value in values),
        tuple(names),
        tuple(bands),
    )


class SiteOperationalMPC(OperationalMPCController):
    """Existing operating MILP and command logic with explicit site PV input."""

    id = "site-forecast-operating-mpc-v1"

    def __init__(self, environment: SiteEnvironment) -> None:
        self.profile = environment.profile
        self.environment = environment
        super().__init__(
            horizon_h=environment.episode_config.horizon_h,
            interval_s=900.0,
            assumptions=environment.economic_assumptions,
            beta_task=environment.beta_task,
        )
        self.settings = replace(self.settings, solver_time_limit_s=2.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "id": self.id,
            "forecast_projection_version": FORECAST_PROJECTION_VERSION,
            "replanning_interval_hours": 3,
            "replanning_events": (
                "newly observed component health change"
                if self.environment.scenario_spec is not None
                else "newly observed DAC outage or recovery"
            ),
            "outage_forecast": "observed DAC outage persists through planning horizon until recovery is observed",
            "weather_source": "fixed downloaded forecast snapshot",
            "electricity": "off-grid; no export or imported electricity; zero marginal solar tariff",
        }

    def reset(self) -> None:
        super().reset()
        self._cached_plan: Any = None
        self._planned_at = -12
        self._observed_dac_offline: bool | None = None
        self._observed_health: tuple[Any, ...] | None = None
        self.solve_count = 0
        self._cached_operation: dict[str, Any] = {}

    def compute_command(
        self,
        observation: Any,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: Any = None,
    ) -> OperatingDecision:
        offline = observation.plant.state.boundary.health["dac"].status == "offline"
        health = tuple(
            (name, item.offline, item.degradation)
            for name, item in sorted(observation.plant.state.boundary.health.items())
        )
        if (
            self._next_decision - self._planned_at >= 12
            or offline != self._observed_dac_offline
            or (self.environment.scenario_spec is not None and health != self._observed_health)
        ):
            at = self._next_decision
            try:
                decision = super().compute_command(
                    observation,
                    constraints=constraints,
                    config=config,
                    commitment_profile=commitment_profile,
                )
            except ServiceMPCInputError as exc:
                if self.environment.scenario_spec is None:
                    raise
                # Fault physics remain authoritative. A planner that cannot
                # represent observed component degradation must say so and let
                # the common safety dispatcher execute the reactive policy.
                decision = None
                self._cached_plan = None
                self._cached_operation = {
                    "policy_id": self.id,
                    "mode": "fallback",
                    "issued_at_hours": at / 4,
                    "horizon_hours": min(24, (self.settings.task_interval_count - at) / 4),
                    "resources": [], "deliveries": [],
                    "projected_shortfall_kg": None,
                    "fallback_controller": LegacyHeuristicSetpointControllerV1.id,
                    "solver_failure": {"reason": "unsupported_observed_state", "detail": str(exc)},
                }
            self._planned_at = at
            self._observed_dac_offline = offline
            self._observed_health = health
            self.solve_count += 1
            if decision is not None:
                self._cached_plan = decision.evidence.plan
                self._cached_operation = decision.evidence.operation()
                return decision
        reference = self._validate(observation, config)
        invalid = False
        if self._cached_plan is None:
            fallback = LegacyHeuristicSetpointControllerV1().compute_command(
                observation.plant,
                constraints=constraints,
                config=config,
                commitment_profile=commitment_profile,
            )
            targets = fallback.command.targets
            invalid = fallback.invalid
            notes = (
                "Prior site plan failed; guarded heuristic fallback until the next three-hour replan or observed fault change.",
                *(
                    (self._cached_operation["solver_failure"]["detail"],)
                    if (self._cached_operation.get("solver_failure") or {}).get("detail")
                    else ()
                ),
            )
        else:
            step = self._cached_plan.steps[self._next_decision - self._planned_at]
            targets = ProcessSetpointTargetsV1(
                electrolyser_target_kw=step.electrolyser_startup_kw
                + step.electrolyser_productive_kw,
                dac_target_kw=step.solar_to_dac_kw,
                methanator_target_kw=step.solar_to_methanator_kw,
                battery_charge_ceiling_kw=step.solar_to_battery_kw,
            )
            notes = (
                "Next 15-minute row of the last site plan; the common dispatcher checks physical state every minute.",
            )
        targets, _ = self._continue_active_startup(
            targets,
            observation,
            constraints=constraints,
            config=config,
            commitment_profile=commitment_profile,
        )
        operation = deepcopy(self._cached_operation)
        operation.update(
            mode="fallback" if self._cached_plan is None else "scheduled",
            planned_at_decision=self._planned_at,
            planned_at_hours=self._planned_at / 4,
            executing_at_hours=self._next_decision / 4,
        )
        state_fields = {
            "battery": "battery_energy_kwh",
            "h2": "h2_level_kg",
            "co2": "co2_level_kg",
            "ch4": "ch4_level_kg",
        }
        for resource in operation.get("resources", []):
            resource["current"] = float(
                getattr(observation.plant.state.boundary, state_fields[resource["id"]])
            )
        evidence = OperatingEvidence(
            self._cached_plan,
            notes,
            canonical_json(
                {
                    "operation": operation,
                    "targets": targets.to_dict(),
                    "observation_sha256": observation.sha256,
                }
            ),
        )
        command = issue_process_setpoint_command(
            targets,
            decision_index=self._next_decision,
            observation_sha256=public_observation_sha256(observation.plant),
            hold_duration_s=900.0,
            issuer_id=self.id,
            source_schema="site-mpc-scheduled-command-v1",
            source_sha256=evidence.sha256,
            constraints=constraints,
            config=config,
        )
        self._reference = reference
        self._next_decision += 1
        self._events.append(evidence)
        return OperatingDecision(evidence, command, invalid)

    def build_model(
        self,
        observation: Any,
        *,
        constraints: Constraints,
        config: PlantConfig,
        commitment_profile: Any,
    ) -> EconomicMPCModelV1:
        reference = self._validate(observation, config)
        horizon = min(
            self.settings.maximum_planning_intervals,
            self.settings.task_interval_count - self._next_decision,
        )
        forecast = observation.plant.forecast
        weather = _trajectory(
            self.profile, forecast, config, observation.plant.state.boundary.time_s
        )
        initial_methane = reference[3] + min(
            reference[1] / config.methanator.h2_kg_per_kg_ch4,
            reference[2] / config.methanator.co2_kg_per_kg_ch4,
        )
        delivery_targets = None
        if self.environment.scenario_spec is not None:
            current_step = observation.plant.state.boundary.step
            horizon_step = current_step + round(horizon * self.settings.interval_s / config.dt_s)
            commitments = self.environment._episode.commitments
            upcoming = tuple(item for item in commitments if current_step < item.due_step <= horizon_step)
            delivery_targets = tuple(
                MPCDeliveryTargetV1(
                    ordinal=index,
                    due_boundary=round((item.due_step - current_step) * config.dt_s / self.settings.interval_s),
                    due_offset_s=(item.due_step - current_step) * config.dt_s,
                    requested_kg=item.requested_kg,
                    source="scenario_schedule",
                )
                for index, item in enumerate(upcoming)
            )
        model = _build_economic_mpc_model_from_prices(
            observation.plant,
            weather,
            settings=self.settings,
            assumptions=self.assumptions,
            constraints=constraints,
            config=config,
            commitment_profile=commitment_profile,
            horizon_intervals=horizon,
            initial_ch4_equivalent_kg=initial_methane,
            initial_battery_energy_kwh=reference[0],
            interval_prices_eur_per_mwh=(0.0,) * horizon,
            price_source_id="site-offgrid-no-electricity-trade-v1",
            causal_current_dac_health=True,
            mandatory_deliveries=True,
            delivery_targets=delivery_targets,
        )
        groups = tuple(
            model.group(name)
            for name in (
                "solar_to_battery_kw",
                "solar_to_electrolyser_kw",
                "solar_to_dac_kw",
                "solar_to_methanator_kw",
            )
        )
        rows = tuple(
            MPCSparseRowV1(
                name=f"site_constant_solar_power[{i}]",
                lower=None,
                upper=weather.available_kw[i],
                coefficients=tuple(sorted((group[i], 1.0) for group in groups)),
            )
            for i in range(horizon)
        )
        # The inherited per-decision planner releases an observed DAC outage
        # after its first interval. A cached three-hour command sequence must
        # not knowingly allocate power to that still-observed offline unit.
        # Persist the observed outage; only a later measurement can clear it.
        dac_offline = observation.plant.state.boundary.health["dac"].status == "offline"
        if dac_offline:
            rows += tuple(
                MPCSparseRowV1(
                    name=f"site_observed_dac_unavailable[{i}]",
                    lower=None,
                    upper=0.0,
                    coefficients=((index, 1.0),),
                )
                for i, index in enumerate(model.group("solar_to_dac_kw"))
            )
        metadata = model.metadata()
        metadata["site_projection"] = {
            "profile_id": self.profile.forecast_id,
            "forecast_source": self.profile.source,
            "forecast_projection_version": FORECAST_PROJECTION_VERSION,
            "dac_outage_assumption": (
                "Observed outage persists for the full planning horizon until recovery is observed."
                if dac_offline
                else "DAC currently available; no future outage schedule provided."
            ),
        }
        return replace(
            model,
            physical_model=replace(
                model.physical_model, rows=(*model.physical_model.rows, *rows)
            ),
            metadata_json=canonical_json(metadata),
        )


def _limiting(state: Any, config: PlantConfig) -> str:
    if not state.dac_online and state.health["dac"].status == "offline":
        return "DAC outage"
    if state.solar_available_kw <= 1.0:
        return "Solar availability"
    if state.h2_level_kg < config.methanator.h2_kg_per_kg_ch4:
        return "Hydrogen supply"
    if state.co2_level_kg < config.methanator.co2_kg_per_kg_ch4:
        return "CO2 supply"
    if state.ch4_fill >= 0.98:
        return "Methane storage"
    if state.solar_curtailed_kw > 1.0 and state.battery_soc >= 0.98:
        return "Battery storage"
    loads = {
        "Electrolyser": state.electrolyser_load_frac,
        "Direct air capture": state.dac_load_frac,
        "Methanator": state.methanator_load_frac,
    }
    most = max(loads, key=lambda key: loads[key])
    return most if loads[most] >= 0.95 else "Solar / dispatch"


def create_site_simulation(
    forecast: dict[str, Any],
    *,
    solar_kw: float = 5000,
    battery_kwh: float = 8000,
    controller: str = "MPC",
    inject_fault: bool = True,
    solar_scale: float = 1.0,
    scenario: dict[str, Any] | None = None,
) -> tuple[SiteEnvironment, Any]:
    """Construct the identical initial plant/controller for live and batch runs."""
    solar_kw = _number("solar_kw", solar_kw, 100, 100000)
    battery_kwh = _number("battery_kwh", battery_kwh, 100, 200000)
    solar_scale = _number("solar_scale", solar_scale, 0.1, 1.0)
    if controller not in {"H", "MPC"}:
        raise ValueError("controller must be H or MPC")
    if type(inject_fault) is not bool:
        raise ValueError("inject_fault must be a boolean")
    if controller == "MPC":
        try:
            importlib.import_module("scipy.optimize")
        except ImportError as exc:
            raise RuntimeError(
                "Site MPC requires the optional scipy optimisation runtime"
            ) from exc
    profile = SitePVProfile.from_forecast(
        forecast, horizon_hours=HORIZON_HOURS if scenario is None else _scenario_horizon(scenario)
    )
    environment = SiteEnvironment(
        profile,
        solar_kw=solar_kw,
        battery_kwh=battery_kwh,
        solar_scale=solar_scale,
        inject_fault=inject_fault,
        scenario=scenario,
    )
    try:
        environment.reset(seed=environment.scenario_seed)
        policy: Any = (
            SiteOperationalMPC(environment)
            if controller == "MPC"
            else LegacyHeuristicSetpointControllerV1()
        )
        policy.reset()
        return environment, policy
    except Exception:
        environment.close()
        raise


def run_site_simulation(
    forecast: dict[str, Any],
    *,
    solar_kw: float = 5000,
    battery_kwh: float = 8000,
    controller: str = "MPC",
    inject_fault: bool = True,
    solar_scale: float = 1.0,
    progress_callback: Callable[[float], None] | None = None,
    scenario: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a complete site projection. All histories are hourly aggregates."""
    started = time.perf_counter()
    environment, policy = create_site_simulation(
        forecast,
        solar_kw=solar_kw,
        battery_kwh=battery_kwh,
        controller=controller,
        inject_fault=inject_fault,
        solar_scale=solar_scale,
        scenario=scenario,
    )
    profile = environment.profile
    horizon_hours = environment.horizon_hours
    solar_kw = environment.config.solar.peak_kw
    battery_kwh = environment.config.battery.capacity_kwh
    try:
        state = environment.telemetry
        initial = state.to_dict()
        history: list[dict[str, Any]] = []
        limiting: Counter[str] = Counter()
        energy = {"electrolyser": 0.0, "dac": 0.0, "methanator": 0.0}
        minimum_soc = state.battery_soc
        fallback_count = 0
        hour_states: list[Any] = []

        def remember(elapsed_h: float) -> None:
            nonlocal hour_states
            samples = hour_states or [environment.telemetry]
            current = environment.telemetry

            def mean(field: str) -> float:
                return math.fsum(getattr(item, field) for item in samples) / len(
                    samples
                )

            interval_start = max(0.0, elapsed_h - (1.0 if hour_states else 0.0))
            label = _limiting(current, environment.config)
            history.append(
                {
                    "time": _iso(profile.start_time + timedelta(hours=elapsed_h)),
                    "elapsed_hours": elapsed_h,
                    "battery_soc": current.battery_soc,
                    "solar_kw": mean("solar_available_kw"),
                    "forecast_solar_kw": profile.at(interval_start * 3600) * solar_kw,
                    "methane_kg_h": mean("ch4_production_kg_h"),
                    "methane_total_kg": current.ch4_produced_total_kg,
                    "curtailed_kw": mean("solar_curtailed_kw"),
                    "electrolyser_kw": mean("electrolyser_power_kw"),
                    "dac_kw": mean("dac_power_kw"),
                    "methanator_kw": mean("methanator_power_kw"),
                    "limiting_subsystem": label,
                    "fault_active": any(
                        health.offline or health.degradation > 0
                        for item in samples for health in item.health.values()
                    ),
                }
            )
            hour_states = []

        remember(0.0)
        if progress_callback:
            progress_callback(0.0)
        while environment.decisions_taken < environment.episode_decisions:
            observation = environment.cyclic_market_controller_observation
            source = observation if controller == "MPC" else observation.plant
            decision = policy.compute_command(
                source,
                constraints=environment.constraints,
                config=environment.config,
                commitment_profile=environment.episode_config.electrolyser_commitment,
            )
            if controller == "MPC" and decision.evidence.plan is None:
                fallback_count += 1
            _, _, terminated, truncated, _ = environment.step_process_setpoints(
                decision.command, controller_invalid=getattr(decision, "invalid", False)
            )
            # The first retained sample is the opening boundary, not another minute.
            for sample in (environment.last_minute_truth_states or ())[1:]:
                hour_states.append(sample)
                minimum_soc = min(minimum_soc, sample.battery_soc)
                limiting[_limiting(sample, environment.config)] += 1
                for name in energy:
                    energy[name] += (
                        getattr(sample, f"{name}_power_kw") * environment.config.dt_h
                    )
            if environment.decisions_taken % 4 == 0:
                remember(environment.decisions_taken / 4.0)
                if progress_callback:
                    progress_callback(
                        environment.decisions_taken / environment.episode_decisions
                    )
            if terminated or truncated:
                break
        state = environment.telemetry
        if environment.decisions_taken != environment.episode_decisions:
            raise RuntimeError("Site simulation stopped before completing the scenario horizon")
        delivery = environment.delivery_observation
        requested = delivery.cumulative_requested_kg
        delivered = delivery.cumulative_delivered_kg
        metrics = {
            "methane_produced_kg": state.ch4_produced_total_kg,
            "solar_energy_kwh": state.energy_produced_kwh,
            "curtailed_energy_kwh": state.energy_curtailed_kwh,
            "curtailment_pct": (
                100 * state.energy_curtailed_kwh / state.energy_produced_kwh
                if state.energy_produced_kwh
                else 0.0
            ),
            "delivery_requested_kg": requested,
            "delivery_delivered_kg": delivered,
            "delivery_service_pct": 100 * delivered / requested if requested else 100.0,
            "battery_final_soc": state.battery_soc,
            "battery_min_soc": minimum_soc,
            "utilisation": {
                name: value
                / (horizon_hours * getattr(environment.config, name).rated_kw)
                for name, value in energy.items()
            },
            "limiting_subsystem": (
                limiting.most_common(1)[0][0] if limiting else "Unknown"
            ),
        }
        return {
            "schema": "site-simulation-v1",
            "forecast_id": profile.forecast_id,
            "site": json.loads(profile.site_json),
            "source": profile.source,
            "start_time": _iso(profile.start_time),
            "end_time": _iso(profile.start_time + timedelta(hours=horizon_hours)),
            "horizon_hours": horizon_hours,
            "controller": {
                "id": policy.id,
                "label": (
                    "Site forecast MPC" if controller == "MPC" else "Reactive heuristic"
                ),
                "uses_forecast": controller == "MPC",
                "planning_horizon_hours": 24 if controller == "MPC" else None,
                "replanning_interval_hours": 3 if controller == "MPC" else None,
                "solve_count": getattr(policy, "solve_count", 0),
                "fallback_count": fallback_count,
            },
            "parameters": {
                "solar_kw": solar_kw,
                "battery_kwh": battery_kwh,
                "solar_scale": solar_scale,
                "inject_fault": environment.inject_fault,
            },
            "metrics": metrics,
            **({
                "scenario": deepcopy(environment.scenario_spec),
                "economic_assumptions": environment.economic_assumptions.to_dict(),
                "economic_accounting": environment.economic_metrics.to_dict(),
                "delivery_schedule": [item.to_dict() for item in environment._episode.commitments],
                "faults": [item.to_dict() for item in environment._episode.scenario.faults],
            } if scenario is not None else {}),
            "history": history,
            "fault": ({
                "enabled": inject_fault,
                "type": "dac_outage",
                "start_hours": FAULT_START_HOURS,
                "duration_hours": FAULT_DURATION_HOURS,
                "note": "DAC power and capture stop for six hours on day five; existing CO2 storage may sustain methane. The controller learns the outage only when it occurs.",
            } if scenario is None else {"enabled": bool(environment._episode.scenario.faults), "type": "scenario_schedule"}),
            "initial_state": initial,
            "assumptions": [
                "Forecast-based projection, not a prediction of measured plant output. Nominal weather equals the fixed downloaded forecast; the stress multiplier changes realization only.",
                "Horizontal PV approximation: AC kW = nameplate kW × 0.85 × shortwave W/m² / 1000, capped at nameplate. No panel tilt, terrain shading, snow or module-temperature model.",
                "Hourly interval-average radiation is held constant for sixty one-minute physics steps; this creates no measured minute-scale cloud variation.",
                "All weather views retain the original provider issue time (or unknown if absent). Views do not become new weather issuances as simulation time advances; deterministic internal quantile slots are not calibrated uncertainty.",
                "Existing plant efficiencies, buffers, electrolyser commitment and common minute safety dispatcher are reused. Temperature, humidity, wind and elevation are context only.",
                "Off-grid electricity: no imports or exports. MPC applies zero marginal solar tariff and inherited illustrative equipment-wear and inventory-recovery weights; no commercial savings claim.",
                (
                    "Fixed starting stocks: battery 50%, H2 300 kg, CO2 1500 kg, CH4 150 kg. Delivery target is 25 kg/h settled every six hours; production excludes opening stocks."
                    if scenario is None else
                    "Initial stocks, delivery deadlines, sunlight windows and equipment events follow the reviewed scenario. Known sunlight changes also enter the forecast; unexpected changes affect physical realization only. Production excludes opening stocks."
                ),
                f"Utilisation is integrated electrical load divided by rated power × {horizon_hours} hours. Limiting subsystem is a descriptive state rule, not a proven optimal bottleneck.",
                "MPC plans over 24 hours every three simulated hours and on newly observed DAC outages/recoveries. It issues successive 15-minute plan rows through the common minute guard. Typed solve/plan failures explicitly use the reactive heuristic until the next replan; fallback intervals are counted.",
                "When the DAC is observed offline, MPC assumes it remains unavailable throughout the planning horizon until recovery is observed. It does not know the injected outage duration.",
            ],
            "runtime_seconds": time.perf_counter() - started,
        }
    finally:
        environment.close()
