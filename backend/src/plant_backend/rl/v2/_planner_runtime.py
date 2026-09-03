"""Plant-specific runtime primitives for the frozen feasible planner.

This privileged module is intentionally separate from causal controllers.  It owns
exact-environment lifetime and resource accounting, while the generic search module
owns ordering and pruning.  Full rollout behavior is layered onto these primitives in
subsequent audited slices.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
import platform
import resource
import struct
import time

from plant_sim import (
    ElectrolyserCommitmentSnapshot,
    PlantConfig,
    PlantState,
    SolarArray,
)

from . import spaces
from ._planner_actions import (
    RelaxationBindingV1,
    action_record_to_action,
    require_attested_binding,
)
from ._planner_search import IntermediateRankV1, PlannerSearchCountersV1
from ._planner_state import accumulate_battery_throughput, extract_final_score
from .boundary import ControlStateV2, IntervalSummaryV2
from .config import EpisodeConfigV2, EpisodeV2, sample_episode
from .contract import contract_digest, describe_checkpoint_contract
from .env import PlantEnvV2
from .metrics import EpisodeMetricsV2
from .planner_schema import (
    A0_CONTROLLER_SHA256,
    FAILURE_CODES,
    FAILURE_PHASES,
    ActionRecordV1,
    CanonicalPayloadV1,
    FinalScoreV1,
    PlannerSettingsV1,
    action_record_trace_payload,
    canonical_json,
    canonical_payload,
    outcome_payload,
    raw_canonical_sha256,
    request_trace_payload,
    require_planner_admission,
    require_sha256,
    revalidate_planner_settings,
    source_trace_payload,
)
from .reward import OBJECTIVE_VERSION, RewardWeightsV2

from ...controllers.base import Constraints


ClockNs = Callable[[], int]
PeakRssRaw = Callable[[], int | float]
PlatformSystem = Callable[[], str]


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum:g}")
    return result


def _bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _payload(name: str, value: object, *, domain: str) -> CanonicalPayloadV1:
    if type(value) is not CanonicalPayloadV1:
        raise TypeError(f"{name} must be an exact CanonicalPayloadV1")
    checked = CanonicalPayloadV1(value.domain, value.canonical_json, value.sha256)
    if checked.domain != domain:
        raise ValueError(f"{name} must use domain {domain!r}")
    return checked


def _binary64_equal(left: object, right: object) -> bool:
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, Real)
        or not isinstance(right, Real)
    ):
        return False
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(left_value) or not math.isfinite(right_value):
        return False
    return struct.pack(">d", left_value) == struct.pack(">d", right_value)


@dataclass(frozen=True, slots=True)
class PlannerRuntimeAdmissionV1:
    """Detached, pre-rollout identity shared by every fresh planner reset."""

    settings: PlannerSettingsV1
    binding: RelaxationBindingV1
    seed: int
    evidence_scope: str
    canonical_episode_verified: bool
    episode_manifest_json: str
    episode_manifest_sha256: str
    exogenous_episode_sha256: str
    checkpoint_contract_json: str
    checkpoint_contract_digest: str
    manifest_json: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        checked = require_planner_admission(
            self.settings,
            seed=self.seed,
            evidence_scope=self.evidence_scope,
            canonical_episode_verified=self.canonical_episode_verified,
        )
        object.__setattr__(self, "settings", checked)
        binding = require_attested_binding(self.binding)
        if binding.binding_scope != self.evidence_scope:
            raise ValueError("relaxation binding scope disagrees with runtime scope")
        object.__setattr__(self, "binding", binding)
        expected_binding_hashes = (
            (binding.model_sha256, checked.relaxation_model_sha256, "model"),
            (binding.result_sha256, checked.relaxation_result_sha256, "result"),
            (binding.witness_sha256, checked.relaxation_witness_sha256, "witness"),
            (
                binding.guidance_sha256,
                checked.relaxation_guidance_sha256,
                "guidance",
            ),
        )
        for actual, expected, name in expected_binding_hashes:
            if actual != expected:
                raise ValueError(f"relaxation {name} identity disagrees with settings")
        episode_json, episode_value = _canonical_json_object(
            "episode_manifest_json", self.episode_manifest_json
        )
        episode_sha256 = require_sha256(
            "episode_manifest_sha256", self.episode_manifest_sha256
        )
        if hashlib.sha256(episode_json.encode("utf-8")).hexdigest() != episode_sha256:
            raise ValueError("episode manifest digest disagrees with its bytes")
        exogenous = require_sha256(
            "exogenous_episode_sha256", self.exogenous_episode_sha256
        )
        if episode_value.get("exogenous_episode_sha256") != exogenous:
            raise ValueError("episode manifest disagrees with exogenous identity")
        if (
            binding.episode_manifest_json != episode_json
            or binding.episode_manifest_sha256 != episode_sha256
            or binding.exogenous_episode_sha256 != exogenous
        ):
            raise ValueError(
                "runtime episode identity disagrees with relaxation binding"
            )
        checkpoint_json, checkpoint_value = _canonical_json_object(
            "checkpoint_contract_json", self.checkpoint_contract_json
        )
        checkpoint_digest = require_sha256(
            "checkpoint_contract_digest", self.checkpoint_contract_digest
        )
        if checkpoint_value.get("checkpoint_contract_digest") != checkpoint_digest:
            raise ValueError("checkpoint contract JSON disagrees with its digest")
        checkpoint_preimage = dict(checkpoint_value)
        checkpoint_preimage.pop("checkpoint_contract_digest")
        if contract_digest(checkpoint_preimage) != checkpoint_digest:
            raise ValueError("checkpoint contract digest disagrees with its payload")
        if (
            checkpoint_digest != checked.checkpoint_contract_digest
            or checkpoint_digest != binding.checkpoint_contract_digest
        ):
            raise ValueError("runtime checkpoint identity is not fully cross-bound")
        manifest_json, manifest_value = _canonical_json_object(
            "manifest_json", self.manifest_json
        )
        manifest_sha256 = require_sha256("manifest_sha256", self.manifest_sha256)
        if hashlib.sha256(manifest_json.encode("utf-8")).hexdigest() != manifest_sha256:
            raise ValueError("environment manifest digest disagrees with its bytes")
        expected_manifest_fields = {
            "manifest_schema_version",
            "environment",
            "checkpoint_contract",
            "objective_version",
            "episode_config",
            "plant_config",
            "constraints",
            "reward_weights",
            "episode",
            "provenance_gaps",
        }
        if set(manifest_value) != expected_manifest_fields:
            raise ValueError("environment manifest has an unexpected field set")
        if (
            manifest_value["manifest_schema_version"] != 3
            or canonical_json(manifest_value["environment"])
            != canonical_json(spaces.describe())
            or manifest_value["objective_version"] != OBJECTIVE_VERSION
            or canonical_json(manifest_value["checkpoint_contract"]) != checkpoint_json
            or canonical_json(manifest_value["episode"]) != episode_json
            or manifest_value["provenance_gaps"]
            != [
                "source revision and dependency-lock digest must be added by the experiment runner"
            ]
        ):
            raise ValueError(
                "environment manifest is not cross-bound to runtime inputs"
            )
        for field_name, expected_sha256 in (
            ("plant_config", checked.plant_config_sha256),
            ("episode_config", checked.episode_config_sha256),
            ("constraints", checked.constraints_sha256),
            ("reward_weights", checked.reward_weights_sha256),
        ):
            if raw_canonical_sha256(manifest_value[field_name]) != expected_sha256:
                raise ValueError(
                    f"environment manifest {field_name} disagrees with settings"
                )

    def validate(self) -> None:
        """Re-run admission invariants after any low-level object mutation."""

        PlannerRuntimeAdmissionV1(
            settings=self.settings,
            binding=self.binding,
            seed=self.seed,
            evidence_scope=self.evidence_scope,
            canonical_episode_verified=self.canonical_episode_verified,
            episode_manifest_json=self.episode_manifest_json,
            episode_manifest_sha256=self.episode_manifest_sha256,
            exogenous_episode_sha256=self.exogenous_episode_sha256,
            checkpoint_contract_json=self.checkpoint_contract_json,
            checkpoint_contract_digest=self.checkpoint_contract_digest,
            manifest_json=self.manifest_json,
            manifest_sha256=self.manifest_sha256,
        )


def _canonical_json_object(name: str, value: object) -> tuple[str, dict[str, object]]:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be canonical JSON text")
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain an object")
    if canonical_json(parsed) != value:
        raise ValueError(f"{name} is not canonical JSON")
    return value, parsed


def _validate_constraints(constraints: Constraints) -> None:
    for name in (
        "electrolyser_max_kw",
        "dac_max_kw",
        "methanator_max_kw",
        "battery_support_max_kw",
    ):
        value = getattr(constraints, name)
        if value is not None:
            _finite(f"constraints.{name}", value, minimum=0.0)
    reserve = _finite(
        "constraints.battery_min_soc", constraints.battery_min_soc, minimum=0.0
    )
    if reserve > 0.95:
        raise ValueError("constraints.battery_min_soc must be <= 0.95")


def admit_planner_runtime(
    *,
    config: PlantConfig,
    episode_config: EpisodeConfigV2,
    constraints: Constraints,
    weights: RewardWeightsV2,
    settings: PlannerSettingsV1,
    binding: RelaxationBindingV1,
    seed: int,
    evidence_scope: str,
    canonical_episode_verified: bool,
) -> PlannerRuntimeAdmissionV1:
    """Bind static inputs, Part-I evidence, and deterministic episode before reset."""

    exact_inputs = (
        ("config", config, PlantConfig),
        ("episode_config", episode_config, EpisodeConfigV2),
        ("constraints", constraints, Constraints),
        ("weights", weights, RewardWeightsV2),
    )
    for name, value, expected_type in exact_inputs:
        if type(value) is not expected_type:
            raise TypeError(
                f"planner {name} must have exact type {expected_type.__name__}"
            )
    checked_settings = require_planner_admission(
        settings,
        seed=seed,
        evidence_scope=evidence_scope,
        canonical_episode_verified=canonical_episode_verified,
    )
    checked_binding = require_attested_binding(binding)
    episode_config.validate(config)
    weights.validate()
    _validate_constraints(constraints)
    static_hashes = (
        ("plant configuration", config.to_dict(), checked_settings.plant_config_sha256),
        (
            "episode configuration",
            episode_config.to_dict(),
            checked_settings.episode_config_sha256,
        ),
        ("constraints", constraints.to_dict(), checked_settings.constraints_sha256),
        ("reward weights", weights.to_dict(), checked_settings.reward_weights_sha256),
    )
    for name, value, expected in static_hashes:
        if raw_canonical_sha256(value) != expected:
            raise ValueError(f"planner {name} identity disagrees with settings")
    profile = episode_config.electrolyser_commitment
    if (
        profile is None
        or profile.profile_sha256 != checked_settings.commitment_profile_sha256
    ):
        raise ValueError("planner commitment profile identity disagrees with settings")
    if checked_binding.binding_scope != evidence_scope:
        raise ValueError("relaxation binding scope disagrees with runtime scope")
    if checked_binding.physics_steps != checked_settings.expected_physics_steps:
        raise ValueError("relaxation horizon disagrees with planner settings")
    if not _binary64_equal(checked_binding.dt_h, config.dt_h):
        raise ValueError("relaxation timestep disagrees with plant configuration")

    checkpoint = describe_checkpoint_contract(
        plant=config,
        episode=episode_config,
        constraints=constraints,
        weights=weights,
    )
    checkpoint_digest = checkpoint.get("checkpoint_contract_digest")
    if (
        checkpoint_digest != checked_settings.checkpoint_contract_digest
        or checkpoint_digest != checked_binding.checkpoint_contract_digest
    ):
        raise ValueError("rebuilt checkpoint contract is not fully cross-bound")

    expected_episode = sample_episode(episode_config, config, seed=seed)
    episode_manifest_json = canonical_json(expected_episode.audit_manifest())
    episode_manifest_sha256 = hashlib.sha256(
        episode_manifest_json.encode("utf-8")
    ).hexdigest()
    if (
        episode_manifest_json != checked_binding.episode_manifest_json
        or episode_manifest_sha256 != checked_binding.episode_manifest_sha256
        or expected_episode.exogenous_episode_sha256
        != checked_binding.exogenous_episode_sha256
    ):
        raise ValueError("regenerated episode does not match relaxation evidence")
    solar = SolarArray(config.solar)
    start_s = expected_episode.scenario.start_hour * 3_600.0
    expected_solar = tuple(
        solar.clear_sky_kw(start_s + step * config.dt_s)
        * expected_episode.weather.tape.for_interval(step)
        for step in range(expected_episode.physics_steps)
    )
    if len(expected_solar) != len(checked_binding.solar_available_kw) or any(
        not _binary64_equal(actual, expected)
        for actual, expected in zip(
            checked_binding.solar_available_kw,
            expected_solar,
            strict=True,
        )
    ):
        raise ValueError("relaxation solar truth disagrees with regenerated episode")

    immutable_manifest = {
        "manifest_schema_version": 3,
        "environment": spaces.describe(),
        "checkpoint_contract": checkpoint,
        "objective_version": OBJECTIVE_VERSION,
        "episode_config": episode_config.to_dict(),
        "plant_config": config.to_dict(),
        "constraints": constraints.to_dict(),
        "reward_weights": weights.to_dict(),
        "episode": expected_episode.audit_manifest(),
        "provenance_gaps": [
            "source revision and dependency-lock digest must be added by the experiment runner"
        ],
    }
    return PlannerRuntimeAdmissionV1(
        settings=checked_settings,
        binding=checked_binding,
        seed=seed,
        evidence_scope=evidence_scope,
        canonical_episode_verified=canonical_episode_verified,
        episode_manifest_json=episode_manifest_json,
        episode_manifest_sha256=episode_manifest_sha256,
        exogenous_episode_sha256=expected_episode.exogenous_episode_sha256,
        checkpoint_contract_json=canonical_json(checkpoint),
        checkpoint_contract_digest=str(checkpoint_digest),
        manifest_json=canonical_json(immutable_manifest),
        manifest_sha256=raw_canonical_sha256(immutable_manifest),
    )


def _validate_reset_environment(
    env: PlantEnvV2,
    *,
    admission: PlannerRuntimeAdmissionV1,
) -> None:
    """Revalidate every static and sampled identity on one fresh reset."""

    if type(env) is not PlantEnvV2:
        raise TypeError("planner reset must produce an exact PlantEnvV2")
    if type(admission) is not PlannerRuntimeAdmissionV1:
        raise TypeError("admission must be an exact PlannerRuntimeAdmissionV1")
    admission.validate()
    expected_seed = _integer("seed", admission.seed)
    checked_settings = admission.settings
    expected_episode_canonical_json = admission.episode_manifest_json
    expected_manifest = admission.manifest_sha256
    expected_exogenous = admission.exogenous_episode_sha256
    episode = env._episode
    if type(episode) is not EpisodeV2 or env.control_state is None:
        raise ValueError("planner environment must be freshly reset")
    if type(episode.seed) is not int or episode.seed != expected_seed:
        raise ValueError("reset episode seed does not match the planner seed")
    if canonical_json(episode.audit_manifest()) != expected_episode_canonical_json:
        raise ValueError("reset episode does not match canonical regeneration")
    if episode.exogenous_episode_sha256 != expected_exogenous:
        raise ValueError("reset episode exogenous identity does not match")

    exact_types = (
        ("config", env.config, PlantConfig),
        ("episode_config", env.episode_config, EpisodeConfigV2),
        ("constraints", env.constraints, Constraints),
        ("weights", env.weights, RewardWeightsV2),
    )
    for name, value, expected_type in exact_types:
        if type(value) is not expected_type:
            raise TypeError(f"planner environment {name} has the wrong exact type")
    static_hashes = (
        (
            "plant configuration",
            raw_canonical_sha256(env.config.to_dict()),
            checked_settings.plant_config_sha256,
        ),
        (
            "episode configuration",
            raw_canonical_sha256(env.episode_config.to_dict()),
            checked_settings.episode_config_sha256,
        ),
        (
            "constraints",
            raw_canonical_sha256(env.constraints.to_dict()),
            checked_settings.constraints_sha256,
        ),
        (
            "reward weights",
            raw_canonical_sha256(env.weights.to_dict()),
            checked_settings.reward_weights_sha256,
        ),
    )
    for name, actual, expected in static_hashes:
        if actual != expected:
            raise ValueError(f"planner {name} identity does not match settings")
    rebuilt_checkpoint = describe_checkpoint_contract(
        plant=env.config,
        episode=env.episode_config,
        constraints=env.constraints,
        weights=env.weights,
    )
    if (
        canonical_json(rebuilt_checkpoint) != admission.checkpoint_contract_json
        or canonical_json(env.checkpoint_contract) != admission.checkpoint_contract_json
        or env.checkpoint_contract_digest != checked_settings.checkpoint_contract_digest
    ):
        raise ValueError("planner checkpoint identity does not match settings")
    profile = env.episode_config.electrolyser_commitment
    if (
        profile is None
        or profile.profile_sha256 != checked_settings.commitment_profile_sha256
    ):
        raise ValueError("planner reset commitment profile identity does not match")
    episode_decisions = _integer(
        "env.episode_decisions", env.episode_decisions, minimum=1
    )
    environment_physics_steps = _integer(
        "env.episode_physics_steps", env.episode_physics_steps, minimum=1
    )
    sampled_physics_steps = _integer(
        "episode.physics_steps", episode.physics_steps, minimum=1
    )
    if (
        episode_decisions != checked_settings.expected_decisions
        or environment_physics_steps != checked_settings.expected_physics_steps
        or sampled_physics_steps != checked_settings.expected_physics_steps
    ):
        raise ValueError("planner reset horizon does not match settings")
    reset_counts = (
        _integer("env.decisions_taken", env.decisions_taken),
        _integer("env.physics_steps_taken", env.physics_steps_taken),
        _integer("simulator.step_index", env.simulator.step_index),
        _integer("metrics.decisions", env.metrics.decisions),
        _integer("metrics.physical_steps", env.metrics.physical_steps),
    )
    if any(reset_counts):
        raise ValueError("planner reset counters must all begin at zero")
    decision_stride = _integer(
        "episode.decision_stride", episode.decision_stride, minimum=1
    )
    expected_decision_stride = _integer(
        "episode_config decision stride",
        env.episode_config.decision_stride(env.config),
        minimum=1,
    )
    if decision_stride != expected_decision_stride:
        raise ValueError("planner reset decision stride is inconsistent")
    delivery_stride = _integer(
        "episode.delivery_stride", episode.delivery_stride, minimum=1
    )
    expected_delivery_stride = _integer(
        "episode_config delivery stride",
        env.episode_config.delivery_stride(env.config),
        minimum=1,
    )
    if delivery_stride != expected_delivery_stride:
        raise ValueError("planner reset delivery stride is inconsistent")
    if env.simulator._solar_factor_tape is not episode.weather.tape:
        raise ValueError("planner reset did not retain the sampled future-truth tape")
    if (
        episode.weather.physics_steps != episode.physics_steps
        or episode.weather.tape.interval_count != episode.physics_steps
        or not _binary64_equal(episode.weather.tape.dt_s, env.config.dt_s)
    ):
        raise ValueError("planner reset weather tape does not cover the horizon")
    solar = SolarArray(env.config.solar)
    start_s = episode.scenario.start_hour * 3_600.0
    if len(admission.binding.solar_available_kw) != episode.physics_steps or any(
        not _binary64_equal(
            admission.binding.solar_available_kw[step],
            solar.clear_sky_kw(start_s + step * env.config.dt_s)
            * episode.weather.tape.for_interval(step),
        )
        for step in range(episode.physics_steps)
    ):
        raise ValueError("planner reset solar truth disagrees with relaxation binding")
    if episode.scenario.faults:
        raise ValueError("planner reset episode contains an unsupported fault")
    if (
        tuple(env.simulator.faults._faults)
        or type(env.simulator.faults._sequence) is not int
        or env.simulator.faults._sequence != 0
    ):
        raise ValueError("planner reset live fault registry is not empty")

    control = env.control_state
    if type(control) is not ControlStateV2:
        raise TypeError("planner reset control state has the wrong exact type")
    telemetry = env.telemetry
    commitment = env.simulator.electrolyser_commitment
    if (
        canonical_json(control.boundary.to_dict())
        != canonical_json(telemetry.to_dict())
        or type(control.previous_interval) is not IntervalSummaryV2
        or canonical_json(control.previous_interval.to_dict())
        != canonical_json(IntervalSummaryV2().to_dict())
        or canonical_json(
            None
            if control.electrolyser_commitment is None
            else control.electrolyser_commitment.to_dict()
        )
        != canonical_json(None if commitment is None else commitment.to_dict())
    ):
        raise ValueError("planner reset control state is not the initial boundary")
    expected_metrics = EpisodeMetricsV2(
        initial_h2_kg=telemetry.h2_level_kg,
        initial_co2_kg=telemetry.co2_level_kg,
        initial_ch4_kg=telemetry.ch4_level_kg,
        initial_battery_soc=telemetry.battery_soc,
        initial_battery_energy_kwh=telemetry.battery_energy_kwh,
        terminal_h2_kg=telemetry.h2_level_kg,
        terminal_co2_kg=telemetry.co2_level_kg,
        terminal_ch4_kg=telemetry.ch4_level_kg,
        terminal_battery_soc=telemetry.battery_soc,
        terminal_battery_energy_kwh=telemetry.battery_energy_kwh,
        min_battery_soc=telemetry.battery_soc,
    )
    if canonical_json(env.metrics.to_dict()) != canonical_json(
        expected_metrics.to_dict()
    ):
        raise ValueError("planner reset metrics are not at their initial values")

    ledger = env.simulator.delivery_ledger
    ledger_audit = ledger.audit()
    if ledger.frozen is not True:
        raise ValueError("planner reset delivery schedule must be frozen")
    if canonical_json(
        [item.to_dict() for item in ledger.commitments]
    ) != canonical_json([item.to_dict() for item in episode.commitments]):
        raise ValueError("planner reset delivery schedule does not match the episode")
    if (
        ledger.settlements
        or not _binary64_equal(ledger_audit.get("cumulative_requested_kg"), 0.0)
        or not _binary64_equal(ledger_audit.get("cumulative_delivered_kg"), 0.0)
        or not _binary64_equal(ledger_audit.get("cumulative_unmet_kg"), 0.0)
        or _integer("delivery pending_count", ledger_audit.get("pending_count"))
        != len(episode.commitments)
    ):
        raise ValueError(
            "planner reset delivery accounting is not at its initial state"
        )

    manifest = env.audit_manifest()
    if not isinstance(manifest, dict):
        raise TypeError("planner reset manifest must be a dictionary")
    manifest_copy = dict(manifest)
    declared_manifest = manifest_copy.pop("manifest_sha256", None)
    declared_exogenous = manifest_copy.pop("exogenous_episode_sha256", None)
    delivery_accounting = manifest_copy.pop("delivery_accounting", None)
    if set(manifest_copy) != {
        "manifest_schema_version",
        "environment",
        "checkpoint_contract",
        "objective_version",
        "episode_config",
        "plant_config",
        "constraints",
        "reward_weights",
        "episode",
        "provenance_gaps",
    }:
        raise ValueError("planner reset manifest has an unexpected static field set")
    recomputed_manifest = raw_canonical_sha256(manifest_copy)
    if (
        canonical_json(manifest_copy) != admission.manifest_json
        or declared_manifest != recomputed_manifest
        or declared_manifest != expected_manifest
    ):
        raise ValueError("planner reset manifest identity does not match")
    if declared_exogenous != expected_exogenous:
        raise ValueError("planner reset manifest exogenous identity does not match")
    if canonical_json(delivery_accounting) != canonical_json(ledger_audit):
        raise ValueError("planner reset manifest delivery accounting does not match")


class PlannerRuntimeError(RuntimeError):
    """Stable plant-runtime failure translated by the future core orchestrator."""

    def __init__(
        self,
        *,
        phase: str,
        code: str,
        message: str,
        observed_resource_value: int | None = None,
    ) -> None:
        if phase not in FAILURE_PHASES:
            raise ValueError(f"unsupported planner failure phase: {phase}")
        if code not in FAILURE_CODES:
            raise ValueError(f"unsupported planner failure code: {code}")
        if not isinstance(message, str) or not message:
            raise ValueError("planner runtime failure message must be non-empty")
        if observed_resource_value is not None:
            observed_resource_value = _integer(
                "observed_resource_value", observed_resource_value
            )
        super().__init__(message)
        self.phase = phase
        self.code = code
        self.observed_resource_value = observed_resource_value


@dataclass(frozen=True, slots=True)
class TerminalSnapshotV1:
    """Detached terminal outcome; it never retains a live environment object."""

    manifest_sha256: str
    exogenous_episode_sha256: str
    checkpoint_contract_digest: str
    plant_config_sha256: str
    action_records: tuple[ActionRecordV1, ...]
    request_trace: CanonicalPayloadV1
    source_trace: CanonicalPayloadV1
    action_record_trace: CanonicalPayloadV1
    metrics: CanonicalPayloadV1
    terminal_truth: CanonicalPayloadV1
    terminal_telemetry: CanonicalPayloadV1
    delivery_audit: CanonicalPayloadV1
    outcome: CanonicalPayloadV1
    final_score: FinalScoreV1
    decision_count: int
    physics_steps: int
    terminated: bool
    truncated: bool
    battery_throughput_kwh: float
    h2_kg_per_kg_ch4: float
    co2_kg_per_kg_ch4: float

    def __post_init__(self) -> None:
        records = tuple(
            _revalidated_action_record(item) for item in self.action_records
        )
        object.__setattr__(self, "action_records", records)
        object.__setattr__(
            self, "decision_count", _integer("decision_count", self.decision_count)
        )
        object.__setattr__(
            self, "physics_steps", _integer("physics_steps", self.physics_steps)
        )
        object.__setattr__(
            self,
            "battery_throughput_kwh",
            _finite(
                "battery_throughput_kwh",
                self.battery_throughput_kwh,
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "h2_kg_per_kg_ch4",
            _finite("h2_kg_per_kg_ch4", self.h2_kg_per_kg_ch4, minimum=0.0),
        )
        object.__setattr__(
            self,
            "co2_kg_per_kg_ch4",
            _finite("co2_kg_per_kg_ch4", self.co2_kg_per_kg_ch4, minimum=0.0),
        )
        self._validate()

    def validate(self) -> None:
        """Re-run every invariant so illicit mutation cannot survive later use."""

        self._validate()

    def _validate(self) -> None:
        for name in (
            "manifest_sha256",
            "exogenous_episode_sha256",
            "checkpoint_contract_digest",
            "plant_config_sha256",
        ):
            require_sha256(name, getattr(self, name))
        domains = (
            ("request_trace", "planner-request-trace-v1"),
            ("source_trace", "planner-source-trace-v1"),
            ("action_record_trace", "planner-action-record-trace-v1"),
            ("metrics", "planner-metrics-v1"),
            ("terminal_truth", "planner-terminal-truth-v1"),
            ("terminal_telemetry", "planner-terminal-telemetry-v1"),
            ("delivery_audit", "planner-delivery-audit-v1"),
            ("outcome", "planner-outcome-v1"),
        )
        for name, domain in domains:
            _payload(name, getattr(self, name), domain=domain)
        if type(self.final_score) is not FinalScoreV1:
            raise TypeError("final_score must be an exact FinalScoreV1")
        checked_score = FinalScoreV1(**self.final_score.to_dict())
        if canonical_json(self.final_score.to_dict()) != canonical_json(
            checked_score.to_dict()
        ):
            raise ValueError("final_score is not canonically normalized")
        decision_count = _integer("decision_count", self.decision_count)
        physics_steps = _integer("physics_steps", self.physics_steps)
        terminated = _bool("terminated", self.terminated)
        truncated = _bool("truncated", self.truncated)
        throughput = _finite(
            "battery_throughput_kwh",
            self.battery_throughput_kwh,
            minimum=0.0,
        )
        h2_ratio = _finite("h2_kg_per_kg_ch4", self.h2_kg_per_kg_ch4, minimum=0.0)
        co2_ratio = _finite("co2_kg_per_kg_ch4", self.co2_kg_per_kg_ch4, minimum=0.0)
        if h2_ratio == 0.0 or co2_ratio == 0.0:
            raise ValueError("terminal methane conversion ratios must be positive")
        if not terminated or truncated:
            raise ValueError("terminal snapshot must terminate without truncation")
        records = tuple(
            _revalidated_action_record(item) for item in self.action_records
        )
        expected_request = request_trace_payload(records)
        expected_source = source_trace_payload(records)
        expected_action_trace = action_record_trace_payload(records)
        if (
            self.request_trace != expected_request
            or self.source_trace != expected_source
            or self.action_record_trace != expected_action_trace
        ):
            raise ValueError("terminal trace payloads disagree with action_records")
        if decision_count != len(records):
            raise ValueError("terminal decision count disagrees with action_records")
        if not _binary64_equal(self.final_score.battery_throughput_kwh, throughput):
            raise ValueError("terminal score throughput disagrees with snapshot")
        self._validate_score_sources(
            h2_ratio=h2_ratio,
            co2_ratio=co2_ratio,
            throughput=throughput,
        )
        expected_outcome = outcome_payload(
            manifest_sha256=self.manifest_sha256,
            exogenous_episode_sha256=self.exogenous_episode_sha256,
            checkpoint_contract_digest=self.checkpoint_contract_digest,
            request_trace=self.request_trace,
            metrics=self.metrics,
            terminal_truth=self.terminal_truth,
            terminal_telemetry=self.terminal_telemetry,
            delivery_audit=self.delivery_audit,
            battery_throughput_kwh=throughput,
            decision_count=decision_count,
            physics_steps=physics_steps,
            terminated=terminated,
            truncated=truncated,
        )
        if self.outcome != expected_outcome:
            raise ValueError("terminal outcome payload disagrees with its snapshot")

    def validate_for_config(self, config: PlantConfig) -> None:
        """Bind score conversion ratios to one exact plant configuration."""

        self.validate()
        if type(config) is not PlantConfig:
            raise TypeError("terminal snapshot config must be an exact PlantConfig")
        config_sha256 = hashlib.sha256(
            canonical_json(config.to_dict()).encode("utf-8")
        ).hexdigest()
        if config_sha256 != self.plant_config_sha256:
            raise ValueError("terminal snapshot plant configuration identity disagrees")
        methanator = config.methanator
        if not _binary64_equal(
            self.h2_kg_per_kg_ch4,
            methanator.h2_kg_per_kg_ch4,
        ) or not _binary64_equal(
            self.co2_kg_per_kg_ch4,
            methanator.co2_kg_per_kg_ch4,
        ):
            raise ValueError(
                "terminal methane conversion ratios disagree with plant configuration"
            )

    def _validate_score_sources(
        self,
        *,
        h2_ratio: float,
        co2_ratio: float,
        throughput: float,
    ) -> None:
        metrics = self.metrics.value()
        truth = self.terminal_truth.value()
        telemetry = self.terminal_telemetry.value()
        audit = self.delivery_audit.value()
        if not all(
            isinstance(value, dict) for value in (metrics, truth, telemetry, audit)
        ):
            raise ValueError("terminal score sources must all contain objects")
        assert isinstance(metrics, dict)
        assert isinstance(truth, dict)
        assert isinstance(telemetry, dict)
        assert isinstance(audit, dict)
        if set(metrics) != set(EpisodeMetricsV2().to_dict()):
            raise ValueError("terminal metrics payload has an incomplete field set")
        plant_fields = set(PlantState().to_dict())
        if set(truth) != plant_fields or set(telemetry) != plant_fields:
            raise ValueError("terminal plant payload has an incomplete field set")
        expected_audit_fields = {
            "frozen",
            "schedule",
            "settlements",
            "current",
            "cumulative_requested_kg",
            "cumulative_delivered_kg",
            "cumulative_unmet_kg",
            "pending_count",
            "next_pending",
        }
        if set(audit) != expected_audit_fields:
            raise ValueError("terminal delivery audit has an incomplete field set")

        score = self.final_score
        float_metrics = (
            (score.delivered_kg, metrics["delivered_kg"], "delivered_kg"),
            (score.worst_fill, metrics["worst_commitment_fill_rate"], "worst_fill"),
            (
                score.terminal_battery_energy_kwh,
                metrics["terminal_battery_energy_kwh"],
                "terminal_battery_energy_kwh",
            ),
            (score.curtailed_kwh, metrics["curtailed_kwh"], "curtailed_kwh"),
        )
        for score_value, source_value, name in float_metrics:
            if not _binary64_equal(score_value, source_value):
                raise ValueError(f"terminal score {name} disagrees with metrics")
        integer_metrics = (
            (score.service_numerator, "commitments_met"),
            (score.service_denominator, "commitments_settled"),
            (score.forced_stops, "electrolyser_forced_stops"),
            (score.aborted_starts, "electrolyser_aborted_starts"),
            (score.cold_starts, "electrolyser_cold_starts"),
        )
        for score_value, name in integer_metrics:
            if _integer(f"metrics.{name}", metrics[name]) != score_value:
                raise ValueError(f"terminal score {name} disagrees with metrics")
        truth_metrics = (
            ("h2_level_kg", "terminal_h2_kg"),
            ("co2_level_kg", "terminal_co2_kg"),
            ("ch4_level_kg", "terminal_ch4_kg"),
            ("battery_soc", "terminal_battery_soc"),
            ("battery_energy_kwh", "terminal_battery_energy_kwh"),
        )
        for truth_name, metric_name in truth_metrics:
            if not _binary64_equal(truth[truth_name], metrics[metric_name]):
                raise ValueError(f"terminal truth {truth_name} disagrees with metrics")
        equivalent = _finite("terminal truth ch4_level_kg", truth["ch4_level_kg"])
        equivalent += min(
            _finite("terminal truth h2_level_kg", truth["h2_level_kg"]) / h2_ratio,
            _finite("terminal truth co2_level_kg", truth["co2_level_kg"]) / co2_ratio,
        )
        if not _binary64_equal(score.terminal_ch4_equivalent_inventory_kg, equivalent):
            raise ValueError("terminal equivalent inventory disagrees with truth")
        if not _binary64_equal(audit["cumulative_delivered_kg"], score.delivered_kg):
            raise ValueError("terminal delivered score disagrees with delivery audit")
        if not _binary64_equal(score.battery_throughput_kwh, throughput):
            raise ValueError("terminal throughput score disagrees with accumulator")


@dataclass(frozen=True, slots=True)
class A0GenerationV1:
    """Detached A0 generation trace and every completed macro prefix identity."""

    action_records: tuple[ActionRecordV1, ...]
    terminal_snapshot: TerminalSnapshotV1
    prefix_state_bytes: tuple[bytes, ...]

    def __post_init__(self) -> None:
        records, prefixes = self._validated_contents()
        object.__setattr__(self, "action_records", records)
        object.__setattr__(self, "prefix_state_bytes", prefixes)

    def validate(self) -> None:
        """Re-run the detached generation contract after low-level mutation."""

        self._validated_contents()

    def _validated_contents(
        self,
    ) -> tuple[tuple[ActionRecordV1, ...], tuple[bytes, ...]]:
        records = tuple(
            _revalidated_action_record(item) for item in self.action_records
        )
        request_trace = request_trace_payload(records)
        source_trace = source_trace_payload(records)
        action_trace = action_record_trace_payload(records)
        if any(
            record.source_phase != "a0_capture"
            or record.source_code != -1
            or record.source_binding_sha256 != A0_CONTROLLER_SHA256
            for record in records
        ):
            raise ValueError("A0 generation records must retain exact A0 lineage")
        if type(self.terminal_snapshot) is not TerminalSnapshotV1:
            raise TypeError("terminal_snapshot must be an exact TerminalSnapshotV1")
        snapshot = self.terminal_snapshot
        snapshot.validate()
        if (
            snapshot.request_trace != request_trace
            or snapshot.source_trace != source_trace
            or snapshot.action_record_trace != action_trace
        ):
            raise ValueError("A0 terminal snapshot does not match its action records")
        if snapshot.decision_count != len(records):
            raise ValueError("A0 decision count does not match its action trace")
        prefixes = tuple(self.prefix_state_bytes)
        expected_prefixes = (len(records) + 3) // 4
        if len(prefixes) != expected_prefixes:
            raise ValueError("A0 prefix count does not match four-decision macros")
        for index, prefix in enumerate(prefixes):
            if not isinstance(prefix, bytes) or not prefix:
                raise TypeError(f"prefix_state_bytes[{index}] must be non-empty bytes")
        return records, prefixes


def _revalidated_action_record(record: ActionRecordV1) -> ActionRecordV1:
    if type(record) is not ActionRecordV1:
        raise TypeError("planner action record must be an exact ActionRecordV1")
    checked = ActionRecordV1(**record.to_dict())
    if canonical_json(checked.to_dict()) != canonical_json(record.to_dict()):
        raise ValueError("planner action record is not canonically normalized")
    return checked


def step_action_record(
    env: PlantEnvV2,
    record: ActionRecordV1,
    *,
    battery_throughput_kwh: float,
) -> tuple[float, bool, bool]:
    """Apply one exact record and verify every counter transition."""

    if type(env) is not PlantEnvV2:
        raise TypeError("planner step requires an exact PlantEnvV2")
    checked = _revalidated_action_record(record)
    throughput = _finite("battery_throughput_kwh", battery_throughput_kwh, minimum=0.0)
    episode = env._episode
    if type(episode) is not EpisodeV2 or env.control_state is None:
        raise ValueError("planner step requires a reset environment")
    before_decisions = _integer("env.decisions_taken", env.decisions_taken)
    before_physics = _integer("env.physics_steps_taken", env.physics_steps_taken)
    if checked.decision_index != before_decisions:
        raise ValueError("action record decision index disagrees with environment")
    if (
        env.metrics.decisions != before_decisions
        or env.metrics.physical_steps != before_physics
        or env.simulator.step_index != before_physics
    ):
        raise ValueError("pre-step environment counters disagree")
    episode_physics_steps = _integer(
        "env.episode_physics_steps", env.episode_physics_steps, minimum=1
    )
    episode_decisions = _integer(
        "env.episode_decisions", env.episode_decisions, minimum=1
    )
    decision_stride = _integer(
        "episode.decision_stride", episode.decision_stride, minimum=1
    )
    remaining = episode_physics_steps - before_physics
    if remaining <= 0 or before_decisions >= episode_decisions:
        raise ValueError("cannot apply an action after planner termination")
    expected_physics_delta = min(decision_stride, remaining)
    result = env.step_action(
        action_record_to_action(checked),
        controller_invalid=checked.controller_invalid,
    )
    if type(result) is not tuple or len(result) != 5:
        raise ValueError("step_action did not return the five-field V2 result")
    terminated = _bool("step_action terminated", result[2])
    truncated = _bool("step_action truncated", result[3])
    after_decisions = _integer("env.decisions_taken", env.decisions_taken)
    metric_decisions = _integer("metrics.decisions", env.metrics.decisions)
    after_physics = _integer("env.physics_steps_taken", env.physics_steps_taken)
    metric_physics = _integer("metrics.physical_steps", env.metrics.physical_steps)
    simulator_step = _integer("simulator.step_index", env.simulator.step_index)
    if after_decisions != before_decisions + 1 or metric_decisions != after_decisions:
        raise ValueError("post-step decision counters disagree")
    if (
        after_physics != before_physics + expected_physics_delta
        or metric_physics != after_physics
        or simulator_step != after_physics
    ):
        raise ValueError("post-step physics counters disagree")
    expected_termination = after_physics == episode_physics_steps
    if terminated != expected_termination:
        raise ValueError("step_action termination flag disagrees with horizon")
    if truncated:
        raise ValueError("planner step unexpectedly truncated")
    return accumulate_battery_throughput(throughput, result), terminated, truncated


def _terminal_manifest(env: PlantEnvV2, admission: PlannerRuntimeAdmissionV1) -> None:
    admission.validate()
    manifest = env.audit_manifest()
    if not isinstance(manifest, dict):
        raise TypeError("terminal manifest must be a dictionary")
    immutable = dict(manifest)
    declared_manifest = immutable.pop("manifest_sha256", None)
    declared_exogenous = immutable.pop("exogenous_episode_sha256", None)
    delivery_accounting = immutable.pop("delivery_accounting", None)
    if (
        declared_manifest != admission.manifest_sha256
        or declared_exogenous != admission.exogenous_episode_sha256
        or canonical_json(immutable) != admission.manifest_json
        or raw_canonical_sha256(immutable) != admission.manifest_sha256
    ):
        raise ValueError("terminal environment identity disagrees with admission")
    if canonical_json(delivery_accounting) != canonical_json(
        env.simulator.delivery_audit()
    ):
        raise ValueError("terminal manifest delivery accounting disagrees with ledger")
    if canonical_json(env.checkpoint_contract) != admission.checkpoint_contract_json:
        raise ValueError("terminal checkpoint contract disagrees with admission")


def capture_terminal_snapshot(
    env: PlantEnvV2,
    *,
    admission: PlannerRuntimeAdmissionV1,
    action_records: tuple[ActionRecordV1, ...],
    battery_throughput_kwh: float,
    terminated: bool,
    truncated: bool,
) -> TerminalSnapshotV1:
    """Detach one complete exact-simulator outcome before releasing its env."""

    if type(env) is not PlantEnvV2:
        raise TypeError("terminal snapshot requires an exact PlantEnvV2")
    if type(admission) is not PlannerRuntimeAdmissionV1:
        raise TypeError("terminal snapshot requires exact runtime admission")
    records = tuple(_revalidated_action_record(item) for item in action_records)
    throughput = _finite("battery_throughput_kwh", battery_throughput_kwh, minimum=0.0)
    ended = _bool("terminated", terminated)
    cut_short = _bool("truncated", truncated)
    if not ended or cut_short:
        raise ValueError("terminal snapshot requires termination without truncation")
    decisions_taken = _integer("env.decisions_taken", env.decisions_taken)
    physics_steps_taken = _integer("env.physics_steps_taken", env.physics_steps_taken)
    episode_decisions = _integer(
        "env.episode_decisions", env.episode_decisions, minimum=1
    )
    episode_physics_steps = _integer(
        "env.episode_physics_steps", env.episode_physics_steps, minimum=1
    )
    if (
        decisions_taken != admission.settings.expected_decisions
        or physics_steps_taken != admission.settings.expected_physics_steps
        or decisions_taken != episode_decisions
        or physics_steps_taken != episode_physics_steps
        or len(records) != decisions_taken
    ):
        raise ValueError("terminal environment does not cover the admitted horizon")
    if (
        _integer("metrics.decisions", env.metrics.decisions) != decisions_taken
        or _integer("metrics.physical_steps", env.metrics.physical_steps)
        != physics_steps_taken
        or _integer("simulator.step_index", env.simulator.step_index)
        != physics_steps_taken
    ):
        raise ValueError("terminal environment counters disagree")
    _terminal_manifest(env, admission)
    request_trace = request_trace_payload(records)
    source_trace = source_trace_payload(records)
    action_trace = action_record_trace_payload(records)
    metrics = canonical_payload("planner-metrics-v1", env.metrics.to_dict())
    truth = canonical_payload(
        "planner-terminal-truth-v1", env.simulator.state.to_dict()
    )
    telemetry = canonical_payload(
        "planner-terminal-telemetry-v1", env.telemetry.to_dict()
    )
    audit = canonical_payload(
        "planner-delivery-audit-v1", env.simulator.delivery_audit()
    )
    score = extract_final_score(env, battery_throughput_kwh=throughput)
    outcome = outcome_payload(
        manifest_sha256=admission.manifest_sha256,
        exogenous_episode_sha256=admission.exogenous_episode_sha256,
        checkpoint_contract_digest=admission.checkpoint_contract_digest,
        request_trace=request_trace,
        metrics=metrics,
        terminal_truth=truth,
        terminal_telemetry=telemetry,
        delivery_audit=audit,
        battery_throughput_kwh=throughput,
        decision_count=decisions_taken,
        physics_steps=physics_steps_taken,
        terminated=ended,
        truncated=cut_short,
    )
    snapshot = TerminalSnapshotV1(
        manifest_sha256=admission.manifest_sha256,
        exogenous_episode_sha256=admission.exogenous_episode_sha256,
        checkpoint_contract_digest=admission.checkpoint_contract_digest,
        plant_config_sha256=raw_canonical_sha256(env.config.to_dict()),
        action_records=records,
        request_trace=request_trace,
        source_trace=source_trace,
        action_record_trace=action_trace,
        metrics=metrics,
        terminal_truth=truth,
        terminal_telemetry=telemetry,
        delivery_audit=audit,
        outcome=outcome,
        final_score=score,
        decision_count=decisions_taken,
        physics_steps=physics_steps_taken,
        terminated=ended,
        truncated=cut_short,
        battery_throughput_kwh=throughput,
        h2_kg_per_kg_ch4=env.config.methanator.h2_kg_per_kg_ch4,
        co2_kg_per_kg_ch4=env.config.methanator.co2_kg_per_kg_ch4,
    )
    snapshot.validate_for_config(env.config)
    return snapshot


@dataclass(eq=False, slots=True)
class PlantPlannerCandidateV1:
    """One explicitly owned mutable environment plus immutable search metadata."""

    env: PlantEnvV2 | None
    action_records: tuple[ActionRecordV1, ...]
    battery_throughput_kwh: float
    macro_depth: int
    terminated: bool
    truncated: bool
    _attestation: tuple[PlantEnvV2, str, bytes, int, bool, bool] | None = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        records, throughput, depth, terminated, truncated = self._validate_contents()
        object.__setattr__(self, "action_records", records)
        object.__setattr__(self, "battery_throughput_kwh", throughput)
        object.__setattr__(self, "macro_depth", depth)
        assert self.env is not None
        object.__setattr__(
            self,
            "_attestation",
            (
                self.env,
                action_record_trace_payload(records).canonical_json,
                struct.pack(">d", throughput),
                depth,
                terminated,
                truncated,
            ),
        )

    def validate(self) -> None:
        """Re-attest immutable candidate metadata before every port operation."""

        expected = self._attestation
        if expected is None:
            raise ValueError("candidate was already released")
        records, throughput, depth, terminated, truncated = self._validate_contents()
        env = self.env
        assert env is not None
        if (
            expected[0] is not env
            or expected[1] != action_record_trace_payload(records).canonical_json
            or expected[2] != struct.pack(">d", throughput)
            or expected[3:] != (depth, terminated, truncated)
        ):
            raise ValueError("candidate no longer matches its creation attestation")

    def mark_released(self) -> PlantEnvV2:
        """Invalidate this ownership handle and return its exact environment once."""

        self.validate()
        return self._dispose_for_cleanup()

    def _dispose_for_cleanup(self) -> PlantEnvV2:
        """Detach the originally attested environment without endorsing metadata.

        The plant runtime uses this only while unwinding an integrity failure.  It must
        still release the returned environment's independent resource-guard token and
        propagate the original validation failure.
        """

        expected = self._attestation
        if expected is None:
            raise ValueError("candidate was already released")
        env = expected[0]
        self.env = None
        self._attestation = None
        return env

    def _validate_contents(
        self,
    ) -> tuple[tuple[ActionRecordV1, ...], float, int, bool, bool]:
        if type(self.env) is not PlantEnvV2:
            raise TypeError("candidate env must be an exact PlantEnvV2")
        env = self.env
        if env._episode is None or env.control_state is None:
            raise ValueError("candidate environment must be reset")
        records = tuple(
            _revalidated_action_record(item) for item in self.action_records
        )
        request_trace_payload(records)
        throughput = _finite(
            "battery_throughput_kwh",
            self.battery_throughput_kwh,
            minimum=0.0,
        )
        depth = _integer("macro_depth", self.macro_depth)
        terminated = _bool("terminated", self.terminated)
        truncated = _bool("truncated", self.truncated)
        if truncated:
            raise ValueError("planner candidates cannot retain truncated environments")
        decisions_taken = _integer("env.decisions_taken", env.decisions_taken)
        physics_steps_taken = _integer(
            "env.physics_steps_taken", env.physics_steps_taken
        )
        simulator_step = _integer("simulator.step_index", env.simulator.step_index)
        metric_decisions = _integer("metrics.decisions", env.metrics.decisions)
        metric_steps = _integer("metrics.physical_steps", env.metrics.physical_steps)
        episode_steps = _integer(
            "env.episode_physics_steps", env.episode_physics_steps, minimum=1
        )
        episode_decisions = _integer(
            "env.episode_decisions", env.episode_decisions, minimum=1
        )
        if decisions_taken != len(records):
            raise ValueError("candidate trace length disagrees with environment")
        if not (physics_steps_taken == simulator_step == metric_steps):
            raise ValueError("candidate physics counters disagree")
        if decisions_taken != metric_decisions:
            raise ValueError("candidate decision counters disagree")
        if physics_steps_taken > episode_steps or decisions_taken > episode_decisions:
            raise ValueError("candidate counters exceed the episode horizon")
        if depth != (len(records) + 3) // 4:
            raise ValueError("candidate macro depth disagrees with its trace")
        if terminated != (physics_steps_taken == episode_steps):
            raise ValueError("candidate termination flag disagrees with environment")
        return records, throughput, depth, terminated, truncated


def extract_intermediate_rank(
    candidate: PlantPlannerCandidateV1,
) -> IntermediateRankV1:
    """Derive the frozen pruning rank solely from authoritative plant state."""

    if type(candidate) is not PlantPlannerCandidateV1:
        raise TypeError("rank candidate must be an exact PlantPlannerCandidateV1")
    candidate.validate()
    env = candidate.env
    assert env is not None
    metrics = env.metrics
    state = env.simulator.state
    methanator = env.config.methanator
    h2_ratio = _finite(
        "methanator.h2_kg_per_kg_ch4",
        methanator.h2_kg_per_kg_ch4,
        minimum=0.0,
    )
    co2_ratio = _finite(
        "methanator.co2_kg_per_kg_ch4",
        methanator.co2_kg_per_kg_ch4,
        minimum=0.0,
    )
    if h2_ratio == 0.0 or co2_ratio == 0.0:
        raise ValueError("methanator conversion ratios must be positive")
    h2_kg = _finite("state.h2_level_kg", state.h2_level_kg, minimum=0.0)
    co2_kg = _finite("state.co2_level_kg", state.co2_level_kg, minimum=0.0)
    ch4_kg = _finite("state.ch4_level_kg", state.ch4_level_kg, minimum=0.0)
    battery_energy_kwh = _finite(
        "state.battery_energy_kwh", state.battery_energy_kwh, minimum=0.0
    )
    feedstock = min(h2_kg / h2_ratio, co2_kg / co2_ratio)
    next_pending = env.simulator.delivery_ledger.next_pending
    readiness = 0.0
    if next_pending is not None:
        requested_kg = _finite(
            "next_pending.requested_kg", next_pending.requested_kg, minimum=0.0
        )
        due_step = _integer("next_pending.due_step", next_pending.due_step)
        simulator_step = _integer("simulator.step_index", env.simulator.step_index)
        remaining_steps = due_step - simulator_step
        if remaining_steps <= 0:
            raise ValueError("next pending commitment is not in the future")
        dt_h = _finite("config.dt_h", env.config.dt_h, minimum=0.0)
        if dt_h == 0.0:
            raise ValueError("config.dt_h must be positive")
        remaining_h = float(remaining_steps) * dt_h
        specific_energy = _finite(
            "methanator.specific_energy_kwh_per_kg",
            methanator.specific_energy_kwh_per_kg,
            minimum=0.0,
        )
        if specific_energy == 0.0:
            raise ValueError("methanator specific energy must be positive")
        conversion = (
            _finite(
                "active methanator cap",
                env.constraints.methanator_limit_kw(env.config),
                minimum=0.0,
            )
            * remaining_h
            / specific_energy
        )
        readiness = min(
            requested_kg,
            ch4_kg + min(feedstock, conversion),
        )
    commitment = env.simulator.electrolyser_commitment
    if type(commitment) is not ElectrolyserCommitmentSnapshot:
        raise ValueError("planner rank requires commitment-enabled simulator truth")
    return IntermediateRankV1(
        delivered_plus_deadline_readiness_kg=metrics.delivered_kg + readiness,
        delivered_kg=metrics.delivered_kg,
        service_numerator=metrics.commitments_met,
        service_denominator=metrics.commitments_settled,
        worst_fill=metrics.worst_commitment_fill_rate,
        ch4_inventory_kg=ch4_kg,
        feedstock_potential_kg=feedstock,
        battery_energy_kwh=battery_energy_kwh,
        commitment_mode=commitment.mode.value,
        startup_remaining_s=commitment.startup_remaining_s,
        cooldown_remaining_s=commitment.cooldown_remaining_s,
        shortfall_elapsed_s=commitment.shortfall_elapsed_s,
        forced_stops=metrics.electrolyser_forced_stops,
        aborted_starts=metrics.electrolyser_aborted_starts,
        cold_starts=metrics.electrolyser_cold_starts,
        curtailed_kwh=metrics.curtailed_kwh,
        battery_throughput_kwh=candidate.battery_throughput_kwh,
    )


@dataclass(frozen=True, slots=True, eq=False)
class _EnvironmentReservationV1:
    owner: object
    serial: int
    phase: str


def _production_peak_rss_raw() -> int | float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


class _ResourceGuardV1:
    """Own exact resource sampling and every logical environment reference."""

    def __init__(
        self,
        *,
        settings: PlannerSettingsV1,
        counters: PlannerSearchCountersV1,
        start_ns: int,
        clock_ns: ClockNs = time.perf_counter_ns,
        peak_rss_raw: PeakRssRaw = _production_peak_rss_raw,
        platform_system: PlatformSystem = platform.system,
    ) -> None:
        if not isinstance(settings, PlannerSettingsV1):
            raise TypeError("settings must be a PlannerSettingsV1")
        if not isinstance(counters, PlannerSearchCountersV1):
            raise TypeError("counters must be PlannerSearchCountersV1")
        if (
            not callable(clock_ns)
            or not callable(peak_rss_raw)
            or not callable(platform_system)
        ):
            raise TypeError("resource readers must be callable")
        checked_settings = revalidate_planner_settings(settings)
        self._counters = counters
        self._start_ns = _integer("start_ns", start_ns)
        self._wall_limit_ns = checked_settings.wall_limit_ns
        self._rss_limit_bytes = checked_settings.rss_limit_bytes
        self._max_live_environments = checked_settings.max_live_environments
        self._clock_ns = clock_ns
        self._peak_rss_raw = peak_rss_raw
        self._platform_system = platform_system
        self._owner = object()
        self._next_serial = 0
        self._reservations: dict[
            int, tuple[_EnvironmentReservationV1, PlantEnvV2 | None]
        ] = {}
        self._environment_ids: dict[int, PlantEnvV2] = {}
        self._peak_rss_bytes = 0
        self._closed = False

    @property
    def live_environments(self) -> int:
        return len(self._environment_ids)

    @property
    def reserved_slots(self) -> int:
        return len(self._reservations)

    @property
    def peak_rss_bytes(self) -> int:
        return self._peak_rss_bytes

    def elapsed_ns(self) -> int:
        self._require_open()
        return self._elapsed_ns(phase="input_validation")

    def _elapsed_ns(self, *, phase: str) -> int:
        now = _integer("clock reading", self._clock_ns())
        if now < self._start_ns:
            raise PlannerRuntimeError(
                phase=phase,
                code="invalid_input",
                message="monotonic clock precedes the supplied worker start",
            )
        return now - self._start_ns

    def reserve_environment(self, *, phase: str) -> _EnvironmentReservationV1:
        self._require_open()
        self._validate_phase(phase)
        proposed = len(self._reservations) + 1
        beam_only = phase == "beam_search" and all(
            item[0].phase == "beam_search" for item in self._reservations.values()
        )
        phase_limit = self._max_live_environments if beam_only else 2
        limit = min(self._max_live_environments, phase_limit)
        if proposed > limit:
            raise PlannerRuntimeError(
                phase=phase,
                code="live_environment_limit",
                message="another environment would exceed the phase live limit",
                observed_resource_value=proposed,
            )
        token = _EnvironmentReservationV1(self._owner, self._next_serial, phase)
        self._next_serial += 1
        self._reservations[id(token)] = (token, None)
        return token

    def bind_environment(
        self, token: _EnvironmentReservationV1, env: PlantEnvV2
    ) -> None:
        self._require_open()
        current = self._lookup_token(token)
        if current is not None:
            raise PlannerRuntimeError(
                phase="input_validation",
                code="result_invariant_error",
                message="environment reservation is already bound",
            )
        if type(env) is not PlantEnvV2:
            raise TypeError("reserved environment must be an exact PlantEnvV2")
        existing = self._environment_ids.get(id(env))
        if existing is not None:
            raise PlannerRuntimeError(
                phase="input_validation",
                code="result_invariant_error",
                message="environment identity is already live",
            )
        try:
            self._reservations[id(token)] = (token, env)
            self._environment_ids[id(env)] = env
        except BaseException:
            registered = self._environment_ids.get(id(env))
            if registered is env:
                del self._environment_ids[id(env)]
            entry = self._reservations.get(id(token))
            if entry is not None and entry[0] is token:
                self._reservations[id(token)] = (token, None)
            raise
        self._counters.max_live_environments_observed = max(
            self._counters.max_live_environments_observed,
            len(self._environment_ids),
        )

    def cancel_reservation(self, token: _EnvironmentReservationV1) -> None:
        self._require_open()
        current = self._lookup_token(token)
        if current is not None:
            raise PlannerRuntimeError(
                phase="input_validation",
                code="result_invariant_error",
                message="a bound environment reservation cannot be cancelled",
            )
        del self._reservations[id(token)]

    def release_environment(
        self, token: _EnvironmentReservationV1, env: PlantEnvV2
    ) -> None:
        self._require_open()
        current = self._lookup_token(token)
        if current is None or current is not env:
            raise PlannerRuntimeError(
                phase="result_serialization",
                code="result_invariant_error",
                message="environment release does not match its reservation",
            )
        registered = self._environment_ids.get(id(env))
        if registered is not env:
            raise PlannerRuntimeError(
                phase="result_serialization",
                code="result_invariant_error",
                message="environment identity is not live",
            )
        del self._environment_ids[id(env)]
        del self._reservations[id(token)]

    def checkpoint(self, *, phase: str) -> tuple[int, int]:
        self._require_open()
        self._validate_phase(phase)
        elapsed = self._elapsed_ns(phase=phase)
        rss = self._read_peak_rss_bytes(phase=phase)
        self._peak_rss_bytes = max(self._peak_rss_bytes, rss)
        if elapsed > self._wall_limit_ns:
            raise PlannerRuntimeError(
                phase=phase,
                code="wall_time_limit",
                message="planner wall-time limit exceeded",
                observed_resource_value=elapsed,
            )
        if rss > self._rss_limit_bytes:
            raise PlannerRuntimeError(
                phase=phase,
                code="rss_limit",
                message="planner peak-RSS limit exceeded",
                observed_resource_value=rss,
            )
        return elapsed, rss

    def close(self) -> None:
        if self._closed:
            return
        leaked = len(self._reservations)
        self._environment_ids.clear()
        self._reservations.clear()
        self._closed = True
        if leaked:
            raise PlannerRuntimeError(
                phase="result_serialization",
                code="result_invariant_error",
                message=f"planner runtime leaked {leaked} environment reservations",
                observed_resource_value=leaked,
            )

    def _lookup_token(self, token: object) -> PlantEnvV2 | None:
        if not isinstance(token, _EnvironmentReservationV1):
            raise TypeError("environment reservation token has the wrong type")
        if token.owner is not self._owner:
            raise PlannerRuntimeError(
                phase="input_validation",
                code="result_invariant_error",
                message="environment reservation belongs to another runtime",
            )
        entry = self._reservations.get(id(token))
        if entry is None or entry[0] is not token:
            raise PlannerRuntimeError(
                phase="result_serialization",
                code="result_invariant_error",
                message="environment reservation is unknown or already released",
            )
        return entry[1]

    def _read_peak_rss_bytes(self, *, phase: str) -> int:
        try:
            raw_value = self._peak_rss_raw()
            system = self._platform_system()
            raw = _finite("peak RSS reading", raw_value, minimum=0.0)
            integer_raw = int(raw)
            if raw != float(integer_raw):
                raise ValueError("peak RSS reading must be integral")
            if not isinstance(system, str) or not system:
                raise ValueError("platform system must be a non-empty string")
            return integer_raw if system == "Darwin" else integer_raw * 1024
        except PlannerRuntimeError:
            raise
        except (OSError, TypeError, ValueError, OverflowError) as exc:
            raise PlannerRuntimeError(
                phase=phase,
                code="invalid_input",
                message="peak RSS is unavailable on this execution platform",
            ) from exc

    @staticmethod
    def _validate_phase(phase: str) -> None:
        if phase not in FAILURE_PHASES:
            raise ValueError(f"unsupported planner resource phase: {phase}")

    def _require_open(self) -> None:
        if self._closed:
            raise PlannerRuntimeError(
                phase="result_serialization",
                code="result_invariant_error",
                message="planner resource guard is closed",
            )


@dataclass(frozen=True, slots=True, eq=False)
class _CandidateLeaseV1:
    candidate: PlantPlannerCandidateV1
    token: _EnvironmentReservationV1
    env: PlantEnvV2


class PlannerCandidateRegistryV1:
    """Authoritative candidate-to-token ownership outside mutable metadata."""

    def __init__(self, guard: _ResourceGuardV1) -> None:
        if type(guard) is not _ResourceGuardV1:
            raise TypeError("candidate registry requires an exact resource guard")
        self._guard = guard
        self._leases: dict[int, _CandidateLeaseV1] = {}
        self._environment_ids: dict[int, PlantEnvV2] = {}

    @property
    def live_candidates(self) -> int:
        return len(self._leases)

    def register(
        self,
        token: _EnvironmentReservationV1,
        candidate: PlantPlannerCandidateV1,
    ) -> None:
        if type(candidate) is not PlantPlannerCandidateV1:
            raise TypeError("registry candidate must be exact PlantPlannerCandidateV1")
        candidate.validate()
        env = candidate.env
        assert env is not None
        if id(candidate) in self._leases:
            raise PlannerRuntimeError(
                phase="beam_search",
                code="result_invariant_error",
                message="candidate identity is already registered",
            )
        if id(env) in self._environment_ids:
            raise PlannerRuntimeError(
                phase="beam_search",
                code="result_invariant_error",
                message="candidate environment identity is already registered",
            )
        if self._guard._lookup_token(token) is not env:
            raise PlannerRuntimeError(
                phase="beam_search",
                code="result_invariant_error",
                message="candidate environment does not match its reservation",
            )
        lease = _CandidateLeaseV1(candidate=candidate, token=token, env=env)
        try:
            self._leases[id(candidate)] = lease
            self._environment_ids[id(env)] = env
        except BaseException:
            if self._leases.get(id(candidate)) is lease:
                del self._leases[id(candidate)]
            if self._environment_ids.get(id(env)) is env:
                del self._environment_ids[id(env)]
            raise

    def validate_candidate(self, candidate: PlantPlannerCandidateV1) -> PlantEnvV2:
        lease = self._lookup(candidate)
        candidate.validate()
        if candidate.env is not lease.env:
            raise ValueError("candidate environment differs from its registered lease")
        if self._guard._lookup_token(lease.token) is not lease.env:
            raise PlannerRuntimeError(
                phase="beam_search",
                code="result_invariant_error",
                message="candidate reservation no longer owns its environment",
            )
        return lease.env

    def release(self, candidate: PlantPlannerCandidateV1) -> None:
        """Release exact registry authority without re-endorsing candidate data."""

        lease = self._lookup(candidate)
        candidate.env = None
        candidate._attestation = None
        self._guard.release_environment(lease.token, lease.env)
        del self._leases[id(candidate)]
        del self._environment_ids[id(lease.env)]

    def release_all(self) -> None:
        first_error: BaseException | None = None
        for lease in tuple(self._leases.values()):
            try:
                self.release(lease.candidate)
            except BaseException as exc:  # cleanup must continue across every lease
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _lookup(self, candidate: object) -> _CandidateLeaseV1:
        if type(candidate) is not PlantPlannerCandidateV1:
            raise TypeError("registry candidate must be exact PlantPlannerCandidateV1")
        lease = self._leases.get(id(candidate))
        if lease is None or lease.candidate is not candidate:
            raise PlannerRuntimeError(
                phase="result_serialization",
                code="result_invariant_error",
                message="candidate is not registered or was already released",
            )
        return lease


__all__ = [
    "A0GenerationV1",
    "PlannerCandidateRegistryV1",
    "PlantPlannerCandidateV1",
    "PlannerRuntimeAdmissionV1",
    "PlannerRuntimeError",
    "TerminalSnapshotV1",
    "admit_planner_runtime",
    "capture_terminal_snapshot",
    "extract_intermediate_rank",
    "step_action_record",
]
