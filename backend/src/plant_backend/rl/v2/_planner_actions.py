"""Frozen privileged action generation for the feasible-planner protocol.

This module is deliberately narrower than the planner itself.  It validates and
detaches the Part-I relaxation guidance, defines the immutable fifteen-template
library, and turns one template invocation into one :class:`ActionRecordV1`.
It never steps, resets, clones, or samples an environment.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import struct
from typing import Iterable, Mapping, Sequence

from plant_sim import Action, ElectrolyserCommitmentProfile

from ...controllers.base import ControlDecision
from .evaluation import LegacyHeuristicControllerV2
from .planner_schema import (
    ActionRecordV1,
    CanonicalPayloadV1,
    canonical_json,
    canonical_payload,
    require_sha256,
)
from .upper_bound import (
    DEFAULT_FEASIBILITY_TOLERANCE,
    DEFAULT_MARGINAL_SIGN_TOLERANCE,
    DEFAULT_OPTIMALITY_TOLERANCE_KG,
    DEFAULT_PHYSICAL_TOLERANCE,
    MODEL_KIND,
    MODEL_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    DeliveryRelaxationModel,
    DeliveryRelaxationResult,
)


ACTION_LIBRARY_DOMAIN = "planner-action-library-v1"
A0_CONTROLLER_DOMAIN = "planner-a0-controller-v1"
RELAXATION_RESULT_DOMAIN = "planner-relaxation-result-v1"
RELAXATION_WITNESS_DOMAIN = "planner-relaxation-witness-v1"
RELAXATION_GUIDANCE_DOMAIN = "planner-relaxation-guidance-v1"
PRIVILEGED_BINDING_SCOPE = "privileged_offline_development"
NONCANONICAL_BINDING_SCOPE = "noncanonical_unit_fixture"
UPPER_BOUND_SOLVER = "scipy.optimize.linprog(method='highs')"

ACTION_LIBRARY_SHA256 = (
    "805bc4a6203026783a0d2d6514d25c83188d9ca4476b7b9c6c7763b6718456f2"
)
A0_CONTROLLER_SHA256 = (
    "d5405d5661b847b67a2999159391fd89fe0543f3459627e7a3a8ab25865a93d5"
)

ACTION_LIBRARY_CANONICAL_JSON = (
    '{"formula_version":"future-truth-router-binary64-v1","schema_version":1,'
    '"templates":[{"generator":"legacy-live-v1","id":0,"name":"legacy_live",'
    '"stack_fraction":null},{"generator":"relaxation-projection-v1","id":1,'
    '"name":"relaxation_projection","stack_fraction":null},{"generator":'
    '"constant-idle-v1","id":2,"name":"idle","stack_fraction":null},'
    '{"generator":"constant-charge-v1","id":3,"name":"charge",'
    '"stack_fraction":null},{"generator":"stack-only-router-v1","id":4,'
    '"name":"warm_hold","stack_fraction":0.02},{"generator":'
    '"stack-only-router-v1","id":5,"name":"cold_start_hold",'
    '"stack_fraction":0.2},{"generator":"stack-only-router-v1","id":6,'
    '"name":"h2_stable","stack_fraction":0.4},{"generator":'
    '"stack-only-router-v1","id":7,"name":"h2_full","stack_fraction":1.0},'
    '{"generator":"dac-only-active-cap-router-v1","id":8,"name":"dac_full",'
    '"stack_fraction":null},{"generator":"feedstocks-router-v1","id":9,'
    '"name":"feedstocks_stable","stack_fraction":0.4},{"generator":'
    '"feedstocks-router-v1","id":10,"name":"feedstocks_full",'
    '"stack_fraction":1.0},{"generator":'
    '"methanator-only-active-cap-router-v1","id":11,"name":'
    '"methanate_full","stack_fraction":null},{"generator":'
    '"clipped-stoich-router-v1","id":12,"name":"clipped_stoich_40",'
    '"stack_fraction":0.4},{"generator":"clipped-stoich-router-v1","id":13,'
    '"name":"clipped_stoich_70","stack_fraction":0.7},{"generator":'
    '"clipped-stoich-router-v1","id":14,"name":"clipped_stoich_100",'
    '"stack_fraction":1.0}]}'
)

A0_CONTROLLER_CANONICAL_JSON = (
    '{"adapter":"legacy_v1_boundary_telemetry","description":"V1 operator '
    "heuristic adapted without behavior changes; it receives only the latest "
    'one-minute boundary telemetry and active safety constraints.","id":'
    '"legacy_heuristic","information_set":"latest one-minute V2 telemetry plus '
    "safety constraints; no preceding-interval aggregate, delivery context, or "
    'issued forecast","label":"Legacy forecast-blind heuristic","uses_forecast":false}'
)


@dataclass(frozen=True, slots=True)
class ActionTemplateV1:
    """One ordered frozen action-template descriptor."""

    id: int
    name: str
    generator: str
    stack_fraction: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "generator": self.generator,
            "stack_fraction": self.stack_fraction,
        }


ACTION_TEMPLATES = (
    ActionTemplateV1(0, "legacy_live", "legacy-live-v1", None),
    ActionTemplateV1(1, "relaxation_projection", "relaxation-projection-v1", None),
    ActionTemplateV1(2, "idle", "constant-idle-v1", None),
    ActionTemplateV1(3, "charge", "constant-charge-v1", None),
    ActionTemplateV1(4, "warm_hold", "stack-only-router-v1", 0.02),
    ActionTemplateV1(5, "cold_start_hold", "stack-only-router-v1", 0.20),
    ActionTemplateV1(6, "h2_stable", "stack-only-router-v1", 0.40),
    ActionTemplateV1(7, "h2_full", "stack-only-router-v1", 1.00),
    ActionTemplateV1(8, "dac_full", "dac-only-active-cap-router-v1", None),
    ActionTemplateV1(9, "feedstocks_stable", "feedstocks-router-v1", 0.40),
    ActionTemplateV1(10, "feedstocks_full", "feedstocks-router-v1", 1.00),
    ActionTemplateV1(
        11, "methanate_full", "methanator-only-active-cap-router-v1", None
    ),
    ActionTemplateV1(12, "clipped_stoich_40", "clipped-stoich-router-v1", 0.40),
    ActionTemplateV1(13, "clipped_stoich_70", "clipped-stoich-router-v1", 0.70),
    ActionTemplateV1(14, "clipped_stoich_100", "clipped-stoich-router-v1", 1.00),
)


def _library_object() -> dict[str, object]:
    return {
        "schema_version": 1,
        "formula_version": "future-truth-router-binary64-v1",
        "templates": [item.to_dict() for item in ACTION_TEMPLATES],
    }


ACTION_LIBRARY_PAYLOAD = canonical_payload(ACTION_LIBRARY_DOMAIN, _library_object())
A0_CONTROLLER_PAYLOAD = canonical_payload(
    A0_CONTROLLER_DOMAIN, json.loads(A0_CONTROLLER_CANONICAL_JSON)
)
if (
    ACTION_LIBRARY_PAYLOAD.canonical_json != ACTION_LIBRARY_CANONICAL_JSON
    or ACTION_LIBRARY_PAYLOAD.sha256 != ACTION_LIBRARY_SHA256
):  # pragma: no cover - an import-time protocol/source corruption guard
    raise RuntimeError("frozen planner action-library identity does not match")
if (
    A0_CONTROLLER_PAYLOAD.canonical_json != A0_CONTROLLER_CANONICAL_JSON
    or A0_CONTROLLER_PAYLOAD.sha256 != A0_CONTROLLER_SHA256
):  # pragma: no cover - an import-time protocol/source corruption guard
    raise RuntimeError("frozen planner A0-controller identity does not match")


_WITNESS_BLOCKS = (
    ("solar_to_battery_kw", "solar_to_battery_kw"),
    ("solar_to_electrolyser_kw", "solar_to_electrolyser_kw"),
    ("solar_to_dac_kw", "solar_to_dac_kw"),
    ("solar_to_methanator_kw", "solar_to_methanator_kw"),
    ("battery_to_electrolyser_kw", "battery_to_electrolyser_kw"),
    ("curtailed_solar_kw", "curtailed_solar_kw"),
    ("h2_produced_kg", "h2_produced_kg"),
    ("co2_captured_kg", "co2_captured_kg"),
    ("ch4_produced_kg", "ch4_produced_kg"),
    ("battery_energy_kwh", "battery_energy_kwh"),
    ("h2_inventory_kg", "h2_inventory_kg"),
    ("co2_inventory_kg", "co2_inventory_kg"),
    ("ch4_inventory_kg", "ch4_inventory_kg"),
)

_BINDING_FACTORY_TOKEN = object()
_ATTESTED_BINDINGS: dict[
    int,
    tuple[
        RelaxationBindingV1,
        str,
        CanonicalPayloadV1,
        CanonicalPayloadV1,
        CanonicalPayloadV1,
        str,
        str,
        str,
    ],
] = {}


def _finite_real(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def _finite_tuple(name: str, values: object, *, size: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{name} must be a finite numeric sequence")
    result = tuple(
        _finite_real(f"{name}[{index}]", item) for index, item in enumerate(values)
    )
    if len(result) != size:
        raise ValueError(f"{name} length does not match its model block")
    return result


def _same_binary64(left: object, right: object) -> bool:
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, Real)
        or not isinstance(right, Real)
    ):
        return False
    try:
        return struct.pack(">d", float(left)) == struct.pack(">d", float(right))
    except (TypeError, ValueError, OverflowError):
        return False


def _solar_digest(values: Sequence[float]) -> str:
    digest = hashlib.sha256()
    digest.update(b"plantanagent-v2-solar-kw-ieee754-v1\0")
    digest.update(struct.pack(">Q", len(values)))
    for value in values:
        digest.update(struct.pack(">d", value))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RelaxationBindingV1:
    """Factory-attested detached guidance for exactly one relaxation result.

    The required init-only token prevents ordinary construction.  The factory
    performs the expensive full canonical graph validation once, then registers
    a compact seal.  Every public action use re-derives that seal from all five
    exact binary64 vectors and every action-relevant scalar/payload identity.
    """

    binding_scope: str
    model_sha256: str
    result_sha256: str
    witness_sha256: str
    checkpoint_contract_digest: str
    exogenous_episode_sha256: str
    solar_truth_sha256: str
    physics_steps: int
    dt_h: float
    solar_available_kw: tuple[float, ...]
    solar_to_electrolyser_kw: tuple[float, ...]
    battery_to_electrolyser_kw: tuple[float, ...]
    solar_to_dac_kw: tuple[float, ...]
    solar_to_methanator_kw: tuple[float, ...]
    relaxation_result: CanonicalPayloadV1
    relaxation_witness: CanonicalPayloadV1
    relaxation_guidance: CanonicalPayloadV1
    _factory_token: InitVar[object]

    def __post_init__(self, _factory_token: object) -> None:
        if _factory_token is not _BINDING_FACTORY_TOKEN:
            raise TypeError(
                "RelaxationBindingV1 must be created by bind_relaxation_guidance"
            )
        _validate_binding_contents(self)

    @property
    def guidance_sha256(self) -> str:
        return str(self.relaxation_guidance.sha256)


def _mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must serialize to an object")
    return value


_PRIMARY_SOLVER_PROVENANCE_GAP = (
    "git source revision and dependency-lock digest must be attached by the "
    "evaluation runner"
)
_HIGHS_SOLVER_PROVENANCE_GAP = (
    "linked HiGHS runtime version is not exposed by this SciPy build"
)
_SOLVE_SETTINGS_FIELDS = {
    "feasibility_tolerance",
    "physical_tolerance",
    "optimality_tolerance_kg",
    "marginal_sign_tolerance",
    "solver_options",
    "highs_runtime_version",
    "provenance_gaps",
}


def _validate_solve_contract(
    *, solver: object, solver_version: object, solve_settings: object
) -> None:
    if solver != UPPER_BOUND_SOLVER:
        raise ValueError("relaxation result does not use the frozen HiGHS method")
    if not isinstance(solver_version, str) or not solver_version:
        raise ValueError("relaxation solver_version must be a non-empty string")
    settings = _mapping("relaxation solve settings", solve_settings)
    if set(settings) != _SOLVE_SETTINGS_FIELDS:
        raise ValueError("relaxation solve settings have an unexpected field set")
    expected_numbers = (
        ("feasibility_tolerance", DEFAULT_FEASIBILITY_TOLERANCE),
        ("physical_tolerance", DEFAULT_PHYSICAL_TOLERANCE),
        ("optimality_tolerance_kg", DEFAULT_OPTIMALITY_TOLERANCE_KG),
        ("marginal_sign_tolerance", DEFAULT_MARGINAL_SIGN_TOLERANCE),
    )
    for name, expected in expected_numbers:
        value = _finite_real(f"solve_settings.{name}", settings[name], minimum=0.0)
        if not _same_binary64(value, expected):
            raise ValueError(f"solve_settings.{name} does not match the frozen value")
    options = settings["solver_options"]
    if not isinstance(options, dict) or options:
        raise ValueError("solve_settings.solver_options must be exactly empty")
    highs_version = settings["highs_runtime_version"]
    if highs_version is not None and (
        not isinstance(highs_version, str) or not highs_version
    ):
        raise ValueError(
            "solve_settings.highs_runtime_version must be non-empty or null"
        )
    expected_gaps = [_PRIMARY_SOLVER_PROVENANCE_GAP]
    if highs_version is None:
        expected_gaps.append(_HIGHS_SOLVER_PROVENANCE_GAP)
    if settings["provenance_gaps"] != expected_gaps:
        raise ValueError("solve_settings.provenance_gaps do not match solver identity")


def _payload_object(
    name: str,
    payload: object,
    *,
    domain: str,
    digest: str | None = None,
) -> dict[str, object]:
    if not isinstance(payload, CanonicalPayloadV1):
        raise TypeError(f"{name} must be a CanonicalPayloadV1")
    CanonicalPayloadV1(payload.domain, payload.canonical_json, payload.sha256)
    if payload.domain != domain or (digest is not None and payload.sha256 != digest):
        raise ValueError(f"{name} identity is inconsistent")
    value = payload.value()
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain an object")
    return value


def _binary64_tuples_equal(left: Sequence[float], right: Sequence[float]) -> bool:
    if len(left) != len(right):
        return False
    return all(_same_binary64(a, b) for a, b in zip(left, right, strict=True))


def _validate_binding_contents(binding: RelaxationBindingV1) -> None:
    if binding.binding_scope not in (
        PRIVILEGED_BINDING_SCOPE,
        NONCANONICAL_BINDING_SCOPE,
    ):
        raise ValueError("relaxation binding_scope is invalid")
    for name in (
        "model_sha256",
        "result_sha256",
        "witness_sha256",
        "checkpoint_contract_digest",
        "exogenous_episode_sha256",
        "solar_truth_sha256",
    ):
        require_sha256(name, getattr(binding, name))
    physics_steps = _integer("physics_steps", binding.physics_steps, minimum=1)
    dt_h = _finite_real("dt_h", binding.dt_h, minimum=0.0)
    if dt_h == 0.0:
        raise ValueError("dt_h must be positive")
    for name in (
        "solar_available_kw",
        "solar_to_electrolyser_kw",
        "battery_to_electrolyser_kw",
        "solar_to_dac_kw",
        "solar_to_methanator_kw",
    ):
        values = getattr(binding, name)
        if not isinstance(values, tuple):
            raise TypeError(f"{name} must be an immutable tuple")
        checked = _finite_tuple(name, values, size=physics_steps)
        if any(value < 0.0 for value in checked):
            raise ValueError(f"{name} must contain non-negative powers")
    if _solar_digest(binding.solar_available_kw) != binding.solar_truth_sha256:
        raise ValueError("relaxation binding solar digest does not match its vector")

    witness = _payload_object(
        "relaxation_witness",
        binding.relaxation_witness,
        domain=RELAXATION_WITNESS_DOMAIN,
        digest=binding.witness_sha256,
    )
    projection_fields = (
        "solar_to_electrolyser_kw",
        "battery_to_electrolyser_kw",
        "solar_to_dac_kw",
        "solar_to_methanator_kw",
    )
    for name in projection_fields:
        witness_values = _finite_tuple(
            f"relaxation_witness.{name}", witness.get(name), size=physics_steps
        )
        if not _binary64_tuples_equal(getattr(binding, name), witness_values):
            raise ValueError(f"binding {name} does not match its witness payload")

    result = _payload_object(
        "relaxation_result",
        binding.relaxation_result,
        domain=RELAXATION_RESULT_DOMAIN,
        digest=binding.result_sha256,
    )
    if "elapsed_s" in result:
        raise ValueError("relaxation result binding must exclude elapsed_s")
    expected_result_fields = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "model_sha256": binding.model_sha256,
        "checkpoint_contract_digest": binding.checkpoint_contract_digest,
        "exogenous_episode_sha256": binding.exogenous_episode_sha256,
        "status": 0,
        "success": True,
        "failure_code": None,
        "preserve_initial_endowment": False,
        "is_task_upper_bound": True,
    }
    for name, expected in expected_result_fields.items():
        if result.get(name) != expected:
            raise ValueError(f"relaxation result {name} is not cross-bound")
    if (
        canonical_json(result.get("witness"))
        != binding.relaxation_witness.canonical_json
    ):
        raise ValueError("relaxation result witness is not byte-exactly cross-bound")
    canonical_verified = result.get("canonical_episode_verified")
    if not isinstance(canonical_verified, bool):
        raise TypeError("canonical_episode_verified must be a bool")
    expected_scope = (
        PRIVILEGED_BINDING_SCOPE if canonical_verified else NONCANONICAL_BINDING_SCOPE
    )
    if binding.binding_scope != expected_scope:
        raise ValueError("binding scope does not match canonical verification")
    certificate = _mapping("relaxation certificate", result.get("certificate"))
    ceiling = _finite_real(
        "relaxation result dual ceiling",
        result.get("dual_certified_delivery_upper_bound_kg"),
        minimum=0.0,
    )
    certificate_ceiling = _finite_real(
        "relaxation certificate dual ceiling",
        certificate.get("dual_certified_delivery_upper_bound_kg"),
        minimum=0.0,
    )
    if not _same_binary64(ceiling, certificate_ceiling):
        raise ValueError("result and certificate dual ceilings do not match")
    _validate_solve_contract(
        solver=result.get("solver"),
        solver_version=result.get("solver_version"),
        solve_settings=result.get("solve_settings"),
    )

    guidance = _payload_object(
        "relaxation_guidance",
        binding.relaxation_guidance,
        domain=RELAXATION_GUIDANCE_DOMAIN,
    )
    expected_guidance = {
        "schema_version": 1,
        "model_sha256": binding.model_sha256,
        "result_sha256": binding.result_sha256,
        "witness_sha256": binding.witness_sha256,
        "status": 0,
        "canonical_episode_verified": canonical_verified,
        "preserve_initial_endowment": False,
        "is_task_upper_bound": True,
        "solver": result["solver"],
        "solver_version": result["solver_version"],
        "solve_settings": result["solve_settings"],
        "checkpoint_contract_digest": binding.checkpoint_contract_digest,
        "exogenous_episode_sha256": binding.exogenous_episode_sha256,
        "solar_truth_sha256": binding.solar_truth_sha256,
    }
    if canonical_json(guidance) != canonical_json(expected_guidance):
        raise ValueError("relaxation guidance payload is not exactly cross-bound")


def _binding_seal(binding: RelaxationBindingV1) -> str:
    """Hash every compact identity and exact action-relevant binary64 vector."""

    digest = hashlib.sha256()
    digest.update(b"plantanagent:planner-relaxation-binding-seal-v1\0")

    def add_text(name: str, value: object) -> None:
        if not isinstance(value, str):
            raise TypeError(f"binding {name} must be a string")
        encoded_name = name.encode("utf-8")
        encoded_value = value.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(struct.pack(">Q", len(encoded_value)))
        digest.update(encoded_value)

    add_text("binding_scope", binding.binding_scope)
    for name in (
        "model_sha256",
        "result_sha256",
        "witness_sha256",
        "checkpoint_contract_digest",
        "exogenous_episode_sha256",
        "solar_truth_sha256",
    ):
        add_text(name, getattr(binding, name))
    digest.update(struct.pack(">Q", _integer("physics_steps", binding.physics_steps)))
    digest.update(struct.pack(">d", _finite_real("dt_h", binding.dt_h)))
    for name in (
        "solar_available_kw",
        "solar_to_electrolyser_kw",
        "battery_to_electrolyser_kw",
        "solar_to_dac_kw",
        "solar_to_methanator_kw",
    ):
        values = getattr(binding, name)
        if not isinstance(values, tuple):
            raise TypeError(f"binding {name} must be an immutable tuple")
        add_text("vector_name", name)
        digest.update(struct.pack(">Q", len(values)))
        for index, value in enumerate(values):
            digest.update(
                struct.pack(">d", _finite_real(f"binding {name}[{index}]", value))
            )
    for name in (
        "relaxation_result",
        "relaxation_witness",
        "relaxation_guidance",
    ):
        payload = getattr(binding, name)
        if not isinstance(payload, CanonicalPayloadV1):
            raise TypeError(f"binding {name} must be a CanonicalPayloadV1")
        add_text(f"{name}.domain", payload.domain)
        add_text(f"{name}.sha256", payload.sha256)
    return digest.hexdigest()


def _require_attested_binding(binding: object) -> RelaxationBindingV1:
    if not isinstance(binding, RelaxationBindingV1):
        raise TypeError("binding must be a RelaxationBindingV1")
    attestation = _ATTESTED_BINDINGS.get(id(binding))
    if attestation is None or attestation[0] is not binding:
        raise ValueError("relaxation binding was not created by the validated factory")
    (
        _,
        expected_seal,
        result_payload,
        witness_payload,
        guidance_payload,
        result_json,
        witness_json,
        guidance_json,
    ) = attestation
    if (
        binding.relaxation_result is not result_payload
        or binding.relaxation_witness is not witness_payload
        or binding.relaxation_guidance is not guidance_payload
        or binding.relaxation_result.canonical_json is not result_json
        or binding.relaxation_witness.canonical_json is not witness_json
        or binding.relaxation_guidance.canonical_json is not guidance_json
        or _binding_seal(binding) != expected_seal
    ):
        raise ValueError("relaxation binding no longer matches its factory attestation")
    return binding


def require_attested_binding(binding: object) -> RelaxationBindingV1:
    """Revalidate and return one factory-attested relaxation binding.

    The plant-specific planner runtime calls this before generating A0 so a
    copied, constructed, replaced, or mutated privileged binding fails before
    any environment outcome is observed.  The public boundary deliberately
    exposes validation, not the process-local attestation registry.
    """

    return _require_attested_binding(binding)


def _model_block_sizes(model: object) -> dict[str, int]:
    try:
        blocks = tuple(model.blocks)  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as exc:
        raise TypeError("relaxation model blocks are unavailable") from exc
    result: dict[str, int] = {}
    expected_start = 0
    for block in blocks:
        name = getattr(block, "name", None)
        start = _integer("model block start", getattr(block, "start", None))
        size = _integer("model block size", getattr(block, "size", None), minimum=1)
        if not isinstance(name, str) or not name or name in result:
            raise ValueError("relaxation model blocks must be named and unique")
        if start != expected_start:
            raise ValueError("relaxation model blocks must be contiguous and ordered")
        result[name] = size
        expected_start += size
    expected_names = {name for name, _ in _WITNESS_BLOCKS} | {"delivered_ch4_kg"}
    if set(result) != expected_names:
        raise ValueError("relaxation model block schema does not match the witness")
    return result


def bind_relaxation_guidance(
    model: DeliveryRelaxationModel,
    result: DeliveryRelaxationResult,
    *,
    allow_noncanonical_fixture: bool = False,
) -> RelaxationBindingV1:
    """Fail closed and attest exact successful task-upper-bound guidance.

    Structurally faithful fakes are admitted only for explicitly noncanonical
    unit fixtures.  Canonical guidance requires the concrete Part-I types and
    the exact frozen HiGHS solve contract; this function never invokes a solver.
    """

    if not isinstance(allow_noncanonical_fixture, bool):
        raise TypeError("allow_noncanonical_fixture must be a bool")
    if _integer("model.schema_version", model.schema_version) != MODEL_SCHEMA_VERSION:
        raise ValueError("relaxation model schema version is unsupported")
    if model.model_kind != MODEL_KIND:
        raise ValueError("relaxation model kind is unsupported")
    if (
        _integer("result.schema_version", result.schema_version)
        != RESULT_SCHEMA_VERSION
    ):
        raise ValueError("relaxation result schema version is unsupported")
    model_sha256 = require_sha256("model.model_sha256", model.model_sha256)
    result_model_sha256 = require_sha256("result.model_sha256", result.model_sha256)
    if model_sha256 != result_model_sha256:
        raise ValueError("relaxation model identity does not match its result")
    status = _integer("result.status", result.status)
    if result.success is not True or status != 0:
        raise ValueError("relaxation result is not a successful status-0 solve")
    if result.failure_code is not None:
        raise ValueError("successful relaxation result must not carry a failure code")
    if model.preserve_initial_endowment is not False:
        raise ValueError("relaxation model must be unconstrained by terminal endowment")
    if result.preserve_initial_endowment is not False:
        raise ValueError(
            "relaxation result must be unconstrained by terminal endowment"
        )
    if model.is_task_upper_bound is not True or result.is_task_upper_bound is not True:
        raise ValueError("relaxation guidance must be the task upper bound")
    if not isinstance(model.canonical_episode_verified, bool) or not isinstance(
        result.canonical_episode_verified, bool
    ):
        raise TypeError("canonical episode verification flags must be bools")
    if model.canonical_episode_verified != result.canonical_episode_verified:
        raise ValueError("relaxation canonical-verification status does not match")
    if model.canonical_episode_verified:
        if allow_noncanonical_fixture:
            raise ValueError("canonical guidance cannot use the fixture-only admission")
        if (
            type(model) is not DeliveryRelaxationModel
            or type(result) is not DeliveryRelaxationResult
        ):
            raise TypeError(
                "canonical guidance requires concrete upper-bound model/result types"
            )
        binding_scope = PRIVILEGED_BINDING_SCOPE
    else:
        if not allow_noncanonical_fixture:
            raise ValueError(
                "noncanonical guidance requires explicit fixture-only admission"
            )
        binding_scope = NONCANONICAL_BINDING_SCOPE
    checkpoint_digest = require_sha256(
        "model.checkpoint_contract_digest", model.checkpoint_contract_digest
    )
    result_checkpoint = require_sha256(
        "result.checkpoint_contract_digest", result.checkpoint_contract_digest
    )
    if checkpoint_digest != result_checkpoint:
        raise ValueError("relaxation checkpoint identities do not match")
    episode_sha256 = require_sha256(
        "model.exogenous_episode_sha256", model.exogenous_episode_sha256
    )
    result_episode = require_sha256(
        "result.exogenous_episode_sha256", result.exogenous_episode_sha256
    )
    if episode_sha256 != result_episode:
        raise ValueError("relaxation episode identities do not match")
    physics_steps = _integer("model.physics_steps", model.physics_steps, minimum=1)
    dt_h = _finite_real("model.dt_h", model.dt_h, minimum=0.0)
    if dt_h == 0.0:
        raise ValueError("model.dt_h must be positive")
    solar_truth_sha256 = require_sha256(
        "model.solar_truth_sha256", model.solar_truth_sha256
    )
    solar = _finite_tuple(
        "model.solar_available_kw", model.solar_available_kw, size=physics_steps
    )
    if any(value < 0.0 for value in solar):
        raise ValueError("model solar truth must be non-negative")
    if _solar_digest(solar) != solar_truth_sha256:
        raise ValueError("model solar truth digest does not match its vector")

    witness = result.witness
    if witness is None:
        raise ValueError("successful relaxation result has no witness")
    if result.certificate is None:
        raise ValueError("successful relaxation result has no dual certificate")
    ceiling = _finite_real(
        "result.dual_certified_delivery_upper_bound_kg",
        result.dual_certified_delivery_upper_bound_kg,
        minimum=0.0,
    )
    certificate_ceiling = _finite_real(
        "result.certificate.dual_certified_delivery_upper_bound_kg",
        result.certificate.dual_certified_delivery_upper_bound_kg,
        minimum=0.0,
    )
    if not _same_binary64(ceiling, certificate_ceiling):
        raise ValueError("result and certificate dual ceilings do not match")

    block_sizes = _model_block_sizes(model)
    witness_vectors: dict[str, tuple[float, ...]] = {}
    for block_name, witness_name in _WITNESS_BLOCKS:
        values = _finite_tuple(
            f"witness.{witness_name}",
            getattr(witness, witness_name, None),
            size=block_sizes[block_name],
        )
        if any(value < 0.0 for value in values):
            raise ValueError(f"witness.{witness_name} must be non-negative")
        witness_vectors[witness_name] = values
    try:
        normalized_size = int(model.program.variable_count)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("relaxation model variable count is unavailable") from exc
    if normalized_size != sum(block_sizes.values()):
        raise ValueError("relaxation variable count does not match its blocks")
    _finite_tuple(
        "witness.normalized_variables",
        witness.normalized_variables,
        size=normalized_size,
    )
    commitments = tuple(model.commitments)
    deliveries = tuple(witness.deliveries)
    if len(deliveries) != len(commitments):
        raise ValueError("witness deliveries do not match model commitments")

    witness_object = _mapping("relaxation witness", witness.to_dict())
    witness_payload = canonical_payload(RELAXATION_WITNESS_DOMAIN, witness_object)
    result_object = dict(_mapping("relaxation result", result.to_dict()))
    if "elapsed_s" not in result_object:
        raise ValueError("relaxation result payload is missing elapsed_s")
    result_object.pop("elapsed_s")
    result_payload = canonical_payload(RELAXATION_RESULT_DOMAIN, result_object)

    solve_settings = _mapping(
        "relaxation solve settings", result.solve_settings.to_dict()
    )
    _validate_solve_contract(
        solver=result.solver,
        solver_version=result.solver_version,
        solve_settings=solve_settings,
    )
    guidance_object = {
        "schema_version": 1,
        "model_sha256": model_sha256,
        "result_sha256": result_payload.sha256,
        "witness_sha256": witness_payload.sha256,
        "status": status,
        "canonical_episode_verified": result.canonical_episode_verified,
        "preserve_initial_endowment": result.preserve_initial_endowment,
        "is_task_upper_bound": result.is_task_upper_bound,
        "solver": result.solver,
        "solver_version": result.solver_version,
        "solve_settings": dict(solve_settings),
        "checkpoint_contract_digest": checkpoint_digest,
        "exogenous_episode_sha256": episode_sha256,
        "solar_truth_sha256": solar_truth_sha256,
    }
    guidance_payload = canonical_payload(RELAXATION_GUIDANCE_DOMAIN, guidance_object)
    binding = RelaxationBindingV1(
        binding_scope=binding_scope,
        model_sha256=model_sha256,
        result_sha256=result_payload.sha256,
        witness_sha256=witness_payload.sha256,
        checkpoint_contract_digest=checkpoint_digest,
        exogenous_episode_sha256=episode_sha256,
        solar_truth_sha256=solar_truth_sha256,
        physics_steps=physics_steps,
        dt_h=dt_h,
        solar_available_kw=solar,
        solar_to_electrolyser_kw=witness_vectors["solar_to_electrolyser_kw"],
        battery_to_electrolyser_kw=witness_vectors["battery_to_electrolyser_kw"],
        solar_to_dac_kw=witness_vectors["solar_to_dac_kw"],
        solar_to_methanator_kw=witness_vectors["solar_to_methanator_kw"],
        relaxation_result=result_payload,
        relaxation_witness=witness_payload,
        relaxation_guidance=guidance_payload,
        _factory_token=_BINDING_FACTORY_TOKEN,
    )
    _ATTESTED_BINDINGS[id(binding)] = (
        binding,
        _binding_seal(binding),
        binding.relaxation_result,
        binding.relaxation_witness,
        binding.relaxation_guidance,
        binding.relaxation_result.canonical_json,
        binding.relaxation_witness.canonical_json,
        binding.relaxation_guidance.canonical_json,
    )
    return binding


def template_descriptor(template_id: int) -> ActionTemplateV1:
    """Return the descriptor at one frozen zero-based template ID."""

    index = _integer("template_id", template_id)
    if index >= len(ACTION_TEMPLATES):
        raise ValueError("template_id must be between 0 and 14")
    descriptor = ACTION_TEMPLATES[index]
    if descriptor.id != index:  # pragma: no cover - source-corruption guard
        raise RuntimeError("planner action-template order is corrupt")
    return descriptor


def action_binary64(action: Action) -> bytes:
    """Return the protocol's packed five-field binary64 action identity."""

    if not isinstance(action, Action):
        raise TypeError("action must be an Action")
    return struct.pack(
        ">5d",
        _finite_real("action.battery_share", action.battery_share),
        _finite_real("action.electrolyser_share", action.electrolyser_share),
        _finite_real("action.dac_share", action.dac_share),
        _finite_real("action.methanator_share", action.methanator_share),
        _finite_real("action.battery_support_kw", action.battery_support_kw),
    )


def actions_binary64_equal(left: Action, right: Action) -> bool:
    return action_binary64(left) == action_binary64(right)


def action_record_to_action(record: ActionRecordV1) -> Action:
    """Reconstruct the exact concrete request held by an action record."""

    if not isinstance(record, ActionRecordV1):
        raise TypeError("record must be an ActionRecordV1")
    return Action(
        battery_share=record.battery_share,
        electrolyser_share=record.electrolyser_share,
        dac_share=record.dac_share,
        methanator_share=record.methanator_share,
        battery_support_kw=record.battery_support_kw,
    )


def proportional_close(
    raw_shares: Iterable[float],
) -> tuple[float, float, float, float]:
    """Apply the frozen bounded common-scale closure to four raw shares."""

    shares = tuple(
        _finite_real(f"raw_shares[{index}]", value, minimum=0.0)
        for index, value in enumerate(raw_shares)
    )
    if len(shares) != 4:
        raise ValueError("raw_shares must contain exactly four values")
    total = 0.0
    for value in shares:
        total += value
        if not math.isfinite(total):
            raise ValueError("raw share sum must remain finite")
    if total <= 1.0:
        return shares[0], shares[1], shares[2], shares[3]
    scale = math.nextafter(1.0 / total, 0.0)
    for _ in range(8):
        closed = (
            shares[0] * scale,
            shares[1] * scale,
            shares[2] * scale,
            shares[3] * scale,
        )
        closed_total = 0.0
        for value in closed:
            closed_total += value
        if closed_total <= 1.0:
            return closed
        scale = math.nextafter(scale, 0.0)
    raise ValueError("proportional share closure did not converge in eight iterations")


def _decision_bounds(env: object, binding: RelaxationBindingV1) -> tuple[int, int]:
    start = _integer(
        "env.physics_steps_taken", getattr(env, "physics_steps_taken", None)
    )
    try:
        stride_raw = env._episode.decision_stride  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise ValueError("environment has no reset episode decision stride") from exc
    stride = _integer("episode.decision_stride", stride_raw, minimum=1)
    if start >= binding.physics_steps:
        raise ValueError("current decision interval is empty")
    stop = min(start + stride, binding.physics_steps)
    if stop <= start:
        raise ValueError("current decision interval is empty")
    return start, stop


def projection_means(
    binding: RelaxationBindingV1,
    start: int,
    stop: int,
) -> tuple[float, float, float]:
    """Compute the exact left-to-right Part-I projection over ``[start, stop)``."""

    binding = _require_attested_binding(binding)
    return _projection_means_validated(binding, start, stop)


def _projection_means_validated(
    binding: RelaxationBindingV1,
    start: int,
    stop: int,
) -> tuple[float, float, float]:
    left = _integer("start", start)
    right = _integer("stop", stop)
    if left >= right or right > binding.physics_steps:
        raise ValueError("projection bounds must be a non-empty in-range interval")
    electrolyser = 0.0
    dac = 0.0
    methanator = 0.0
    for index in range(left, right):
        term = (
            binding.solar_to_electrolyser_kw[index]
            + binding.battery_to_electrolyser_kw[index]
        )
        _finite_real(f"projection electrolyser term[{index}]", term)
        electrolyser += term
        dac += binding.solar_to_dac_kw[index]
        methanator += binding.solar_to_methanator_kw[index]
        _finite_real("projection electrolyser accumulator", electrolyser)
        _finite_real("projection DAC accumulator", dac)
        _finite_real("projection methanator accumulator", methanator)
    count = right - left
    return electrolyser / count, dac / count, methanator / count


def _active_caps(env: object) -> tuple[float, float, float]:
    try:
        config = env.config  # type: ignore[attr-defined]
        constraints = env.constraints  # type: ignore[attr-defined]
        caps = (
            constraints.electrolyser_limit_kw(config),
            constraints.dac_limit_kw(config),
            constraints.methanator_limit_kw(config),
        )
    except AttributeError as exc:
        raise TypeError(
            "environment lacks planner configuration or constraints"
        ) from exc
    return (
        _finite_real("active electrolyser cap", caps[0], minimum=0.0),
        _finite_real("active DAC cap", caps[1], minimum=0.0),
        _finite_real("active methanator cap", caps[2], minimum=0.0),
    )


def target_powers(*, env: object, stack_fraction: float) -> tuple[float, float, float]:
    """Return cap-resolved ``(E, D, M)`` for one named stack fraction."""

    q = _finite_real("stack_fraction", stack_fraction, minimum=0.0)
    if q > 1.0:
        raise ValueError("stack_fraction must not exceed one")
    ce, cd, cm = _active_caps(env)
    try:
        config = env.config  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("environment lacks the plant configuration") from exc
    profile = _commitment_profile(env)
    rated = _finite_real(
        "electrolyser rated power", config.electrolyser.rated_kw, minimum=0.0
    )
    if rated == 0.0:
        raise ValueError("electrolyser rated power must be positive")
    requested_electrolyser = _finite_real(
        "fractional electrolyser target", q * rated, minimum=0.0
    )
    electrolyser = min(ce, requested_electrolyser)
    configured_specific_energy = _finite_real(
        "configured stack specific energy",
        config.electrolyser.specific_energy_kwh_per_kg,
        minimum=0.0,
    )
    nominal_specific_energy = _finite_real(
        "profile nominal specific energy",
        profile.nominal_specific_energy_kwh_per_kg,
        minimum=0.0,
    )
    specific_energy_base = max(configured_specific_energy, nominal_specific_energy)
    multiplier = _finite_real(
        "profile specific-energy multiplier",
        profile.specific_energy_multiplier(electrolyser / rated),
        minimum=0.0,
    )
    specific_energy = _finite_real(
        "effective stack specific energy",
        specific_energy_base * multiplier,
        minimum=0.0,
    )
    if specific_energy == 0.0:
        raise ValueError("effective stack specific energy must be positive")
    hydrogen = _finite_real(
        "hydrogen target", electrolyser / specific_energy, minimum=0.0
    )
    h2_per_ch4 = _finite_real(
        "methanator H2 ratio",
        config.methanator.h2_kg_per_kg_ch4,
        minimum=0.0,
    )
    if h2_per_ch4 == 0.0:
        raise ValueError("methanator H2 ratio must be positive")
    methane = _finite_real("methane target", hydrogen / h2_per_ch4, minimum=0.0)
    co2_per_ch4 = _finite_real(
        "methanator CO2 ratio",
        config.methanator.co2_kg_per_kg_ch4,
        minimum=0.0,
    )
    dac_specific_energy = _finite_real(
        "DAC specific energy", config.dac.specific_energy_kwh_per_kg, minimum=0.0
    )
    methanator_specific_energy = _finite_real(
        "methanator specific energy",
        config.methanator.specific_energy_kwh_per_kg,
        minimum=0.0,
    )
    dac_uncapped = _finite_real(
        "uncapped DAC target",
        methane * co2_per_ch4 * dac_specific_energy,
        minimum=0.0,
    )
    methanator_uncapped = _finite_real(
        "uncapped methanator target",
        methane * methanator_specific_energy,
        minimum=0.0,
    )
    dac = min(cd, dac_uncapped)
    methanator = min(cm, methanator_uncapped)
    return electrolyser, dac, methanator


def _commitment_profile(env: object) -> ElectrolyserCommitmentProfile:
    try:
        profile = env.episode_config.electrolyser_commitment  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("environment lacks the exact commitment profile") from exc
    if profile is None:
        raise ValueError("planner templates require an electrolyser commitment profile")
    if not isinstance(profile, ElectrolyserCommitmentProfile):
        raise TypeError("environment commitment profile has the wrong concrete type")
    expected = (
        ("hot_standby_load_frac", 0.02),
        ("start_load_frac", 0.20),
        ("min_stable_load_frac", 0.40),
    )
    for name, frozen in expected:
        value = _finite_real(f"profile.{name}", getattr(profile, name, None))
        if not _same_binary64(value, frozen):
            raise ValueError(f"profile.{name} does not match the frozen action library")
    return profile


def future_truth_router(
    *,
    env: object,
    binding: RelaxationBindingV1,
    start: int,
    stop: int,
    electrolyser_kw: float,
    dac_kw: float,
    methanator_kw: float,
) -> Action:
    """Route a non-negative target triple with the frozen future-solar rule."""

    binding = _require_attested_binding(binding)
    return _future_truth_router_validated(
        env=env,
        binding=binding,
        start=start,
        stop=stop,
        electrolyser_kw=electrolyser_kw,
        dac_kw=dac_kw,
        methanator_kw=methanator_kw,
    )


def _future_truth_router_validated(
    *,
    env: object,
    binding: RelaxationBindingV1,
    start: int,
    stop: int,
    electrolyser_kw: float,
    dac_kw: float,
    methanator_kw: float,
) -> Action:
    left = _integer("start", start)
    right = _integer("stop", stop)
    if left >= right or right > binding.physics_steps:
        raise ValueError("router bounds must be a non-empty in-range interval")
    ce, cd, cm = _active_caps(env)
    electrolyser = min(
        _finite_real("electrolyser target", electrolyser_kw, minimum=0.0), ce
    )
    dac = min(_finite_real("DAC target", dac_kw, minimum=0.0), cd)
    methanator = min(_finite_real("methanator target", methanator_kw, minimum=0.0), cm)

    solar = binding.solar_available_kw[left]
    for index in range(left + 1, right):
        candidate = binding.solar_available_kw[index]
        if candidate > solar:
            solar = candidate
    try:
        config = env.config  # type: ignore[attr-defined]
        constraints = env.constraints  # type: ignore[attr-defined]
        state = env.simulator.state  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("environment lacks the current planner boundary") from exc
    capacity = _finite_real(
        "boundary battery capacity", state.battery_capacity_kwh, minimum=0.0
    )
    reserve_energy = constraints.reserve_soc(config) * capacity
    reserve_energy = _finite_real("reserve energy", reserve_energy, minimum=0.0)
    battery_energy = _finite_real(
        "boundary battery energy", state.battery_energy_kwh, minimum=0.0
    )
    usable_energy = max(0.0, battery_energy - reserve_energy)
    config_dt_h = _finite_real("config.dt_h", config.dt_h, minimum=0.0)
    if config_dt_h == 0.0:
        raise ValueError("config.dt_h must be positive")
    if not _same_binary64(config_dt_h, binding.dt_h):
        raise ValueError("environment timestep does not match relaxation guidance")
    held_h = _finite_real(
        "decision hold hours", (right - left) * config_dt_h, minimum=0.0
    )
    if held_h == 0.0:
        raise ValueError("decision hold hours must be positive")
    discharge_efficiency = _finite_real(
        "battery discharge efficiency",
        config.battery.discharge_efficiency,
        minimum=0.0,
    )
    support_limit = _finite_real(
        "active battery support cap",
        constraints.support_limit_kw(config),
        minimum=0.0,
    )
    sustainable_support = _finite_real(
        "sustainable battery support",
        usable_energy * discharge_efficiency / held_h,
        minimum=0.0,
    )
    energy_support = math.nextafter(sustainable_support, 0.0)
    support_max = min(support_limit, ce, energy_support)
    support = min(electrolyser, support_max)
    process_route = _finite_real(
        "combined process route", (electrolyser + dac) + methanator, minimum=0.0
    )
    routed = max(solar, process_route)
    if routed == 0.0:
        return Action.idle()
    electrolyser_share = _finite_real(
        "electrolyser share", (electrolyser - support) / routed, minimum=0.0
    )
    dac_share = _finite_real("DAC share", dac / routed, minimum=0.0)
    methanator_share = _finite_real(
        "methanator share", methanator / routed, minimum=0.0
    )
    battery_share = max(
        0.0,
        1.0 - ((electrolyser_share + dac_share) + methanator_share),
    )
    closed = proportional_close(
        (battery_share, electrolyser_share, dac_share, methanator_share)
    )
    action = Action(
        battery_share=closed[0],
        electrolyser_share=closed[1],
        dac_share=closed[2],
        methanator_share=closed[3],
        battery_support_kw=support,
    )
    sanitised = Action.sanitised(action)
    if not actions_binary64_equal(action, sanitised):
        raise ValueError("generated action is not closed under Action.sanitised")
    return action


route_targets = future_truth_router


def _validated_legacy_action(env: object) -> Action:
    controller = LegacyHeuristicControllerV2()
    _require_a0_descriptor(controller)
    controller.reset()
    _require_a0_descriptor(controller)
    try:
        observation = env.controller_observation.without_forecast()  # type: ignore[attr-defined]
        constraints = env.constraints  # type: ignore[attr-defined]
        config = env.config  # type: ignore[attr-defined]
        commitment_profile = env.episode_config.electrolyser_commitment  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("environment lacks the forecast-blind legacy seam") from exc
    if not hasattr(observation, "forecast") or observation.forecast is not None:
        raise ValueError("legacy_live observation must be forecast-blind")
    decision = controller.compute_action(
        observation,
        constraints=constraints,
        config=config,
        commitment_profile=commitment_profile,
    )
    _require_a0_descriptor(controller)
    if not isinstance(decision, ControlDecision):
        raise TypeError("legacy_live must return a ControlDecision")
    if not isinstance(decision.action, Action):
        raise TypeError("legacy_live ControlDecision must contain an Action")
    action_binary64(decision.action)
    return decision.action


def _require_a0_descriptor(controller: object) -> None:
    try:
        descriptor = controller.to_dict()  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise TypeError("legacy controller has no descriptor") from exc
    if canonical_json(descriptor) != A0_CONTROLLER_CANONICAL_JSON:
        raise ValueError("legacy controller descriptor does not match frozen A0")


def _targets_for_template(
    template_id: int,
    *,
    env: object,
    binding: RelaxationBindingV1,
    start: int,
    stop: int,
) -> tuple[float, float, float]:
    ce, cd, cm = _active_caps(env)
    if template_id == 1:
        return _projection_means_validated(binding, start, stop)
    if template_id in (4, 5):
        profile = _commitment_profile(env)
        q = (
            profile.hot_standby_load_frac
            if template_id == 4
            else profile.start_load_frac
        )
        rated = _finite_real(
            "electrolyser rated power",
            env.config.electrolyser.rated_kw,  # type: ignore[attr-defined]
            minimum=0.0,
        )
        requested = _finite_real(
            "hold electrolyser target",
            _finite_real("profile stack fraction", q, minimum=0.0) * rated,
            minimum=0.0,
        )
        return min(ce, requested), 0.0, 0.0
    if template_id in (6, 9):
        profile = _commitment_profile(env)
        target = target_powers(env=env, stack_fraction=profile.min_stable_load_frac)
        return target if template_id == 9 else (target[0], 0.0, 0.0)
    if template_id in (7, 10):
        target = target_powers(env=env, stack_fraction=1.0)
        return target if template_id == 10 else (target[0], 0.0, 0.0)
    if template_id == 8:
        return 0.0, cd, 0.0
    if template_id == 11:
        return 0.0, 0.0, cm
    if template_id in (12, 13, 14):
        fraction = ACTION_TEMPLATES[template_id].stack_fraction
        if fraction is None:  # pragma: no cover - frozen-source corruption guard
            raise RuntimeError("clipped-stoichiometric template lacks its fraction")
        return target_powers(env=env, stack_fraction=fraction)
    raise ValueError("template does not use routed target powers")


def generate_action_record(
    *,
    template_id: int,
    env: object,
    decision_index: int,
    source_phase: str,
    binding: RelaxationBindingV1,
) -> ActionRecordV1:
    """Generate one frozen template request without advancing ``env``."""

    descriptor = template_descriptor(template_id)
    index = _integer("decision_index", decision_index)
    env_index = _integer("env.decisions_taken", getattr(env, "decisions_taken", None))
    if index != env_index:
        raise ValueError("decision_index must equal the environment decision count")
    if source_phase not in ("beam", "coarse_polish", "fine_polish"):
        raise ValueError("planner template source_phase is invalid")
    binding = _require_attested_binding(binding)
    start, stop = _decision_bounds(env, binding)

    if descriptor.id == 0:
        action = _validated_legacy_action(env)
        source_binding_sha256 = A0_CONTROLLER_PAYLOAD.sha256
    elif descriptor.id == 2:
        action = Action.idle()
        source_binding_sha256 = ACTION_LIBRARY_PAYLOAD.sha256
    elif descriptor.id == 3:
        action = Action.all_to_battery()
        source_binding_sha256 = ACTION_LIBRARY_PAYLOAD.sha256
    else:
        targets = _targets_for_template(
            descriptor.id,
            env=env,
            binding=binding,
            start=start,
            stop=stop,
        )
        action = _future_truth_router_validated(
            env=env,
            binding=binding,
            start=start,
            stop=stop,
            electrolyser_kw=targets[0],
            dac_kw=targets[1],
            methanator_kw=targets[2],
        )
        source_binding_sha256 = (
            binding.guidance_sha256
            if descriptor.id == 1
            else ACTION_LIBRARY_PAYLOAD.sha256
        )
    action_binary64(action)
    return ActionRecordV1(
        decision_index=index,
        source_phase=source_phase,
        source_code=descriptor.id,
        source_binding_sha256=source_binding_sha256,
        controller_invalid=False,
        battery_share=action.battery_share,
        electrolyser_share=action.electrolyser_share,
        dac_share=action.dac_share,
        methanator_share=action.methanator_share,
        battery_support_kw=action.battery_support_kw,
    )


__all__ = [
    "A0_CONTROLLER_CANONICAL_JSON",
    "A0_CONTROLLER_DOMAIN",
    "A0_CONTROLLER_PAYLOAD",
    "A0_CONTROLLER_SHA256",
    "ACTION_LIBRARY_CANONICAL_JSON",
    "ACTION_LIBRARY_DOMAIN",
    "ACTION_LIBRARY_PAYLOAD",
    "ACTION_LIBRARY_SHA256",
    "ACTION_TEMPLATES",
    "ActionTemplateV1",
    "NONCANONICAL_BINDING_SCOPE",
    "PRIVILEGED_BINDING_SCOPE",
    "RelaxationBindingV1",
    "UPPER_BOUND_SOLVER",
    "action_binary64",
    "action_record_to_action",
    "actions_binary64_equal",
    "bind_relaxation_guidance",
    "future_truth_router",
    "generate_action_record",
    "projection_means",
    "proportional_close",
    "require_attested_binding",
    "route_targets",
    "target_powers",
    "template_descriptor",
]
