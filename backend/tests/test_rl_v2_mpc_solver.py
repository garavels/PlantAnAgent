"""Fail-closed tests for the optional two-stage SciPy/HiGHS adapter."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest

from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.mpc_forecast import project_issued_mean_solar
from plant_backend.rl.v2.mpc_model import (
    MPCSparseRowV1,
    MPCVariableGroupV1,
    ServiceMPCModelV1,
)
from plant_backend.rl.v2.mpc_schema import (
    DEFAULT_SERVICE_MPC_SETPOINT_SETTINGS_V1,
    DEFAULT_SERVICE_MPC_SETTINGS_V1,
    MPCPlanV1,
    SERVICE_MPC_MODEL_VERSION,
    ServiceMPCInputError,
    ServiceMPCPlanError,
    ServiceMPCSolveError,
)
from plant_backend.rl.v2.mpc_solver import solve_service_mpc_model
import plant_backend.rl.v2.mpc_solver as mpc_solver


SETTINGS = DEFAULT_SERVICE_MPC_SETTINGS_V1
SETPOINT_SETTINGS = DEFAULT_SERVICE_MPC_SETPOINT_SETTINGS_V1


def _two_variable_model(
    *,
    names: tuple[str, str],
    upper: tuple[float, float],
    integrality: tuple[int, int] = (0, 0),
    rows: tuple[MPCSparseRowV1, ...],
) -> ServiceMPCModelV1:
    return ServiceMPCModelV1(
        model_version=SERVICE_MPC_MODEL_VERSION,
        settings_sha256=SETPOINT_SETTINGS.sha256,
        forecast_sha256="f" * 64,
        interval_s=900.0,
        horizon_intervals=1,
        deliveries=(),
        variable_names=names,
        variable_lower=(0.0, 0.0),
        variable_upper=upper,
        integrality=integrality,
        primary_objective=(0.0, 0.0),
        secondary_objective=(0.0, 0.0),
        groups=(MPCVariableGroupV1(name="fixture", indices=(0, 1)),),
        rows=rows,
        metadata_json="{}",
    )


@pytest.fixture(scope="module")
def tiny_model() -> ServiceMPCModelV1:
    env = PlantEnvV2(episode=EpisodeConfigV2(horizon_h=72.0, delivery_rate_kg_h=35.0))
    env.reset(seed=0)
    observation = env.controller_observation
    assert observation.forecast is not None
    forecast = project_issued_mean_solar(
        observation.forecast,
        boundary_time_s=observation.state.boundary.time_s,
        config=env.config,
        interval_s=SETTINGS.interval_s,
    )
    from plant_backend.rl.v2.mpc_model import build_service_mpc_model

    return build_service_mpc_model(
        observation,
        forecast,
        settings=SETTINGS,
        constraints=env.constraints,
        config=env.config,
        commitment_profile=env.episode_config.electrolyser_commitment,
        horizon_intervals=8,
    )


@pytest.fixture(scope="module")
def tiny_plan(tiny_model: ServiceMPCModelV1) -> MPCPlanV1:
    pytest.importorskip("scipy")
    return solve_service_mpc_model(tiny_model, settings=SETTINGS)


def test_tiny_model_completes_both_frozen_stages_and_publishes_evidence(
    tiny_model: ServiceMPCModelV1,
    tiny_plan: MPCPlanV1,
) -> None:
    assert tiny_plan.model_sha256 == tiny_model.sha256
    assert tiny_plan.settings_sha256 == SETTINGS.sha256
    assert tiny_plan.forecast_sha256 == tiny_model.forecast_sha256
    assert tiny_plan.horizon_intervals == 8
    assert len(tiny_plan.steps) == 8
    assert tiny_plan.deliveries == ()
    assert tiny_plan.primary_delivered_optimum_kg == 0.0
    assert tiny_plan.selected_delivered_kg == 0.0
    assert tiny_plan.minimum_commitment_fill == 1.0
    assert tiny_plan.stages[0].stage == "primary_delivery"
    assert tiny_plan.stages[1].stage == "secondary_tiebreak"
    assert all(stage.success and stage.status == 0 for stage in tiny_plan.stages)
    assert all(stage.elapsed_ns > 0 for stage in tiny_plan.stages)
    assert all(stage.scipy_version for stage in tiny_plan.stages)
    assert all(stage.highs_version for stage in tiny_plan.stages)
    assert all(stage.solution_sha256 is not None for stage in tiny_plan.stages)
    assert tiny_plan.stages[0].solution_sha256 != tiny_plan.stages[1].solution_sha256
    assert tiny_plan.stages[1].objective_value == pytest.approx(
        -tiny_plan.secondary_utility,
        abs=1e-9,
    )
    assert len(tiny_plan.sha256) == 64


def test_selected_steps_preserve_plan_schema_and_terminal_reserve(
    tiny_model: ServiceMPCModelV1,
    tiny_plan: MPCPlanV1,
) -> None:
    terminal_floor = tiny_model.metadata()["terminal_battery_energy_floor_kwh"]
    assert tiny_plan.steps[-1].battery_energy_end_kwh >= terminal_floor - 1e-7
    for step in tiny_plan.steps:
        assert step.solar_to_battery_kw >= 0.0
        assert step.battery_support_kw >= 0.0
        assert not (step.battery_charging and step.battery_support_kw > 1e-7)
        assert step.warmup is False or step.running is False


def test_solution_canonicalization_projects_scaled_noise_to_a_declared_bound(
    tiny_model: ServiceMPCModelV1,
) -> None:
    support_index = tiny_model.group("battery_support_kw")[0]
    raw = list(tiny_model.variable_lower)
    raw[support_index] = 3.9e-5

    projected = mpc_solver._canonical_solution(
        tiny_model,
        raw,
        absolute_tolerance=SETTINGS.solver_mip_feasibility_tolerance,
        relative_tolerance=SETTINGS.solver_mip_feasibility_tolerance,
    )

    assert projected[support_index] == 0.0
    raw[support_index] = 1.0
    retained = mpc_solver._canonical_solution(
        tiny_model,
        raw,
        absolute_tolerance=SETTINGS.solver_mip_feasibility_tolerance,
        relative_tolerance=SETTINGS.solver_mip_feasibility_tolerance,
    )
    assert retained[support_index] == 1.0


def test_feasible_raw_co2_state_survives_an_invalid_scaled_bound_projection() -> None:
    observed_state = 3.495807901682279e-05
    model = _two_variable_model(
        names=("co2_inventory_kg[1]", "co2_captured_kg[0]"),
        upper=(6_000.0, 1.0),
        rows=(
            MPCSparseRowV1(
                name="co2_balance[1]",
                lower=0.0,
                upper=0.0,
                coefficients=((0, 1.0), (1, -1.0)),
            ),
        ),
    )

    selected = mpc_solver._select_feasible_solution(
        model,
        (observed_state, observed_state),
        stage="secondary_tiebreak",
        settings=SETPOINT_SETTINGS,
    )

    assert selected.solution == (observed_state, observed_state)
    assert selected.evidence is not None
    assert selected.evidence.selected == "raw_feasible_projection_rejected"
    assert selected.evidence.projection_count == 1
    assert selected.evidence.max_projection_delta == observed_state
    assert selected.evidence.raw_audit.valid is True
    assert selected.evidence.projected_audit.valid is False
    assert selected.evidence.projected_audit.worst_row == "co2_balance[1]"


def test_feasible_gated_support_noise_still_selects_the_projected_vector() -> None:
    observed_support = 3.935447649175992e-05
    model = _two_variable_model(
        names=("battery_support_kw[0]", "running[0]"),
        upper=(1_250.0, 1.0),
        integrality=(0, 1),
        rows=(
            MPCSparseRowV1(
                name="battery_support_gated[0]",
                lower=None,
                upper=0.0,
                coefficients=((0, 1.0), (1, -1_250.0)),
            ),
        ),
    )

    selected = mpc_solver._select_feasible_solution(
        model,
        (observed_support, 0.0),
        stage="primary_delivery",
        settings=SETPOINT_SETTINGS,
    )

    assert selected.solution == (0.0, 0.0)
    assert selected.evidence is not None
    assert selected.evidence.selected == "projected"
    assert selected.evidence.projection_count == 1
    assert selected.evidence.max_projection_delta == observed_support
    assert selected.evidence.raw_audit.valid is True
    assert selected.evidence.projected_audit.valid is True


def test_materially_invalid_raw_vector_remains_rejected() -> None:
    model = _two_variable_model(
        names=("co2_inventory_kg[1]", "co2_captured_kg[0]"),
        upper=(6_000.0, 1.0),
        rows=(
            MPCSparseRowV1(
                name="co2_balance[1]",
                lower=0.0,
                upper=0.0,
                coefficients=((0, 1.0), (1, -1.0)),
            ),
        ),
    )

    with pytest.raises(ServiceMPCPlanError, match="raw solution.*residual audit"):
        mpc_solver._select_feasible_solution(
            model,
            (1e-3, 0.0),
            stage="primary_delivery",
            settings=SETPOINT_SETTINGS,
        )


def test_projection_that_breaks_a_stage_lock_selects_the_raw_vector() -> None:
    locked_value = 3.495807901682279e-05
    model = _two_variable_model(
        names=("locked_objective_value", "fixed_auxiliary"),
        upper=(6_000.0, 1.0),
        rows=(
            MPCSparseRowV1(
                name="fixed_auxiliary",
                lower=0.0,
                upper=0.0,
                coefficients=((1, 1.0),),
            ),
        ),
    )
    lock = MPCSparseRowV1(
        name="primary_delivery_lock",
        lower=locked_value,
        upper=locked_value,
        coefficients=((0, 1.0),),
    )

    selected = mpc_solver._select_feasible_solution(
        model,
        (locked_value, 0.0),
        stage="secondary_tiebreak",
        settings=SETPOINT_SETTINGS,
        extra_rows=(lock,),
    )

    assert selected.solution == (locked_value, 0.0)
    assert selected.evidence is not None
    assert selected.evidence.selected == "raw_feasible_projection_rejected"
    assert selected.evidence.projected_audit.worst_row == "primary_delivery_lock"


def test_historical_settings_retain_the_strict_projection_failure_contract() -> None:
    observed_state = 3.495807901682279e-05
    model = _two_variable_model(
        names=("co2_inventory_kg[1]", "co2_captured_kg[0]"),
        upper=(6_000.0, 1.0),
        rows=(
            MPCSparseRowV1(
                name="co2_balance[1]",
                lower=0.0,
                upper=0.0,
                coefficients=((0, 1.0), (1, -1.0)),
            ),
        ),
    )

    with pytest.raises(ServiceMPCPlanError, match="solution failed.*residual audit"):
        mpc_solver._select_feasible_solution(
            model,
            (observed_state, observed_state),
            stage="secondary_tiebreak",
            settings=SETTINGS,
        )


def test_setpoint_settings_publish_selection_evidence_for_both_stages(
    tiny_model: ServiceMPCModelV1,
) -> None:
    pytest.importorskip("scipy")
    setpoint_model = replace(tiny_model, settings_sha256=SETPOINT_SETTINGS.sha256)

    plan = solve_service_mpc_model(setpoint_model, settings=SETPOINT_SETTINGS)

    for stage in plan.stages:
        assert stage.solution_selection is not None
        selection = stage.solution_selection
        assert selection.raw_audit.valid is True
        assert selection.projected_audit.valid is True
        assert selection.selected == "projected"
        assert stage.to_dict()["solution_selection"] == selection.to_dict()


def test_solver_rejects_a_settings_identity_change_before_runtime_load(
    tiny_model: ServiceMPCModelV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = replace(SETTINGS, terminal_battery_soc_target=0.30)
    loaded = False

    def forbidden_load() -> object:
        nonlocal loaded
        loaded = True
        raise AssertionError("runtime must not load")

    monkeypatch.setattr(mpc_solver, "_load_scipy_runtime", forbidden_load)
    with pytest.raises(ServiceMPCInputError, match="identity mismatch"):
        solve_service_mpc_model(tiny_model, settings=changed)
    assert not loaded


def test_solver_rejects_a_model_outside_the_frozen_group_contract(
    tiny_model: ServiceMPCModelV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    malformed = replace(tiny_model, groups=tuple(reversed(tiny_model.groups)))
    loaded = False

    def forbidden_load() -> object:
        nonlocal loaded
        loaded = True
        raise AssertionError("runtime must not load")

    monkeypatch.setattr(mpc_solver, "_load_scipy_runtime", forbidden_load)
    with pytest.raises(ServiceMPCInputError, match="frozen formulation"):
        solve_service_mpc_model(malformed, settings=SETTINGS)
    assert not loaded


def test_missing_optional_runtime_raises_without_a_fallback(
    tiny_model: ServiceMPCModelV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = mpc_solver.importlib.import_module

    def selective_import(name: str) -> Any:
        if name == "scipy":
            raise ImportError("deliberately unavailable")
        return original_import(name)

    monkeypatch.setattr(mpc_solver.importlib, "import_module", selective_import)
    with pytest.raises(ServiceMPCSolveError, match="optional optimisation") as caught:
        solve_service_mpc_model(tiny_model, settings=SETTINGS)
    assert caught.value.stage_evidence is None


def test_infeasible_primary_stage_retains_failed_solver_evidence(
    tiny_model: ServiceMPCModelV1,
) -> None:
    pytest.importorskip("scipy")
    fixed_fill = tiny_model.group("minimum_commitment_fill")[0]
    impossible = replace(
        tiny_model,
        rows=(
            *tiny_model.rows,
            MPCSparseRowV1(
                name="deliberately_infeasible_fixture",
                lower=None,
                upper=0.0,
                coefficients=((fixed_fill, 1.0),),
            ),
        ),
    )

    with pytest.raises(ServiceMPCSolveError, match="primary_delivery") as caught:
        solve_service_mpc_model(impossible, settings=SETTINGS)
    evidence = caught.value.stage_evidence
    assert evidence is not None
    assert evidence.stage == "primary_delivery"
    assert not evidence.success
    assert evidence.status != 0
    assert evidence.elapsed_ns > 0
    assert evidence.solution_sha256 is None
    assert evidence.scipy_version
    assert evidence.highs_version


def test_solver_exception_is_typed_and_retains_stage_evidence(
    tiny_model: ServiceMPCModelV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    runtime = mpc_solver._load_scipy_runtime()

    def explode(**_: object) -> object:
        raise RuntimeError("deliberate solver exception")

    monkeypatch.setattr(
        mpc_solver,
        "_load_scipy_runtime",
        lambda: replace(runtime, milp=explode),
    )
    with pytest.raises(ServiceMPCSolveError, match="stage raised") as caught:
        solve_service_mpc_model(tiny_model, settings=SETTINGS)
    evidence = caught.value.stage_evidence
    assert evidence is not None
    assert evidence.stage == "primary_delivery"
    assert evidence.status == -1
    assert not evidence.success
    assert "deliberate solver exception" in evidence.message
    assert evidence.solution_sha256 is None


def test_status_zero_solution_must_pass_the_independent_residual_audit(
    tiny_model: ServiceMPCModelV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    runtime = mpc_solver._load_scipy_runtime()

    def invalid_success(**_: object) -> object:
        return SimpleNamespace(
            status=0,
            success=True,
            message="deliberately invalid optimal fixture",
            x=runtime.numpy.zeros(tiny_model.variable_count),
            fun=0.0,
            mip_dual_bound=0.0,
            mip_gap=0.0,
            mip_node_count=0,
        )

    monkeypatch.setattr(
        mpc_solver,
        "_load_scipy_runtime",
        lambda: replace(runtime, milp=invalid_success),
    )
    with pytest.raises(ServiceMPCSolveError, match="residual audit") as caught:
        solve_service_mpc_model(tiny_model, settings=SETTINGS)
    evidence = caught.value.stage_evidence
    assert evidence is not None
    assert caught.value.stages == (evidence,)
    assert evidence.stage == "primary_delivery"
    assert evidence.status == 0
    assert evidence.success is False
    assert evidence.solution_sha256 is None
    assert "independent adapter validation" in evidence.message


def test_time_limit_status_is_retained_without_a_solution_or_fallback(
    tiny_model: ServiceMPCModelV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    runtime = mpc_solver._load_scipy_runtime()

    def time_limit(**_: object) -> object:
        return SimpleNamespace(
            status=1,
            success=False,
            message="Time limit reached deliberately",
            x=None,
            fun=-2.0,
            mip_dual_bound=-3.0,
            mip_gap=0.5,
            mip_node_count=7,
        )

    monkeypatch.setattr(
        mpc_solver,
        "_load_scipy_runtime",
        lambda: replace(runtime, milp=time_limit),
    )
    with pytest.raises(ServiceMPCSolveError, match="status 1") as caught:
        solve_service_mpc_model(tiny_model, settings=SETTINGS)

    evidence = caught.value.stage_evidence
    assert evidence is not None
    assert caught.value.stages == (evidence,)
    assert evidence.stage == "primary_delivery"
    assert evidence.status == 1
    assert evidence.success is False
    assert evidence.objective_value == -2.0
    assert evidence.mip_dual_bound == -3.0
    assert evidence.mip_gap == 0.5
    assert evidence.mip_node_count == 7
    assert evidence.solution_sha256 is None


def test_solver_passes_the_hash_bound_highs_feasibility_contract(
    tiny_model: ServiceMPCModelV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    runtime = mpc_solver._load_scipy_runtime()
    observed_options: dict[str, object] = {}

    def stop_after_capture(**kwargs: object) -> object:
        options = kwargs.get("options")
        assert isinstance(options, dict)
        observed_options.update(options)
        return SimpleNamespace(
            status=1,
            success=False,
            message="captured declared solver options",
            x=None,
            fun=None,
            mip_dual_bound=None,
            mip_gap=None,
            mip_node_count=0,
        )

    monkeypatch.setattr(
        mpc_solver,
        "_load_scipy_runtime",
        lambda: replace(runtime, milp=stop_after_capture),
    )
    with pytest.raises(ServiceMPCSolveError, match="status 1"):
        solve_service_mpc_model(tiny_model, settings=SETTINGS)

    assert observed_options["mip_feasibility_tolerance"] == (
        SETTINGS.solver_mip_feasibility_tolerance
    )


def test_nonfinite_status_zero_vector_is_retained_as_validation_failure(
    tiny_model: ServiceMPCModelV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    runtime = mpc_solver._load_scipy_runtime()

    def nonfinite_success(**_: object) -> object:
        vector = runtime.numpy.zeros(tiny_model.variable_count)
        vector[0] = runtime.numpy.nan
        return SimpleNamespace(
            status=0,
            success=True,
            message="solver claimed optimality for non-finite fixture",
            x=vector,
            fun=0.0,
            mip_dual_bound=0.0,
            mip_gap=0.0,
            mip_node_count=0,
        )

    monkeypatch.setattr(
        mpc_solver,
        "_load_scipy_runtime",
        lambda: replace(runtime, milp=nonfinite_success),
    )
    with pytest.raises(ServiceMPCSolveError, match="not finite") as caught:
        solve_service_mpc_model(tiny_model, settings=SETTINGS)

    evidence = caught.value.stage_evidence
    assert evidence is not None
    assert evidence.status == 0
    assert evidence.success is False
    assert evidence.solution_sha256 is None
    assert "independent adapter validation" in evidence.message


def test_secondary_failure_retains_successful_primary_stage(
    tiny_model: ServiceMPCModelV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")
    runtime = mpc_solver._load_scipy_runtime()
    real_milp = runtime.milp
    calls = 0

    def fail_second(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_milp(**kwargs)
        return SimpleNamespace(
            status=1,
            success=False,
            message="deliberate secondary timeout",
            x=None,
            fun=None,
            mip_dual_bound=None,
            mip_gap=None,
            mip_node_count=3,
        )

    monkeypatch.setattr(
        mpc_solver,
        "_load_scipy_runtime",
        lambda: replace(runtime, milp=fail_second),
    )
    with pytest.raises(ServiceMPCSolveError, match="secondary_tiebreak") as caught:
        solve_service_mpc_model(tiny_model, settings=SETTINGS)

    assert calls == 2
    assert len(caught.value.stages) == 2
    primary, secondary = caught.value.stages
    assert primary.stage == "primary_delivery"
    assert primary.success is True
    assert primary.solution_sha256 is not None
    assert secondary.stage == "secondary_tiebreak"
    assert secondary.success is False
    assert secondary.status == 1
    assert caught.value.stage_evidence == secondary


def test_plan_extraction_failure_retains_both_successful_stages(
    tiny_model: ServiceMPCModelV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("scipy")

    def reject_plan(*_: object, **__: object) -> NoReturn:
        raise ServiceMPCPlanError("deliberate extraction rejection")

    monkeypatch.setattr(mpc_solver, "_extract_plan", reject_plan)
    with pytest.raises(ServiceMPCPlanError, match="extraction rejection") as caught:
        solve_service_mpc_model(tiny_model, settings=SETTINGS)

    assert len(caught.value.stages) == 2
    assert tuple(stage.stage for stage in caught.value.stages) == (
        "primary_delivery",
        "secondary_tiebreak",
    )
    assert all(stage.success for stage in caught.value.stages)
