"""Fail-closed two-stage solver adapter for the cyclic-market beta MPC."""

from __future__ import annotations

from .beta_mpc_model import BetaMPCModelV1
from .beta_mpc_schema import (
    BETA_MPC_SOLUTION_DOMAIN,
    BetaMPCInputError,
    BetaMPCPlanError,
    BetaMPCPlanV1,
    BetaMPCSettingsV1,
    BetaMPCSolveError,
)
from .economic_mpc_schema import (
    EconomicMPCInputError,
    EconomicMPCPlanError,
    EconomicMPCSolveError,
)
from .economic_mpc_solver import (
    _economic_mpc_plan_values_from_solve_result,
    solve_economic_mpc_stages,
)
from .mpc_schema import canonical_json


def solve_beta_mpc_model(
    model: BetaMPCModelV1,
    *,
    settings: BetaMPCSettingsV1,
) -> BetaMPCPlanV1:
    """Enforce returns in both stages and publish only beta-native plan evidence."""

    if not isinstance(model, BetaMPCModelV1):
        raise TypeError("model must be BetaMPCModelV1")
    if not isinstance(settings, BetaMPCSettingsV1):
        raise TypeError("settings must be BetaMPCSettingsV1")
    if model.settings_sha256 != settings.sha256:
        raise BetaMPCInputError("beta model/settings identity mismatch")
    try:
        solved = solve_economic_mpc_stages(
            model,
            settings=settings,
            solution_domain=BETA_MPC_SOLUTION_DOMAIN,
        )
    except EconomicMPCInputError as exc:
        raise BetaMPCInputError(str(exc)) from exc
    except EconomicMPCSolveError as exc:
        raise BetaMPCSolveError(str(exc), stages=exc.stages) from exc

    try:
        values = _economic_mpc_plan_values_from_solve_result(
            model,
            settings=settings,
            solved=solved,
            expected_solution_domain=BETA_MPC_SOLUTION_DOMAIN,
        )
        plan = BetaMPCPlanV1(
            model_sha256=values.model_sha256,
            forecast_sha256=values.forecast_sha256,
            settings_sha256=values.settings_sha256,
            assumptions_sha256=values.assumptions_sha256,
            horizon_intervals=values.horizon_intervals,
            deliveries=values.deliveries,
            stages=values.stages,
            steps=values.steps,
            primary_delivered_optimum_kg=values.primary_delivered_optimum_kg,
            selected_delivered_kg=values.selected_delivered_kg,
            delivery_lock_tolerance_kg=values.delivery_lock_tolerance_kg,
            predicted_cost_components_eur=values.predicted_cost_components_eur,
            terminal_ch4_equivalent_deficit_kg=(
                values.terminal_ch4_equivalent_deficit_kg
            ),
            terminal_battery_deficit_kwh=values.terminal_battery_deficit_kwh,
            price_projection=model.price_projection,
            market_issue_sha256=model.market_issue_sha256,
            cycle_observation_sha256=model.cycle_observation_sha256,
            beta_task_sha256=model.beta_task_sha256,
            cycle_return=model.cycle_return,
            selected_solution_vector=solved.economic_solution,
            reserve_evidence_json=(
                canonical_json(model.metadata()["uncertainty_policy"])
                if "uncertainty_policy" in model.metadata()
                else None
            ),
        )
    except (EconomicMPCPlanError, TypeError, ValueError) as exc:
        raise BetaMPCPlanError(
            f"beta-MPC selected plan failed independent validation: {exc}"
        ) from exc
    if plan.model_sha256 != model.sha256:
        raise BetaMPCPlanError("beta plan/model identity mismatch")
    return plan


__all__ = ["solve_beta_mpc_model"]
