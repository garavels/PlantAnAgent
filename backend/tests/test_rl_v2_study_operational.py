"""Independent arithmetic, missingness and censoring fixtures; no plant resets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from plant_backend.rl.v2 import beta_spaces
from plant_backend.rl.v2.beta_config import CyclicMarketTaskConfigV1, RESOURCE_NAMES
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.economics import EconomicEpisodeMetricsV1
from plant_backend.rl.v2.metrics import EpisodeMetricsV2
from plant_backend.rl.v2.study import operational as op
from plant_backend.rl.v2.study import training_scorecards as training
from plant_backend.rl.v2.study.contract import (
    BETA_CORE_ALGORITHM_IDS,
    BETA_OPTIMIZER_SEEDS,
    COMPONENT_NAMES,
)
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.training.custody_json import authenticate

SEEDS = list(range(9_000_100, 9_000_132))


@pytest.fixture(autouse=True)
def no_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("operational arithmetic must not instantiate or reset a plant")

    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "__init__", forbidden)
    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "reset", forbidden)


@pytest.fixture
def arithmetic_episode() -> dict[str, Any]:
    config = ControllerNeutralTrainingConfigV1(
        environment_id=beta_spaces.ENVIRONMENT_ID,
        episode=EpisodeConfigV2(horizon_h=168.0, decision_interval_s=3600.0),
        beta_task=CyclicMarketTaskConfigV1(),
    )
    physical = EpisodeMetricsV2(
        decisions=168,
        physical_steps=10080,
        hours=168,
        corrected_physics_steps=3,
        guard_interventions=7,
    ).to_dict()
    economic = EconomicEpisodeMetricsV1(0, 0, 1).to_dict()
    economic["components_eur"] = {
        key: 168.0 if i == 0 else 0.0 for i, key in enumerate(COMPONENT_NAMES)
    }
    economic["net_operating_value_eur"] = 168.0
    settlements = [
        {
            "cycle_index": index + 1,
            "boundary_decision": (index + 1) * 24,
            "targets": {r: 10.0 for r in RESOURCE_NAMES},
            "actual": {r: 10.0 for r in RESOURCE_NAMES},
            "deficits": {r: 0.0 for r in RESOURCE_NAMES},
            "tolerances": {r: 1e-6 for r in RESOURCE_NAMES},
            "feasible": True,
            "total_constraint_cost": 0.0,
        }
        for index in range(7)
    ]
    interval = {
        "reward": {
            "total_eur": 1.0,
            "components_eur": {
                name: 1.0 if i == 0 else 0.0 for i, name in enumerate(COMPONENT_NAMES)
            },
            "signals": {name: 1.0 for name in op.CYCLE_SIGNALS},
        }
    }
    run = {
        "arm": "H",
        "status": "complete",
        "failure": None,
        "execution_complete": True,
        "hard_safety_passed": True,
        "cycle_feasible": True,
        "completed_decisions": 168,
        "physical_steps": 10080,
        "physical_metrics": physical,
        "economic_metrics": economic,
        "hard_safety_metrics": {"unexpected_critical_or_offline_steps": 0},
        "cycle_settlements": settlements,
        "evidence": {
            "boundary_ledger": {"decision_boundaries": [{} for _ in range(168)]},
            "intervals": [deepcopy(interval) for _ in range(168)],
            "minute_truth_states": [
                {"step": i, "co2_capture_kg_h": 0.0} for i in range(10081)
            ],
        },
    }
    manifest = {
        "disturbances": {"schedule": [{"start_s": 60.0, "duration_s": 60.0}]},
        "market": {"selected_regime": "calm"},
    }
    return {"config": config, "run": run, "manifest": manifest}


def project(fixture: dict[str, Any]) -> dict[str, Any]:
    return op.episode_scorecard(
        seed=SEEDS[0],
        actor={"arm": "H", "algorithm": None, "optimizer_seed": None},
        **fixture,
    )


def test_distribution_has_independent_linear_quantiles_and_empty_nulls() -> None:
    summary = op.distribution([0, 10, 20, 30])
    assert [
        summary[k] for k in ("minimum", "p25", "median", "p75", "p95", "maximum")
    ] == pytest.approx([0, 7.5, 15, 22.5, 28.5, 30])
    assert summary["values"] == [0, 10, 20, 30]
    assert op.distribution([])["mean"] is None


@pytest.mark.parametrize("value", [True, "4", float("nan"), float("inf"), 10**1000])
def test_non_numeric_or_unrepresentable_values_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        op.distribution([value])


@pytest.mark.parametrize(
    "count,total", [(True, 1), (1, False), (-1, 3), (4, 3), (1, 0), (0.5, 1)]
)
def test_exposure_does_not_silently_coerce_or_invent_trials(
    count: object, total: object
) -> None:
    with pytest.raises(ValueError):
        op.exposure_rate(count, total)


def test_serial_exposure_rates_have_no_binomial_interval() -> None:
    assert op.exposure_rate(0, 0) == {"count": 0, "exposure": 0, "rate": None}
    assert op.exposure_rate(3, 10) == {"count": 3, "exposure": 10, "rate": 0.3}


@pytest.mark.parametrize(
    "steps,productive,status,delay,reason",
    [
        (1, None, "not_reached", None, None),
        (2, None, "outage_incomplete", None, None),
        (3, None, "censored_no_capture", None, "execution_end"),
        (4, 4, "resumed", 0, None),
        (6, 6, "resumed", 120, None),
        (10, None, "censored_no_capture", None, "task_end"),
    ],
)
def test_recovery_and_right_censoring(
    steps: int,
    productive: int | None,
    status: str,
    delay: float | None,
    reason: str | None,
) -> None:
    states = [
        {"step": i, "co2_capture_kg_h": 2 if i == productive else 0}
        for i in range(steps + 1)
    ]
    rows = op.outage_recovery(
        [{"start_s": 60, "duration_s": 120}], states, physics_dt_s=60, horizon_s=600
    )
    row = rows[0]
    assert row["status"] == status
    assert row["recovery_delay_s"] == delay
    assert row["censor_reason"] == reason
    assert row["encountered_duration_s"] == min(120, max(0, steps * 60 - 60))


def test_next_outage_stops_followup_and_capture_threshold_is_strict() -> None:
    states = [
        {"step": i, "co2_capture_kg_h": 1e-9 if i == 4 else 1.0 if i == 8 else 0}
        for i in range(11)
    ]
    rows = op.outage_recovery(
        [{"start_s": 60, "duration_s": 120}, {"start_s": 300, "duration_s": 60}],
        states,
        physics_dt_s=60,
        horizon_s=600,
    )
    assert rows[0]["status"] == "censored_no_capture"
    assert rows[0]["censor_reason"] == "next_outage"
    assert rows[0]["followup_end_s"] == 300
    assert rows[1]["recovery_delay_s"] == 60


@pytest.mark.parametrize(
    "schedule",
    [
        [{"start_s": -60, "duration_s": 60}],
        [{"start_s": 1, "duration_s": 60}],
        [{"start_s": 60, "duration_s": 0}],
        [{"start_s": 60, "duration_s": 601}],
        [{"start_s": 60, "duration_s": 180}, {"start_s": 120, "duration_s": 60}],
    ],
)
def test_invalid_outage_geometry_rejected(schedule: list[dict[str, int]]) -> None:
    with pytest.raises(ValueError):
        op.outage_recovery(schedule, [{"step": 0}], physics_dt_s=60, horizon_s=600)


def test_episode_retains_verified_fields_and_counts_not_pair_eligibility(
    arithmetic_episode: dict[str, Any]
) -> None:
    row = project(arithmetic_episode)
    assert row["physical_metrics"] == arithmetic_episode["run"]["physical_metrics"]
    assert row["economic_metrics"] == arithmetic_episode["run"]["economic_metrics"]
    assert sum(c["net_operating_value_eur"] for c in row["cycles"]) == 168
    assert all(c["signals"]["delivered_kg"] == 24 for c in row["cycles"])
    assert row["exposure_rates"]["guard_correction"]["count"] == 3
    assert row["physical_metrics"]["guard_interventions"] == 7
    assert row["cycles"][0]["outage_stratum"] == "outage"
    assert row["cycles"][1]["outage_stratum"] == "no_outage"
    assert row["outages"][0]["status"] == "censored_no_capture"
    assert row["source_evidence_verified"] is False and row["result_claim"] is None


@pytest.mark.parametrize("offset,feasible", [(0.5e-12, True), (2e-12, False)])
def test_cycle_feasibility_preserves_simulator_epsilon(
    arithmetic_episode: dict[str, Any], offset: float, feasible: bool
) -> None:
    settlement = arithmetic_episode["run"]["cycle_settlements"][0]
    settlement["deficits"]["h2"] = settlement["tolerances"]["h2"] + offset
    assert (
        project(arithmetic_episode)["cycles"][0]["resources"]["h2"]["feasible"]
        is feasible
    )


def test_all_slots_and_per_cycle_denominators_with_failures(
    arithmetic_episode: dict[str, Any]
) -> None:
    row = project(arithmetic_episode)
    bad = deepcopy(row)
    bad.update(
        seed=SEEDS[1],
        execution_complete=False,
        cycle_feasible=False,
        status="failed",
        failure={"exception_type": "RuntimeError"},
    )
    for cycle in bad["cycles"][1:]:
        cycle.update(complete=False, resources=None, settlement=None)
    bad["cycles"][0]["resources"]["h2"].update(feasible=False, deficit=4.0)
    summary = op.policy_scorecard(SEEDS, [row, authenticate(bad)] + [None] * 30)
    resource = summary["resources"]["h2"]
    assert resource["per_cycle"][0]["verified_feasible_over_registered"]["total"] == 32
    assert resource["per_cycle"][0]["feasible_over_observed"]["rate"] == 0.5
    assert resource["per_cycle"][1]["feasible_over_observed"]["total"] == 1
    assert resource["missing_settlements"] == 224 - 8
    assert resource["observed_deficits"]["maximum"] == 4
    assert summary["execution_complete"]["successes"] == 1
    assert summary["hard_safety_complete_episode"]["successes"] == 1
    assert summary["hard_safety_observed_prefix"]["successes"] == 2
    assert summary["missing_episode_seeds"] == SEEDS[2:]
    assert summary["strata"]["market"]["calm"]["incomplete_unit_count"] == 1
    assert summary["strata"]["outage"]["no_outage"]["incomplete_unit_count"] == 6
    metric = summary["episode_metric_distributions"][
        "economic_metrics.net_operating_value_eur"
    ]
    assert metric["observed_prefix"]["distribution"]["count"] == 2
    assert metric["complete_episode"]["distribution"]["count"] == 1
    assert summary["source_evidence_verified"] is False
    assert summary["outage_recovery"]["known_manifest_episode_count"] == 2
    assert summary["outage_recovery"]["status_counts"]["censored_no_capture"] == 2
    assert summary["outage_recovery"]["observed_recovery_delay_s"]["mean"] is None


def test_all_missing_has_explicit_empty_metric_fields() -> None:
    summary = op.policy_scorecard(SEEDS, [None] * 32)
    assert summary["execution_complete"]["total"] == 32
    assert (
        summary["episode_metric_distributions"][
            "physical_metrics.electrolyser_cold_starts"
        ]["complete_episode"]["missing_count"]
        == 32
    )
    assert (
        summary["resources"]["battery"]["per_cycle"][0]["feasible_over_observed"][
            "wilson_95"
        ]
        is None
    )
    assert (
        summary["strata"]["outage"]["outage"]["metrics"]["net_operating_value_eur"][
            "bootstrap_undefined_replicates"
        ]
        == 10000
    )


def test_preflight_has_no_observed_safety_or_economic_trial(
    arithmetic_episode: dict[str, Any]
) -> None:
    from plant_backend.rl.v2 import beta_controller_benchmark as benchmark

    arithmetic_episode["manifest"] = None
    arithmetic_episode["run"] = benchmark._preflight_failed_cell(
        SEEDS[0], ValueError("fixture preflight")
    )["runs"]["H"]
    row = project(arithmetic_episode)
    assert row["economic_metrics"] is None
    assert row["exposure_rates"] is None
    assert all(c["net_operating_value_eur"] is None for c in row["cycles"])
    result = op.policy_scorecard(SEEDS, [row] + [None] * 31)
    assert result["hard_safety_observed_prefix"]["total"] == 0
    assert result["hard_safety_complete_episode"]["successes"] == 0
    assert (
        result["episode_metric_distributions"][
            "economic_metrics.components_eur.delivered_service_value"
        ]["observed_prefix"]["missing_count"]
        == 32
    )


def test_initialized_zero_prefix_is_observed_but_no_cycle_is_complete(
    arithmetic_episode: dict[str, Any]
) -> None:
    run = arithmetic_episode["run"]
    run.update(
        status="failed",
        failure={"exception_type": "RuntimeError"},
        execution_complete=False,
        cycle_feasible=False,
        completed_decisions=0,
        physical_steps=0,
        cycle_settlements=[],
        physical_metrics=EpisodeMetricsV2().to_dict(),
        economic_metrics=EconomicEpisodeMetricsV1(0, 0, 1).to_dict(),
    )
    run["evidence"] = {
        "boundary_ledger": {"decision_boundaries": [{}]},
        "intervals": [],
        "minute_truth_states": [{"step": 0, "co2_capture_kg_h": 0.0}],
    }
    row = project(arithmetic_episode)
    result = op.policy_scorecard(SEEDS, [row] + [None] * 31)
    component = result["episode_metric_distributions"][
        "economic_metrics.components_eur.delivered_service_value"
    ]
    assert component["observed_prefix"]["values_by_episode"][0] == 0.0
    assert component["complete_episode"]["values_by_episode"][0] is None
    assert result["hard_safety_observed_prefix"]["successes"] == 1
    assert result["hard_safety_complete_episode"]["successes"] == 0


def test_cluster_bootstrap_oracle_preserves_two_correlated_cycles() -> None:
    values = [[0, 0], [100, 100]] + [[] for _ in range(30)]
    draws = np.random.Generator(np.random.PCG64(20260905)).integers(0, 32, (10000, 32))
    expected = []
    for draw in draws:
        zeros = sum(index == 0 for index in draw)
        hundreds = sum(index == 1 for index in draw)
        if zeros + hundreds:
            expected.append(100 * hundreds / (zeros + hundreds))
    result = op.clustered_summary(values, draws)
    assert result["unit_distribution"]["mean"] == 50
    assert result["observed_cluster_count"] == 2
    assert result["bootstrap_defined_replicates"] == len(expected)
    assert result["conditional_mean_interval_95"] == pytest.approx(
        np.quantile(expected, [0.025, 0.975], method="linear")
    )


@pytest.mark.parametrize(
    "mutation", ["short", "duplicate", "seed", "actor", "cycle", "digest", "claim"]
)
def test_policy_registry_and_projection_identity_rejected(
    arithmetic_episode: dict[str, Any], mutation: str
) -> None:
    row = project(arithmetic_episode)
    seeds = SEEDS.copy()
    records = [row] + [None] * 31
    if mutation == "short":
        records.pop()
    elif mutation == "duplicate":
        seeds[1] = seeds[0]
    elif mutation == "seed":
        row["seed"] += 3
    elif mutation == "actor":
        second = deepcopy(row)
        second.update(seed=SEEDS[1], actor={"arm": "PPO"})
        records[1] = authenticate(second)
    elif mutation == "cycle":
        row["cycles"][0]["cycle_index"] = 2
        records[0] = authenticate(row)
    elif mutation == "digest":
        row["status"] = "invented"
    else:
        row["result_claim"] = "registered"
        records[0] = authenticate(row)
    with pytest.raises(ValueError):
        op.policy_scorecard(seeds, records)


def _rollouts(algorithm: str) -> list[dict[str, Any]]:
    from plant_backend.rl.v2.training.constrained_ppo import (
        RolloutLedgerV1,
        PILagrangeStateV1,
    )

    rows = []
    for index in range(1, 3):
        row = RolloutLedgerV1.build(
            algorithm_id=algorithm,
            rollout_index=index,
            transition_count=8,
            completed_episode_count=1 if index == 1 else 0,
            complete_cycle_count=1 if index == 1 else 0,
            reward_return=-2.0,
            raw_cost=(2, 4, 6, 8),
            training_cost=(1, 2, 3, 4),
            mean_completed_episode_cost=(1, 2, 3, 4) if index == 1 else None,
            cost_value_losses=(4, 3, 2, 1),
            multiplier_state=PILagrangeStateV1(
                proportional=(1, 2, 3, 4),
                integral=(2, 3, 4, 5),
                multipliers=(0, 50, 100, 2),
                completed_update_count=1,
            ),
            multiplier_min=0,
            multiplier_max=100,
            approximate_kl=0.01,
            entropy_loss=-1,
            clip_fraction=0.25,
            optimizer_steps=2,
            elapsed_compute_s=0.1,
            previous_sha256=None if not rows else rows[-1]["sha256"],
        )
        rows.append(row.to_dict())
    return rows


def test_training_costs_trajectories_and_saturation_retain_all_fields() -> None:
    algorithm = BETA_CORE_ALGORITHM_IDS[0]
    rows = _rollouts(algorithm)
    result = training.rollout_scorecard(algorithm, rows, expected_rollouts=2)
    assert result["rollouts"] == rows
    assert result["resources"]["h2"]["raw_cost_total"] == 8
    assert result["resources"]["h2"]["training_cost_total"] == 4
    assert result["resources"]["h2"]["completed_episode_cost_mean_by_rollout"] == [
        2,
        None,
    ]
    assert result["resources"]["co2"]["saturation_fraction"] == 1
    assert result["resources"]["h2"]["saturation_fraction"] == 0
    assert result["optimizer_steps"] == 4
    assert result["registered_checkpoint_verified"] is False


@pytest.mark.parametrize("mutation", ["missing", "order", "algorithm", "cost", "chain"])
def test_training_incomplete_or_substituted_rollouts_rejected(mutation: str) -> None:
    algorithm = BETA_CORE_ALGORITHM_IDS[0]
    rows = _rollouts(algorithm)
    if mutation == "missing":
        rows.pop()
    elif mutation == "order":
        rows.reverse()
    elif mutation == "algorithm":
        algorithm = BETA_CORE_ALGORITHM_IDS[1]
    elif mutation == "cost":
        rows[0]["raw_cost"][0] = 5
    else:
        rows[1]["previous_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        training.rollout_scorecard(algorithm, rows, expected_rollouts=2)


def _slots() -> list[training.TrainingSlot]:
    return [
        training.TrainingSlot(family, seed, False, False, "not_started", None)
        for family in BETA_CORE_ALGORITHM_IDS
        for seed in BETA_OPTIMIZER_SEEDS
    ]


def test_training_completion_and_checkpoint_admission_are_distinct() -> None:
    slots = _slots()
    slots[0] = replace(slots[0], training_completed=True, failure="checkpoint_rejected")
    slots[1] = replace(
        slots[1],
        training_completed=True,
        checkpoint_admitted=True,
        failure=None,
        rollouts=tuple(_rollouts(slots[1].algorithm)),
    )
    result = training.training_scorecard(slots, expected_rollouts=2)
    family = result["families"][BETA_CORE_ALGORITHM_IDS[0]]
    assert family["training_completed"]["successes"] == 2
    assert family["checkpoint_admitted"]["successes"] == 1
    assert family["checkpoint_admitted"]["total"] == 5
    assert family["slots"][0]["failure"] == "checkpoint_rejected"
    assert result["registered_slot_count"] == 15
    assert result["source_evidence_verified"] is False


@pytest.mark.parametrize(
    "mutation", ["missing", "order", "boolean", "admission", "failure", "rollouts"]
)
def test_training_registry_and_custody_declarations_rejected(mutation: str) -> None:
    slots = _slots()
    if mutation == "missing":
        slots.pop()
    elif mutation == "order":
        slots.reverse()
    elif mutation == "boolean":
        slots[0] = replace(slots[0], training_completed=1)
    elif mutation == "admission":
        slots[0] = replace(slots[0], checkpoint_admitted=True)
    elif mutation == "failure":
        slots[0] = replace(slots[0], failure=None)
    else:
        slots[0] = replace(slots[0], rollouts=tuple(_rollouts(slots[0].algorithm)))
    with pytest.raises(ValueError):
        training.training_scorecard(slots, expected_rollouts=2)
