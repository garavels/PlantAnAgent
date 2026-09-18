"""Fabricated batch002 audit packets only: no simulator, model or live outcomes."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    "effort_report", str(Path(__file__).with_name("plant-effort-report.py"))
)
assert spec is not None and spec.loader is not None
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def fake_training_data(arm):
    gradient = {
        "reward_critic_preclip_gradient_norm": 10.0,
        "actor_total_preclip_gradient_norm": 2.0,
        "implied_common_clip_factor": 0.05,
        "reward_value_mse": 4.0,
        "weighted_cost_actor_gradient_norm": None,
        "reward_actor_gradient_norm": 2.0,
        "reward_vs_weighted_cost_gradient_cosine": None,
        "cost_actor_components": None,
    }
    sample = {
        "reward_critic_tanh": [],
        "costs": None,
        "gaussian": {
            "mean_by_action": [{"mean": 0.0}] * 4,
            "std_by_action": [{"mean": 0.2}] * 4,
            "expected_clipped_fraction_all_action_coordinates": 0.317,
            "observed_clipped_fraction_all_action_coordinates": 0.3,
        },
        "probability_ratio": {"mean": 1.0},
    }
    diagnostics = [
        {
            "num_timesteps": n,
            "gradient_attribution": deepcopy(gradient),
            "sample": deepcopy(sample),
            "full_rollout": {
                "reward_explained_variance": None,
                "costs": None,
                "reward": {"mean": n / 1000},
            },
        }
        for n in range(8192, 131073, 8192)
    ]
    rollouts = [
        {
            "rollout_index": i,
            "transition_count": 8192,
            "reward_return": -4.0 * i,
            "completed_episode_count": 12,
            "complete_cycle_count": 84,
            "raw_cost": [1.0, 2.0, 3.0, 4.0],
            "training_cost": [0.1, 0.2, 0.3, 0.4],
            "mean_completed_episode_cost": None,
            "cost_value_losses": [0.0] * 4,
            "multipliers": [0.0] * 4,
            "approximate_kl": 0.05,
            "entropy_loss": -0.2,
            "clip_fraction": 0.11,
            "optimizer_steps": 640,
            "elapsed_compute_s": 10.0,
            "sha256": f"{i:064x}",
        }
        for i in range(1, 17)
    ]
    return {
        "closed_successfully": True,
        "receipt": {"fixture": True, "codes": []},
        "result": {
            "arm": {
                "name": arm,
                "reward_scale": 0.001,
                "initial_std": 0.2,
                "residual": "residual" in arm,
            }
        },
        "diagnostics": diagnostics,
        "rollouts": rollouts,
        "first_rollout": rollouts[0],
        "final_rollout": rollouts[-1],
        "failures": [],
    }


def packet():
    physical = {name: 0.0 for name in reporter.PHYSICAL}
    physical.update(
        requested_kg=6720.0,
        delivered_kg=6000.0,
        delivery_fill_rate=6000 / 6720,
        unmet_kg=720.0,
        commitments_settled=28,
        commitments_met=24,
        min_battery_soc=0.15,
    )
    summary = {
        "execution_complete": True,
        "physical_metrics": physical,
        "economic_metrics": {
            "net_operating_value_eur": -10000.0,
            "components_eur": {"service": 2000.0, "operating": -12000.0},
        },
        "hard_safety_passed": True,
        "hard_safety_classifier_failure": None,
        "hard_safety_metrics": {
            "minimum_allowed_battery_soc": 0.05,
            "battery_reserve_respected": True,
        },
        "failure": None,
        "all_cycles_feasible": True,
        "resource_order": list(reporter.RESOURCES),
        "cycle_deficit_vectors": [[0.0] * 4 for _ in range(7)],
        "fallback_decisions": 0,
        "recovery_decisions": 0,
    }
    actors = {
        name: {
            "slots": [
                {
                    "seed": seed,
                    "summary": deepcopy(summary),
                    "eligible_for_descriptive_comparison": True,
                    "failures": [],
                }
                for seed in reporter.SEEDS
            ]
        }
        for name in (*reporter.COMPARATORS, *reporter.ARMS)
    }
    equivalence = {
        "all_eight_pairs_available": True,
        "all_available_pairs_equal": True,
        "all_eight_pairs_equal": True,
        "slots": [
            {
                "seed": seed,
                "available": True,
                "equal": True,
                "reason": None,
                "comparisons": {
                    "minute_physics": True,
                    "economics": True,
                    "daily_settlements": True,
                },
                "archive_records": {
                    "H": {"sha256": "1" * 64},
                    "initial-residual": {"sha256": "2" * 64},
                },
            }
            for seed in reporter.SEEDS
        ],
    }
    return {
        "schema": reporter.AUDIT_SCHEMA,
        "integrity_verified": True,
        "batch_closed": True,
        "registered_evaluation_eligible": False,
        "source": {"git_head": reporter.PINNED_HEAD, "fixture": True},
        "report": {"path": "/fabricated/report.json", "sha256": "0" * 64},
        "actors": actors,
        "training": {arm: fake_training_data(arm) for arm in reporter.ARMS},
        "initial_residual_heuristic_equivalence": equivalence,
    }


def get_pair(result, candidate, comparator, *, field="operational_comparisons"):
    return next(
        p
        for p in result[field]
        if p["candidate"] == candidate and p["comparator"] == comparator
    )


def test_fixed_design_all_slots_contrasts_and_no_automatic_selection():
    p = packet()
    before = deepcopy(p)
    result = reporter.make_report(p)
    assert p == before
    assert len(result["actors"]) == 10
    assert result["fixed_development_seeds"] == list(range(9700100, 9700108))
    assert [
        (p["candidate"], p["comparator"]) for p in result["method_comparisons"]
    ] == list(reporter.METHOD_PAIRS)
    assert len(result["operational_comparisons"]) == 20
    for arm in reporter.ARMS:
        pairs = [
            p["comparator"]
            for p in result["operational_comparisons"]
            if p["candidate"] == arm
        ]
        assert pairs == [reporter.OWN_INITIAL[arm], *reporter.BASELINES]
    assert result["selected_arm"] is None
    assert result["operational_benefit_claim"] is False
    assert result["confirmation_or_materiality_threshold"] is None
    assert result["optimizer_seed"] == 101
    assert (
        result["initial_residual_heuristic_equivalence"]
        == p["initial_residual_heuristic_equivalence"]
    )


def test_exact_paired_signs_worse_case_and_every_daily_resource():
    p = packet()
    for i, slot in enumerate(p["actors"]["C-residual-pi"]["slots"]):
        slot["summary"]["economic_metrics"]["net_operating_value_eur"] += (
            2000 if i else -5000
        )
        slot["summary"]["physical_metrics"]["delivered_kg"] -= 100
        slot["summary"]["physical_metrics"]["commitments_met"] -= 1
        slot["summary"]["cycle_deficit_vectors"][2] = [0.1 * i, 3, 2, 1]
        slot["summary"]["all_cycles_feasible"] = False
    result = reporter.make_report(p)
    pair = get_pair(result, "C-residual-pi", "OperatingMPC")
    assert pair["paired_differences"]["net_operating_value_eur"]["mean"] == 1125
    assert pair["paired_differences"]["net_operating_value_eur"]["minimum"] == -5000
    assert (
        pair["paired_differences"]["net_operating_value_eur"]["values"]
        == [-5000] + [2000] * 7
    )
    assert pair["paired_differences"]["delivered_kg"]["minimum"] == -100
    assert pair["paired_differences"]["commitments_met"]["minimum"] == -1
    assert len(pair["daily_resource_differences"]["all_56_boundaries"]) == 56
    assert pair["daily_resource_differences"]["each_resource"]["h2"]["maximum"] == 3
    assert not pair["candidate_all_cases_daily_returns_passed"]
    assert pair["confirmation_or_material_benefit_claim"] is False
    for actor in result["actors"].values():
        assert len(actor["all_56_daily_resource_slots"]) == 56


def test_stopped_baseline_keeps_prefix_and_null_boundaries_without_subset_means():
    p = packet()
    slot = p["actors"]["E-beta"]["slots"][4]
    slot["eligible_for_descriptive_comparison"] = False
    slot["summary"]["execution_complete"] = False
    slot["summary"]["cycle_deficit_vectors"] = [[0.0, 1.0, 0.0, 0.0]]
    slot["failures"] = [
        {"kind": "opening-infeasibility", "trace": "fabricated native exception"}
    ]
    result = reporter.make_report(p)
    baseline = result["actors"]["E-beta"]
    assert baseline["aggregate"] is None
    assert (
        baseline["hard_safety_pass_count"]
        == baseline["all_daily_returns_pass_count"]
        == 7
    )
    assert baseline["slots"][4] == slot
    grid = [
        r
        for r in baseline["all_56_daily_resource_slots"]
        if r["seed"] == reporter.SEEDS[4]
    ]
    assert grid[0]["deficits"]["h2"] == 1
    assert all(r["deficits"] is None for r in grid[1:])
    for pair in result["operational_comparisons"]:
        if pair["comparator"] == "E-beta":
            assert pair["paired_differences"] is None
            assert pair["daily_resource_differences"] is None
            assert pair["unavailable_seeds"] == [reporter.SEEDS[4]]
    assert "opening-infeasibility" in reporter.markdown(result)
    assert "unavailable" in reporter.markdown(result)


def test_one_forced_stop_retains_strict_failure_even_with_allowed_SOC_pass():
    p = packet()
    s = p["actors"]["D-residual-scalar"]["slots"][7]["summary"]
    s["hard_safety_passed"] = False
    s["physical_metrics"]["electrolyser_forced_stops"] = 1
    result = reporter.make_report(p)
    actor = result["actors"]["D-residual-scalar"]
    assert actor["hard_safety_pass_count"] == 7
    assert (
        actor["slots"][7]["summary"]["hard_safety_metrics"][
            "minimum_allowed_battery_soc"
        ]
        == 0.05
    )
    assert actor["aggregate"]["electrolyser_forced_stops"]["maximum"] == 1
    assert not get_pair(result, "D-residual-scalar", "H")[
        "candidate_all_cases_hard_safety_passed"
    ]


@pytest.mark.parametrize(
    "change",
    [
        "unclosed",
        "unaudited",
        "old_schema",
        "wrong_source",
        "eligible_registered",
        "missing_actor",
        "extra_actor",
        "missing_seed",
        "duplicate_seed",
        "resource_order",
        "nonfinite",
        "short_daily",
        "nonbool_eligibility",
    ],
)
def test_inconsistent_or_unclosed_audit_refused(change):
    p = packet()
    if change == "unclosed":
        p["batch_closed"] = False
    elif change == "unaudited":
        p["integrity_verified"] = False
    elif change == "old_schema":
        p["schema"] = "plant-adaptive-closed-batch-audit-v1"
    elif change == "wrong_source":
        p["source"]["git_head"] = "b" * 40
    elif change == "eligible_registered":
        p["registered_evaluation_eligible"] = True
    elif change == "missing_actor":
        del p["actors"]["H"]
    elif change == "extra_actor":
        p["actors"]["new-arm"] = deepcopy(p["actors"]["H"])
    elif change == "missing_seed":
        p["actors"]["H"]["slots"].pop()
    elif change == "duplicate_seed":
        p["actors"]["H"]["slots"][1]["seed"] = reporter.SEEDS[0]
    elif change == "resource_order":
        p["actors"]["H"]["slots"][0]["summary"]["resource_order"].reverse()
    elif change == "nonfinite":
        p["actors"]["H"]["slots"][0]["summary"]["economic_metrics"][
            "net_operating_value_eur"
        ] = float("nan")
    elif change == "short_daily":
        p["actors"]["H"]["slots"][0]["summary"]["cycle_deficit_vectors"].pop()
    else:
        p["actors"]["H"]["slots"][0]["eligible_for_descriptive_comparison"] = 1
    with pytest.raises(ValueError):
        reporter.make_report(p)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "missing_pair",
        "false_equal_claim",
        "false_available",
        "empty_checks",
        "unavailable_equal",
        "missing_archives",
    ],
)
def test_inconsistent_initial_H_equivalence_refused(change):
    p = packet()
    eq = p["initial_residual_heuristic_equivalence"]
    if change == "missing":
        del p["initial_residual_heuristic_equivalence"]
    elif change == "missing_pair":
        eq["slots"].pop()
    elif change == "false_equal_claim":
        eq["slots"][0]["comparisons"]["minute_physics"] = False
    elif change == "false_available":
        eq["slots"][0]["available"] = False
    elif change == "empty_checks":
        eq["slots"][0]["comparisons"] = {}
    elif change == "unavailable_equal":
        p["actors"]["initial-residual"]["slots"][0][
            "eligible_for_descriptive_comparison"
        ] = False
        eq["slots"][0]["available"] = False
    else:
        eq["slots"][0]["archive_records"].pop("H")
    with pytest.raises((ValueError, KeyError)):
        reporter.make_report(p)


def test_physical_equivalence_mismatch_is_reported_not_reclassified_as_success():
    p = packet()
    eq = p["initial_residual_heuristic_equivalence"]
    eq["slots"][2]["comparisons"]["minute_physics"] = False
    eq["slots"][2]["equal"] = False
    eq["all_available_pairs_equal"] = eq["all_eight_pairs_equal"] = False
    result = reporter.make_report(p)
    assert (
        result["initial_residual_heuristic_equivalence"]["all_eight_pairs_equal"]
        is False
    )
    assert (
        result["actors"]["H"]["aggregate"]["net_operating_value_eur"]
        == result["actors"]["initial-residual"]["aggregate"]["net_operating_value_eur"]
    )
    assert "All eight pairs equal: false" in reporter.markdown(result)


def test_all_sixteen_rows_unit_conversion_and_different_clipping_measures():
    data = fake_training_data("B-absolute-scalar")
    result = reporter.training_report("B-absolute-scalar", data)
    assert len(result["diagnostic_rows"]) == len(result["optimizer_rows"]) == 16
    first, optimizer = result["diagnostic_rows"][0], result["optimizer_rows"][0]
    assert first["critic_mse_eur_squared"] == 4_000_000
    assert first["diagnostic_actor_norm_after_common_clip"] == 0.1
    assert first["costs"] is None and first["reward_explained_variance"] is None
    assert first["observed_action_space_clipping"] == 0.3
    assert optimizer["ppo_ratio_clip_fraction"] == 0.11
    assert optimizer["optimizer_reward_sum"] == -4
    assert optimizer["optimizer_reward_sum_eur_equivalent"] == -4000
    assert optimizer["raw_cost"] == [1, 2, 3, 4]
    assert optimizer["learner_cost_value_losses"] is None
    assert (
        result["raw_training_evidence"]["rollouts"][0]["cost_value_losses"] == [0] * 4
    )
    assert result["raw_training_evidence"] == data


@pytest.mark.parametrize(
    "change",
    [
        "missing_diag",
        "missing_ledger",
        "skipped_prefix",
        "ledger_reorder",
        "wrong_scale",
        "wrong_std",
        "wrong_interface",
        "wrong_arm",
        "wrong_count",
        "seventeenth",
    ],
)
def test_training_design_or_complete_curves_cannot_be_changed(change):
    arm = "A-absolute-pi"
    d = fake_training_data(arm)
    if change == "missing_diag":
        d["diagnostics"].pop()
    elif change == "missing_ledger":
        d["rollouts"].pop()
    elif change == "skipped_prefix":
        d["diagnostics"].pop(0)
    elif change == "ledger_reorder":
        d["rollouts"].reverse()
    elif change == "wrong_scale":
        d["result"]["arm"]["reward_scale"] = 1.0
    elif change == "wrong_std":
        d["result"]["arm"]["initial_std"] = 1.0
    elif change == "wrong_interface":
        d["result"]["arm"]["residual"] = True
    elif change == "wrong_arm":
        d["result"]["arm"]["name"] = "C-residual-pi"
    elif change == "wrong_count":
        d["rollouts"][1]["transition_count"] = 1
    else:
        d["diagnostics"].append(deepcopy(d["diagnostics"][-1]))
    with pytest.raises(ValueError):
        reporter.training_report(arm, d)


def fail_training(p, arm):
    d = p["training"][arm]
    d.update(
        closed_successfully=False,
        result=None,
        diagnostics=d["diagnostics"][:3],
        rollouts=d["rollouts"][:2],
        failures=[{"type": "optimizer_failure", "phase": "third_update"}],
    )
    for actor in (
        arm,
        *(name for name, source in reporter.INITIAL_SOURCES.items() if source == arm),
    ):
        for slot in p["actors"][actor]["slots"]:
            slot.update(
                summary=None,
                eligible_for_descriptive_comparison=False,
                failures=["training unavailable"],
            )
    if arm == "C-residual-pi":
        eq = p["initial_residual_heuristic_equivalence"]
        eq.update(
            all_eight_pairs_available=False,
            all_available_pairs_equal=None,
            all_eight_pairs_equal=None,
        )
        for slot in eq["slots"]:
            slot.update(
                available=False,
                equal=None,
                reason="initial unavailable",
                comparisons={},
                archive_records={},
            )


@pytest.mark.parametrize("arm", reporter.ARMS)
def test_failed_training_keeps_every_retained_prefix_and_correct_initial_unavailable(
    arm,
):
    p = packet()
    fail_training(p, arm)
    result = reporter.make_report(p)
    training = result["training"][arm]
    assert len(training["diagnostic_rows"]) == 3
    assert len(training["optimizer_rows"]) == 2
    assert training["raw_training_evidence"] == p["training"][arm]
    assert training["failure_evidence"] == [
        {"type": "optimizer_failure", "phase": "third_update"}
    ]
    assert result["actors"][arm]["aggregate"] is None
    assert len(result["actors"][arm]["slots"]) == 8
    if arm == "C-residual-pi":
        assert (
            result["initial_residual_heuristic_equivalence"]["all_eight_pairs_equal"]
            is None
        )
    assert "optimizer_failure" in reporter.markdown(result)
    assert "null" in reporter.markdown(result)


def test_failed_training_cannot_supply_an_eligible_final_or_initializer():
    p = packet()
    p["training"]["A-absolute-pi"].update(closed_successfully=False, result=None)
    with pytest.raises(ValueError, match="failed training closure"):
        reporter.make_report(p)


def test_unavailable_equivalence_pair_has_no_successful_subset_all_eight_claim():
    p = packet()
    p["actors"]["H"]["slots"][0]["eligible_for_descriptive_comparison"] = False
    eq = p["initial_residual_heuristic_equivalence"]
    eq.update(all_eight_pairs_available=False, all_eight_pairs_equal=None)
    eq["slots"][0].update(
        available=False,
        equal=None,
        reason="replay missing",
        comparisons={},
        archive_records={},
    )
    result = reporter.make_report(p)
    assert (
        result["initial_residual_heuristic_equivalence"]["all_available_pairs_equal"]
        is True
    )
    assert (
        result["initial_residual_heuristic_equivalence"]["all_eight_pairs_equal"]
        is None
    )
    assert result["actors"]["H"]["aggregate"] is None


def test_report_has_no_environment_or_live_data_dependency_and_cli_no_overwrite(
    tmp_path, monkeypatch
):
    p = packet()
    audit = tmp_path / "fabricated-audit.json"
    audit.write_text(json.dumps(p))
    output = tmp_path / "report"
    original = Path.read_bytes
    allowed = {audit, Path(reporter.__file__)}

    def read(path):
        assert path in allowed, f"unexpected input read: {path}"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(
        sys, "argv", [reporter.__file__, str(audit), "--output-directory", str(output)]
    )
    reporter.main()
    result = json.loads((output / "analysis.json").read_text())
    assert result["audited_packet_file"]["bytes"] == len(audit.read_bytes())
    text = (output / "analysis.md").read_text()
    assert "PPO ratio clip" in text and "Every daily resource vector" in text
    assert all(arm in text for arm in reporter.ARMS)
    with pytest.raises(FileExistsError):
        reporter.main()


def test_bool_and_empty_metrics_are_not_silently_numeric():
    with pytest.raises(ValueError):
        reporter.numeric(True)
    with pytest.raises(ValueError):
        reporter.describe([])


def test_fabricated_auditor_to_report_integration(tmp_path, monkeypatch):
    """Run both schemas together, guarding production reads, resets and model use."""
    import gzip

    audit_test_path = Path(__file__).with_name("test_plant_effort_closure_audit.py")
    module_spec = importlib.util.spec_from_file_location(
        "effort_audit_report_fixture", str(audit_test_path)
    )
    assert module_spec is not None and module_spec.loader is not None
    fixtures = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(fixtures)
    fixtures.forbid_production.__wrapped__(monkeypatch)
    root = fixtures.fabricate(tmp_path / "fabricated-batch")
    audit = fixtures.audit
    # The auditor's graph fixture deliberately uses schema-minimal diagnostics
    # and metrics. Enrich only those fabricated artifacts before the audit so
    # the reporter receives an untouched, self-consistent audited packet.
    for arm in reporter.ARMS:
        for index, detail in enumerate(fake_training_data(arm)["diagnostics"], start=1):
            path = (
                root / "training" / arm / "diagnostics" / f"diagnostic-{index:03d}.json"
            )
            record = audit.decode(path.read_bytes())
            record.update(detail)
            fixtures.write(path, record)
    for seed in reporter.SEEDS:
        for actor in (*reporter.COMPARATORS, *reporter.ARMS):
            cell = root / "evaluations" / f"{seed}-{actor}"
            archive = cell / "episode.json.gz"
            episode = audit.decode(gzip.decompress(archive.read_bytes()))
            summary = episode["summary"]
            physical = {name: 0.0 for name in reporter.PHYSICAL}
            physical.update(
                requested_kg=6720.0,
                delivered_kg=3360.0,
                unmet_kg=3360.0,
                min_battery_soc=0.15,
            )
            physical.update(summary["physical_metrics"])
            summary["physical_metrics"] = physical
            summary["economic_metrics"]["components_eur"] = {
                "fixture_accounting": summary["economic_metrics"][
                    "net_operating_value_eur"
                ]
            }
            summary.update(fallback_decisions=0, recovery_decisions=0)
            raw = audit.canonical_bytes(episode)
            archive.write_bytes(gzip.compress(raw, mtime=0))
            outer = audit.decode((cell / "summary.json").read_bytes())
            outer.update(summary=summary, archive=fixtures.identity(archive))
            fixtures.write(cell / "summary.json", outer)
            replay = audit.decode((cell / "replay.json").read_bytes())
            replay["evidence_sha256"] = audit.sha(raw)
            fixtures.write(cell / "replay.json", replay)
    audited = audit.audit_closed_batch(root, fixture=True)
    result = reporter.make_report(audited)
    assert result["audit_provenance"]["source"]["fixture"] is True
    assert (
        result["initial_residual_heuristic_equivalence"]["all_eight_pairs_equal"]
        is True
    )
    assert all(
        len(t["diagnostic_rows"]) == len(t["optimizer_rows"]) == 16
        for t in result["training"].values()
    )
    assert all(
        len(actor["all_56_daily_resource_slots"]) == 56
        for actor in result["actors"].values()
    )
    assert result["actors"]["H"]["aggregate_available"]
    assert result["actors"]["initial-residual"]["aggregate_available"]
    assert result["actors"]["E-beta"]["aggregate"] is None
    assert "fabricated failure" in reporter.markdown(result)
    assert all(
        pair["paired_differences"] is None for pair in result["operational_comparisons"]
    )
