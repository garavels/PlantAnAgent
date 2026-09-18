"""Focused tests for the Environment V1 methane-tank ceiling diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plant_backend.rl.checkpoints import write_fixed_checkpoint
from plant_backend.rl.episodes import EpisodeConfig
from plant_backend.rl.evaluation.ceiling import (
    CeilingDiagnostic,
    CeilingRun,
    diagnose,
    main,
)
from plant_backend.rl.metrics import EpisodeMetrics

VERY_SHORT = EpisodeConfig(horizon_h=0.05)


def _synthetic_run(
    *, seed: int, initial_ch4_kg: float, produced_ch4_kg: float, terminal_fill: float
) -> CeilingRun:
    capacity = 1_000.0
    metrics = EpisodeMetrics()
    metrics.steps = 1
    metrics.hours = 1.0 / 60.0
    metrics.reward_total = produced_ch4_kg / 100.0
    metrics.reward_components = {"methane": produced_ch4_kg / 100.0}
    metrics.ch4_kg = produced_ch4_kg
    metrics.solar_kwh = 100.0
    metrics.curtailed_kwh = 10.0
    metrics.min_battery_soc = 0.5
    return CeilingRun(
        controller="heuristic",
        seed=seed,
        scenario="synthetic",
        ch4_capacity_kg=capacity,
        initial_ch4_kg=initial_ch4_kg,
        final_ch4_kg=initial_ch4_kg + produced_ch4_kg,
        theoretical_ch4_headroom_kg=capacity - initial_ch4_kg,
        terminal_ch4_headroom_kg=capacity * (1.0 - terminal_fill),
        produced_ch4_kg=produced_ch4_kg,
        terminal_fill=terminal_fill,
        reward_total=metrics.reward_total,
        metrics=metrics,
    )


def test_summary_reports_distributions_and_ceiling_correlation() -> None:
    runs = [
        _synthetic_run(seed=1, initial_ch4_kg=0.0, produced_ch4_kg=1_000.0, terminal_fill=1.0),
        _synthetic_run(seed=2, initial_ch4_kg=100.0, produced_ch4_kg=900.0, terminal_fill=1.0),
        _synthetic_run(seed=3, initial_ch4_kg=200.0, produced_ch4_kg=800.0, terminal_fill=1.0),
        # Deliberately excluded from the correlation by the terminal-fill criterion.
        _synthetic_run(seed=4, initial_ch4_kg=900.0, produced_ch4_kg=0.0, terminal_fill=0.90),
    ]
    diagnostic = CeilingDiagnostic(
        seeds=[1, 2, 3, 4],
        horizon_h=48.0,
        terminal_fill_threshold=0.96,
        smoke_test=True,
        baseline_only=True,
        runs=runs,
    )

    summary = diagnostic.summary()["heuristic"]
    produced = summary["diagnostics"]["produced_ch4_kg"]
    assert produced == pytest.approx(
        {"mean": 675.0, "std": 396.074487, "min": 0.0, "max": 1_000.0},
        abs=1e-6,
    )
    ceiling = summary["ceiling_limited"]
    assert ceiling["count"] == 3
    assert ceiling["fraction"] == pytest.approx(0.75)
    assert ceiling["pearson_initial_vs_produced_ch4"] == pytest.approx(-1.0)
    assert set(summary["metrics"]["solar_utilisation"]) == {"mean", "std", "min", "max"}
    assert summary["reward_components"]["methane"]["max"] == pytest.approx(10.0)


def test_identical_seeds_give_heuristic_and_checkpoint_identical_initial_plant(
    tmp_path: Path,
) -> None:
    checkpoint = write_fixed_checkpoint(tmp_path / "fixed", [0.0, 0.0, 0.0, 0.0, 0.0])
    result = diagnose(seeds=[7, 8], checkpoint=checkpoint, episode=VERY_SHORT)

    assert result.smoke_test and not result.baseline_only
    assert result.evidence_scope == "smoke_test"
    assert result.controllers == ["heuristic", "rl"]
    for seed in result.seeds:
        paired = [run for run in result.runs if run.seed == seed]
        assert len(paired) == 2
        assert paired[0].scenario == paired[1].scenario
        assert paired[0].initial_ch4_kg == paired[1].initial_ch4_kg
        assert paired[0].ch4_capacity_kg == paired[1].ch4_capacity_kg
        for run in paired:
            assert run.metrics.steps == 3
            assert run.produced_ch4_kg == pytest.approx(run.metrics.ch4_kg)
            assert run.initial_ch4_kg + run.produced_ch4_kg == pytest.approx(run.final_ch4_kg)
            assert run.theoretical_ch4_headroom_kg == pytest.approx(
                run.ch4_capacity_kg - run.initial_ch4_kg
            )

    repeated = diagnose(seeds=[7, 8], checkpoint=checkpoint, episode=VERY_SHORT)
    assert result.to_dict() == repeated.to_dict()


def test_baseline_report_and_json_are_honest_and_complete(tmp_path: Path, capsys) -> None:
    target = tmp_path / "ceiling.json"
    exit_code = main(
        [
            "--seed-start",
            "12",
            "--count",
            "1",
            "--hours",
            "0.016666666666666666",
            "--json",
            str(target),
        ]
    )

    assert exit_code == 0
    assert "baseline_only" in capsys.readouterr().out
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["evidence_scope"] == "baseline_only"
    assert payload["smoke_test"] is True
    assert "does not establish heuristic/RL parity" in payload["disclaimer"]
    assert payload["seeds"] == [12]
    assert payload["runs"][0]["metrics"]["steps"] == 1
    assert payload["summary"]["heuristic"]["diagnostics"]["reward_total"]["std"] == 0.0


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_diagnostic_rejects_invalid_thresholds(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        diagnose(seeds=[1], episode=VERY_SHORT, terminal_fill_threshold=threshold)


def test_cli_rejects_an_empty_seed_range(capsys) -> None:
    assert main(["--count", "0"]) == 2
    assert "count must be positive" in capsys.readouterr().err
