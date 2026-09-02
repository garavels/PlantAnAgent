"""Auditable paired-baseline evaluation for Environment V2 weather alpha."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from plant_sim import Action

from plant_backend.rl.v2.__main__ import main
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.evaluation import (
    BASELINE_ONLY_DISCLAIMER,
    Baseline,
    evaluate,
    write,
)

VERY_SHORT = EpisodeConfigV2(horizon_h=0.5, delivery_period_h=0.5)


def test_all_baselines_receive_the_same_immutable_episode() -> None:
    result = evaluate(seeds=[3, 7], episode=VERY_SHORT)

    assert result.controllers == ["heuristic", "idle", "balanced", "production_mix"]
    assert len(result.runs) == 8
    assert result.pairing()["manifest_match_by_seed"] is True
    assert result.pairing()["all_arms_present_by_seed"] is True
    for seed in result.seeds:
        assert len({run.manifest_sha256 for run in result.runs if run.seed == seed}) == 1


def test_summary_has_population_distributions_and_physical_metrics() -> None:
    result = evaluate(seeds=[1, 2], episode=VERY_SHORT)
    summary = result.summary()

    assert summary["heuristic"]["runs"] == 2
    fill = summary["heuristic"]["metrics"]["delivery_fill_rate"]
    assert set(fill) == {"count", "mean", "std", "min", "max"}
    assert fill["count"] == 2
    assert 0.0 <= fill["min"] <= fill["max"] <= 1.0
    assert "shortfall" in summary["idle"]["reward_components"]


def test_result_is_explicitly_baseline_only_and_records_provenance() -> None:
    result = evaluate(seeds=[5], episode=VERY_SHORT)
    payload = result.to_dict()

    assert payload["evidence_scope"] == "baseline_only"
    assert payload["disclaimer"] == BASELINE_ONLY_DISCLAIMER
    assert "no trained RL policy" in result.report()
    assert payload["provenance"]["environment"]["environment_id"].endswith(
        "weather-alpha"
    )
    assert len(payload["provenance"]["dependency_lock"]["sha256"]) == 64
    assert payload["provenance"]["source"]["git_revision"]
    assert isinstance(payload["provenance"]["source"]["git_dirty"], bool)
    assert "no preceding-interval aggregate" in payload["information_sets"]["heuristic"]
    assert payload["labels"]["heuristic"] == "Legacy forecast-blind heuristic"


def test_paired_evaluation_is_reproducible_except_live_provenance() -> None:
    first = evaluate(seeds=[11], episode=VERY_SHORT)
    second = evaluate(seeds=[11], episode=VERY_SHORT)

    assert first.summary() == second.summary()
    assert [run.to_dict() for run in first.runs] == [run.to_dict() for run in second.runs]
    assert first.pairing() == second.pairing()


@pytest.mark.parametrize("seed", [True, 1.9, "2"])
def test_evaluation_rejects_coercible_non_integer_seeds(seed: object) -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        evaluate(seeds=[seed], episode=VERY_SHORT)  # type: ignore[list-item]


def test_evaluation_rejects_duplicate_and_negative_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        evaluate(seeds=[2, 2], episode=VERY_SHORT)
    with pytest.raises(ValueError, match="non-negative integers"):
        evaluate(seeds=[-1], episode=VERY_SHORT)


def test_custom_baseline_kind_cannot_disagree_with_its_information_set() -> None:
    with pytest.raises(ValueError, match="unsupported baseline kind"):
        Baseline("custom", "Custom", "unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="requires an action"):
        Baseline("custom", "Custom", "fixed_action")
    with pytest.raises(ValueError, match="cannot define"):
        Baseline("custom", "Custom", "legacy_heuristic", action=Action.idle())

    result = evaluate(
        seeds=[2],
        episode=VERY_SHORT,
        baselines=(Baseline("custom", "Custom", "legacy_heuristic"),),
    )
    assert result.information_sets["custom"] == (
        "latest one-minute V2 telemetry plus safety constraints; no "
        "preceding-interval aggregate, delivery context, or issued forecast"
    )


def test_json_contains_complete_runs_and_configuration(tmp_path: Path) -> None:
    result = evaluate(seeds=[13], episode=VERY_SHORT)
    target = write(result, tmp_path / "v2.json")
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert payload["study"] == "environment_v2_weather_alpha_paired_baselines"
    assert payload["episode_config"]["horizon_h"] == 0.5
    assert len(payload["runs"]) == 4
    assert {run["metrics"]["stage"] for run in payload["runs"]} == {
        "weather-forecast-alpha"
    }
    assert payload["pairing"]["manifest_match_by_seed"] is True


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--count", "0"], "count must be positive"),
        (["--seed-start", "-1"], "seed-start must be non-negative"),
        (["--delivery-rate", "0"], "delivery_rate"),
    ],
)
def test_cli_rejects_invalid_inputs(
    args: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(args) == 2
    assert message in capsys.readouterr().err
