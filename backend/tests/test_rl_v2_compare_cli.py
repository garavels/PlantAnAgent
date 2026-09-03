"""Command-line contract for the causal V2 development comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plant_backend.rl.v2 import compare


def test_compare_cli_runs_declared_arms_and_writes_schema_six(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / "causal-comparison.json"
    assert compare.main(
        [
            "--seed-start",
            "0",
            "--count",
            "1",
            "--hours",
            "6",
            "--delivery-rate",
            "35",
            "--allow-dirty-development",
            "--json",
            str(target),
        ]
    ) == 0

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 6
    assert payload["evidence_scope"] == "development_comparator"
    assert list(payload["labels"]) == [
        "legacy_heuristic",
        "delivery-aware-climatology-v2",
        "issued-forecast-v2",
        "risk-adjusted-issued-forecast-v2",
    ]
    assert [item["id"] for item in payload["contrasts"]] == [
        "delivery-aware-climatology-v2_minus_legacy_heuristic",
        "issued-forecast-v2_minus_delivery-aware-climatology-v2",
        "risk-adjusted-issued-forecast-v2_minus_issued-forecast-v2",
    ]
    assert payload["pairing"]["exogenous_episode_match_by_seed"] is True
    assert payload["pairing"]["full_manifest_match_by_seed"] is True
    output = capsys.readouterr().out
    assert "causal controller comparison" in output
    assert "DEVELOPMENT COMPARISON ONLY" in output


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (["--count", "0"], "count must be positive"),
        (["--seed-start", "-1"], "seed-start must be non-negative"),
        (["--delivery-rate", "0"], "delivery_rate"),
    ),
)
def test_compare_cli_rejects_invalid_inputs(
    arguments: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert compare.main(arguments) == 2
    assert message in capsys.readouterr().err


def test_compare_cli_requires_acknowledgement_outside_development_seeds(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert compare.main(["--seed-start", "3", "--count", "1"]) == 2
    assert "--acknowledge-seed-contamination" in capsys.readouterr().err


def test_compare_cli_returns_nonzero_when_any_planned_arm_failed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    class FailedResult:
        def report(self) -> str:
            return "invalid comparison"

        def execution_summary(self) -> dict[str, bool]:
            return {"all_planned_arms_succeeded": False}

    written: list[Path] = []
    monkeypatch.setattr(compare, "evaluate_controllers", lambda **_: FailedResult())
    monkeypatch.setattr(
        compare,
        "write",
        lambda _result, path: written.append(Path(path)) or Path(path),
    )

    target = tmp_path / "failed.json"
    assert compare.main(
        ["--allow-dirty-development", "--json", str(target)]
    ) == 1
    captured = capsys.readouterr()
    assert "invalid comparison" in captured.out
    assert written == [target]
    assert "ineligible" in captured.err


def test_compare_cli_source_audit_abort_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    writes: list[object] = []

    def abort(**_: object) -> object:
        raise RuntimeError("source changed during execution")

    monkeypatch.setattr(compare, "evaluate_controllers", abort)
    monkeypatch.setattr(compare, "write", lambda *args: writes.append(args))

    target = tmp_path / "must-not-exist.json"
    assert compare.main(
        ["--allow-dirty-development", "--json", str(target)]
    ) == 1
    assert writes == []
    assert not target.exists()
    assert "failed closed" in capsys.readouterr().err
