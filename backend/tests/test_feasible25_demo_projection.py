"""The demo is a faithful projection of closed evidence, not a new simulation."""

from copy import deepcopy
import json

import pytest

from tools import publish_feasible25_demo as export


@pytest.fixture(scope="module")
def source():
    return export.read_source()


@pytest.fixture(scope="module")
def projected(source):
    return export.project(source)


def test_published_demo_is_reproducible_and_preserves_all_states(source, projected):
    path = export.ROOT / "demo/recordings" / f"{export.RUN_ID}.json"
    assert path.read_bytes() == export.canonical(projected)
    assert len(projected["frames"]) == 673
    for actual, retained in zip(
        projected["frames"], source["decision_frames"], strict=True
    ):
        assert actual["state"] == retained["state"]
        assert actual["commitment"] == retained["commitment"]
    assert projected["snapshot"]["frame"] == projected["frames"][-1]
    assert projected["snapshot"]["environment"]["fixture_seed"] == 9930000
    assert projected["snapshot"]["environment"]["observation_count"] == 139


def test_settlements_and_economics_are_visible_at_the_correct_boundary(
    source, projected
):
    frames = projected["frames"]
    assert frames[23]["delivery"]["commitments"][0]["delivered_kg"] is None
    assert frames[24]["delivery"]["commitments"][0]["delivered_kg"] == 150
    assert frames[95]["cycle_settlements"] == []
    assert frames[96]["cycle_settlements"] == source["cycle_settlements"][:1]
    assert frames[672]["cycle_settlements"] == source["cycle_settlements"]
    for before, after, command in zip(frames, frames[1:], source["commands"]):
        delta = (
            after["economics"]["net_value_eur"] - before["economics"]["net_value_eur"]
        )
        assert delta == pytest.approx(command["reward_eur"], abs=1e-7)
    assert frames[-1]["economics"]["net_value_eur"] == pytest.approx(15044.791835428674)


def test_forecast_uses_recorded_policy_input_and_keeps_terminal_issue(
    source, projected
):
    frame = projected["frames"][17]
    vector = source["commands"][17]["evidence"]["policy_observation"]
    assert frame["forecasts"]["solar"][0]["mean_kw"] == vector[34] * 5000
    assert frame["forecasts"]["solar"][0]["p10_kw"] == vector[35] * 5000
    assert frame["forecasts"]["issued_at_hours"] == 17 / 4
    assert projected["frames"][-1]["forecasts"]["issued_at_hours"] == 167.75
    assert all(
        p["mean"] == p["p10"] == p["p90"] == 0 for p in frame["forecasts"]["prices"]
    )


def test_public_projection_omits_private_manifest_and_does_not_invent_logs(
    source, projected
):
    assert "manifest" not in projected
    assert "minute_truth_states" not in projected
    assert "commands" not in projected
    for frame in projected["frames"][1:]:
        decision = frame["decision"]
        assert decision["computation_ms"] is None
        assert decision["execution_digest"] is None
        assert decision["guard_adjustments"] is None
        assert all(m["adjustments"] is None for m in decision["minutes"])
    raw = json.dumps(projected)
    assert '"hourly_eur_per_mwh": [0.0, 0.0' in raw
    assert source["manifest"]["market"] != projected["frames"][0]["forecasts"]


def test_corrupted_source_is_rejected_before_projection(tmp_path):
    path = tmp_path / "altered.json.gz"
    path.write_bytes(b"altered")
    with pytest.raises(ValueError, match="approved evaluation"):
        export.read_source(path)


def test_reward_mismatch_cannot_be_hidden_by_reconstructing_a_ledger(source):
    altered = deepcopy(source)
    altered["commands"][4]["reward_eur"] += 1
    with pytest.raises(ValueError, match="economic reward"):
        export.project(altered)
