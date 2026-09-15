"""Projection preserves recorded units, sparse diagnostics and failed prefixes."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from plant_backend.publish_adaptive_learning import (
    project_failures,
    project_rollouts,
    read_pinned,
)


def ledger(index: int, *, older: bool) -> dict[str, Any]:
    return {
        "rollout_index": index,
        "transition_count": 8192,
        "completed_episode_count": 12,
        "complete_cycle_count": 84,
        "optimizer_reward_sum" if older else "reward_return": -29.844895333581093,
        "approximate_kl": 0.006305,
        "ppo_ratio_clip_fraction" if older else "clip_fraction": 0.066626,
        "mean_completed_episode_cost": [0.4, 0.8, 0.9, 0.9],
        "learner_multipliers" if older else "multipliers": [0.0, 0.0, 0.0, 0.0],
    }


@pytest.mark.parametrize("older", [False, True])
def test_sparse_preupdate_diagnostics_join_on_transitions(older: bool) -> None:
    # A missing first diagnostic must not slide the second value backwards or
    # invent an initial reward. Summed reward stays scaled, even for gamma1.
    diagnostic = {
        "num_timesteps": 16384,
        "full_rollout_training_statistics" if older else "full_rollout": {
            "reward": {"mean": -0.00364},
            "reward_explained_variance": -0.31,
        },
        "reward_explained_variance": -0.31,
        "critic_mse_optimizer_units_squared": 0.177,
        "gradient_attribution": {"reward_value_mse": 0.177},
    }
    data = {
        "optimizer_rows" if older else "rollouts": [
            ledger(1, older=older),
            ledger(2, older=older),
        ],
        "diagnostic_rows" if older else "diagnostics": [diagnostic],
    }
    result = project_rollouts(data, older=older)
    assert [row["transitions"] for row in result] == [8192, 16384]
    assert result[0]["value_loss"] is None
    assert result[0]["explained_variance"] is None
    assert result[1]["value_loss"] == 0.177
    assert result[1]["training_mean_reward"] == -0.00364
    assert result[1]["explained_variance"] == -0.31
    assert result[0]["reward_sum"] == -29.844895333581093
    assert result[1]["mean_episode_cost"] == [0.4, 0.8, 0.9, 0.9]
    if not older:
        assert all(row["multipliers"] is None for row in result)


def test_pi_multipliers_are_recorded_and_scalar_placeholders_stay_inactive() -> None:
    pi = ledger(1, older=True)
    pi["learner_multipliers"] = [1.0, 2.0, 3.0, 4.0]
    scalar = ledger(1, older=True)
    scalar["learner_multipliers"] = None
    for record in (pi, scalar):
        result = project_rollouts(
            {"optimizer_rows": [record], "diagnostic_rows": []}, older=True
        )
        assert result[0]["multipliers"] == record["learner_multipliers"]


def test_failure_projection_retains_kind_phase_type_and_unknown_details() -> None:
    failure = {
        "kind": "controller_or_execution",
        "detail": {
            "phase": "controller_compute",
            "type": "ServiceMPCInputError",
            "message": "service MPC V1 does not model degraded/offline dac",
        },
    }
    projected = project_failures([failure, {"new_failure": {"code": 86}}, "stopped"])
    assert (
        projected[0]
        == "controller_or_execution / controller_compute / ServiceMPCInputError: service MPC V1 does not model degraded/offline dac"
    )
    assert json.loads(projected[1]) == {"new_failure": {"code": 86}}
    assert projected[2] == "stopped"


def test_changed_source_is_rejected_before_json_parse(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    raw = b'{"closed": true}\n'
    source.write_bytes(raw)
    expected = hashlib.sha256(raw).hexdigest()
    data, artifact = read_pinned(source, expected, "batch/analysis.json")
    assert data == {"closed": True}
    assert artifact == {
        "id": "batch/analysis.json",
        "bytes": len(raw),
        "sha256": expected,
    }
    source.write_bytes(b"not valid JSON")
    with pytest.raises(ValueError, match="source digest mismatch"):
        read_pinned(source, expected, "batch/analysis.json")
