"""Independent reservation arithmetic; no measured-capacity or execution claim."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from plant_backend.rl.v2.study import execution_budget as budget, registry
from plant_backend.rl.v2.training.custody_json import authenticate


def example() -> budget.ExecutionBudget:
    return budget.ExecutionBudget(
        child_timeout_seconds=11,
        log_limit_bytes=17,
        arm_limits=(
            budget.EvidenceLimits(100, 1_000, 10, 200),
            budget.EvidenceLimits(200, 2_000, 20, 300),
            budget.EvidenceLimits(300, 3_000, 30, 400),
            budget.EvidenceLimits(400, 4_000, 40, 500),
        ),
        report_stored_bytes=1_000,
        report_decoded_bytes=10_000,
        filesystem_reserve_bytes=123,
    )


def test_complete_reservation_oracle_and_no_permission(tmp_path: Path) -> None:
    model = example()
    # 32 episodes, three baselines once and 15 PPO policies. Include every
    # failed-arm log allowance, two controls/cell, five root/authority documents.
    evidence_per_episode = (
        (100 + 10) + (200 + 20) + (300 + 30) + 15 * (400 + 40) + 18 * 2 * 17
    )
    controls = 5 * 8 * 1024**2 + 576 * 2 * 64 * 1024
    one_stage = 32 * evidence_per_episode + controls + 1_000
    assert model.stage_allocation_bound_bytes == one_stage
    assert model.opening_free_bytes("development") == 2 * one_stage + 123
    assert model.opening_free_bytes("heldout") == one_stage + 123
    raw = model.to_dict()
    assert raw["all_child_timeouts_seconds"] == 576 * 11
    assert raw["capacity_measured"] is raw["execution_authorized"] is False
    assert budget.ExecutionBudget.from_dict(raw) == model
    for arm in registry.actors():
        if arm["arm"] == "PPO":
            assert model.for_arm(arm["arm"]) is model.arm_limits[3]
    for stage in registry.STAGES:
        record = registry.stage_registry(tmp_path / "training", stage)
        assert (
            record["execution_contract_sha256"] == budget.execution_contract()["sha256"]
        )
        assert record["report_path"].endswith("/report.json.gz")
        assert (
            record["authorization_input_path"]
            == record["stage_root"] + ".authorization.json"
        )
        assert not Path(record["stage_root"]).exists()


@pytest.mark.parametrize(
    "field",
    [
        "child_timeout_seconds",
        "log_limit_bytes",
        "report_stored_bytes",
        "report_decoded_bytes",
        "filesystem_reserve_bytes",
    ],
)
@pytest.mark.parametrize("bad", [0, -1, True, 0.5, "10", None])
def test_limits_cannot_coerce_or_disable_caps(field: str, bad: object) -> None:
    raw = example().to_dict()
    raw[field] = bad
    with pytest.raises(ValueError):
        budget.ExecutionBudget.from_dict(authenticate(raw))


@pytest.mark.parametrize(
    "field",
    [
        "stage_allocation_bound_bytes",
        "development_opening_free_bytes",
        "heldout_opening_free_bytes",
        "all_child_timeouts_seconds",
        "execution_contract_sha256",
        "execution_authorized",
        "capacity_measured",
        "schema",
        "extra",
        "missing_profile",
        "extra_profile",
    ],
)
def test_rehashed_budget_and_scope_substitution_rejected(field: str) -> None:
    raw = deepcopy(example().to_dict())
    if field == "missing_profile":
        del raw["arm_limits"]["PPO"]
    elif field == "extra_profile":
        raw["arm_limits"]["pi-lagrangian"] = raw["arm_limits"]["PPO"]
    elif field in {"capacity_measured", "execution_authorized"}:
        raw[field] = True
    else:
        raw[field] = "tampered"
    with pytest.raises(ValueError):
        budget.ExecutionBudget.from_dict(authenticate(raw))


def test_control_caps_are_inclusive_and_profile_types_are_exact() -> None:
    # The canonical representation of a string adds quotes and a newline.
    text = "x" * (budget.CELL_CONTROL_LIMIT_BYTES - 3)
    assert len(budget.control_bytes(text, cell=True)) == budget.CELL_CONTROL_LIMIT_BYTES
    with pytest.raises(ValueError, match="byte limit"):
        budget.control_bytes(text + "x", cell=True)
    with pytest.raises(ValueError, match="exact boolean"):
        budget.control_bytes({}, cell=1)
    with pytest.raises(ValueError, match="exact H/M24"):
        replace(example(), arm_limits=list(example().arm_limits))
    with pytest.raises(ValueError):
        budget.EvidenceLimits(True, 1, 1, 1)
    with pytest.raises(ValueError):
        example().for_arm("pi-lagrangian")
    with pytest.raises(ValueError):
        example().opening_free_bytes("training")
