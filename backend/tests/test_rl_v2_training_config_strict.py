"""Strict constraint parsing at the controller-neutral training boundary."""

from __future__ import annotations

import math

import pytest

from plant_backend.controllers.base import Constraints
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1


CAPS = (
    "electrolyser_max_kw",
    "dac_max_kw",
    "methanator_max_kw",
    "battery_support_max_kw",
)


@pytest.mark.parametrize("name", CAPS)
@pytest.mark.parametrize("value", [0, 12.5])
def test_constraint_caps_accept_finite_nonnegative_reals(
    name: str, value: float
) -> None:
    constraints = ControllerNeutralTrainingConfigV1.from_dict(
        {"constraints": {name: value}}
    ).constraints
    assert getattr(constraints, name) == float(value)


@pytest.mark.parametrize("name", CAPS)
def test_constraint_caps_accept_null(name: str) -> None:
    assert (
        getattr(
            ControllerNeutralTrainingConfigV1.from_dict(
                {"constraints": {name: None}}
            ).constraints,
            name,
        )
        is None
    )


@pytest.mark.parametrize("name", (*CAPS, "battery_min_soc"))
@pytest.mark.parametrize("value", [True, False, "12", "junk", [], {}])
def test_constraints_reject_non_real_values(name: str, value: object) -> None:
    with pytest.raises(TypeError, match=rf"constraint settings\.{name}"):
        ControllerNeutralTrainingConfigV1.from_dict({"constraints": {name: value}})


@pytest.mark.parametrize("name", (*CAPS, "battery_min_soc"))
@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_constraints_reject_non_finite_reals(name: str, value: float) -> None:
    with pytest.raises(
        ValueError, match=rf"constraint settings\.{name} must be finite"
    ):
        ControllerNeutralTrainingConfigV1.from_dict({"constraints": {name: value}})


@pytest.mark.parametrize("name", CAPS)
def test_constraint_caps_reject_negative_reals(name: str) -> None:
    with pytest.raises(ValueError, match=rf"{name} cannot be negative"):
        ControllerNeutralTrainingConfigV1.from_dict({"constraints": {name: -0.01}})


@pytest.mark.parametrize("value", [0, 0.25, 0.95])
def test_battery_min_soc_accepts_closed_operating_range(value: float) -> None:
    constraints = ControllerNeutralTrainingConfigV1.from_dict(
        {"constraints": {"battery_min_soc": value}}
    ).constraints
    assert constraints.battery_min_soc == float(value)


@pytest.mark.parametrize("value", [None, -0.01, 0.950001])
def test_battery_min_soc_rejects_null_and_out_of_range_values(value: object) -> None:
    expected = TypeError if value is None else ValueError
    with pytest.raises(expected, match="battery_min_soc"):
        ControllerNeutralTrainingConfigV1.from_dict(
            {"constraints": {"battery_min_soc": value}}
        )


def test_constraints_reject_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown constraint settings: imaginary_cap"):
        ControllerNeutralTrainingConfigV1.from_dict(
            {"constraints": {"imaginary_cap": 1.0}}
        )


def test_direct_training_config_cannot_bypass_strict_constraint_validation() -> None:
    invalid = Constraints(electrolyser_max_kw="12")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="electrolyser_max_kw"):
        ControllerNeutralTrainingConfigV1(constraints=invalid).validate()
