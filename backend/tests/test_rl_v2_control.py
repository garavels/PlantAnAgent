"""Algorithm-neutral, no-leakage control boundary for Environment V2."""

from __future__ import annotations

import json

import numpy as np
import pytest

from plant_backend.controllers import Constraints
from plant_backend.rl.v2.config import EpisodeConfigV2
from plant_backend.rl.v2.control import ControllerObservationV2, DeliveryObservation
from plant_backend.rl.v2.env import PlantEnvV2
from plant_backend.rl.v2.weather import ForecastIssueV2, PublicForecastIssueV2


SHORT = EpisodeConfigV2(horizon_h=1.0, delivery_period_h=1.0)


def test_env_exposes_detached_controller_source_used_by_rl_encoder() -> None:
    env = PlantEnvV2(episode=SHORT)
    reset_vector, _ = env.reset(seed=23)

    source = env.controller_observation
    assert isinstance(source, ControllerObservationV2)
    assert isinstance(source.forecast, PublicForecastIssueV2)
    assert not isinstance(source.forecast, ForecastIssueV2)
    assert source.delivery == env.delivery_observation
    assert source.decision_interval_s == SHORT.decision_interval_s
    np.testing.assert_array_equal(
        reset_vector,
        env.encoder.encode_controller_observation(
            source,
            constraints=env.constraints,
        ),
    )


def test_controller_source_has_no_private_episode_or_replay_identity() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=29)

    payload = env.controller_observation.to_dict()
    serialized = json.dumps(payload, sort_keys=True).lower()
    forbidden = (
        "seed",
        "scenario",
        "schedule",
        "commitments",
        "boundary_factors",
        "truth_sha256",
        "forecast_issue_stream",
        "issue_stream_sha256",
        "episode_sha256",
    )
    assert all(name not in serialized for name in forbidden)
    assert set(payload) == {
        "state",
        "delivery",
        "forecast",
        "progress",
        "decision_interval_s",
    }


def test_controller_source_deep_copies_mutable_health_mapping() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=31)

    detached = env.controller_observation
    detached.state.boundary.health.clear()

    assert env.controller_observation.state.boundary.health
    assert env.telemetry.health


def test_forecast_blind_projection_does_not_mutate_public_source() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=37)

    public = env.controller_observation
    blind = public.without_forecast()

    assert blind.forecast is None
    assert public.forecast is not None
    assert blind.state == public.state
    assert blind.delivery == public.delivery
    with pytest.raises(ValueError, match="requires an issued forecast"):
        env.encoder.encode_controller_observation(
            blind,
            constraints=Constraints(),
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"steps_to_next": -1},
        {"delivery_period_steps": 0},
        {"next_requested_kg": float("nan")},
        {"cumulative_requested_kg": 1.0, "cumulative_delivered_kg": 2.0},
        {"steps_to_next": None, "next_requested_kg": 1.0},
    ),
)
def test_delivery_observation_rejects_invalid_ledger_views(
    changes: dict[str, object],
) -> None:
    fields: dict[str, object] = {
        "steps_to_next": 60,
        "delivery_period_steps": 360,
        "next_requested_kg": 210.0,
        "cumulative_requested_kg": 0.0,
        "cumulative_delivered_kg": 0.0,
    }
    fields.update(changes)
    with pytest.raises((TypeError, ValueError)):
        DeliveryObservation(**fields)  # type: ignore[arg-type]


def test_controller_observation_rejects_private_forecast_object() -> None:
    env = PlantEnvV2(episode=SHORT)
    env.reset(seed=41)
    source = env.controller_observation
    private = env._episode  # noqa: SLF001 - explicit adversarial contract test
    assert private is not None

    with pytest.raises(TypeError, match="PublicForecastIssueV2"):
        ControllerObservationV2(
            state=source.state,
            delivery=source.delivery,
            forecast=private.weather.current_issue(0),  # type: ignore[arg-type]
            progress=source.progress,
            decision_interval_s=source.decision_interval_s,
        )
