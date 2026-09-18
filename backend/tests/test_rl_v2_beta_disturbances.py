"""Contract tests for the private Environment V2 beta disturbance tape."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from plant_sim import FaultRegistry, FaultSpec

from plant_backend.rl.v2.beta_disturbances import (
    DISTURBANCE_GENERATOR_VERSION,
    DisturbanceConfigV1,
    generate_disturbance_episode,
)


def _episode(
    seed: int | np.random.SeedSequence = 17,
    *,
    config: DisturbanceConfigV1 | None = None,
    cycle_count: int = 8,
    physics_dt_s: float = 300.0,
):
    return generate_disturbance_episode(
        config or DisturbanceConfigV1(),
        seed=seed,
        cycle_h=24.0,
        cycle_count=cycle_count,
        physics_dt_s=physics_dt_s,
    )


def test_replay_is_deterministic_and_serialization_is_detached() -> None:
    first = _episode(81)
    replay = _episode(81)

    assert first == replay
    assert first.faults == replay.faults
    assert first.schedule_sha256 == replay.schedule_sha256
    assert first.episode_sha256 == replay.episode_sha256
    assert first.to_dict() == replay.to_dict()
    assert first.private_audit_manifest() == replay.private_audit_manifest()
    assert json.dumps(first.private_audit_manifest(), allow_nan=False, sort_keys=True)

    detached = first.private_audit_manifest()
    detached["schedule"].clear()  # type: ignore[union-attr]
    assert first.private_audit_manifest()["schedule"]


def test_seed_and_config_changes_are_bound_into_episode_identity() -> None:
    base = _episode(101)
    changed_seed = _episode(102)
    changed_config = _episode(
        101,
        config=DisturbanceConfigV1(per_cycle_probability=0.75),
    )

    assert changed_seed.episode_sha256 != base.episode_sha256
    assert changed_config.config_sha256 != base.config_sha256
    assert changed_config.episode_sha256 != base.episode_sha256


def test_config_and_schedule_hashes_use_strict_canonical_json() -> None:
    episode = _episode(103)

    def digest(payload: object) -> str:
        canonical = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    assert episode.config_sha256 == digest(episode.config.to_dict())
    assert episode.schedule_sha256 == digest(
        [fault.to_dict() for fault in episode.faults]
    )


def test_extending_the_horizon_does_not_perturb_existing_cycle_draws() -> None:
    short = _episode(104, cycle_count=3)
    long = _episode(104, cycle_count=9)
    cutoff_s = 3 * 24 * 3_600.0

    assert tuple(fault for fault in long.faults if fault.start_s < cutoff_s) == (
        short.faults
    )
    short_streams = short.private_audit_manifest()["random_streams"][  # type: ignore[index]
        "cycles"
    ]
    long_streams = long.private_audit_manifest()["random_streams"][  # type: ignore[index]
        "cycles"
    ]
    assert short_streams == long_streams[:3]  # type: ignore[index]


def test_disabled_profile_has_the_exact_empty_schedule() -> None:
    first = _episode(2, config=DisturbanceConfigV1(enabled=False))
    other_seed = _episode(999, config=DisturbanceConfigV1(enabled=False))

    assert first.faults == ()
    assert first.private_audit_manifest()["schedule"] == []
    assert first.schedule_sha256 == other_seed.schedule_sha256
    assert first.episode_sha256 != other_seed.episode_sha256


def test_each_cycle_has_at_most_one_aligned_finite_outage() -> None:
    dt_s = 300.0
    cycle_s = 24.0 * 3_600.0
    episode = _episode(
        44,
        config=DisturbanceConfigV1(
            per_cycle_probability=1.0,
            duration_h=(0.5, 3.0),
        ),
        cycle_count=20,
        physics_dt_s=dt_s,
    )

    assert len(episode.faults) == 20
    occupied: set[int] = set()
    for fault in episode.faults:
        assert fault.start_s is not None
        assert fault.duration_s is not None
        cycle_index = int(fault.start_s // cycle_s)
        assert cycle_index not in occupied
        occupied.add(cycle_index)
        assert fault.start_s / dt_s == round(fault.start_s / dt_s)
        assert fault.duration_s / dt_s == round(fault.duration_s / dt_s)
        assert 0.5 * 3_600.0 <= fault.duration_s <= 3.0 * 3_600.0
        assert cycle_index * cycle_s <= fault.start_s
        assert fault.start_s + fault.duration_s <= (cycle_index + 1) * cycle_s - dt_s


def test_generated_specs_use_simulator_normalized_outage_semantics() -> None:
    fault = _episode(
        7,
        config=DisturbanceConfigV1(
            per_cycle_probability=1.0,
            duration_h=(1.0, 1.0),
        ),
        cycle_count=1,
        physics_dt_s=3_600.0,
    ).faults[0]

    assert isinstance(fault, FaultSpec)
    assert fault == FaultSpec.parse(fault)
    assert fault.type == "dac_outage"
    assert fault.kind.target == "dac"
    assert fault.kind.mode == "outage"
    assert fault.severity == 1.0
    assert fault.duration_s == 3_600.0

    registry = FaultRegistry()
    installed = registry.inject(fault)
    assert installed.end_s is not None
    assert not registry.effect_for_at("dac", installed.start_s - 1.0).offline
    assert registry.effect_for_at("dac", installed.start_s).offline
    assert not registry.effect_for_at("dac", installed.end_s).offline


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": 1},
        {"per_cycle_probability": True},
        {"per_cycle_probability": float("nan")},
        {"per_cycle_probability": -0.01},
        {"per_cycle_probability": 1.01},
        {"duration_h": [1.0, 2.0]},
        {"duration_h": (0.0, 2.0)},
        {"duration_h": (3.0, 2.0)},
        {"duration_h": (1.0, float("inf"))},
    ],
)
def test_invalid_configuration_values_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        DisturbanceConfigV1(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"seed": True}, ValueError),
        ({"seed": -1}, ValueError),
        ({"seed": "17"}, TypeError),
        ({"seed": np.random.default_rng(17)}, TypeError),
        ({"cycle_h": 12.0}, ValueError),
        ({"cycle_h": float("nan")}, ValueError),
        ({"cycle_count": 0}, ValueError),
        ({"cycle_count": True}, ValueError),
        ({"cycle_count": 1.5}, ValueError),
        ({"physics_dt_s": 0.0}, ValueError),
        ({"physics_dt_s": float("inf")}, ValueError),
        ({"physics_dt_s": 700.0}, ValueError),
    ],
)
def test_invalid_seed_and_geometry_fail_closed(
    overrides: dict[str, object],
    error: type[Exception],
) -> None:
    arguments: dict[str, object] = {
        "seed": 17,
        "cycle_h": 24.0,
        "cycle_count": 2,
        "physics_dt_s": 300.0,
    }
    arguments.update(overrides)
    with pytest.raises(error):
        generate_disturbance_episode(
            DisturbanceConfigV1(),
            **arguments,  # type: ignore[arg-type]
        )


def test_duration_must_fit_and_align_to_the_physics_lattice() -> None:
    with pytest.raises(ValueError, match="align"):
        _episode(
            config=DisturbanceConfigV1(duration_h=(0.51, 1.0)),
            physics_dt_s=300.0,
        )
    with pytest.raises(ValueError, match="leave one physics timestep"):
        _episode(config=DisturbanceConfigV1(duration_h=(1.0, 25.0)))
    with pytest.raises(ValueError, match="leave one physics timestep"):
        _episode(config=DisturbanceConfigV1(duration_h=(24.0, 24.0)))


def test_seed_sequence_input_is_snapshotted_without_consuming_it() -> None:
    seed = np.random.SeedSequence([4, 8, 15, 16, 23, 42], spawn_key=(9,))
    seed.spawn(2)
    before = deepcopy(seed.state)

    first = _episode(seed)
    after_first = deepcopy(seed.state)
    replay = _episode(seed)

    assert after_first == before
    assert seed.state == before
    assert replay == first
    root = first.private_audit_manifest()["random_streams"]["root"]  # type: ignore[index]
    assert root["spawn_key"] == [9]  # type: ignore[index]
    assert root["initial_state"]["n_children_spawned"] == 2  # type: ignore[index]


def test_frozen_config_and_episode_have_stable_to_dict_forms() -> None:
    config = DisturbanceConfigV1()
    episode = _episode(config=config)

    assert replace(config).to_dict() == config.to_dict()
    assert episode.to_dict()["disturbance_generator_version"] == (
        DISTURBANCE_GENERATOR_VERSION
    )
    assert "schedule" not in episode.to_dict()
    with pytest.raises(FrozenInstanceError):
        episode.faults = ()  # type: ignore[misc]
