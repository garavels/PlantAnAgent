"""Opt-in alkaline commitment dynamics, accounting, and compatibility."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from plant_sim import (
    COMMITMENT_DYNAMICS_VERSION,
    COMMITMENT_INTENT_DEADBAND_FRAC,
    DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
    Action,
    ComponentEffect,
    Electrolyser,
    ElectrolyserCommitmentProfile,
    ElectrolyserMode,
    PlantConfig,
    PlantSimulator,
    StorageConfig,
)


DT_H = 1.0 / 60.0
RATED_KW = 3_000.0
START_KW = 0.20 * RATED_KW
STANDBY_KW = 0.02 * RATED_KW
MIN_RUNNING_KW = 0.40 * RATED_KW


def _start_cold(
    electrolyser: Electrolyser,
    *,
    offered_kw: float = RATED_KW,
) -> list[float]:
    profile = electrolyser.commitment_profile
    assert profile is not None
    steps = round(profile.cold_start_s / (DT_H * 3_600.0))
    masses = []
    for _ in range(steps):
        result = electrolyser.advance_commitment(
            offered_kw,
            DT_H,
            h2_headroom_kg=1_000.0,
            commitment_intent=True,
        )
        masses.append(result.h2_produced_kg)
    return masses


def _short_start_profile(**changes: object) -> ElectrolyserCommitmentProfile:
    values: dict[str, object] = {"cold_start_s": 60.0, "min_up_s": 60.0}
    values.update(changes)
    return replace(DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE, **values)


def test_default_profile_is_frozen_labelled_and_json_ready() -> None:
    profile = DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
    assert profile.profile_id == "ael_pressurised_fixed_bop_alpha_v1"
    assert profile.technology == "pressurised alkaline"
    assert "not vendor-calibrated" in profile.calibration_basis
    assert profile.dynamics_version == COMMITMENT_DYNAMICS_VERSION
    assert COMMITMENT_INTENT_DEADBAND_FRAC == 0.01
    assert len(profile.profile_sha256) == 64
    assert (
        replace(profile, cold_start_s=7_200.0).profile_sha256
        != profile.profile_sha256
    )
    assert profile.to_dict()["specific_energy_multiplier_anchors"] == [
        [0.4, 1.12],
        [0.6, 1.06],
        [0.8, 1.02],
        [1.0, 1.0],
    ]
    with pytest.raises(FrozenInstanceError):
        profile.cold_start_s = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"profile_id": ""}, ValueError),
        ({"technology": 4}, TypeError),
        ({"dynamics_version": "unknown"}, ValueError),
        ({"cold_start_s": True}, TypeError),
        ({"cold_start_s": 0.0}, ValueError),
        ({"hot_start_s": float("nan")}, ValueError),
        ({"cooldown_s": -1.0}, ValueError),
        ({"shortfall_grace_s": float("inf")}, ValueError),
        ({"hot_standby_load_frac": 0.20}, ValueError),
        ({"start_load_frac": 0.40}, ValueError),
        ({"low_load_threshold_frac": 0.39}, ValueError),
        ({"ramp_up_frac_per_min": 0.0}, ValueError),
        ({"nominal_specific_energy_kwh_per_kg": 51.9}, ValueError),
        ({"specific_energy_multiplier_anchors": [(0.4, 1.1), (1.0, 1.0)]}, TypeError),
        ({"specific_energy_multiplier_anchors": ((0.4, 1.1),)}, ValueError),
        (
            {"specific_energy_multiplier_anchors": ((0.4, 1.0), (1.0, 1.1))},
            ValueError,
        ),
        (
            {"specific_energy_multiplier_anchors": ((0.5, 1.1), (1.0, 1.0))},
            ValueError,
        ),
        (
            {"specific_energy_multiplier_anchors": ((0.4, 1.1), (0.9, 1.0))},
            ValueError,
        ),
    ],
)
def test_profile_validation_rejects_ambiguous_or_nonphysical_values(
    changes: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        replace(DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE, **changes)


def test_commitment_constructor_rejects_unlabelled_values() -> None:
    with pytest.raises(TypeError, match="ElectrolyserCommitmentProfile"):
        PlantSimulator(electrolyser_commitment=True)  # type: ignore[arg-type]


def test_profile_time_boundaries_and_first_running_ramp_must_fit_physics_grid() -> None:
    with pytest.raises(ValueError, match="align exactly"):
        PlantSimulator(
            PlantConfig(dt_s=70.0),
            electrolyser_commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
        )
    with pytest.raises(ValueError, match="too short"):
        PlantSimulator(
            PlantConfig(dt_s=30.0),
            electrolyser_commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
        )
    with pytest.raises(ValueError, match="shortfall_grace_s"):
        PlantSimulator(
            electrolyser_commitment=replace(
                DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE,
                shortfall_grace_s=100.0,
            )
        )


def test_cold_start_duration_energy_and_next_interval_hydrogen_boundary() -> None:
    electrolyser = Electrolyser(
        commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
    )

    for minute in range(59):
        result = electrolyser.advance_commitment(
            RATED_KW,
            DT_H,
            h2_headroom_kg=1_000.0,
            commitment_intent=True,
        )
        assert result.accepted_power_kw == pytest.approx(START_KW)
        assert result.h2_produced_kg == 0.0
        assert electrolyser.commitment_snapshot is not None
        assert electrolyser.commitment_snapshot.mode is ElectrolyserMode.COLD_STARTING
        assert electrolyser.commitment_snapshot.startup_remaining_s == pytest.approx(
            3_600.0 - (minute + 1) * 60.0
        )

    final_start = electrolyser.advance_commitment(
        RATED_KW,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=True,
    )
    snapshot = electrolyser.commitment_snapshot
    assert snapshot is not None
    assert final_start.accepted_power_kw == pytest.approx(START_KW)
    assert final_start.h2_produced_kg == 0.0
    assert snapshot.mode is ElectrolyserMode.RUNNING
    assert snapshot.startup_energy_kwh == pytest.approx(600.0)
    assert snapshot.min_up_remaining_s == pytest.approx(3_600.0)
    assert snapshot.mode_time_s == (
        (ElectrolyserMode.OFF, 0.0),
        (ElectrolyserMode.COLD_STARTING, 3_600.0),
        (ElectrolyserMode.RUNNING, 0.0),
        (ElectrolyserMode.HOT_STANDBY, 0.0),
        (ElectrolyserMode.HOT_STARTING, 0.0),
        (ElectrolyserMode.COOLDOWN, 0.0),
    )
    assert not electrolyser.online
    assert electrolyser.efficiency == 0.0
    assert snapshot.ramp_up_headroom_kw == pytest.approx(750.0)
    assert snapshot.ramp_down_headroom_kw == 0.0

    first_running = electrolyser.advance_commitment(
        RATED_KW,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=True,
    )
    assert first_running.accepted_power_kw == pytest.approx(1_350.0)
    assert first_running.h2_produced_kg > 0.0
    assert electrolyser.online


def test_hot_standby_and_five_minute_hot_start_make_no_hydrogen() -> None:
    electrolyser = Electrolyser(
        commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
    )
    _start_cold(electrolyser)

    standby = electrolyser.advance_commitment(
        STANDBY_KW,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=True,
    )
    assert standby.accepted_power_kw == pytest.approx(STANDBY_KW)
    assert standby.h2_produced_kg == 0.0
    assert not electrolyser.online
    assert electrolyser.commitment_snapshot is not None
    assert electrolyser.commitment_snapshot.mode is ElectrolyserMode.HOT_STANDBY

    for minute in range(5):
        hot_start = electrolyser.advance_commitment(
            RATED_KW,
            DT_H,
            h2_headroom_kg=1_000.0,
            commitment_intent=True,
        )
        assert hot_start.accepted_power_kw == pytest.approx(START_KW)
        assert hot_start.h2_produced_kg == 0.0
        assert not electrolyser.online
        assert electrolyser.commitment_snapshot is not None
        expected = (
            ElectrolyserMode.RUNNING
            if minute == 4
            else ElectrolyserMode.HOT_STARTING
        )
        assert electrolyser.commitment_snapshot.mode is expected

    snapshot = electrolyser.commitment_snapshot
    assert snapshot is not None
    assert snapshot.hot_starts == 1
    assert snapshot.min_up_remaining_s == pytest.approx(3_240.0)
    assert snapshot.standby_energy_kwh == pytest.approx(1.0)
    assert snapshot.startup_energy_kwh == pytest.approx(650.0)
    assert electrolyser.advance_commitment(
        RATED_KW,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=True,
    ).h2_produced_kg > 0.0


def test_shortfall_pauses_once_then_aborts_at_two_minutes_and_blocks_restart() -> None:
    electrolyser = Electrolyser(
        commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
    )
    electrolyser.advance_commitment(
        START_KW,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=True,
    )

    first_pause = electrolyser.advance_commitment(
        0.0,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=True,
    )
    snapshot = electrolyser.commitment_snapshot
    assert first_pause.accepted_power_kw == 0.0
    assert snapshot is not None
    assert snapshot.mode is ElectrolyserMode.COLD_STARTING
    assert snapshot.startup_remaining_s == pytest.approx(3_540.0)
    assert snapshot.shortfall_elapsed_s == pytest.approx(60.0)

    second_pause = electrolyser.advance_commitment(
        0.0,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=True,
    )
    snapshot = electrolyser.commitment_snapshot
    assert second_pause.accepted_power_kw == 0.0
    assert snapshot is not None
    assert snapshot.mode is ElectrolyserMode.COOLDOWN
    assert snapshot.cooldown_remaining_s == pytest.approx(900.0)
    assert snapshot.min_down_remaining_s == pytest.approx(1_800.0)
    assert snapshot.aborted_starts == 1
    assert snapshot.forced_stops == 1

    # Fifteen cooldown minutes elapse before mode becomes off; minimum-down
    # then blocks a further fifteen minutes.  Eligibility is sampled only at
    # the next boundary, so starting begins on the following call.
    for _ in range(15):
        result = electrolyser.advance_commitment(
            RATED_KW,
            DT_H,
            h2_headroom_kg=1_000.0,
            commitment_intent=True,
        )
        assert result.accepted_power_kw == 0.0
    assert electrolyser.commitment_snapshot is not None
    assert electrolyser.commitment_snapshot.mode is ElectrolyserMode.OFF
    assert electrolyser.commitment_snapshot.min_down_remaining_s == pytest.approx(900.0)

    for _ in range(15):
        result = electrolyser.advance_commitment(
            RATED_KW,
            DT_H,
            h2_headroom_kg=1_000.0,
            commitment_intent=True,
        )
        assert result.accepted_power_kw == 0.0
    restarted = electrolyser.advance_commitment(
        RATED_KW,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=True,
    )
    assert restarted.accepted_power_kw == pytest.approx(START_KW)
    assert electrolyser.commitment_snapshot is not None
    assert electrolyser.commitment_snapshot.cold_starts == 2


def test_premature_stop_is_forced_but_post_minimum_up_stop_is_controlled() -> None:
    premature_profile = _short_start_profile(min_up_s=3_600.0)
    premature = Electrolyser(commitment=premature_profile)
    _start_cold(premature)
    premature.advance_commitment(
        0.0,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=False,
    )
    snapshot = premature.commitment_snapshot
    assert snapshot is not None
    assert snapshot.forced_stops == 1
    assert snapshot.controlled_stops == 0
    assert snapshot.mode is ElectrolyserMode.COOLDOWN

    controlled = Electrolyser(commitment=_short_start_profile())
    _start_cold(controlled)
    controlled.advance_commitment(
        MIN_RUNNING_KW,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=True,
    )
    controlled.advance_commitment(
        0.0,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=False,
    )
    snapshot = controlled.commitment_snapshot
    assert snapshot is not None
    assert snapshot.forced_stops == 0
    assert snapshot.controlled_stops == 1


def test_finite_command_deadband_makes_near_endpoint_stop_orderly() -> None:
    profile = _short_start_profile()
    near_endpoint = PlantSimulator(electrolyser_commitment=profile)
    exact_endpoint = PlantSimulator(electrolyser_commitment=profile)
    running = Action(battery_support_kw=RATED_KW)
    for simulator in (near_endpoint, exact_endpoint):
        simulator.step(running)
        simulator.step(running)
        snapshot = simulator.electrolyser_commitment
        assert snapshot is not None
        assert snapshot.mode is ElectrolyserMode.RUNNING
        assert snapshot.min_up_remaining_s == 0.0

    # These physical commands correspond to a raw policy output very close to
    # -1, but not exactly at the endpoint, on both commitment-bearing channels.
    near_endpoint.step(
        Action(electrolyser_share=5e-7, battery_support_kw=0.0015)
    )
    exact_endpoint.step(Action.idle())

    near_snapshot = near_endpoint.electrolyser_commitment
    exact_snapshot = exact_endpoint.electrolyser_commitment
    assert near_snapshot is not None and exact_snapshot is not None
    assert near_snapshot.mode is ElectrolyserMode.COOLDOWN
    assert near_snapshot.forced_stops == 0
    assert near_snapshot.controlled_stops == 1
    assert near_snapshot.to_dict() == exact_snapshot.to_dict()


def test_running_up_and_down_settling_respects_the_frozen_ramp() -> None:
    electrolyser = Electrolyser(
        commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
    )
    _start_cold(electrolyser)
    powers = []
    for offered_kw in (RATED_KW, RATED_KW, RATED_KW, RATED_KW, 2_500.0, 2_000.0):
        result = electrolyser.advance_commitment(
            offered_kw,
            DT_H,
            h2_headroom_kg=1_000.0,
            commitment_intent=True,
        )
        powers.append(result.accepted_power_kw)
    assert powers == pytest.approx([1_350.0, 2_100.0, 2_850.0, 3_000.0, 2_500.0, 2_000.0])
    assert all(abs(right - left) <= 750.0 + 1e-9 for left, right in zip([600.0, *powers], powers))
    snapshot = electrolyser.commitment_snapshot
    assert snapshot is not None
    assert snapshot.ramp_limited_intervals == 3
    assert snapshot.ramp_up_headroom_kw == pytest.approx(750.0)
    assert snapshot.ramp_down_headroom_kw == pytest.approx(750.0)


def test_unfunded_down_ramp_falls_back_to_standby_and_is_counted() -> None:
    electrolyser = Electrolyser(commitment=_short_start_profile())
    _start_cold(electrolyser)
    for _ in range(4):
        electrolyser.advance_commitment(
            RATED_KW,
            DT_H,
            h2_headroom_kg=1_000.0,
            commitment_intent=True,
        )

    before = electrolyser.commitment_snapshot
    assert before is not None
    assert before.previous_power_kw == pytest.approx(RATED_KW)

    # 1.5 MW is a stable operating point but lies below the 2.25 MW next-step
    # floor implied by the prior 3 MW load.  Offered power is a hard bus ceiling:
    # the component cannot manufacture the missing 0.75 MW merely to ramp down.
    fallback = electrolyser.advance_commitment(
        1_500.0,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=True,
    )
    after = electrolyser.commitment_snapshot
    assert after is not None
    assert fallback.accepted_power_kw == pytest.approx(STANDBY_KW)
    assert fallback.h2_produced_kg == 0.0
    assert after.mode is ElectrolyserMode.HOT_STANDBY
    assert after.ramp_limited_intervals == before.ramp_limited_intervals + 1
    assert after.forced_stops == before.forced_stops


def test_part_load_curve_and_external_degradation_multiply_specific_energy() -> None:
    healthy = Electrolyser(commitment=_short_start_profile())
    degraded = Electrolyser(commitment=_short_start_profile())
    degraded.apply_effect(ComponentEffect(degradation=0.5))
    _start_cold(healthy)
    _start_cold(degraded)

    healthy_step = healthy.advance_commitment(
        MIN_RUNNING_KW,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=True,
    )
    degraded_step = degraded.advance_commitment(
        MIN_RUNNING_KW,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=True,
    )
    expected = MIN_RUNNING_KW * DT_H / (52.0 * 1.12)
    assert healthy_step.h2_produced_kg == pytest.approx(expected)
    assert degraded_step.h2_produced_kg == pytest.approx(expected * 0.5)
    assert healthy.specific_energy_multiplier == pytest.approx(1.12)
    assert degraded.specific_energy_kwh_per_kg == pytest.approx(52.0 * 1.12 / 0.5)
    snapshot = degraded.commitment_snapshot
    assert snapshot is not None
    assert snapshot.cold_starts == 1  # faults do not invent degradation/cycle updates


def test_conservative_headroom_cap_never_discards_hydrogen() -> None:
    electrolyser = Electrolyser(commitment=_short_start_profile())
    _start_cold(electrolyser)

    headroom_kg = 0.40
    result = electrolyser.advance_commitment(
        RATED_KW,
        DT_H,
        h2_headroom_kg=headroom_kg,
        commitment_intent=True,
    )
    assert result.accepted_power_kw == pytest.approx(headroom_kg * 52.0 / DT_H)
    assert 0.0 < result.h2_produced_kg <= headroom_kg

    # Less room than one minimum-load minute moves the committed unit to hot
    # standby instead of overproducing and silently discarding mass.
    no_room = electrolyser.advance_commitment(
        RATED_KW,
        DT_H,
        h2_headroom_kg=0.10,
        commitment_intent=True,
    )
    assert no_room.accepted_power_kw == pytest.approx(STANDBY_KW)
    assert no_room.h2_produced_kg == 0.0
    assert electrolyser.commitment_snapshot is not None
    assert electrolyser.commitment_snapshot.mode is ElectrolyserMode.HOT_STANDBY


@pytest.mark.parametrize("during_start", [True, False])
def test_component_outage_aborts_start_or_trips_running_cleanly(
    during_start: bool,
) -> None:
    profile = (
        DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
        if during_start
        else _short_start_profile()
    )
    electrolyser = Electrolyser(commitment=profile)
    if during_start:
        electrolyser.advance_commitment(
            START_KW,
            DT_H,
            h2_headroom_kg=1_000.0,
            commitment_intent=True,
        )
    else:
        _start_cold(electrolyser)

    electrolyser.apply_effect(ComponentEffect(offline=True))
    result = electrolyser.advance_commitment(
        RATED_KW,
        DT_H,
        h2_headroom_kg=1_000.0,
        commitment_intent=True,
    )
    snapshot = electrolyser.commitment_snapshot
    assert result.accepted_power_kw == 0.0
    assert result.h2_produced_kg == 0.0
    assert not electrolyser.online
    assert snapshot is not None
    assert snapshot.mode is ElectrolyserMode.COOLDOWN
    assert snapshot.forced_stops == 1
    assert snapshot.aborted_starts == int(during_start)


def test_reset_clears_all_commitment_timers_and_counters_and_replay_matches() -> None:
    action = Action(battery_support_kw=1_000.0)

    def trace(simulator: PlantSimulator) -> list[tuple[dict, dict]]:
        output = []
        for _ in range(75):
            state = simulator.step(action)
            snapshot = simulator.electrolyser_commitment
            assert snapshot is not None
            output.append((state.to_dict(), snapshot.to_dict()))
        return output

    first = PlantSimulator(
        electrolyser_commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
    )
    second = PlantSimulator(
        electrolyser_commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
    )
    assert trace(first) == trace(second)

    first.reset()
    snapshot = first.electrolyser_commitment
    assert snapshot is not None
    assert snapshot.mode is ElectrolyserMode.OFF
    assert snapshot.cold_starts == 0
    assert snapshot.startup_energy_kwh == 0.0
    assert snapshot.mode_time_s == tuple((mode, 0.0) for mode in ElectrolyserMode)


def test_committed_dispatch_retracts_unused_support_before_battery_discharge() -> None:
    simulator = PlantSimulator(
        electrolyser_commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
    )
    energy_before = simulator.battery.energy_kwh
    state = simulator.step(Action(battery_support_kw=3_000.0))

    assert state.solar_available_kw == 0.0
    assert state.electrolyser_power_kw == pytest.approx(START_KW)
    assert state.battery_to_electrolyser_kw == pytest.approx(START_KW)
    assert energy_before - simulator.battery.energy_kwh == pytest.approx(
        START_KW * simulator.config.dt_h
        / simulator.config.battery.discharge_efficiency
    )


def test_committed_dispatch_conserves_solar_and_electrolyser_bus_energy() -> None:
    simulator = PlantSimulator(
        electrolyser_commitment=DEFAULT_ELECTROLYSER_COMMITMENT_PROFILE
    )
    simulator.run(12 * 60, Action.idle())
    action = Action(
        battery_share=0.25,
        electrolyser_share=0.55,
        dac_share=0.10,
        methanator_share=0.10,
        battery_support_kw=1_000.0,
    )
    for _ in range(180):
        state = simulator.step(action)
        routed_solar_kw = (
            state.solar_to_battery_kw
            + state.solar_to_electrolyser_kw
            + state.solar_to_dac_kw
            + state.solar_to_methanator_kw
            + state.solar_curtailed_kw
        )
        assert routed_solar_kw == pytest.approx(state.solar_available_kw, abs=1e-8)
        assert state.electrolyser_power_kw == pytest.approx(
            state.solar_to_electrolyser_kw
            + state.battery_to_electrolyser_kw,
            abs=1e-8,
        )
        assert 0.0 <= state.h2_level_kg <= state.h2_capacity_kg + 1e-9


def test_tiny_hydrogen_tank_cannot_be_overfilled_by_nonlinear_yield() -> None:
    simulator = PlantSimulator(
        PlantConfig(h2_tank=StorageConfig(capacity_kg=0.40, initial_kg=0.0)),
        electrolyser_commitment=_short_start_profile(),
    )
    action = Action(battery_support_kw=3_000.0)
    for _ in range(10):
        simulator.step(action)
        assert simulator.h2_tank.level_kg <= simulator.h2_tank.capacity_kg + 1e-12


def test_explicit_none_is_exactly_the_omitted_legacy_path() -> None:
    omitted = PlantSimulator()
    explicit_none = PlantSimulator(electrolyser_commitment=None)
    assert omitted.electrolyser_commitment is None
    assert explicit_none.electrolyser_commitment is None
    assert omitted.electrolyser.snapshot() == explicit_none.electrolyser.snapshot()
    assert "commitment" not in omitted.electrolyser.snapshot()
    assert "specific_energy_multiplier" not in omitted.electrolyser.snapshot()

    actions = [
        Action.idle(),
        Action(battery_share=0.3, electrolyser_share=0.7, battery_support_kw=700.0),
        Action(battery_share=0.1, electrolyser_share=0.4, dac_share=0.3, methanator_share=0.2),
        Action.all_to_electrolyser(),
    ]
    for step in range(360):
        action = actions[step % len(actions)]
        left = omitted.step(action)
        right = explicit_none.step(action)
        assert left.to_dict() == right.to_dict()
        assert omitted.electrolyser.snapshot() == explicit_none.electrolyser.snapshot()
        assert omitted.delivery_audit() == explicit_none.delivery_audit()
