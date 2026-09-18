"""Short exact checks for delivery-only variants of retained plant episodes.

The external recording is intentionally optional in ordinary checkouts. These
tests exercise the real reconstruction and dispatcher when it is available;
they never substitute synthetic evidence for a missing retained episode.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
from plant_sim import ElectrolyserMode

from docs.research.tools.feasible_mission_controller import DailyCoverageController
from docs.research.tools.feasible_mission_replay import (
    RESEARCH_ROOT,
    FixedMissionEnv,
    identity_smoke,
    load_frozen_scenario,
    make_environment,
    run_controller,
    state_differences,
)
from plant_backend.rl.v2.beta_control import CyclicMarketControllerObservationV1
from plant_backend.rl.v2.setpoint_control import ProcessSetpointTargetsV1


SEED = 9700300


@pytest.fixture(scope="module")
def frozen_source():
    recording = RESEARCH_ROOT / f"{SEED}-OperatingMPC/episode.json.gz"
    if not recording.is_file():
        pytest.skip(
            "Retained research episode 9700300 is unavailable; exact historical "
            f"identity checks require {recording}"
        )
    return load_frozen_scenario(SEED)


@pytest.fixture
def variant_env(frozen_source):
    env = FixedMissionEnv(frozen_source, 30.0)
    env.reset(seed=SEED)
    try:
        yield env
    finally:
        env.close()


@pytest.mark.parametrize("rate", [25.0, 30.0])
def test_changed_rate_preserves_the_original_physical_case(frozen_source, rate):
    env = make_environment(SEED, rate)
    try:
        original = frozen_source.original_manifest
        changed = env.audit_manifest()
        state_differences(
            frozen_source.source_payload["minute_truth_states"][0],
            env.simulator.state.to_dict(),
        )
        assert env.simulator.state.ch4_level_kg == original["cycle_reference"]["ch4_kg"]
        for field in (
            "market",
            "disturbances",
            "plant_config",
            "constraints",
            "cycle_reference",
            "beta_task",
            "economic_assumptions",
        ):
            assert changed[field] == original[field], field

        old_episode = original["episode"]
        new_episode = changed["episode"]
        assert {
            key: value
            for key, value in new_episode.items()
            if key not in {"commitments", "exogenous_episode_sha256"}
        } == {
            key: value
            for key, value in old_episode.items()
            if key not in {"commitments", "exogenous_episode_sha256"}
        }
        assert len(new_episode["commitments"]) == 28
        for before, after in zip(
            old_episode["commitments"], new_episode["commitments"], strict=True
        ):
            assert after == {**before, "requested_kg": rate * 6.0}
        assert changed["episode_config"] == {
            **original["episode_config"],
            "delivery_rate_kg_h": rate,
        }
        assert changed["manifest_sha256"] != original["manifest_sha256"]
        assert changed["checkpoint_contract"] != original["checkpoint_contract"]
    finally:
        env.close()


def test_a_clean_partial_run_cannot_claim_any_complete_mission_pass(variant_env):
    observations = []

    def idle(public, **kwargs):
        assert isinstance(public, CyclicMarketControllerObservationV1)
        assert not hasattr(public, "simulator")
        assert not hasattr(public, "episode")
        assert not hasattr(public, "seed")
        assert set(kwargs) == {"constraints", "config", "commitment_profile"}
        observations.append(public)
        return ProcessSetpointTargetsV1(0.0, 0.0, 0.0, 0.0)

    result = run_controller(variant_env, idle, max_decisions=2)
    assert result["failure"] is None
    assert result["decisions"] == 2
    assert result["physical_steps"] == 30
    assert len(result["minute_truth_states"]) == 31
    assert len(observations) == 2
    assert observations[0].plant.state.boundary.step == 0
    assert observations[1].plant.state.boundary.step == 15
    assert result["minute_checks"]["exogenous_forcing_exact"]
    for flag in (
        "execution_complete",
        "hard_safety_passed",
        "every_delivery_passed",
        "all_daily_returns_passed",
        "mission_passed",
    ):
        assert result[flag] is False, flag


def test_a_controller_exception_is_retained_as_a_failed_prefix(variant_env):
    def failing_controller(public, **kwargs):
        raise RuntimeError("intentional control failure")

    result = run_controller(variant_env, failing_controller, max_decisions=2)
    assert result["failure"] == {
        "type": "RuntimeError",
        "message": "intentional control failure",
        "decision": 0,
        "physics_steps": 0,
    }
    assert result["commands"] == []
    assert result["execution_complete"] is False
    assert result["mission_passed"] is False


@pytest.mark.parametrize(
    ("decision_in_cycle", "steps_to_next", "pending_kg"),
    [
        (0, 360, 720.0),  # Four six-hour deliveries, including the 24 h item.
        (72, 360, 180.0),  # At 18 h, the daily boundary delivery still remains.
        (95, 15, 180.0),  # Due at the end of the last controller interval.
        (95, 16, 0.0),  # Due strictly after this day's boundary.
    ],
)
def test_public_coverage_includes_delivery_at_the_daily_boundary(
    variant_env, decision_in_cycle, steps_to_next, pending_kg
):
    public = variant_env.cyclic_market_controller_observation
    plant = replace(
        deepcopy(public.plant).without_forecast(),
        delivery=replace(public.plant.delivery, steps_to_next=steps_to_next),
    )
    cycle = replace(
        public.cycle,
        decision_in_cycle=decision_in_cycle,
        progress=decision_in_cycle / public.cycle.cycle_decisions,
    )
    # Give the controller only its declared public inputs, without an environment,
    # market tape, weather forecast or retained scenario identity to consult.
    detached = SimpleNamespace(plant=plant, cycle=cycle)
    decision = DailyCoverageController().compute_command(
        detached,
        constraints=variant_env.constraints,
        config=variant_env.config,
        commitment_profile=variant_env.episode_config.electrolyser_commitment,
    )
    assert decision.evidence["remaining_deliveries_kg"] == pending_kg
    assert decision.invalid is False


def test_original_commands_reproduce_the_retained_minute_physics(frozen_source):
    replay = identity_smoke(seed=frozen_source.seed, decisions=8)
    assert replay["decisions"] == 8
    assert replay["physical_steps"] == 120
    assert replay["all_physical_states_match"] is True
    assert replay["minute_checks"]["exogenous_forcing_exact"] is True
    assert replay["minute_checks"]["resource_bounds_passed"] is True


def test_a_cloud_dip_cannot_cancel_an_already_started_cold_sequence(variant_env):
    public = variant_env.cyclic_market_controller_observation
    profile = variant_env.episode_config.electrolyser_commitment
    start_power = profile.start_load_frac * variant_env.config.electrolyser.rated_kw
    boundary = replace(public.plant.state.boundary, solar_available_kw=300.0)
    original_snapshot = public.plant.state.electrolyser_commitment

    def command_at(snapshot):
        plant = replace(
            public.plant,
            state=replace(
                public.plant.state,
                boundary=boundary,
                electrolyser_commitment=snapshot,
            ),
        )
        return (
            DailyCoverageController()
            .compute_command(
                replace(public, plant=plant),
                constraints=variant_env.constraints,
                config=variant_env.config,
                commitment_profile=profile,
            )
            .command
        )

    # This solar level is too low to initiate a fresh cold start.
    assert command_at(original_snapshot).targets.electrolyser_target_kw == 0.0
    started = replace(
        original_snapshot,
        mode=ElectrolyserMode.COLD_STARTING,
        mode_elapsed_s=900.0,
        startup_remaining_s=2700.0,
        previous_power_kw=start_power,
        cold_starts=1,
    )
    # Once started, retain the required 600 kW; the ordinary dispatcher and guard
    # decide how much support is physically available, without canceling intent.
    assert command_at(started).targets.electrolyser_target_kw == start_power
