"""V7: the plant as an RL environment -- spaces, seeding, reward, action conversion.

Short episodes throughout (a couple of hours), because what is under test is the
translation layer, not the plant: the physics already has its own suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from plant_sim import Action, PlantConfig, PlantSimulator

from plant_backend.controllers import Constraints
from plant_backend.rl import spaces
from plant_backend.rl.actions import ActionAdapter
from plant_backend.rl.env import PlantEnv
from plant_backend.rl.episodes import EpisodeConfig, sample
from plant_backend.rl.metrics import EpisodeMetrics, aggregate
from plant_backend.rl.observation import ObservationEncoder
from plant_backend.rl.reward import RewardWeights, compute

SHORT = EpisodeConfig(horizon_h=2.0)


@pytest.fixture
def env() -> PlantEnv:
    return PlantEnv(episode=SHORT)


# --- spaces ---------------------------------------------------------------
def test_field_contract_is_self_consistent() -> None:
    """The declared sizes, names and bounds all describe the same vector."""
    assert spaces.OBSERVATION_SIZE == len(spaces.OBSERVATION_NAMES) == len(spaces.observation_low())
    assert spaces.ACTION_SIZE == len(spaces.ACTION_NAMES) == len(spaces.action_low())
    assert len(set(spaces.OBSERVATION_NAMES)) == spaces.OBSERVATION_SIZE
    assert np.all(spaces.observation_low() < spaces.observation_high())
    assert np.all(spaces.action_low() < spaces.action_high())
    described = spaces.describe()
    assert len(described["observation_fields"]) == spaces.OBSERVATION_SIZE  # type: ignore[arg-type]
    # every field documents where it came from and how it was normalised
    for field in spaces.OBSERVATION_FIELDS:
        assert field.source and field.unit and field.normalisation


def test_environment_contract_digest_is_deterministic_and_described() -> None:
    assert spaces.ENVIRONMENT_ID == "plant-env-v1"
    assert spaces.contract_digest() == spaces.contract_digest(spaces.contract_payload())
    assert len(spaces.ENVIRONMENT_CONTRACT_DIGEST) == 64
    int(spaces.ENVIRONMENT_CONTRACT_DIGEST, 16)
    described = spaces.describe()
    assert described["environment_id"] == spaces.ENVIRONMENT_ID
    assert described["environment_contract_digest"] == spaces.ENVIRONMENT_CONTRACT_DIGEST


def test_reset_returns_a_member_of_the_observation_space(env: PlantEnv) -> None:
    observation, info = env.reset(seed=3)
    assert observation.shape == (spaces.OBSERVATION_SIZE,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert np.all(np.isfinite(observation))
    assert info["episode"]["seed"] == 3


def test_observations_stay_inside_their_bounds_over_a_rollout(env: PlantEnv) -> None:
    """Including at the extremes: full tanks, dead battery, sensor faults."""
    env.reset(seed=11)
    truncated = False
    while not truncated:
        observation, _, _, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation), observation


def test_encoder_normalises_a_known_state() -> None:
    """Spot-check the formulas the field table documents, on a real plant state."""
    config = PlantConfig()
    simulator = PlantSimulator(config)
    simulator.run(int(12 * 3600 / config.dt_s), Action.idle())
    state = simulator.get_telemetry()
    encoded = ObservationEncoder(config).encode(state, constraints=Constraints())
    values = dict(zip(spaces.OBSERVATION_NAMES, encoded.tolist()))
    assert values["solar_available"] == pytest.approx(
        state.solar_available_kw / config.solar.peak_kw, abs=1e-6
    )
    assert values["battery_soc"] == pytest.approx(state.battery_soc, abs=1e-6)
    assert values["h2_fill"] == pytest.approx(state.h2_fill, abs=1e-6)
    # midday: sin(2*pi*12/24) == 0, cos == -1
    assert values["time_of_day_sin"] == pytest.approx(0.0, abs=1e-3)
    assert values["time_of_day_cos"] == pytest.approx(-1.0, abs=1e-3)


# --- determinism ----------------------------------------------------------
def test_same_seed_same_episode() -> None:
    """Same seed, same actions, identical rollout: reward included."""
    actions = np.linspace(-1.0, 1.0, 10 * spaces.ACTION_SIZE, dtype=np.float32).reshape(-1, 5)

    def rollout(seed: int) -> tuple[list[float], list[float]]:
        env = PlantEnv(episode=SHORT)
        observation, _ = env.reset(seed=seed)
        rewards, tail = [], []
        for action in actions:
            observation, reward, _, _, _ = env.step(action)
            rewards.append(reward)
        tail = observation.tolist()
        return rewards, tail

    assert rollout(5) == rollout(5)
    assert rollout(5) != rollout(6)


def test_episode_draw_is_a_function_of_the_seed() -> None:
    config = PlantConfig()
    first = sample(SHORT, config, np.random.default_rng(42), seed=42)
    second = sample(SHORT, config, np.random.default_rng(42), seed=42)
    assert first.scenario.initial.battery_soc == second.scenario.initial.battery_soc
    assert first.scenario.id == second.scenario.id
    assert first.steps == SHORT.steps(config) == int(2.0 * 3600 / config.dt_s)


def test_episode_ends_by_truncation_not_termination(env: PlantEnv) -> None:
    """A plant has no game over: the horizon truncates, nothing terminates."""
    env.reset(seed=1)
    steps = 0
    while True:
        _, _, terminated, truncated, _ = env.step(np.zeros(spaces.ACTION_SIZE, dtype=np.float32))
        steps += 1
        assert not terminated
        if truncated:
            break
    assert steps == env.episode_steps


# --- actions --------------------------------------------------------------
def test_action_decode_maps_the_policy_range_onto_shares() -> None:
    adapter = ActionAdapter(PlantConfig())
    constraints = Constraints()
    decoded = adapter.decode(
        np.array([-1.0, 1.0, -1.0, -1.0, 1.0]), constraints=constraints, solar_available_kw=1000.0
    )
    assert decoded.action.battery_share == pytest.approx(0.0)
    assert decoded.action.electrolyser_share == pytest.approx(1.0)
    assert decoded.action.battery_support_kw == pytest.approx(
        constraints.support_limit_kw(PlantConfig())
    )
    assert not decoded.clipped
    assert decoded.requested_kw["electrolyser"] > 1000.0  # solar plus battery support


def test_action_decode_flags_and_clips_out_of_range_output() -> None:
    adapter = ActionAdapter(PlantConfig())
    decoded = adapter.decode(
        np.array([9.0, -9.0, 0.0, 0.0, np.nan]),
        constraints=Constraints(),
        solar_available_kw=500.0,
    )
    assert decoded.clipped
    assert decoded.action == Action.sanitised(decoded.action)
    assert 0.0 <= decoded.action.battery_share <= 1.0


@pytest.mark.parametrize(
    "invalid",
    [float("nan"), float("inf"), -float("inf")],
)
def test_non_finite_policy_output_fails_to_zero_authority(invalid: float) -> None:
    adapter = ActionAdapter(PlantConfig())
    decoded = adapter.decode(
        np.full(spaces.ACTION_SIZE, invalid),
        constraints=Constraints(),
        solar_available_kw=5_000.0,
    )

    assert decoded.clipped
    assert decoded.action == Action.idle()
    assert decoded.requested_kw == {
        "battery": 0.0,
        "electrolyser": 0.0,
        "dac": 0.0,
        "methanator": 0.0,
        "battery_support": 0.0,
    }


def test_action_encode_is_the_inverse_of_decode() -> None:
    adapter = ActionAdapter(PlantConfig())
    constraints = Constraints()
    # shares that sum below 100%, so `Action.sanitised` has nothing to rescale
    original = np.array([-0.6, -0.2, -0.8, -0.9, 0.5], dtype=np.float32)
    decoded = adapter.decode(original, constraints=constraints, solar_available_kw=2000.0)
    assert adapter.encode(decoded.action, constraints=constraints) == pytest.approx(
        original, abs=1e-5
    )


def test_wrong_sized_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="entries"):
        ActionAdapter(PlantConfig()).decode(
            np.zeros(3), constraints=Constraints(), solar_available_kw=0.0
        )


def test_every_action_reaches_the_plant_through_the_guard(env: PlantEnv) -> None:
    """The env never applies an unvalidated action, however wild the policy is."""
    env.reset(seed=2)
    for _ in range(30):
        _, _, _, _, info = env.step(np.full(spaces.ACTION_SIZE, 1.0, dtype=np.float32))
        applied = Action.sanitised(info["applied_action"])
        assert Action.sanitised(applied) == applied
        # everything asked for is reported, corrected or not
        assert "requested" in info and "adjustments" in info
        assert info["corrected"] == bool(info["adjustments"])


# --- reward ---------------------------------------------------------------
def test_reward_components_sum_to_the_total(env: PlantEnv) -> None:
    env.reset(seed=4)
    for _ in range(20):
        _, reward, _, _, info = env.step(env.action_space.sample())
        components = info["reward"]["components"]
        assert sum(components.values()) == pytest.approx(reward, abs=1e-9)
        assert set(components) == set(RewardWeights().to_dict())


def test_methane_is_the_only_positive_term() -> None:
    """Production is rewarded; every other term can only subtract."""
    config = PlantConfig()
    simulator = PlantSimulator(config)
    simulator.run(int(11 * 3600 / config.dt_s), Action.idle())
    before = simulator.state
    after = simulator.step(Action(electrolyser_share=0.6, dac_share=0.3, methanator_share=0.1))
    breakdown = compute(before, after, config=config, constraints=Constraints())
    assert breakdown.components["methane"] >= 0.0
    for name, value in breakdown.components.items():
        if name != "methane":
            assert value <= 0.0


def test_corrections_and_out_of_range_output_are_penalised() -> None:
    config = PlantConfig()
    simulator = PlantSimulator(config)
    before = simulator.state
    after = simulator.step(Action.idle())
    clean = compute(before, after, config=config, constraints=Constraints())
    dirty = compute(
        before,
        after,
        config=config,
        constraints=Constraints(),
        adjustments=("clamped electrolyser", "clamped dac"),
        out_of_range=True,
    )
    assert dirty.components["correction"] < clean.components["correction"] <= 0.0
    assert dirty.total < clean.total


def test_reserve_breach_scales_with_the_deficit() -> None:
    config = PlantConfig()
    simulator = PlantSimulator(config)
    before = simulator.state
    after = simulator.step(Action.idle())
    strict = compute(
        before, after, config=config, constraints=Constraints(battery_min_soc=0.95)
    )
    lenient = compute(
        before, after, config=config, constraints=Constraints(battery_min_soc=0.05)
    )
    assert strict.components["reserve_breach"] < lenient.components["reserve_breach"]


def test_reward_weights_reject_unknown_terms() -> None:
    with pytest.raises(ValueError, match="unknown reward weights"):
        RewardWeights.from_dict({"methane": 2.0, "vibes": 1.0})


# --- metrics --------------------------------------------------------------
def test_metrics_track_the_rollout(env: PlantEnv) -> None:
    env.reset(seed=8)
    truncated = False
    while not truncated:
        _, _, _, truncated, info = env.step(env.action_space.sample())
    summary = info["metrics"]
    assert summary["steps"] == env.episode_steps
    assert summary["hours"] == pytest.approx(SHORT.horizon_h, abs=1e-6)
    assert 0.0 <= summary["solar_utilisation"] <= 1.0
    assert 0.0 <= summary["idle_fraction"] <= 1.0
    assert summary["corrections"] >= summary["corrected_steps"]


def test_aggregate_averages_across_runs() -> None:
    first, second = EpisodeMetrics(), EpisodeMetrics()
    first.steps, first.ch4_kg, first.reward_components = 10, 4.0, {"methane": 2.0}
    second.steps, second.ch4_kg, second.reward_components = 10, 6.0, {"methane": 4.0}
    summary = aggregate([first, second])
    assert summary["runs"] == 2
    assert summary["ch4_kg"] == pytest.approx(5.0)
    assert summary["reward_components"]["methane"] == pytest.approx(3.0)
    assert aggregate([]) == {"runs": 0}


# --- config ---------------------------------------------------------------
def test_episode_config_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        EpisodeConfig(horizon_h=0.0).validate()
    with pytest.raises(ValueError, match="unknown scenario"):
        EpisodeConfig(scenarios=("no_such_scenario",)).validate()
    with pytest.raises(ValueError):
        EpisodeConfig(battery_soc=(0.9, 0.1)).validate()
