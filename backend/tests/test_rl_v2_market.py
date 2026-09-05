"""Causality, replay, validation, and shape contracts for V2 beta market."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from plant_backend.rl.v2.economics import (
    DEFAULT_HOURLY_TARIFF_EUR_PER_MWH,
    DayAheadTariffV1,
    MAX_ABS_TARIFF_EUR_PER_MWH,
)
from plant_backend.rl.v2.market import (
    FORECAST_HORIZON_H,
    MARKET_GENERATOR_VERSION,
    NORMALIZED_MARKET_ISSUE_SIZE,
    MarketConfigV1,
    MarketRegimeV1,
    generate_market_episode,
)


DECISIONS = 12
DECISION_INTERVAL_S = 900.0
START_HOUR = 6.0


def _episode(
    seed: int | np.random.SeedSequence = 41,
    *,
    config: MarketConfigV1 | None = None,
    prices: tuple[float, ...] | None = None,
    baseline: DayAheadTariffV1 | None = None,
):
    return generate_market_episode(
        config or MarketConfigV1(forecast_ensemble_members=16),
        seed=seed,
        decision_count=DECISIONS,
        decision_interval_s=DECISION_INTERVAL_S,
        start_hour=START_HOUR,
        baseline=baseline or DayAheadTariffV1(),
        realized_prices_eur_per_mwh=prices,
    )


def test_market_episode_is_exactly_replayable_and_frozen() -> None:
    first = _episode(81)
    replay = _episode(81)

    assert first == replay
    assert first.prices_eur_per_mwh == replay.prices_eur_per_mwh
    assert first.issues == replay.issues
    assert first.private_audit_manifest() == replay.private_audit_manifest()
    assert first.price_at_decision(3) == first.prices_eur_per_mwh[3]
    assert first.issue_at_decision(3) is first.issues[3]
    assert len(first.episode_sha256) == 64
    int(first.episode_sha256, 16)
    with pytest.raises(FrozenInstanceError):
        first.start_hour = 7.0  # type: ignore[misc]


def test_seed_sequence_is_snapshotted_without_consuming_the_caller() -> None:
    seed = np.random.SeedSequence(117, spawn_key=(3,))

    first = _episode(seed)
    replay = _episode(seed)

    assert first == replay
    assert seed.state["n_children_spawned"] == 0
    streams = first.private_audit_manifest()["random_streams"]
    assert isinstance(streams, dict)
    assert streams["market.root"]["spawn_key"] == [3]  # type: ignore[index]


@pytest.mark.parametrize(
    "regime",
    (
        MarketRegimeV1.CALM,
        MarketRegimeV1.VOLATILE,
        MarketRegimeV1.SCARCITY,
    ),
)
def test_explicit_regime_is_respected_and_mixed_is_seed_selected(
    regime: MarketRegimeV1,
) -> None:
    explicit = _episode(
        4, config=MarketConfigV1(regime=regime, forecast_ensemble_members=10)
    )
    assert explicit.selected_regime is regime

    selected = {
        generate_market_episode(
            MarketConfigV1(forecast_ensemble_members=10),
            seed=seed,
            decision_count=1,
            decision_interval_s=3_600.0,
            start_hour=0.0,
        ).selected_regime
        for seed in range(12)
    }
    assert selected == {
        MarketRegimeV1.CALM,
        MarketRegimeV1.VOLATILE,
        MarketRegimeV1.SCARCITY,
    }
    assert MarketRegimeV1.MIXED not in selected


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (MarketConfigV1(regime="calm"), "MarketRegimeV1"),  # type: ignore[arg-type]
        (MarketConfigV1(forecast_ensemble_members=9), "at least 10"),
        (MarketConfigV1(forecast_ensemble_members=True), "at least 10"),  # type: ignore[arg-type]
        (MarketConfigV1(forecast_innovation_scale=0.0), "positive"),
        (MarketConfigV1(forecast_innovation_scale=float("nan")), "finite"),
    ],
)
def test_market_configuration_validation_is_strict(
    config: MarketConfigV1,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()

    with pytest.raises(FrozenInstanceError):
        MarketConfigV1().regime = MarketRegimeV1.CALM  # type: ignore[misc]


def test_truth_and_every_public_forecast_are_complete_finite_and_bounded() -> None:
    episode = _episode(29)
    bound = MAX_ABS_TARIFF_EUR_PER_MWH

    assert len(episode.prices_eur_per_mwh) == DECISIONS
    assert len(episode.issues) == DECISIONS
    assert all(-bound <= price <= bound for price in episode.prices_eur_per_mwh)
    for decision, issue in enumerate(episode.issues):
        assert issue.issued_at_decision == decision
        assert issue.issued_at_offset_s == decision * DECISION_INTERVAL_S
        assert issue.current_eur_per_mwh == episode.prices_eur_per_mwh[decision]
        assert len(issue.bins) == FORECAST_HORIZON_H
        assert tuple(item.hour_index for item in issue.bins) == tuple(range(24))
        for forecast_bin in issue.bins:
            assert -bound <= forecast_bin.p10_eur_per_mwh
            assert (
                forecast_bin.p10_eur_per_mwh
                <= forecast_bin.mean_eur_per_mwh
                <= forecast_bin.p90_eur_per_mwh
            )
            assert forecast_bin.p90_eur_per_mwh <= bound

        vector = issue.normalized()
        assert len(vector) == NORMALIZED_MARKET_ISSUE_SIZE == 73
        assert vector[0] == issue.current_eur_per_mwh / bound
        assert vector[1:4] == (
            issue.bins[0].mean_eur_per_mwh / bound,
            issue.bins[0].p10_eur_per_mwh / bound,
            issue.bins[0].p90_eur_per_mwh / bound,
        )
        assert all(np.isfinite(value) and -1.0 <= value <= 1.0 for value in vector)


def test_public_issue_is_controller_safe_and_has_a_canonical_hash() -> None:
    issue = _episode(13).issues[0]
    payload = issue.to_dict()

    assert tuple(payload) == (
        "schema",
        "market_id",
        "issued_at_decision",
        "issued_at_offset_s",
        "issue_time_basis",
        "current_eur_per_mwh",
        "bins",
    )
    assert "regime" not in str(payload)
    assert "seed" not in str(payload)
    assert "stream" not in str(payload)
    assert len(issue.sha256) == 64
    assert issue.sha256 == _episode(13).issues[0].sha256
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_hourly_means_project_exactly_onto_any_whole_hour_subdivision() -> None:
    issue = _episode(53).issues[0]

    quarter_hour = issue.project_mean_prices(interval_s=900.0, horizon_intervals=96)
    twenty_minutes = issue.project_mean_prices(
        interval_s=1_200.0,
        horizon_intervals=7,
    )

    assert len(quarter_hour) == 96
    assert quarter_hour == tuple(
        forecast_bin.mean_eur_per_mwh for forecast_bin in issue.bins for _ in range(4)
    )
    assert twenty_minutes == (
        (issue.bins[0].mean_eur_per_mwh,) * 3
        + (issue.bins[1].mean_eur_per_mwh,) * 3
        + (issue.bins[2].mean_eur_per_mwh,)
    )
    with pytest.raises(ValueError, match="whole subdivision"):
        issue.project_mean_prices(interval_s=1_000.0, horizon_intervals=1)
    with pytest.raises(ValueError, match="24-hour"):
        issue.project_mean_prices(interval_s=900.0, horizon_intervals=97)


def test_changing_strictly_future_truth_cannot_change_an_existing_issue() -> None:
    original = _episode(515)
    issue_decision = 4
    counterfactual_prices = list(original.prices_eur_per_mwh)
    for decision in range(issue_decision + 1, DECISIONS):
        original_price = counterfactual_prices[decision]
        counterfactual_prices[decision] = (
            original_price - 40.0 if original_price > 0.0 else original_price + 40.0
        )
    counterfactual = _episode(515, prices=tuple(counterfactual_prices))

    assert original.truth_sha256 != counterfactual.truth_sha256
    assert (
        original.issue_at_decision(issue_decision).to_dict()
        == counterfactual.issue_at_decision(issue_decision).to_dict()
    )
    assert (
        original.issue_at_decision(issue_decision + 1).to_dict()
        != counterfactual.issue_at_decision(issue_decision + 1).to_dict()
    )
    assert original.episode_sha256 != counterfactual.episode_sha256


def test_canonical_hashes_change_with_seed_config_baseline_and_truth() -> None:
    base = _episode(901)
    changed_seed = _episode(902)
    changed_config = _episode(
        901,
        config=MarketConfigV1(
            regime=MarketRegimeV1.VOLATILE,
            forecast_ensemble_members=16,
        ),
    )
    changed_prices = list(base.prices_eur_per_mwh)
    changed_prices[-1] += -1.0 if changed_prices[-1] > 299.0 else 1.0
    external = _episode(901, prices=tuple(changed_prices))
    hourly = list(DEFAULT_HOURLY_TARIFF_EUR_PER_MWH)
    hourly[0] += 1.0
    changed_baseline = _episode(
        901,
        baseline=DayAheadTariffV1(
            id="changed-baseline-v1",
            hourly_eur_per_mwh=tuple(hourly),
        ),
    )

    assert (
        len(
            {
                base.episode_sha256,
                changed_seed.episode_sha256,
                changed_config.episode_sha256,
                external.episode_sha256,
                changed_baseline.episode_sha256,
            }
        )
        == 5
    )
    assert base.baseline_sha256 != changed_baseline.baseline_sha256
    assert base.market_id != changed_baseline.market_id


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"decision_count": 0}, "positive integer"),
        ({"decision_count": True}, "positive integer"),
        ({"decision_interval_s": 0.0}, "positive"),
        ({"decision_interval_s": 1_000.0}, "whole subdivision"),
        ({"start_hour": -0.1}, r"\[0, 24\)"),
        ({"start_hour": 24.0}, r"\[0, 24\)"),
        ({"seed": -1}, "non-negative"),
    ],
)
def test_generation_arguments_are_strictly_validated(
    changes: dict[str, object],
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "seed": 3,
        "decision_count": DECISIONS,
        "decision_interval_s": DECISION_INTERVAL_S,
        "start_hour": START_HOUR,
    }
    kwargs.update(changes)
    with pytest.raises(ValueError, match=message):
        generate_market_episode(MarketConfigV1(), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("prices", "message"),
    [
        ((0.0,) * (DECISIONS - 1), "decision_count"),
        (
            (0.0,) * (DECISIONS - 1) + (MAX_ABS_TARIFF_EUR_PER_MWH + 0.1,),
            "at most",
        ),
        ((0.0,) * (DECISIONS - 1) + (float("nan"),), "finite"),
    ],
)
def test_external_truth_override_is_validated(
    prices: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _episode(9, prices=prices)


def test_private_manifest_binds_version_profiles_replay_and_hashes() -> None:
    episode = _episode(61)
    manifest = episode.private_audit_manifest()

    assert MARKET_GENERATOR_VERSION == "synthetic-ar1-signed-jumps-v1"
    assert manifest["market_generator_version"] == MARKET_GENERATOR_VERSION
    assert set(manifest["config"]["regime_parameters"]) == {  # type: ignore[index]
        "calm",
        "volatile",
        "scarcity",
    }
    assert manifest["visibility"]["public_vector_size"] == 73  # type: ignore[index]
    assert manifest["replay"]["truth_tape_embedded"] is False  # type: ignore[index]
    assert "prices_eur_per_mwh" not in str(manifest)
    assert (
        manifest["private_hashes"]["episode_sha256"]  # type: ignore[index]
        == episode.episode_sha256
    )
    json.dumps(manifest, allow_nan=False, sort_keys=True)


def test_realized_residual_is_serially_correlated_on_a_flat_baseline() -> None:
    flat = DayAheadTariffV1(
        id="flat-zero-v1",
        hourly_eur_per_mwh=(0.0,) * 24,
    )
    episode = generate_market_episode(
        MarketConfigV1(
            regime=MarketRegimeV1.CALM,
            forecast_ensemble_members=10,
        ),
        seed=1_729,
        decision_count=2_000,
        decision_interval_s=900.0,
        start_hour=0.0,
        baseline=flat,
    )
    prices = np.asarray(episode.prices_eur_per_mwh)
    correlation = float(np.corrcoef(prices[:-1], prices[1:])[0, 1])

    assert 0.80 < correlation < 1.0
    assert float(np.var(prices)) > 1.0
    assert np.all(np.abs(prices) <= MAX_ABS_TARIFF_EUR_PER_MWH)


def test_replacing_a_frozen_identity_field_is_rejected_by_hash_validation() -> None:
    episode = _episode(71)

    with pytest.raises(ValueError, match="episode_sha256"):
        replace(episode, episode_sha256="0" * 64)
