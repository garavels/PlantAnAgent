"""Causal synthetic electricity-market truth and public forecasts for V2 beta.

The deterministic :class:`~plant_backend.rl.v2.economics.DayAheadTariffV1`
remains the civil-time baseline.  This module adds a bounded stochastic residual
at controller cadence and issues a new 24-hour ensemble forecast at every
decision.  An issue uses the current observed price and its own random stream;
it never reads a later value from the realized tape.

This is an engineering stress fixture, not historical market data or a
calibrated model for a real bidding zone.  The immutable private episode owns
truth and replay evidence.  Controllers receive only
:class:`PublicPriceForecastIssueV1`.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt

from .economics import DayAheadTariffV1, MAX_ABS_TARIFF_EUR_PER_MWH


MARKET_GENERATOR_VERSION = "synthetic-ar1-signed-jumps-v1"
PUBLIC_PRICE_FORECAST_SCHEMA = "public-price-forecast-issue-v1"
FORECAST_HORIZON_H = 24
FORECAST_BIN_H = 1
FORECAST_QUANTILES = (0.10, 0.90)
NORMALIZED_MARKET_ISSUE_SIZE = 1 + FORECAST_HORIZON_H * 3
SECONDS_PER_HOUR = 3_600.0

ROOT_STREAM_NAME = "market.root"
REGIME_STREAM_NAME = "market.regime"
TRUTH_STREAM_NAME = "market.truth"
FORECAST_STREAM_NAME = "market.forecast"

SeedSource: TypeAlias = int | np.random.SeedSequence


def _json_ready(value: object) -> object:
    """Detach NumPy values into the strict JSON domain used for evidence."""
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _canonical_json(payload: object) -> str:
    return json.dumps(
        _json_ready(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc
    return value


def _finite(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum:g}")
    return number


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _decision_grid(decision_interval_s: object) -> tuple[float, int]:
    interval_s = _finite("decision_interval_s", decision_interval_s, positive=True)
    subdivisions = int(round(SECONDS_PER_HOUR / interval_s))
    if subdivisions <= 0 or not math.isclose(
        subdivisions * interval_s,
        SECONDS_PER_HOUR,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("decision_interval_s must be a whole subdivision of one hour")
    return interval_s, subdivisions


def _checked_start_hour(value: object) -> float:
    start_hour = _finite("start_hour", value)
    if not 0.0 <= start_hour < 24.0:
        raise ValueError("start_hour must lie within [0, 24)")
    return start_hour


def _checked_price(name: str, value: object) -> float:
    return _finite(
        name,
        value,
        minimum=-MAX_ABS_TARIFF_EUR_PER_MWH,
        maximum=MAX_ABS_TARIFF_EUR_PER_MWH,
    )


class MarketRegimeV1(str, Enum):
    """Requested or privately selected synthetic market regime."""

    CALM = "calm"
    VOLATILE = "volatile"
    SCARCITY = "scarcity"
    MIXED = "mixed"


@dataclass(frozen=True)
class _RegimeParametersV1:
    deviation_mean_eur_per_mwh: float
    deviation_stationary_sd_eur_per_mwh: float
    ar1_half_life_h: float
    jump_arrival_rate_per_h: float
    jump_mean_magnitude_eur_per_mwh: float
    jump_decay_half_life_h: float
    positive_jump_probability: float

    def __post_init__(self) -> None:
        _finite("deviation_mean_eur_per_mwh", self.deviation_mean_eur_per_mwh)
        _finite(
            "deviation_stationary_sd_eur_per_mwh",
            self.deviation_stationary_sd_eur_per_mwh,
            positive=True,
        )
        _finite("ar1_half_life_h", self.ar1_half_life_h, positive=True)
        _finite(
            "jump_arrival_rate_per_h",
            self.jump_arrival_rate_per_h,
            positive=True,
        )
        _finite(
            "jump_mean_magnitude_eur_per_mwh",
            self.jump_mean_magnitude_eur_per_mwh,
            positive=True,
        )
        _finite(
            "jump_decay_half_life_h",
            self.jump_decay_half_life_h,
            positive=True,
        )
        probability = _finite(
            "positive_jump_probability",
            self.positive_jump_probability,
        )
        if not 0.0 < probability < 1.0:
            raise ValueError(
                "positive_jump_probability must lie strictly within (0, 1)"
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "deviation_mean_eur_per_mwh": float(self.deviation_mean_eur_per_mwh),
            "deviation_stationary_sd_eur_per_mwh": float(
                self.deviation_stationary_sd_eur_per_mwh
            ),
            "ar1_half_life_h": float(self.ar1_half_life_h),
            "jump_arrival_rate_per_h": float(self.jump_arrival_rate_per_h),
            "jump_mean_magnitude_eur_per_mwh": float(
                self.jump_mean_magnitude_eur_per_mwh
            ),
            "jump_decay_half_life_h": float(self.jump_decay_half_life_h),
            "positive_jump_probability": float(self.positive_jump_probability),
        }


# These immutable profiles are benchmark assumptions.  Changing a profile
# requires a new generator version because the full table enters episode hashes.
_REGIME_PARAMETERS: Mapping[MarketRegimeV1, _RegimeParametersV1] = MappingProxyType(
    {
        MarketRegimeV1.CALM: _RegimeParametersV1(
            deviation_mean_eur_per_mwh=0.0,
            deviation_stationary_sd_eur_per_mwh=8.0,
            ar1_half_life_h=4.0,
            jump_arrival_rate_per_h=0.015,
            jump_mean_magnitude_eur_per_mwh=40.0,
            jump_decay_half_life_h=0.75,
            positive_jump_probability=0.50,
        ),
        MarketRegimeV1.VOLATILE: _RegimeParametersV1(
            deviation_mean_eur_per_mwh=0.0,
            deviation_stationary_sd_eur_per_mwh=28.0,
            ar1_half_life_h=2.0,
            jump_arrival_rate_per_h=0.040,
            jump_mean_magnitude_eur_per_mwh=85.0,
            jump_decay_half_life_h=1.50,
            positive_jump_probability=0.50,
        ),
        MarketRegimeV1.SCARCITY: _RegimeParametersV1(
            deviation_mean_eur_per_mwh=35.0,
            deviation_stationary_sd_eur_per_mwh=20.0,
            ar1_half_life_h=3.0,
            jump_arrival_rate_per_h=0.060,
            jump_mean_magnitude_eur_per_mwh=135.0,
            jump_decay_half_life_h=3.0,
            positive_jump_probability=0.85,
        ),
    }
)
_CONCRETE_REGIMES = (
    MarketRegimeV1.CALM,
    MarketRegimeV1.VOLATILE,
    MarketRegimeV1.SCARCITY,
)


@dataclass(frozen=True)
class MarketConfigV1:
    """Version-bound market and forecast settings.

    ``mixed`` samples exactly one of the three concrete profiles from the
    private regime stream.  The public issue never contains that selection.
    """

    regime: MarketRegimeV1 = MarketRegimeV1.MIXED
    forecast_ensemble_members: int = 64
    forecast_innovation_scale: float = 1.0

    def validate(self) -> "MarketConfigV1":
        if not isinstance(self.regime, MarketRegimeV1):
            raise ValueError("regime must be a MarketRegimeV1 value")
        members = self.forecast_ensemble_members
        if (
            isinstance(members, bool)
            or not isinstance(members, Integral)
            or members < 10
        ):
            raise ValueError(
                "forecast_ensemble_members must be an integer of at least 10"
            )
        _finite(
            "forecast_innovation_scale",
            self.forecast_innovation_scale,
            positive=True,
        )
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "regime": self.regime.value,
            "forecast_ensemble_members": int(self.forecast_ensemble_members),
            "forecast_innovation_scale": float(self.forecast_innovation_scale),
            "forecast_horizon_h": FORECAST_HORIZON_H,
            "forecast_bin_h": FORECAST_BIN_H,
            "forecast_quantiles": list(FORECAST_QUANTILES),
            "normalized_public_field_order": (
                "current, hour_00(mean,p10,p90), ..., hour_23(mean,p10,p90)"
            ),
            "regime_parameters": {
                regime.value: _REGIME_PARAMETERS[regime].to_dict()
                for regime in _CONCRETE_REGIMES
            },
        }


@dataclass(frozen=True)
class PriceForecastBinV1:
    """One controller-visible one-hour ensemble summary."""

    hour_index: int
    mean_eur_per_mwh: float
    p10_eur_per_mwh: float
    p90_eur_per_mwh: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.hour_index, bool)
            or not isinstance(self.hour_index, Integral)
            or not 0 <= self.hour_index < FORECAST_HORIZON_H
        ):
            raise ValueError("forecast hour_index must be an integer within [0, 24)")
        mean = _checked_price("forecast mean_eur_per_mwh", self.mean_eur_per_mwh)
        p10 = _checked_price("forecast p10_eur_per_mwh", self.p10_eur_per_mwh)
        p90 = _checked_price("forecast p90_eur_per_mwh", self.p90_eur_per_mwh)
        if not p10 <= mean <= p90:
            raise ValueError("forecast prices must satisfy p10 <= mean <= p90")

    @property
    def start_offset_h(self) -> float:
        return float(self.hour_index)

    @property
    def end_offset_h(self) -> float:
        return float(self.hour_index + 1)

    @property
    def start_offset_s_from_issue(self) -> float:
        return self.start_offset_h * SECONDS_PER_HOUR

    @property
    def end_offset_s_from_issue(self) -> float:
        return self.end_offset_h * SECONDS_PER_HOUR

    def to_dict(self) -> dict[str, object]:
        return {
            "hour_index": int(self.hour_index),
            "start_offset_s_from_issue": self.start_offset_s_from_issue,
            "end_offset_s_from_issue": self.end_offset_s_from_issue,
            "offset_interval": "half-open [start, end)",
            "mean_eur_per_mwh": float(self.mean_eur_per_mwh),
            "p10_eur_per_mwh": float(self.p10_eur_per_mwh),
            "p90_eur_per_mwh": float(self.p90_eur_per_mwh),
        }


@dataclass(frozen=True)
class PublicPriceForecastIssueV1:
    """The complete market information available at one decision boundary."""

    issued_at_decision: int
    issued_at_offset_s: float
    current_eur_per_mwh: float
    bins: tuple[PriceForecastBinV1, ...]
    market_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.issued_at_decision, bool)
            or not isinstance(self.issued_at_decision, Integral)
            or self.issued_at_decision < 0
        ):
            raise ValueError("issued_at_decision must be a non-negative integer")
        _finite("issued_at_offset_s", self.issued_at_offset_s, minimum=0.0)
        _checked_price("current_eur_per_mwh", self.current_eur_per_mwh)
        if not isinstance(self.bins, tuple):
            raise TypeError("forecast bins must be an immutable tuple")
        if len(self.bins) != FORECAST_HORIZON_H:
            raise ValueError("forecast issue must contain exactly 24 hourly bins")
        if any(not isinstance(item, PriceForecastBinV1) for item in self.bins):
            raise TypeError("forecast bins must contain PriceForecastBinV1 values")
        if tuple(item.hour_index for item in self.bins) != tuple(
            range(FORECAST_HORIZON_H)
        ):
            raise ValueError("forecast bins must be ordered hour indices 0 through 23")
        if not isinstance(self.market_id, str) or not self.market_id.strip():
            raise ValueError("market_id must be a non-empty string")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def normalized(self) -> tuple[float, ...]:
        """Return the fixed 73-value controller vector in protocol order."""
        scale = MAX_ABS_TARIFF_EUR_PER_MWH
        values = [float(self.current_eur_per_mwh) / scale]
        for forecast_bin in self.bins:
            values.extend(
                (
                    float(forecast_bin.mean_eur_per_mwh) / scale,
                    float(forecast_bin.p10_eur_per_mwh) / scale,
                    float(forecast_bin.p90_eur_per_mwh) / scale,
                )
            )
        if len(values) != NORMALIZED_MARKET_ISSUE_SIZE:  # pragma: no cover
            raise RuntimeError("public market vector has an invalid size")
        return tuple(values)

    def project_mean_prices(
        self,
        *,
        interval_s: float,
        horizon_intervals: int,
    ) -> tuple[float, ...]:
        """Project hourly means onto a whole subdivision of one hour.

        Each public hourly mean is piecewise constant.  The requested horizon
        may end within an hour but cannot exceed this issue's 24-hour horizon.
        """
        _, subdivisions = _decision_grid(interval_s)
        intervals = _positive_integer("horizon_intervals", horizon_intervals)
        maximum = FORECAST_HORIZON_H * subdivisions
        if intervals > maximum:
            raise ValueError(
                "horizon_intervals must not project beyond the 24-hour forecast"
            )
        return tuple(
            float(self.bins[index // subdivisions].mean_eur_per_mwh)
            for index in range(intervals)
        )

    def to_dict(self) -> dict[str, object]:
        """Return only controller-safe fields; no stream or regime metadata."""
        return {
            "schema": PUBLIC_PRICE_FORECAST_SCHEMA,
            "market_id": self.market_id,
            "issued_at_decision": int(self.issued_at_decision),
            "issued_at_offset_s": float(self.issued_at_offset_s),
            "issue_time_basis": "seconds since episode start",
            "current_eur_per_mwh": float(self.current_eur_per_mwh),
            "bins": [forecast_bin.to_dict() for forecast_bin in self.bins],
        }


@dataclass(frozen=True)
class _RandomStreamKeyV1:
    name: str
    source_kind: str
    spawn_key: tuple[int, ...]
    state_sha256: str
    initial_state_json: str

    def __post_init__(self) -> None:
        if not self.name or not self.source_kind:
            raise ValueError("random stream name and source kind must not be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in self.spawn_key
        ):
            raise ValueError(
                "random stream spawn key must contain non-negative integers"
            )
        try:
            state = json.loads(self.initial_state_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "random stream initial state must be canonical JSON"
            ) from exc
        if _canonical_json(state) != self.initial_state_json:
            raise ValueError(
                "random stream initial state must use canonical JSON encoding"
            )
        _require_sha256("random stream state_sha256", self.state_sha256)
        if _canonical_sha256(state) != self.state_sha256:
            raise ValueError("random stream digest does not match its state")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_kind": self.source_kind,
            "spawn_key": list(self.spawn_key),
            "state_sha256": self.state_sha256,
            "initial_state": json.loads(self.initial_state_json),
        }


def _stream_identity(
    sequence: np.random.SeedSequence,
    name: str,
    *,
    source_kind: str = "numpy.random.SeedSequence",
) -> _RandomStreamKeyV1:
    state = _json_ready(deepcopy(sequence.state))
    return _RandomStreamKeyV1(
        name=name,
        source_kind=source_kind,
        spawn_key=tuple(int(value) for value in sequence.spawn_key),
        state_sha256=_canonical_sha256(state),
        initial_state_json=_canonical_json(state),
    )


def _clone_seed_sequence(source: np.random.SeedSequence) -> np.random.SeedSequence:
    return np.random.SeedSequence(**deepcopy(source.state))


def _root_sequence(seed: SeedSource) -> tuple[np.random.SeedSequence, str]:
    if isinstance(seed, bool):
        raise ValueError("seed must be a non-negative integer or numpy SeedSequence")
    if isinstance(seed, Integral):
        if seed < 0:
            raise ValueError("seed must be non-negative")
        return np.random.SeedSequence(int(seed)), "integer_seed"
    if isinstance(seed, np.random.SeedSequence):
        return _clone_seed_sequence(seed), "numpy.random.SeedSequence"
    raise TypeError("seed must be an integer or numpy.random.SeedSequence")


def _market_contract_id(
    config: MarketConfigV1,
    baseline: DayAheadTariffV1,
    *,
    decision_interval_s: float,
    start_hour: float,
) -> str:
    # Deliberately excludes seed, selected mixed regime, realized truth, and
    # episode length.  The public identifier cannot become a private-label leak.
    return _canonical_sha256(
        {
            "market_generator_version": MARKET_GENERATOR_VERSION,
            "config": config.to_dict(),
            "baseline": baseline.to_dict(),
            "decision_interval_s": decision_interval_s,
            "start_hour": start_hour,
        }
    )


def _truth_hash(
    prices: tuple[float, ...],
    *,
    decision_interval_s: float,
    start_hour: float,
) -> str:
    return _canonical_sha256(
        {
            "semantics": (
                "price at decision k applies to controller interval [k, k+1)"
            ),
            "unit": "EUR/MWh",
            "decision_interval_s": decision_interval_s,
            "start_hour": start_hour,
            "prices_eur_per_mwh": prices,
        }
    )


def _episode_identity_payload(
    *,
    config: MarketConfigV1,
    selected_regime: MarketRegimeV1,
    decision_count: int,
    decision_interval_s: float,
    start_hour: float,
    baseline_sha256: str,
    market_id: str,
    truth_origin_kind: str,
    root_stream_key: _RandomStreamKeyV1,
    regime_stream_key: _RandomStreamKeyV1,
    truth_stream_key: _RandomStreamKeyV1,
    forecast_stream_key: _RandomStreamKeyV1,
    issue_stream_sha256: tuple[str, ...],
    truth_sha256: str,
    forecasts_sha256: str,
) -> dict[str, object]:
    return {
        "market_generator_version": MARKET_GENERATOR_VERSION,
        "config": config.to_dict(),
        "selected_regime": selected_regime.value,
        "selected_regime_parameters": _REGIME_PARAMETERS[selected_regime].to_dict(),
        "decision_count": decision_count,
        "decision_interval_s": decision_interval_s,
        "start_hour": start_hour,
        "baseline_sha256": baseline_sha256,
        "market_id": market_id,
        "truth_origin_kind": truth_origin_kind,
        "random_streams": {
            ROOT_STREAM_NAME: root_stream_key.to_dict(),
            REGIME_STREAM_NAME: regime_stream_key.to_dict(),
            TRUTH_STREAM_NAME: truth_stream_key.to_dict(),
            FORECAST_STREAM_NAME: forecast_stream_key.to_dict(),
        },
        "issue_stream_sha256": issue_stream_sha256,
        "truth_sha256": truth_sha256,
        "forecasts_sha256": forecasts_sha256,
    }


@dataclass(frozen=True)
class MarketEpisodeV1:
    """Immutable private market truth, public issues, and replay evidence."""

    config: MarketConfigV1
    selected_regime: MarketRegimeV1
    prices_eur_per_mwh: tuple[float, ...]
    issues: tuple[PublicPriceForecastIssueV1, ...]
    decision_count: int
    decision_interval_s: float
    start_hour: float
    baseline: DayAheadTariffV1
    market_id: str
    truth_origin_kind: str
    baseline_sha256: str
    truth_sha256: str
    forecasts_sha256: str
    episode_sha256: str
    _root_stream_key: _RandomStreamKeyV1
    _regime_stream_key: _RandomStreamKeyV1
    _truth_stream_key: _RandomStreamKeyV1
    _forecast_stream_key: _RandomStreamKeyV1
    _issue_stream_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        self.config.validate()
        if self.selected_regime not in _CONCRETE_REGIMES:
            raise ValueError("selected_regime must be calm, volatile, or scarcity")
        count = _positive_integer("decision_count", self.decision_count)
        interval_s, _ = _decision_grid(self.decision_interval_s)
        start_hour = _checked_start_hour(self.start_hour)
        if not isinstance(self.baseline, DayAheadTariffV1):
            raise TypeError("baseline must be DayAheadTariffV1")
        self.baseline.validate()
        if not isinstance(self.prices_eur_per_mwh, tuple):
            raise TypeError("market price tape must be an immutable tuple")
        if len(self.prices_eur_per_mwh) != count:
            raise ValueError("market price tape must match decision_count")
        for index, price in enumerate(self.prices_eur_per_mwh):
            _checked_price(f"prices_eur_per_mwh[{index}]", price)
        if not isinstance(self.issues, tuple):
            raise TypeError("market issues must be an immutable tuple")
        if len(self.issues) != count:
            raise ValueError("market must issue one forecast at every decision")
        for index, issue in enumerate(self.issues):
            if not isinstance(issue, PublicPriceForecastIssueV1):
                raise TypeError("issues must contain PublicPriceForecastIssueV1 values")
            if issue.issued_at_decision != index:
                raise ValueError("market issues must be ordered at every decision")
            if not math.isclose(
                issue.issued_at_offset_s,
                index * interval_s,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("market issue time does not match decision cadence")
            if issue.current_eur_per_mwh != self.prices_eur_per_mwh[index]:
                raise ValueError("market issue current price does not match truth tape")
            if issue.market_id != self.market_id:
                raise ValueError("market issue has the wrong market_id")
        if self.truth_origin_kind not in {"synthetic_generator", "external_override"}:
            raise ValueError("truth_origin_kind is invalid")
        expected_stream_names = (
            (self._root_stream_key, ROOT_STREAM_NAME),
            (self._regime_stream_key, REGIME_STREAM_NAME),
            (self._truth_stream_key, TRUTH_STREAM_NAME),
            (self._forecast_stream_key, FORECAST_STREAM_NAME),
        )
        if any(key.name != name for key, name in expected_stream_names):
            raise ValueError("market random stream has the wrong name")
        if len(self._issue_stream_sha256) != count:
            raise ValueError(
                "market must retain one private issue-stream digest per issue"
            )
        for digest in self._issue_stream_sha256:
            _require_sha256("issue stream identity", digest)
        if len(set(self._issue_stream_sha256)) != count:
            raise ValueError("issue-specific forecast streams must be independent")
        for name, digest in (
            ("baseline_sha256", self.baseline_sha256),
            ("truth_sha256", self.truth_sha256),
            ("forecasts_sha256", self.forecasts_sha256),
            ("episode_sha256", self.episode_sha256),
        ):
            _require_sha256(name, digest)
        expected_baseline_sha256 = _canonical_sha256(self.baseline.to_dict())
        if self.baseline_sha256 != expected_baseline_sha256:
            raise ValueError("baseline_sha256 does not match the baseline")
        expected_market_id = _market_contract_id(
            self.config,
            self.baseline,
            decision_interval_s=interval_s,
            start_hour=start_hour,
        )
        if self.market_id != expected_market_id:
            raise ValueError("market_id does not match the public market contract")
        expected_truth_sha256 = _truth_hash(
            self.prices_eur_per_mwh,
            decision_interval_s=interval_s,
            start_hour=start_hour,
        )
        if self.truth_sha256 != expected_truth_sha256:
            raise ValueError("truth_sha256 does not match the realized price tape")
        expected_forecasts_sha256 = _canonical_sha256(
            [issue.to_dict() for issue in self.issues]
        )
        if self.forecasts_sha256 != expected_forecasts_sha256:
            raise ValueError("forecasts_sha256 does not match the public issues")
        identity = _episode_identity_payload(
            config=self.config,
            selected_regime=self.selected_regime,
            decision_count=count,
            decision_interval_s=interval_s,
            start_hour=start_hour,
            baseline_sha256=self.baseline_sha256,
            market_id=self.market_id,
            truth_origin_kind=self.truth_origin_kind,
            root_stream_key=self._root_stream_key,
            regime_stream_key=self._regime_stream_key,
            truth_stream_key=self._truth_stream_key,
            forecast_stream_key=self._forecast_stream_key,
            issue_stream_sha256=self._issue_stream_sha256,
            truth_sha256=self.truth_sha256,
            forecasts_sha256=self.forecasts_sha256,
        )
        if self.episode_sha256 != _canonical_sha256(identity):
            raise ValueError(
                "episode_sha256 does not match the private market identity"
            )

    @property
    def regime(self) -> MarketRegimeV1:
        """Private selected regime (never include this in an observation)."""
        return self.selected_regime

    def price_at_decision(self, index: int) -> float:
        checked = self._checked_decision(index)
        return float(self.prices_eur_per_mwh[checked])

    def issue_at_decision(self, index: int) -> PublicPriceForecastIssueV1:
        checked = self._checked_decision(index)
        return self.issues[checked]

    def _checked_decision(self, index: object) -> int:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ValueError(
                f"decision index must be an integer within [0, {self.decision_count})"
            )
        checked = int(index)
        if not 0 <= checked < self.decision_count:
            raise ValueError(
                f"decision index must be an integer within [0, {self.decision_count})"
            )
        return checked

    def private_audit_manifest(self) -> dict[str, object]:
        """Return offline provenance without materialising the private truth tape."""
        return {
            "market_generator_version": MARKET_GENERATOR_VERSION,
            "config": self.config.to_dict(),
            "selected_regime": self.selected_regime.value,
            "selected_regime_parameters": _REGIME_PARAMETERS[
                self.selected_regime
            ].to_dict(),
            "market_id": self.market_id,
            "time_convention": {
                "decision_count": self.decision_count,
                "decision_interval_s": self.decision_interval_s,
                "start_hour": self.start_hour,
                "truth_interval": (
                    "price at decision k applies to controller interval [k, k+1)"
                ),
                "forecast_issue_cadence_s": self.decision_interval_s,
                "forecast_bins": "24 consecutive one-hour half-open bins",
            },
            "baseline": {
                **self.baseline.to_dict(),
                "sha256": self.baseline_sha256,
            },
            "random_streams": {
                ROOT_STREAM_NAME: self._root_stream_key.to_dict(),
                REGIME_STREAM_NAME: self._regime_stream_key.to_dict(),
                TRUTH_STREAM_NAME: self._truth_stream_key.to_dict(),
                FORECAST_STREAM_NAME: self._forecast_stream_key.to_dict(),
            },
            "visibility": {
                "public_issue_schema": PUBLIC_PRICE_FORECAST_SCHEMA,
                "public_vector_size": NORMALIZED_MARKET_ISSUE_SIZE,
                "selected_regime_is_private": True,
                "future_truth_is_private": True,
                "issue_random_streams_are_private": True,
            },
            "replay": {
                "truth_origin_kind": self.truth_origin_kind,
                "random_initial_states_embedded": True,
                "truth_tape_embedded": False,
                "requirements": (
                    [
                        "matching source revision and dependency lock",
                        "regenerate truth and issues from the embedded RNG initial states",
                        "retain the declared deterministic baseline",
                    ]
                    if self.truth_origin_kind == "synthetic_generator"
                    else [
                        "matching source revision and dependency lock",
                        "retain the external realized price tape matching truth_sha256",
                        "regenerate issues from the embedded forecast RNG initial state",
                        "retain the declared deterministic baseline",
                    ]
                ),
            },
            "private_hashes": {
                "baseline_sha256": self.baseline_sha256,
                "truth_sha256": self.truth_sha256,
                "forecasts_sha256": self.forecasts_sha256,
                "forecast_issue_stream_sha256": list(self._issue_stream_sha256),
                "episode_sha256": self.episode_sha256,
            },
        }


def _process_coefficients(
    parameters: _RegimeParametersV1,
    interval_h: float,
) -> tuple[float, float, float, float]:
    phi = 2.0 ** (-interval_h / parameters.ar1_half_life_h)
    innovation_sd = parameters.deviation_stationary_sd_eur_per_mwh * math.sqrt(
        max(0.0, 1.0 - phi * phi)
    )
    jump_decay = 2.0 ** (-interval_h / parameters.jump_decay_half_life_h)
    jump_probability = -math.expm1(-parameters.jump_arrival_rate_per_h * interval_h)
    return phi, innovation_sd, jump_decay, jump_probability


def _signed_jump(
    rng: np.random.Generator,
    parameters: _RegimeParametersV1,
    *,
    size: int | None = None,
) -> npt.NDArray[np.float64]:
    magnitudes = np.asarray(
        rng.exponential(parameters.jump_mean_magnitude_eur_per_mwh, size=size),
        dtype=np.float64,
    )
    positive = np.asarray(
        rng.random(size=size) < parameters.positive_jump_probability,
        dtype=np.bool_,
    )
    return np.asarray(np.where(positive, magnitudes, -magnitudes), dtype=np.float64)


def _generate_truth_prices(
    *,
    parameters: _RegimeParametersV1,
    decision_count: int,
    decision_interval_s: float,
    start_hour: float,
    baseline: DayAheadTariffV1,
    rng: np.random.Generator,
) -> tuple[float, ...]:
    interval_h = decision_interval_s / SECONDS_PER_HOUR
    phi, innovation_sd, jump_decay, jump_probability = _process_coefficients(
        parameters,
        interval_h,
    )
    deviation = parameters.deviation_mean_eur_per_mwh + float(
        rng.normal(0.0, parameters.deviation_stationary_sd_eur_per_mwh)
    )
    jump_state = 0.0
    prices: list[float] = []
    for decision in range(decision_count):
        if decision:
            deviation = (
                parameters.deviation_mean_eur_per_mwh
                + phi * (deviation - parameters.deviation_mean_eur_per_mwh)
                + innovation_sd * float(rng.normal())
            )
            jump_state *= jump_decay
            if float(rng.random()) < jump_probability:
                jump_state += float(_signed_jump(rng, parameters).item())
        civil_hour = start_hour + decision * interval_h
        price = baseline.price_at(civil_hour) + deviation + jump_state
        prices.append(
            float(
                np.clip(
                    price,
                    -MAX_ABS_TARIFF_EUR_PER_MWH,
                    MAX_ABS_TARIFF_EUR_PER_MWH,
                )
            )
        )
    return tuple(prices)


def _forecast_paths(
    *,
    config: MarketConfigV1,
    parameters: _RegimeParametersV1,
    current_eur_per_mwh: float,
    issued_at_decision: int,
    decision_interval_s: float,
    start_hour: float,
    baseline: DayAheadTariffV1,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Draw paths from current truth without consulting the later truth tape."""
    members = int(config.forecast_ensemble_members)
    _, subdivisions = _decision_grid(decision_interval_s)
    horizon_steps = FORECAST_HORIZON_H * subdivisions
    interval_h = decision_interval_s / SECONDS_PER_HOUR
    phi, innovation_sd, jump_decay, jump_probability = _process_coefficients(
        parameters,
        interval_h,
    )
    innovation_sd *= config.forecast_innovation_scale
    issue_hour = start_hour + issued_at_decision * interval_h
    current_baseline = baseline.price_at(issue_hour)
    # The observed residual is the Markov analysis state.  No decomposition of
    # the private truth's AR and jump states, and no later truth, is available.
    deviation = np.full(
        members,
        current_eur_per_mwh - current_baseline,
        dtype=np.float64,
    )
    jump_state = np.zeros(members, dtype=np.float64)
    paths = np.empty((members, horizon_steps), dtype=np.float64)
    paths[:, 0] = current_eur_per_mwh

    for step in range(1, horizon_steps):
        deviation = (
            parameters.deviation_mean_eur_per_mwh
            + phi * (deviation - parameters.deviation_mean_eur_per_mwh)
            + innovation_sd * rng.normal(size=members)
        )
        jump_state *= jump_decay
        arrivals = rng.random(size=members) < jump_probability
        arrival_count = int(np.count_nonzero(arrivals))
        if arrival_count:
            jump_state[arrivals] += np.asarray(
                _signed_jump(rng, parameters, size=arrival_count),
                dtype=np.float64,
            )
        civil_hour = issue_hour + step * interval_h
        paths[:, step] = np.clip(
            baseline.price_at(civil_hour) + deviation + jump_state,
            -MAX_ABS_TARIFF_EUR_PER_MWH,
            MAX_ABS_TARIFF_EUR_PER_MWH,
        )
    return paths


def _forecast_issue(
    *,
    config: MarketConfigV1,
    parameters: _RegimeParametersV1,
    current_eur_per_mwh: float,
    issued_at_decision: int,
    decision_interval_s: float,
    start_hour: float,
    baseline: DayAheadTariffV1,
    market_id: str,
    issue_seed: np.random.SeedSequence,
) -> PublicPriceForecastIssueV1:
    paths = _forecast_paths(
        config=config,
        parameters=parameters,
        current_eur_per_mwh=current_eur_per_mwh,
        issued_at_decision=issued_at_decision,
        decision_interval_s=decision_interval_s,
        start_hour=start_hour,
        baseline=baseline,
        rng=np.random.default_rng(issue_seed),
    )
    _, subdivisions = _decision_grid(decision_interval_s)
    bins: list[PriceForecastBinV1] = []
    for hour_index in range(FORECAST_HORIZON_H):
        start = hour_index * subdivisions
        end = start + subdivisions
        member_means = paths[:, start:end].mean(axis=1)
        mean = float(np.mean(member_means))
        p10 = min(mean, float(np.quantile(member_means, FORECAST_QUANTILES[0])))
        p90 = max(mean, float(np.quantile(member_means, FORECAST_QUANTILES[1])))
        bins.append(
            PriceForecastBinV1(
                hour_index=hour_index,
                mean_eur_per_mwh=float(
                    np.clip(
                        mean,
                        -MAX_ABS_TARIFF_EUR_PER_MWH,
                        MAX_ABS_TARIFF_EUR_PER_MWH,
                    )
                ),
                p10_eur_per_mwh=float(
                    np.clip(
                        p10,
                        -MAX_ABS_TARIFF_EUR_PER_MWH,
                        MAX_ABS_TARIFF_EUR_PER_MWH,
                    )
                ),
                p90_eur_per_mwh=float(
                    np.clip(
                        p90,
                        -MAX_ABS_TARIFF_EUR_PER_MWH,
                        MAX_ABS_TARIFF_EUR_PER_MWH,
                    )
                ),
            )
        )
    return PublicPriceForecastIssueV1(
        issued_at_decision=issued_at_decision,
        issued_at_offset_s=issued_at_decision * decision_interval_s,
        current_eur_per_mwh=current_eur_per_mwh,
        bins=tuple(bins),
        market_id=market_id,
    )


def _normalize_external_prices(
    prices: Sequence[float],
    *,
    decision_count: int,
) -> tuple[float, ...]:
    if isinstance(prices, (str, bytes)):
        raise TypeError("realized_prices_eur_per_mwh must be a numeric sequence")
    try:
        length = len(prices)
    except TypeError as exc:
        raise TypeError(
            "realized_prices_eur_per_mwh must be a sized numeric sequence"
        ) from exc
    if length != decision_count:
        raise ValueError("realized price override must contain decision_count values")
    return tuple(
        _checked_price(f"realized_prices_eur_per_mwh[{index}]", value)
        for index, value in enumerate(prices)
    )


def generate_market_episode(
    config: MarketConfigV1 = MarketConfigV1(),
    *,
    seed: SeedSource,
    decision_count: int,
    decision_interval_s: float,
    start_hour: float,
    baseline: DayAheadTariffV1 = DayAheadTariffV1(),
    realized_prices_eur_per_mwh: Sequence[float] | None = None,
) -> MarketEpisodeV1:
    """Generate one replayable price tape and all causal public vintages.

    ``realized_prices_eur_per_mwh`` is the deliberate causality-test seam.  It
    replaces the complete truth tape but retains regime selection and all issue
    streams from ``seed``.  External values are validated, never silently
    clipped.
    """
    if not isinstance(config, MarketConfigV1):
        raise TypeError("config must be MarketConfigV1")
    config.validate()
    count = _positive_integer("decision_count", decision_count)
    interval_s, _ = _decision_grid(decision_interval_s)
    checked_start_hour = _checked_start_hour(start_hour)
    if not isinstance(baseline, DayAheadTariffV1):
        raise TypeError("baseline must be DayAheadTariffV1")
    baseline.validate()

    root_sequence, root_source_kind = _root_sequence(seed)
    root_key = _stream_identity(
        root_sequence,
        ROOT_STREAM_NAME,
        source_kind=root_source_kind,
    )
    regime_sequence, truth_sequence, forecast_sequence = root_sequence.spawn(3)
    regime_key = _stream_identity(regime_sequence, REGIME_STREAM_NAME)
    truth_key = _stream_identity(truth_sequence, TRUTH_STREAM_NAME)
    forecast_key = _stream_identity(forecast_sequence, FORECAST_STREAM_NAME)

    if config.regime is MarketRegimeV1.MIXED:
        regime_index = int(
            np.random.default_rng(regime_sequence).integers(len(_CONCRETE_REGIMES))
        )
        selected_regime = _CONCRETE_REGIMES[regime_index]
    else:
        selected_regime = config.regime
    parameters = _REGIME_PARAMETERS[selected_regime]

    if realized_prices_eur_per_mwh is None:
        prices = _generate_truth_prices(
            parameters=parameters,
            decision_count=count,
            decision_interval_s=interval_s,
            start_hour=checked_start_hour,
            baseline=baseline,
            rng=np.random.default_rng(truth_sequence),
        )
        truth_origin_kind = "synthetic_generator"
    else:
        prices = _normalize_external_prices(
            realized_prices_eur_per_mwh,
            decision_count=count,
        )
        truth_origin_kind = "external_override"

    market_id = _market_contract_id(
        config,
        baseline,
        decision_interval_s=interval_s,
        start_hour=checked_start_hour,
    )
    issue_sequences = tuple(_clone_seed_sequence(forecast_sequence).spawn(count))
    issue_stream_sha256 = tuple(
        _stream_identity(
            issue_seed,
            f"{FORECAST_STREAM_NAME}.issue-{index}",
        ).state_sha256
        for index, issue_seed in enumerate(issue_sequences)
    )
    issues = tuple(
        _forecast_issue(
            config=config,
            parameters=parameters,
            current_eur_per_mwh=prices[index],
            issued_at_decision=index,
            decision_interval_s=interval_s,
            start_hour=checked_start_hour,
            baseline=baseline,
            market_id=market_id,
            issue_seed=issue_seed,
        )
        for index, issue_seed in enumerate(issue_sequences)
    )
    baseline_sha256 = _canonical_sha256(baseline.to_dict())
    truth_sha256 = _truth_hash(
        prices,
        decision_interval_s=interval_s,
        start_hour=checked_start_hour,
    )
    forecasts_sha256 = _canonical_sha256([issue.to_dict() for issue in issues])
    identity = _episode_identity_payload(
        config=config,
        selected_regime=selected_regime,
        decision_count=count,
        decision_interval_s=interval_s,
        start_hour=checked_start_hour,
        baseline_sha256=baseline_sha256,
        market_id=market_id,
        truth_origin_kind=truth_origin_kind,
        root_stream_key=root_key,
        regime_stream_key=regime_key,
        truth_stream_key=truth_key,
        forecast_stream_key=forecast_key,
        issue_stream_sha256=issue_stream_sha256,
        truth_sha256=truth_sha256,
        forecasts_sha256=forecasts_sha256,
    )
    return MarketEpisodeV1(
        config=config,
        selected_regime=selected_regime,
        prices_eur_per_mwh=prices,
        issues=issues,
        decision_count=count,
        decision_interval_s=interval_s,
        start_hour=checked_start_hour,
        baseline=baseline,
        market_id=market_id,
        truth_origin_kind=truth_origin_kind,
        baseline_sha256=baseline_sha256,
        truth_sha256=truth_sha256,
        forecasts_sha256=forecasts_sha256,
        episode_sha256=_canonical_sha256(identity),
        _root_stream_key=root_key,
        _regime_stream_key=regime_key,
        _truth_stream_key=truth_key,
        _forecast_stream_key=forecast_key,
        _issue_stream_sha256=issue_stream_sha256,
    )


__all__ = [
    "FORECAST_BIN_H",
    "FORECAST_HORIZON_H",
    "FORECAST_QUANTILES",
    "MARKET_GENERATOR_VERSION",
    "NORMALIZED_MARKET_ISSUE_SIZE",
    "MarketConfigV1",
    "MarketEpisodeV1",
    "MarketRegimeV1",
    "PriceForecastBinV1",
    "PublicPriceForecastIssueV1",
    "generate_market_episode",
]
