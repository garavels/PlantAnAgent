"""Causal synthetic weather truth and controller-visible PV forecasts for V2.

The stochastic process in this module is deliberately an engineering fixture,
not a calibrated weather model.  It separates three quantities which are easy
to conflate:

* ``SolarFactorTape`` stores the bounded resource factor multiplying the
  deterministic clear-sky curve;
* the caller supplies that deterministic curve as a fraction of PV nameplate
  power for every physical interval; and
* issued forecasts report interval-average available PV/nameplate fractions.

All physical intervals are half-open.  A factor at boundary ``k + 1`` applies
to interval ``[k, k + 1)``.  A forecast issued at boundary ``k`` may analyse
the truth at boundary ``k`` and the deterministic future clear-sky envelope,
but it never reads resource factors after ``k``.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np
from plant_sim import SolarFactorTape


WEATHER_GENERATOR_VERSION = "synthetic-logit-ou-v1"
FORECAST_WINDOWS_H: tuple[tuple[float, float], ...] = (
    (0.0, 1.0),
    (1.0, 6.0),
    (6.0, 12.0),
    (12.0, 24.0),
)
TRUTH_STREAM_NAME = "weather.truth"
FORECAST_STREAM_NAME = "weather.forecast"
SECONDS_PER_HOUR = 3_600.0

RandomSource: TypeAlias = np.random.SeedSequence | np.random.Generator


def _json_ready(value: object) -> object:
    """Detach NumPy state into the strict JSON domain used for evidence hashes."""
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
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
    encoded = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc
    return value


def _finite(name: str, value: float, *, positive: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or (positive and value <= 0.0)
    ):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _aligned_steps(seconds: float, dt_s: float, *, name: str) -> int:
    steps = int(round(seconds / dt_s))
    if steps <= 0 or not math.isclose(
        seconds,
        steps * dt_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{name} must align to the {dt_s:g} s physics timestep")
    return steps


def _sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    non_negative = array >= 0.0
    result[non_negative] = 1.0 / (1.0 + np.exp(-array[non_negative]))
    exponential = np.exp(array[~non_negative])
    result[~non_negative] = exponential / (1.0 + exponential)
    return result


def _logit(value: float) -> float:
    # Tape values can legally include exact endpoints even though this generator
    # does not produce them.  Clipping makes externally supplied counterfactual
    # tapes analyzable without infinities.
    bounded = min(max(float(value), 1e-12), 1.0 - 1e-12)
    return math.log(bounded / (1.0 - bounded))


@dataclass(frozen=True)
class WeatherConfigV2:
    """Parameters for a bounded discrete-Ornstein--Uhlenbeck test fixture.

    The latent Gaussian process is transformed through a logistic function.
    These defaults are hypotheses for controller discrimination, not a claim
    about the climate at a real project site.
    """

    latent_median_resource_factor: float = 0.70
    latent_stationary_sd: float = 0.75
    autocorrelation_half_life_h: float = 1.0
    forecast_ensemble_members: int = 32
    forecast_analysis_latent_sd: float = 0.05
    forecast_innovation_scale: float = 1.0
    forecast_windows_h: tuple[tuple[float, float], ...] = FORECAST_WINDOWS_H

    def validate(self) -> "WeatherConfigV2":
        median = self.latent_median_resource_factor
        _finite("latent_median_resource_factor", median)
        if not 0.0 < median < 1.0:
            raise ValueError("latent_median_resource_factor must lie strictly within (0, 1)")
        _finite("latent_stationary_sd", self.latent_stationary_sd, positive=True)
        _finite(
            "autocorrelation_half_life_h",
            self.autocorrelation_half_life_h,
            positive=True,
        )
        members = self.forecast_ensemble_members
        if isinstance(members, bool) or not isinstance(members, Integral) or members < 10:
            raise ValueError("forecast_ensemble_members must be an integer of at least 10")
        _finite("forecast_analysis_latent_sd", self.forecast_analysis_latent_sd)
        if self.forecast_analysis_latent_sd < 0.0:
            raise ValueError("forecast_analysis_latent_sd must be non-negative")
        _finite(
            "forecast_innovation_scale",
            self.forecast_innovation_scale,
            positive=True,
        )
        if self.forecast_windows_h != FORECAST_WINDOWS_H:
            raise ValueError(
                "forecast_windows_h must be the canonical 0-1, 1-6, 6-12, "
                "and 12-24 hour windows"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_median_resource_factor": float(
                self.latent_median_resource_factor
            ),
            "latent_stationary_sd": float(self.latent_stationary_sd),
            "autocorrelation_half_life_h": float(
                self.autocorrelation_half_life_h
            ),
            "forecast_ensemble_members": int(self.forecast_ensemble_members),
            "forecast_analysis_latent_sd": float(
                self.forecast_analysis_latent_sd
            ),
            "forecast_innovation_scale": float(self.forecast_innovation_scale),
            "forecast_windows_h": tuple(
                (float(start_h), float(end_h))
                for start_h, end_h in self.forecast_windows_h
            ),
        }


@dataclass(frozen=True)
class RandomStreamKeyV2:
    """Auditable initial state snapshotted without consuming the caller's stream."""

    name: str
    source_kind: str
    spawn_key: tuple[int, ...]
    state_sha256: str
    initial_state_json: str

    def __post_init__(self) -> None:
        if not self.name or not self.source_kind:
            raise ValueError("random stream name and source kind must not be empty")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < 0
            for value in self.spawn_key
        ):
            raise ValueError("random stream spawn key must contain non-negative integers")
        try:
            initial_state = json.loads(self.initial_state_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("random stream initial state must be canonical JSON") from exc
        if _canonical_json(initial_state) != self.initial_state_json:
            raise ValueError("random stream initial state must use canonical JSON encoding")
        _require_sha256("random stream state_sha256", self.state_sha256)
        if _canonical_sha256(initial_state) != self.state_sha256:
            raise ValueError("random stream initial-state digest does not match its state")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_kind": self.source_kind,
            "spawn_key": list(self.spawn_key),
            "state_sha256": self.state_sha256,
            "initial_state": json.loads(self.initial_state_json),
        }


@dataclass(frozen=True)
class ForecastWindowV2:
    """One forecast band; offsets are seconds from its issue boundary."""

    name: str
    start_offset_s_from_issue: float
    end_offset_s_from_issue: float
    mean_peak_power_fraction: float
    p10_peak_power_fraction: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("forecast window name must not be empty")
        start = self.start_offset_s_from_issue
        end = self.end_offset_s_from_issue
        if (
            isinstance(start, bool)
            or not isinstance(start, Real)
            or isinstance(end, bool)
            or not isinstance(end, Real)
            or not (
                math.isfinite(start)
                and math.isfinite(end)
                and 0.0 <= start < end
            )
        ):
            raise ValueError("forecast window offsets must be finite, ordered seconds")
        mean = self.mean_peak_power_fraction
        p10 = self.p10_peak_power_fraction
        if (
            isinstance(mean, bool)
            or not isinstance(mean, Real)
            or isinstance(p10, bool)
            or not isinstance(p10, Real)
            or not (math.isfinite(mean) and math.isfinite(p10))
        ):
            raise ValueError("forecast peak-power fractions must be finite")
        if not 0.0 <= p10 <= mean <= 1.0:
            raise ValueError("forecast fractions must satisfy 0 <= p10 <= mean <= 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "start_offset_s_from_issue": self.start_offset_s_from_issue,
            "end_offset_s_from_issue": self.end_offset_s_from_issue,
            "offset_interval": "half-open [start, end)",
            "mean_peak_power_fraction": self.mean_peak_power_fraction,
            "p10_peak_power_fraction": self.p10_peak_power_fraction,
        }


@dataclass(frozen=True)
class ForecastIssueV2:
    """Immutable forecasts available from one controller decision boundary."""

    issued_at_physics_boundary_step: int
    issued_at_offset_s: float
    windows: tuple[ForecastWindowV2, ...]
    _issue_stream_sha256: str

    def __post_init__(self) -> None:
        step = self.issued_at_physics_boundary_step
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError("forecast issue boundary step must be a non-negative integer")
        if (
            isinstance(self.issued_at_offset_s, bool)
            or not isinstance(self.issued_at_offset_s, Real)
            or not math.isfinite(self.issued_at_offset_s)
            or self.issued_at_offset_s < 0.0
        ):
            raise ValueError("forecast issue offset must be finite non-negative seconds")
        expected_windows = tuple(
            (_window_name(start_h, end_h), start_h, end_h)
            for start_h, end_h in FORECAST_WINDOWS_H
        )
        actual_windows = tuple(
            (
                window.name,
                window.start_offset_s_from_issue / SECONDS_PER_HOUR,
                window.end_offset_s_from_issue / SECONDS_PER_HOUR,
            )
            for window in self.windows
        )
        if actual_windows != expected_windows:
            raise ValueError("forecast issue must contain the canonical ordered windows")
        _require_sha256("issue stream identity", self._issue_stream_sha256)

    def window(self, name: str) -> ForecastWindowV2:
        for window in self.windows:
            if window.name == name:
                return window
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        """Controller-safe issuance; random-stream identity remains private."""
        return {
            "issued_at_physics_boundary_step": self.issued_at_physics_boundary_step,
            "issued_at_offset_s": self.issued_at_offset_s,
            "issue_time_basis": "seconds since episode start",
            "windows": [window.to_dict() for window in self.windows],
        }


@dataclass(frozen=True)
class WeatherEpisodeV2:
    """Immutable private truth, public issuances, and replay identities."""

    config: WeatherConfigV2
    tape: SolarFactorTape
    issues: tuple[ForecastIssueV2, ...]
    physics_steps: int
    decision_stride_steps: int
    truth_stream_key: RandomStreamKeyV2
    forecast_stream_key: RandomStreamKeyV2
    clear_sky_envelope_id: str
    clear_sky_envelope_sha256: str
    truth_origin_kind: str
    truth_sha256: str
    forecasts_sha256: str
    episode_sha256: str

    def __post_init__(self) -> None:
        self.config.validate()
        if self.physics_steps != self.tape.interval_count:
            raise ValueError("weather physics_steps must match the truth tape")
        if self.decision_stride_steps <= 0 or self.physics_steps % self.decision_stride_steps:
            raise ValueError("weather decision stride must divide the physics horizon")
        expected_issues = tuple(
            range(0, self.physics_steps, self.decision_stride_steps)
        )
        if self.issue_steps != expected_issues:
            raise ValueError("weather issues must cover every decision boundary exactly")
        if self.truth_stream_key.name != TRUTH_STREAM_NAME:
            raise ValueError("weather truth stream has the wrong name")
        if self.forecast_stream_key.name != FORECAST_STREAM_NAME:
            raise ValueError("weather forecast stream has the wrong name")
        if self.truth_origin_kind not in {"synthetic_generator", "external_tape"}:
            raise ValueError("weather truth origin kind is invalid")
        if not self.clear_sky_envelope_id:
            raise ValueError("weather clear-sky envelope id must not be empty")
        for name, value in (
            ("clear_sky_envelope_sha256", self.clear_sky_envelope_sha256),
            ("truth_sha256", self.truth_sha256),
            ("forecasts_sha256", self.forecasts_sha256),
            ("episode_sha256", self.episode_sha256),
        ):
            _require_sha256(name, value)

    @property
    def dt_s(self) -> float:
        return float(self.tape.dt_s)

    @property
    def issue_steps(self) -> tuple[int, ...]:
        return tuple(issue.issued_at_physics_boundary_step for issue in self.issues)

    @property
    def stream_keys(self) -> dict[str, RandomStreamKeyV2]:
        # Return a new mapping so the frozen episode never retains mutable state.
        return {
            TRUTH_STREAM_NAME: self.truth_stream_key,
            FORECAST_STREAM_NAME: self.forecast_stream_key,
        }

    def issue_at_boundary(self, boundary_step: int) -> ForecastIssueV2 | None:
        step = self._checked_boundary_step(boundary_step)
        index = bisect_right(self.issue_steps, step) - 1
        if index >= 0 and self.issues[index].issued_at_physics_boundary_step == step:
            return self.issues[index]
        return None

    def current_issue(self, boundary_step: int) -> ForecastIssueV2:
        """Return the latest issue available at a physical boundary."""
        step = self._checked_boundary_step(boundary_step)
        index = bisect_right(self.issue_steps, step) - 1
        if index < 0:  # Defensive: a generated episode always has an issue at zero.
            raise LookupError("no forecast has been issued yet")
        return self.issues[index]

    def _checked_boundary_step(self, boundary_step: int) -> int:
        if (
            isinstance(boundary_step, bool)
            or not isinstance(boundary_step, Integral)
            or not 0 <= boundary_step <= self.physics_steps
        ):
            raise ValueError(
                f"boundary_step must be an integer within [0, {self.physics_steps}]"
            )
        return int(boundary_step)

    def private_audit_manifest(self) -> dict[str, object]:
        """Offline replay metadata; this object must not enter agent observations."""
        return {
            "weather_generator_version": WEATHER_GENERATOR_VERSION,
            "config": self.config.to_dict(),
            "time_convention": {
                "physics_timestep_s": self.dt_s,
                "physics_steps": self.physics_steps,
                "decision_stride_steps": self.decision_stride_steps,
                "forecast_issue_cadence_s": self.decision_stride_steps * self.dt_s,
                "issue_time_basis": "seconds since episode start",
                "window_offsets": "seconds from issue boundary, half-open [start, end)",
                "truth_interval_factor": "boundary k+1 applies to interval [k, k+1)",
            },
            "random_streams": {
                TRUTH_STREAM_NAME: self.truth_stream_key.to_dict(),
                FORECAST_STREAM_NAME: self.forecast_stream_key.to_dict(),
            },
            "clear_sky_envelope": {
                "id": self.clear_sky_envelope_id,
                "unit": "fraction of PV nameplate peak power",
                "sha256": self.clear_sky_envelope_sha256,
            },
            "replay": {
                "truth_origin_kind": self.truth_origin_kind,
                "random_initial_states_embedded": True,
                "truth_tape_embedded": False,
                "clear_sky_envelope_embedded": False,
                "requirements": (
                    [
                        "matching source revision and dependency lock",
                        "regenerate truth and forecasts from the embedded RNG initial states",
                        "regenerate the declared deterministic clear-sky envelope",
                    ]
                    if self.truth_origin_kind == "synthetic_generator"
                    else [
                        "matching source revision and dependency lock",
                        "retain the external SolarFactorTape matching truth_sha256",
                        "retain the external clear-sky envelope matching its sha256",
                        "regenerate forecasts from the embedded forecast RNG initial state",
                    ]
                ),
            },
            "private_hashes": {
                "truth_sha256": self.truth_sha256,
                "forecasts_sha256": self.forecasts_sha256,
                "forecast_issue_stream_sha256": [
                    issue._issue_stream_sha256 for issue in self.issues
                ],
                "episode_sha256": self.episode_sha256,
            },
        }


def _window_name(start_h: float, end_h: float) -> str:
    return f"{start_h:g}-{end_h:g}h"


def _stream_identity(source: RandomSource, name: str) -> RandomStreamKeyV2:
    if isinstance(source, np.random.SeedSequence):
        state = _json_ready(deepcopy(source.state))
        return RandomStreamKeyV2(
            name=name,
            source_kind="numpy.random.SeedSequence",
            spawn_key=tuple(int(value) for value in source.spawn_key),
            state_sha256=_canonical_sha256(state),
            initial_state_json=_canonical_json(state),
        )
    if isinstance(source, np.random.Generator):
        state = _json_ready(deepcopy(source.bit_generator.state))
        return RandomStreamKeyV2(
            name=name,
            source_kind=f"numpy.random.{type(source.bit_generator).__name__}",
            spawn_key=(),
            state_sha256=_canonical_sha256(state),
            initial_state_json=_canonical_json(state),
        )
    raise TypeError(f"{name} source must be a numpy SeedSequence or Generator")


def _clone_seed_sequence(source: np.random.SeedSequence) -> np.random.SeedSequence:
    """Copy a seed descriptor so weather generation never consumes its caller."""
    return np.random.SeedSequence(**deepcopy(source.state))


def _as_generator(source: RandomSource, name: str) -> np.random.Generator:
    if isinstance(source, np.random.SeedSequence):
        return np.random.default_rng(_clone_seed_sequence(source))
    if isinstance(source, np.random.Generator):
        return deepcopy(source)
    raise TypeError(f"{name} source must be a numpy SeedSequence or Generator")


def _issue_seed_sequences(
    source: RandomSource,
    count: int,
) -> tuple[np.random.SeedSequence, ...]:
    if isinstance(source, np.random.SeedSequence):
        return tuple(_clone_seed_sequence(source).spawn(count))
    if isinstance(source, np.random.Generator):
        entropy = deepcopy(source).integers(
            0,
            np.iinfo(np.uint32).max,
            size=(count, 4),
            dtype=np.uint32,
            endpoint=True,
        )
        return tuple(
            np.random.SeedSequence([int(word) for word in row]) for row in entropy
        )
    raise TypeError("forecast source must be a numpy SeedSequence or Generator")


def _process_parameters(
    config: WeatherConfigV2,
    step_s: float,
) -> tuple[float, float, float]:
    latent_mean = _logit(config.latent_median_resource_factor)
    phi = 2.0 ** (-(step_s / SECONDS_PER_HOUR) / config.autocorrelation_half_life_h)
    innovation_sd = config.latent_stationary_sd * math.sqrt(max(0.0, 1.0 - phi * phi))
    return latent_mean, phi, innovation_sd


def generate_truth_tape(
    config: WeatherConfigV2,
    *,
    physics_steps: int,
    dt_s: float,
    truth_seed: RandomSource,
) -> SolarFactorTape:
    """Generate resource factors at physics boundaries from a caller-owned stream."""
    config.validate()
    steps = _positive_integer("physics_steps", physics_steps)
    _finite("dt_s", dt_s, positive=True)
    rng = _as_generator(truth_seed, TRUTH_STREAM_NAME)
    latent_mean, phi, innovation_sd = _process_parameters(config, dt_s)
    latent = latent_mean + config.latent_stationary_sd * float(rng.normal())
    boundary_factors = np.empty(steps + 1, dtype=np.float64)
    boundary_factors[0] = float(_sigmoid(latent))
    for step in range(1, steps + 1):
        latent = latent_mean + phi * (latent - latent_mean) + innovation_sd * float(
            rng.normal()
        )
        boundary_factors[step] = float(_sigmoid(latent))
    return SolarFactorTape(dt_s=float(dt_s), boundary_factors=tuple(boundary_factors))


def _forecast_latent_paths(
    *,
    config: WeatherConfigV2,
    current_resource_factor: float,
    horizon_steps: int,
    dt_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw conditional latent paths without consulting a future truth tape."""
    members = int(config.forecast_ensemble_members)
    latent_mean, phi, innovation_sd = _process_parameters(config, dt_s)
    current_latent = _logit(current_resource_factor)
    state = np.full(members, current_latent, dtype=np.float64)
    if config.forecast_analysis_latent_sd:
        state += rng.normal(
            0.0,
            config.forecast_analysis_latent_sd,
            size=members,
        )
    innovation_sd *= config.forecast_innovation_scale
    paths = np.empty((members, horizon_steps), dtype=np.float64)

    if phi < 1e-6:
        # Avoid overflow in the vectorized closed form below for an extreme,
        # near-memoryless caller configuration.  This branch remains the exact
        # recurrence rather than approximating a small non-zero ``phi`` as zero.
        for step in range(horizon_steps):
            state = (
                latent_mean
                + phi * (state - latent_mean)
                + innovation_sd * rng.normal(size=members)
            )
            paths[:, step] = state
        return paths

    # A chunked closed-form scan avoids one Python iteration per physics step,
    # while resetting powers often enough to remain numerically stable for short
    # half-lives.  Within a chunk this is algebraically identical to AR(1).
    # Keep the smallest inverse power well below overflow.  The fixed 128-step
    # block is fast for operational half-lives; unusual short half-lives receive
    # smaller, still-vectorized blocks instead of dividing by underflowed powers.
    safe_inverse_limit = 1e100
    max_safe_exponent = (
        128
        if phi >= 1.0
        else int(
            math.floor(math.log(1.0 / safe_inverse_limit) / math.log(phi))
        )
    )
    chunk_size = max(1, min(128, max_safe_exponent))
    for start in range(0, horizon_steps, chunk_size):
        stop = min(start + chunk_size, horizon_steps)
        count = stop - start
        powers = phi ** np.arange(1, count + 1, dtype=np.float64)
        innovations = rng.normal(size=(members, count))
        weighted = np.cumsum(innovations / powers, axis=1)
        block = (
            latent_mean
            + (state - latent_mean)[:, None] * powers[None, :]
            + innovation_sd * weighted * powers[None, :]
        )
        paths[:, start:stop] = block
        state = block[:, -1]
    return paths


def _normalize_envelope(
    clear_sky_peak_fractions: Sequence[float] | None,
    *,
    required_intervals: int,
) -> tuple[float, ...]:
    if clear_sky_peak_fractions is None:
        return (1.0,) * required_intervals
    if len(clear_sky_peak_fractions) < required_intervals:
        raise ValueError(
            "clear_sky_peak_fractions must cover every episode interval plus the "
            "full forecast horizon after the final issue"
        )
    result: list[float] = []
    for index, raw in enumerate(clear_sky_peak_fractions[:required_intervals]):
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise ValueError(
                f"clear_sky_peak_fractions[{index}] must be a real number"
            )
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"clear_sky_peak_fractions[{index}] must be finite and within [0, 1]"
            )
        result.append(value)
    return tuple(result)


def _forecast_issue(
    config: WeatherConfigV2,
    *,
    tape: SolarFactorTape,
    issue_step: int,
    dt_s: float,
    window_steps: tuple[tuple[int, int], ...],
    envelope: np.ndarray,
    issue_seed: np.random.SeedSequence,
) -> ForecastIssueV2:
    issue_key = _stream_identity(issue_seed, f"{FORECAST_STREAM_NAME}.issue-{issue_step}")
    rng = np.random.default_rng(issue_seed)
    horizon_steps = window_steps[-1][1]
    latent_paths = _forecast_latent_paths(
        config=config,
        current_resource_factor=tape.at_boundary(issue_step),
        horizon_steps=horizon_steps,
        dt_s=dt_s,
        rng=rng,
    )
    resource_paths = _sigmoid(latent_paths)
    future_envelope = envelope[issue_step : issue_step + horizon_steps]
    peak_power_paths = resource_paths * future_envelope[None, :]

    windows: list[ForecastWindowV2] = []
    for (start_h, end_h), (start_step, end_step) in zip(
        config.forecast_windows_h,
        window_steps,
        strict=True,
    ):
        member_means = peak_power_paths[:, start_step:end_step].mean(axis=1)
        mean = float(np.clip(member_means.mean(), 0.0, 1.0))
        p10 = float(np.clip(np.quantile(member_means, 0.10), 0.0, mean))
        windows.append(
            ForecastWindowV2(
                name=_window_name(start_h, end_h),
                start_offset_s_from_issue=start_h * SECONDS_PER_HOUR,
                end_offset_s_from_issue=end_h * SECONDS_PER_HOUR,
                mean_peak_power_fraction=mean,
                p10_peak_power_fraction=p10,
            )
        )
    return ForecastIssueV2(
        issued_at_physics_boundary_step=issue_step,
        issued_at_offset_s=issue_step * dt_s,
        windows=tuple(windows),
        _issue_stream_sha256=issue_key.state_sha256,
    )


def build_weather_episode_from_tape(
    config: WeatherConfigV2,
    *,
    tape: SolarFactorTape,
    decision_stride: int,
    truth_stream_key: RandomStreamKeyV2,
    forecast_seed: RandomSource,
    clear_sky_peak_fractions: Sequence[float] | None = None,
    clear_sky_envelope_id: str = "unit-envelope-v1",
    truth_origin_kind: str = "external_tape",
) -> WeatherEpisodeV2:
    """Issue causal forecasts for an immutable tape.

    This seam is intentionally public for counterfactual leakage tests: callers
    can replace truth strictly after an issue boundary and prove that issuance
    remains unchanged.
    """
    config.validate()
    steps = _positive_integer("tape.interval_count", tape.interval_count)
    stride = _positive_integer("decision_stride", decision_stride)
    dt_s = float(tape.dt_s)
    _finite("tape.dt_s", dt_s, positive=True)
    if steps % stride:
        raise ValueError("tape intervals must contain a whole number of decisions")
    if not isinstance(clear_sky_envelope_id, str) or not clear_sky_envelope_id.strip():
        raise ValueError("clear_sky_envelope_id must be a non-empty string")
    if truth_origin_kind not in {"synthetic_generator", "external_tape"}:
        raise ValueError(
            "truth_origin_kind must be 'synthetic_generator' or 'external_tape'"
        )
    if truth_stream_key.name != TRUTH_STREAM_NAME:
        raise ValueError(f"truth_stream_key.name must be {TRUTH_STREAM_NAME!r}")

    window_steps = tuple(
        (
            0 if start_h == 0.0 else _aligned_steps(
                start_h * SECONDS_PER_HOUR,
                dt_s,
                name=f"forecast window start {start_h:g} h",
            ),
            _aligned_steps(
                end_h * SECONDS_PER_HOUR,
                dt_s,
                name=f"forecast window end {end_h:g} h",
            ),
        )
        for start_h, end_h in config.forecast_windows_h
    )
    issue_steps = tuple(range(0, steps, stride))
    required_intervals = issue_steps[-1] + window_steps[-1][1]
    normalized_envelope = _normalize_envelope(
        clear_sky_peak_fractions,
        required_intervals=required_intervals,
    )
    envelope = np.asarray(normalized_envelope, dtype=np.float64)
    forecast_stream_key = _stream_identity(forecast_seed, FORECAST_STREAM_NAME)
    if truth_stream_key.state_sha256 == forecast_stream_key.state_sha256:
        raise ValueError("truth and forecast must use independent random streams")
    issue_seeds = _issue_seed_sequences(forecast_seed, len(issue_steps))
    issues = tuple(
        _forecast_issue(
            config,
            tape=tape,
            issue_step=issue_step,
            dt_s=dt_s,
            window_steps=window_steps,
            envelope=envelope,
            issue_seed=issue_seed,
        )
        for issue_step, issue_seed in zip(issue_steps, issue_seeds, strict=True)
    )

    truth_sha256 = _canonical_sha256(
        {
            "semantics": "factor at boundary k+1 applies to interval [k, k+1)",
            **tape.to_dict(),
        }
    )
    forecasts_sha256 = _canonical_sha256([issue.to_dict() for issue in issues])
    envelope_sha256 = _canonical_sha256(
        {
            "unit": "fraction of PV nameplate peak power",
            "interval_values": normalized_envelope,
        }
    )
    identity_payload = {
        "weather_generator_version": WEATHER_GENERATOR_VERSION,
        "config": config.to_dict(),
        "dt_s": dt_s,
        "physics_steps": steps,
        "decision_stride_steps": stride,
        "truth_stream": truth_stream_key.to_dict(),
        "forecast_stream": forecast_stream_key.to_dict(),
        "clear_sky_envelope_id": clear_sky_envelope_id,
        "clear_sky_envelope_sha256": envelope_sha256,
        "truth_origin_kind": truth_origin_kind,
        "truth_sha256": truth_sha256,
        "forecasts_sha256": forecasts_sha256,
        "issue_stream_sha256": [issue._issue_stream_sha256 for issue in issues],
    }
    return WeatherEpisodeV2(
        config=config,
        tape=tape,
        issues=issues,
        physics_steps=steps,
        decision_stride_steps=stride,
        truth_stream_key=truth_stream_key,
        forecast_stream_key=forecast_stream_key,
        clear_sky_envelope_id=clear_sky_envelope_id,
        clear_sky_envelope_sha256=envelope_sha256,
        truth_origin_kind=truth_origin_kind,
        truth_sha256=truth_sha256,
        forecasts_sha256=forecasts_sha256,
        episode_sha256=_canonical_sha256(identity_payload),
    )


def generate_weather_episode(
    config: WeatherConfigV2,
    *,
    physics_steps: int,
    dt_s: float,
    decision_stride: int,
    truth_seed: RandomSource,
    forecast_seed: RandomSource,
    clear_sky_peak_fractions: Sequence[float] | None = None,
    clear_sky_envelope_id: str = "unit-envelope-v1",
) -> WeatherEpisodeV2:
    """Generate one replayable truth tape and all causally issued forecasts."""
    config.validate()
    truth_stream_key = _stream_identity(truth_seed, TRUTH_STREAM_NAME)
    forecast_stream_key = _stream_identity(forecast_seed, FORECAST_STREAM_NAME)
    if truth_stream_key.state_sha256 == forecast_stream_key.state_sha256:
        raise ValueError("truth and forecast must use independent random streams")
    tape = generate_truth_tape(
        config,
        physics_steps=physics_steps,
        dt_s=dt_s,
        truth_seed=truth_seed,
    )
    return build_weather_episode_from_tape(
        config,
        tape=tape,
        decision_stride=decision_stride,
        truth_stream_key=truth_stream_key,
        forecast_seed=forecast_seed,
        clear_sky_peak_fractions=clear_sky_peak_fractions,
        clear_sky_envelope_id=clear_sky_envelope_id,
        truth_origin_kind="synthetic_generator",
    )


__all__ = [
    "FORECAST_STREAM_NAME",
    "FORECAST_WINDOWS_H",
    "TRUTH_STREAM_NAME",
    "WEATHER_GENERATOR_VERSION",
    "ForecastIssueV2",
    "ForecastWindowV2",
    "RandomStreamKeyV2",
    "WeatherConfigV2",
    "WeatherEpisodeV2",
    "build_weather_episode_from_tape",
    "generate_truth_tape",
    "generate_weather_episode",
]
