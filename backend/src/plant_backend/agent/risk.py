"""V10: what HEPA said, read as risk the supervisor can act on.

HEPA answers one question -- `P(event within Δt)` -- and this module is everything
between that number and a supervisor deciding whether to look:

    policy      thresholds per event, and per horizon where they differ
    levels      LOW / ELEVATED / HIGH / CRITICAL
    trends      stable / rising / falling, against the previous forecast
    cadence     how often the model is asked at all
    triggers    which risks are worth waking the supervisor for

Deliberately unsophisticated, in three places. A level is a threshold comparison,
not a calibrated alarm. A trend is this forecast against the previous one with a
tolerance, not a fitted slope. And the output of all of it is *attention*, never
an action: nothing here caps a load, switches a controller or stops a machine.
Crossing HIGH means the supervisor investigates, and the supervisor -- with the
telemetry and the diagnostics in front of it -- decides what, if anything, to do.

The numbers are configurable, because a threshold is an operating decision:

    PLANT_RISK_CONFIG=configs/risk.json   # see agent/configs/risk.json
    PLANT_RISK_REFRESH_S=900              # simulated seconds between forecasts

With no checkpoint configured (the default) a `RiskMonitor` reports
`status: unavailable`, holds no probabilities and raises no triggers, and the
supervisor behaves exactly as it did in V8.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..hepa import service
from ..hepa.events import EVENT_NAMES
from .trace import AgentTrace
from .triggers import Trigger

log = logging.getLogger(__name__)

#: Risk levels, least to most urgent. `LOW` is "not worth mentioning".
LOW = "low"
ELEVATED = "elevated"
HIGH = "high"
CRITICAL = "critical"
LEVELS: tuple[str, ...] = (LOW, ELEVATED, HIGH, CRITICAL)
LEVEL_RANK = {level: rank for rank, level in enumerate(LEVELS)}

#: How a probability moved since the previous forecast.
STABLE = "stable"
RISING = "rising"
FALLING = "falling"

#: Horizons the supervisor asks about by default. 24h is available but too vague
#: to steer a shift by, so it is not requested unless someone asks for it.
DEFAULT_HORIZONS: tuple[str, ...] = ("1h", "3h", "6h", "12h")

#: Simulated seconds between two forecasts. HEPA is not a per-timestep tool: the
#: plant moves in minutes, the horizons are hours.
REFRESH_S = 900.0
#: Probability change below which a movement is called `stable`.
TREND_TOLERANCE = 0.05
#: Frames of history offered to the model; it resamples and truncates what it needs.
HISTORY_FRAMES = 1440

#: Environment overrides.
CONFIG_ENV = "PLANT_RISK_CONFIG"
REFRESH_ENV = "PLANT_RISK_REFRESH_S"


@dataclass(frozen=True)
class Thresholds:
    """Where a probability stops being background noise, per event."""

    elevated: float = 0.30
    high: float = 0.55
    critical: float = 0.80

    def level(self, probability: float) -> str:
        if probability >= self.critical:
            return CRITICAL
        if probability >= self.high:
            return HIGH
        if probability >= self.elevated:
            return ELEVATED
        return LOW

    def to_dict(self) -> dict[str, float]:
        return {"elevated": self.elevated, "high": self.high, "critical": self.critical}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None, base: "Thresholds") -> "Thresholds":
        if not raw:
            return base
        return cls(
            elevated=float(raw.get("elevated", base.elevated)),
            high=float(raw.get("high", base.high)),
            critical=float(raw.get("critical", base.critical)),
        )


#: Defaults, per event. Lower thresholds mean "tell me earlier": a forced
#: production shutdown is worth a look at a lower probability than the battery
#: touching a reserve the plant is allowed to touch.
DEFAULT_THRESHOLDS: dict[str, Thresholds] = {
    "battery_reserve_violation": Thresholds(elevated=0.30, high=0.55, critical=0.80),
    "h2_starvation": Thresholds(elevated=0.30, high=0.55, critical=0.80),
    "co2_starvation": Thresholds(elevated=0.35, high=0.60, critical=0.85),
    "forced_production_shutdown": Thresholds(elevated=0.20, high=0.45, critical=0.75),
    "critical_plant_state": Thresholds(elevated=0.15, high=0.35, critical=0.60),
}


@dataclass(frozen=True)
class RiskPolicy:
    """The configurable half of V10: when a prediction is worth attention."""

    horizons: tuple[str, ...] = DEFAULT_HORIZONS
    refresh_s: float = REFRESH_S
    trend_tolerance: float = TREND_TOLERANCE
    #: Level at or above which the supervisor is woken.
    attention: str = HIGH
    #: A risk *rising* into this level is worth a look before it reaches
    #: `attention`: the movement is the information, not the single number.
    rising_attention: str = ELEVATED
    default: Thresholds = Thresholds()
    events: Mapping[str, Thresholds] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    #: Per event, per horizon overrides, for the cases where 1 h and 12 h are not
    #: the same question at all.
    per_horizon: Mapping[str, Mapping[str, Thresholds]] = field(default_factory=dict)

    def thresholds(self, event: str, horizon: str) -> Thresholds:
        base = self.events.get(event, self.default)
        return self.per_horizon.get(event, {}).get(horizon, base)

    def level(self, event: str, horizon: str, probability: float) -> str:
        return self.thresholds(event, horizon).level(probability)

    def wakes(self, level: str, trend: str) -> bool:
        """Whether one risk entry is, on its own, worth an investigation."""
        rank = LEVEL_RANK.get(level, 0)
        if rank >= LEVEL_RANK[self.attention]:
            return True
        return trend == RISING and rank >= LEVEL_RANK[self.rising_attention]

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizons": list(self.horizons),
            "refresh_s": self.refresh_s,
            "trend_tolerance": self.trend_tolerance,
            "attention": self.attention,
            "rising_attention": self.rising_attention,
            "default": self.default.to_dict(),
            "events": {name: value.to_dict() for name, value in self.events.items()},
            "per_horizon": {
                event: {horizon: value.to_dict() for horizon, value in per.items()}
                for event, per in self.per_horizon.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RiskPolicy":
        """A policy from plain JSON. Unknown keys are ignored, defaults survive."""
        default = Thresholds.from_dict(raw.get("default"), Thresholds())
        events = dict(DEFAULT_THRESHOLDS)
        per_horizon: dict[str, dict[str, Thresholds]] = {}
        for name, entry in (raw.get("events") or {}).items():
            if not isinstance(entry, Mapping):
                continue
            base = Thresholds.from_dict(entry, events.get(name, default))
            events[name] = base
            horizons = entry.get("horizons")
            if isinstance(horizons, Mapping):
                per_horizon[name] = {
                    str(label): Thresholds.from_dict(values, base)
                    for label, values in horizons.items()
                    if isinstance(values, Mapping)
                }
        horizons_raw = raw.get("horizons")
        return cls(
            horizons=tuple(str(label) for label in horizons_raw) if horizons_raw else DEFAULT_HORIZONS,
            refresh_s=float(raw.get("refresh_s", REFRESH_S)),
            trend_tolerance=float(raw.get("trend_tolerance", TREND_TOLERANCE)),
            attention=str(raw.get("attention", HIGH)),
            rising_attention=str(raw.get("rising_attention", ELEVATED)),
            default=default,
            events=events,
            per_horizon=per_horizon,
        )

    @classmethod
    def from_env(cls) -> "RiskPolicy":
        """The configured policy, or the defaults. A bad config never stops the plant."""
        policy = cls()
        path = os.environ.get(CONFIG_ENV, "").strip()
        if path:
            try:
                policy = cls.from_dict(json.loads(Path(path).read_text()))
            except (OSError, ValueError, TypeError) as reason:
                log.warning("ignoring %s=%s: %s", CONFIG_ENV, path, reason)
        refresh = os.environ.get(REFRESH_ENV, "").strip()
        if refresh:
            try:
                policy = replace_refresh(policy, float(refresh))
            except ValueError:
                log.warning("ignoring %s=%s: not a number", REFRESH_ENV, refresh)
        return policy


def replace_refresh(policy: RiskPolicy, refresh_s: float) -> RiskPolicy:
    """The same policy on a different cadence."""
    return RiskPolicy(
        horizons=policy.horizons,
        refresh_s=max(0.0, refresh_s),
        trend_tolerance=policy.trend_tolerance,
        attention=policy.attention,
        rising_attention=policy.rising_attention,
        default=policy.default,
        events=policy.events,
        per_horizon=policy.per_horizon,
    )


@dataclass(frozen=True)
class EventRisk:
    """One event at one horizon: the number, what it means, where it is going."""

    event: str
    horizon: str
    probability: float
    level: str
    trend: str
    previous: float | None

    @property
    def delta(self) -> float | None:
        return None if self.previous is None else round(self.probability - self.previous, 4)

    def percent(self) -> int:
        return int(round(100.0 * self.probability))

    def sentence(self) -> str:
        """One shift-log line: what may happen, when, and how it is moving."""
        label = self.event.replace("_", " ")
        moving = "" if self.trend == STABLE else f" and {self.trend}"
        return f"{label} within {self.horizon}: {self.percent()}% ({self.level}{moving})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "horizon": self.horizon,
            "probability": round(self.probability, 4),
            "level": self.level,
            "trend": self.trend,
            "previous": None if self.previous is None else round(self.previous, 4),
            "delta": self.delta,
        }


class RiskMonitor:
    """Keeps the plant's current predicted risk, and says when it matters.

    Owns the cadence (`policy.refresh_s` of simulated time between forecasts), the
    previous forecast (which is what makes a trend possible) and nothing else. It
    reads telemetry history and writes only to the trace.
    """

    def __init__(
        self,
        runtime: Any,
        forecaster: service.Forecaster | None = None,
        policy: RiskPolicy | None = None,
        *,
        trace: AgentTrace | None = None,
    ) -> None:
        self.runtime = runtime
        self.forecaster = forecaster or service.CheckpointForecaster()
        self.policy = policy or RiskPolicy.from_env()
        self.trace = trace
        self.status = service.UNAVAILABLE
        self.reason: str | None = None
        self.source = service.SOURCE_CHECKPOINT
        self.updated_time_s: float | None = None
        self.updated_clock: str | None = None
        self.forecasts = 0
        #: event -> horizon -> latest risk.
        self.risks: dict[str, dict[str, EventRisk]] = {}

    # --- asking -----------------------------------------------------------
    def due(self, time_s: float) -> bool:
        """Whether enough simulated time has passed to ask HEPA again."""
        if self.updated_time_s is None:
            return True
        return time_s - self.updated_time_s >= self.policy.refresh_s

    def update(
        self,
        frames: Sequence[dict[str, Any]] | None = None,
        *,
        force: bool = False,
    ) -> service.Forecast | None:
        """Ask HEPA if it is time to, absorb the answer, and log what moved.

        Returns the forecast, or `None` when the cadence says it is not due yet.
        Never raises: a forecaster that throws becomes `status: error` and the
        supervisor carries on with telemetry, events and diagnostics.
        """
        observation = self.runtime.session.telemetry
        now_s = float(observation.time_s)
        if not force and not self.due(now_s):
            return None
        history = list(frames) if frames is not None else self.runtime.session.recent(HISTORY_FRAMES)
        forecast = self._ask(history)
        self._absorb(forecast, now_s, observation.clock)
        return forecast

    def _ask(self, frames: Sequence[dict[str, Any]]) -> service.Forecast:
        try:
            return self.forecaster.forecast(frames, self.policy.horizons)
        except Exception as reason:  # noqa: BLE001 - the plant outranks the forecast
            log.warning("HEPA forecast failed: %s", reason, exc_info=True)
            return service.Forecast.failed(f"{type(reason).__name__}: {reason}")

    def _absorb(self, forecast: service.Forecast, time_s: float, clock: str) -> None:
        self.status = forecast.status
        self.reason = forecast.reason
        self.source = forecast.source
        self.updated_time_s = time_s
        self.updated_clock = clock
        if not forecast.available:
            self.risks = {}
            return
        self.forecasts += 1
        previous = self.risks
        risks: dict[str, dict[str, EventRisk]] = {}
        for event, per_horizon in forecast.probabilities.items():
            for horizon, value in per_horizon.items():
                probability = float(value)
                earlier = previous.get(event, {}).get(horizon)
                before = earlier.probability if earlier else None
                risks.setdefault(event, {})[horizon] = EventRisk(
                    event=event,
                    horizon=horizon,
                    probability=probability,
                    level=self.policy.level(event, horizon, probability),
                    trend=self._trend(probability, before),
                    previous=before,
                )
        self.risks = risks
        self._log(time_s, clock)

    def _trend(self, probability: float, previous: float | None) -> str:
        if previous is None or abs(probability - previous) < self.policy.trend_tolerance:
            return STABLE
        return RISING if probability > previous else FALLING

    def _log(self, time_s: float, clock: str) -> None:
        """Trace the risks that moved. Predictions, marked as predictions."""
        if self.trace is None:
            return
        moved = [
            risk
            for risk in self.notable()
            if risk.trend != STABLE or (risk.previous is None and risk.level != LOW)
        ]
        if not moved:
            return
        self.trace.at(time_s, clock)
        for risk in moved[:3]:
            direction = {RISING: "increased to", FALLING: "decreased to", STABLE: "at"}[risk.trend]
            self.trace.record(
                "prediction",
                f"HEPA: {risk.event.replace('_', ' ')} {risk.horizon} risk "
                f"{direction} {risk.percent()}%",
                **risk.to_dict(),
            )

    # --- reading ----------------------------------------------------------
    @property
    def available(self) -> bool:
        return self.status == service.ACTIVE and bool(self.risks)

    def entries(self) -> list[EventRisk]:
        """Every risk in the latest forecast, most urgent first."""
        flat = [risk for per_horizon in self.risks.values() for risk in per_horizon.values()]
        return sorted(flat, key=_urgency, reverse=True)

    def notable(self) -> list[EventRisk]:
        """The risks above LOW, most urgent first. Usually a short list."""
        return [risk for risk in self.entries() if risk.level != LOW]

    def concerns(self) -> list[EventRisk]:
        """The risks that justify waking the supervisor, most urgent first."""
        return [risk for risk in self.entries() if self.policy.wakes(risk.level, risk.trend)]

    def trigger(self) -> Trigger | None:
        """The predicted-risk trigger, if the latest forecast raises one."""
        concerns = self.concerns()
        if not concerns:
            return None
        worst = concerns[0]
        priority = _PRIORITY.get(worst.level, 35)
        if worst.trend == RISING:
            priority += 5
        return Trigger(
            reason="predicted_risk",
            detail=f"HEPA predicts {worst.sentence()}",
            key=f"risk:{worst.event}",
            priority=priority,
        )

    def brief(self, limit: int = 4) -> list[str]:
        """The forecast as sentences, for the supervisory brief and the trace."""
        return [risk.sentence() for risk in self.notable()[:limit]]

    def view(self, horizons: Sequence[str] | None = None) -> dict[str, Any]:
        """The tool's answer: semantic probabilities, levels, trends. No internals."""
        wanted = [str(label) for label in horizons] if horizons else list(self.policy.horizons)
        probabilities: dict[str, dict[str, float]] = {}
        levels: dict[str, dict[str, str]] = {}
        trends: dict[str, dict[str, str]] = {}
        for event, per_horizon in self.risks.items():
            for horizon, risk in per_horizon.items():
                if horizon not in wanted:
                    continue
                probabilities.setdefault(event, {})[horizon] = round(risk.probability, 4)
                levels.setdefault(event, {})[horizon] = risk.level
                trends.setdefault(event, {})[horizon] = risk.trend
        return {
            "status": self.status,
            "reason": self.reason,
            "observed_clock": self.updated_clock,
            "horizons": [label for label in wanted if any(label in per for per in probabilities.values())],
            "probabilities": probabilities,
            "levels": levels,
            "trends": trends,
            "attention": [risk.to_dict() for risk in self.concerns()],
            "note": (
                "Probabilities, not certainties, and they depend on the horizon. "
                "Confirm with telemetry and diagnostics before changing strategy."
            ),
        }

    def snapshot(self) -> dict[str, Any]:
        """What the control room shows: status, the table, the trends, the policy."""
        return {
            "status": self.status,
            "available": self.available,
            "reason": self.reason,
            "source": self.source,
            "updated_clock": self.updated_clock,
            "updated_time_s": self.updated_time_s,
            "forecasts": self.forecasts,
            "events": list(EVENT_NAMES),
            "horizons": list(self.policy.horizons),
            "risks": [risk.to_dict() for risk in self.entries()],
            "attention": [risk.to_dict() for risk in self.concerns()],
            "refresh_s": self.policy.refresh_s,
            "thresholds": {
                name: self.policy.thresholds(name, self.policy.horizons[-1]).to_dict()
                for name in EVENT_NAMES
            },
        }

    def reset(self) -> None:
        """Forget the forecast history. Called when the plant itself is reset."""
        self.status = service.UNAVAILABLE
        self.reason = None
        self.source = service.SOURCE_CHECKPOINT
        self.updated_time_s = None
        self.updated_clock = None
        self.forecasts = 0
        self.risks = {}


#: Predicted risk sits below a plant that is already in trouble (a critical event
#: is 90, plant CRITICAL 80, controller health 70) and above a production drop:
#: foresight is worth acting on, but not before the present.
_PRIORITY = {CRITICAL: 75, HIGH: 55, ELEVATED: 35}


def _urgency(risk: EventRisk) -> tuple[int, float, float]:
    """Level first, then probability, then the shorter horizon of two equals."""
    return (LEVEL_RANK.get(risk.level, 0), risk.probability, -_hours(risk.horizon))


def _hours(horizon: str) -> float:
    try:
        return float(horizon.rstrip("h"))
    except ValueError:
        return 0.0


__all__ = [
    "CRITICAL",
    "DEFAULT_HORIZONS",
    "DEFAULT_THRESHOLDS",
    "ELEVATED",
    "EventRisk",
    "FALLING",
    "HIGH",
    "LEVELS",
    "LOW",
    "RISING",
    "REFRESH_S",
    "RiskMonitor",
    "RiskPolicy",
    "STABLE",
    "Thresholds",
    "replace_refresh",
]
