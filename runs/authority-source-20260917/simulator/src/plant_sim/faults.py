"""Faults and disturbances: how the plant is allowed to misbehave.

A fault never changes the simulator's control flow. It resolves into a small
`ComponentEffect` -- a derating and/or an outage -- that the affected component
applies to its own physics through `Component.apply_effect()`. Adding a fault
type therefore means adding one `FaultKind` to `CATALOGUE`, not a special case
in the dispatch loop.

    registry = FaultRegistry()
    registry.inject(FaultSpec(type="electrolyser_degradation", severity=0.4), now=0.0)
    electrolyser.apply_effect(registry.effect_for("electrolyser"))

Sensor faults are the one exception: they leave physics untouched and are
applied to the *observed* telemetry instead (see `telemetry.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal

#: How a fault reaches its target.
FaultMode = Literal["degradation", "outage", "sensor"]

#: The most a fault is allowed to derate a component; keeps physics finite.
MAX_DEGRADATION = 0.95


@dataclass(frozen=True)
class FaultKind:
    """One injectable fault type: what it hits, and how."""

    type: str
    label: str
    #: Component name the effect lands on, or `sensors` for observation faults.
    target: str
    mode: FaultMode
    description: str
    default_severity: float = 0.4
    #: Sensor faults only: which telemetry channel is biased by default.
    default_channel: str | None = None


CATALOGUE: dict[str, FaultKind] = {
    kind.type: kind
    for kind in (
        FaultKind(
            type="electrolyser_degradation",
            label="Electrolyser degradation",
            target="electrolyser",
            mode="degradation",
            description="The stack makes less H2 from the same power, and runs hotter.",
        ),
        FaultKind(
            type="dac_degradation",
            label="DAC degradation",
            target="dac",
            mode="degradation",
            description="Sorbent performance falls: less CO2 captured per kWh.",
        ),
        FaultKind(
            type="dac_outage",
            label="DAC outage",
            target="dac",
            mode="outage",
            description="The capture unit is unavailable: no new CO2 at all.",
            default_severity=1.0,
        ),
        FaultKind(
            type="battery_degradation",
            label="Battery capacity fade",
            target="battery",
            mode="degradation",
            description="Effective capacity shrinks; stored energy is clamped to it.",
            default_severity=0.3,
        ),
        FaultKind(
            type="methanator_outage",
            label="Methanator outage",
            target="methanator",
            mode="outage",
            description="The reactor trips: no CH4, and no H2/CO2 consumed.",
            default_severity=1.0,
        ),
        FaultKind(
            type="solar_drop",
            label="Solar generation drop",
            target="solar",
            mode="degradation",
            description="A disturbance, not a fault: cloud cover cuts generation.",
            default_severity=0.6,
        ),
        FaultKind(
            type="sensor_drift",
            label="Sensor drift",
            target="sensors",
            mode="sensor",
            description="A level transmitter over-reports: telemetry stops matching reality.",
            default_severity=0.17,
            default_channel="h2_level_kg",
        ),
    )
}

#: Ordered for menus: process faults first, then the disturbance, then sensors.
FAULT_TYPES = tuple(CATALOGUE)


@dataclass(frozen=True)
class ComponentEffect:
    """What a component should do differently this timestep."""

    degradation: float = 0.0
    offline: bool = False

    @property
    def health(self) -> float:
        """Fraction of nominal performance left, 0..1."""
        return 0.0 if self.offline else 1.0 - self.degradation

    @property
    def faulted(self) -> bool:
        return self.offline or self.degradation > 0.0


NO_EFFECT = ComponentEffect()


@dataclass(frozen=True)
class SensorBias:
    """A biased observation channel: `reported = true + severity * full scale`."""

    channel: str
    severity: float


@dataclass(frozen=True)
class FaultSpec:
    """A fault as requested: type, how bad, when it starts, how long it lasts.

    `start_s` and `duration_s` are simulator seconds. `start_s=None` means "as
    soon as it is injected"; `duration_s=None` means "until cleared".
    """

    type: str
    severity: float = 0.4
    start_s: float | None = None
    duration_s: float | None = None
    #: Sensor faults only: the telemetry channel to bias.
    channel: str | None = None

    @classmethod
    def parse(cls, raw: "FaultSpec | dict[str, Any]") -> "FaultSpec":
        """Build a spec from user input, refusing unknown types."""
        if isinstance(raw, FaultSpec):
            spec = raw
        else:
            fault_type = str(raw.get("type", ""))
            if fault_type not in CATALOGUE:
                raise ValueError(f"unknown fault type: {fault_type!r}")
            spec = cls(
                type=fault_type,
                severity=_number(raw.get("severity"), CATALOGUE[fault_type].default_severity),
                start_s=_optional_number(raw.get("start_s")),
                duration_s=_optional_number(raw.get("duration_s")),
                channel=raw.get("channel"),
            )
        if spec.type not in CATALOGUE:
            raise ValueError(f"unknown fault type: {spec.type!r}")
        kind = CATALOGUE[spec.type]
        severity = _clamp(spec.severity, 0.0, 1.0)
        if kind.mode == "outage":
            severity = 1.0
        elif kind.mode == "degradation":
            severity = min(severity, MAX_DEGRADATION)
        return replace(
            spec,
            severity=severity,
            channel=spec.channel or kind.default_channel,
            duration_s=None if spec.duration_s is None else max(0.0, spec.duration_s),
        )

    @property
    def kind(self) -> FaultKind:
        return CATALOGUE[self.type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "start_s": self.start_s,
            "duration_s": self.duration_s,
            "channel": self.channel,
        }


@dataclass
class Fault:
    """A scheduled fault with a runtime lifecycle: pending -> active -> cleared."""

    id: str
    spec: FaultSpec
    start_s: float
    active: bool = False
    cleared: bool = False

    @property
    def end_s(self) -> float | None:
        return None if self.spec.duration_s is None else self.start_s + self.spec.duration_s

    def should_run(self, now_s: float) -> bool:
        if self.cleared or now_s < self.start_s:
            return False
        end = self.end_s
        return end is None or now_s < end

    def to_dict(self) -> dict[str, Any]:
        kind = self.spec.kind
        return {
            "id": self.id,
            "type": self.spec.type,
            "label": kind.label,
            "target": kind.target,
            "mode": kind.mode,
            "severity": self.spec.severity,
            "channel": self.spec.channel,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.spec.duration_s,
            "active": self.active,
            "cleared": self.cleared,
        }


class FaultRegistry:
    """Every fault the plant has been given, and what they currently imply.

    The registry is pure bookkeeping: it answers "what is active right now" and
    "what does that mean for component X". It never touches components itself.
    """

    def __init__(self) -> None:
        self._faults: list[Fault] = []
        self._sequence = 0

    # --- lifecycle --------------------------------------------------------
    def inject(self, spec: FaultSpec | dict[str, Any], *, now_s: float = 0.0) -> Fault:
        """Schedule a fault. Without a `start_s` it begins immediately."""
        parsed = FaultSpec.parse(spec)
        self._sequence += 1
        fault = Fault(
            id=f"f{self._sequence}",
            spec=parsed,
            start_s=now_s if parsed.start_s is None else parsed.start_s,
        )
        self._faults.append(fault)
        return fault

    def clear(self, fault_id: str) -> bool:
        """Deactivate one fault; returns whether it existed."""
        for fault in self._faults:
            if fault.id == fault_id and not fault.cleared:
                fault.cleared = True
                fault.active = False
                return True
        return False

    def clear_all(self) -> None:
        self._faults.clear()
        self._sequence = 0

    def update(self, now_s: float) -> list[tuple[str, Fault]]:
        """Advance the lifecycle to `now_s`; returns `(transition, fault)` pairs."""
        transitions: list[tuple[str, Fault]] = []
        for fault in self._faults:
            running = fault.should_run(now_s)
            if running and not fault.active:
                fault.active = True
                transitions.append(("activated", fault))
            elif not running and fault.active:
                fault.active = False
                if fault.end_s is not None and now_s >= fault.end_s:
                    fault.cleared = True
                transitions.append(("ended", fault))
        return transitions

    # --- read-only views --------------------------------------------------
    @property
    def faults(self) -> tuple[Fault, ...]:
        return tuple(self._faults)

    def active(self) -> tuple[Fault, ...]:
        return tuple(fault for fault in self._faults if fault.active)

    def pending(self) -> tuple[Fault, ...]:
        return tuple(f for f in self._faults if not f.active and not f.cleared)

    def effect_for(self, component: str) -> ComponentEffect:
        """Combine every active fault aimed at `component` (worst one wins)."""
        return self._combined_effect(self.active(), component)

    def effect_for_at(self, component: str, time_s: float) -> ComponentEffect:
        """Project an effect at ``time_s`` without advancing fault lifecycle.

        This read-only view is used by the actuator guard to inspect the same
        solar health that the next physical step will apply, including a fault
        that begins or ends exactly on the current boundary.
        """
        return self._combined_effect(
            (fault for fault in self._faults if fault.should_run(time_s)),
            component,
        )

    @staticmethod
    def _combined_effect(faults: Iterable[Fault], component: str) -> ComponentEffect:
        """Combine one fault view for ``component`` (the worst effect wins)."""
        degradation = 0.0
        offline = False
        for fault in faults:
            kind = fault.spec.kind
            if kind.target != component:
                continue
            if kind.mode == "outage":
                offline = True
            elif kind.mode == "degradation":
                degradation = max(degradation, fault.spec.severity)
        return ComponentEffect(degradation=degradation, offline=offline)

    def sensor_biases(self) -> tuple[SensorBias, ...]:
        """Active observation faults, as channel biases for `telemetry.observe`."""
        return tuple(
            SensorBias(channel=fault.spec.channel or "", severity=fault.spec.severity)
            for fault in self.active()
            if fault.spec.kind.mode == "sensor" and fault.spec.channel
        )

    def to_list(self) -> list[dict[str, Any]]:
        return [fault.to_dict() for fault in self._faults if not fault.cleared]


def catalogue() -> list[dict[str, Any]]:
    """The injectable fault types, for menus and API discovery."""
    return [
        {
            "type": kind.type,
            "label": kind.label,
            "target": kind.target,
            "mode": kind.mode,
            "description": kind.description,
            "default_severity": kind.default_severity,
            "default_channel": kind.default_channel,
        }
        for kind in CATALOGUE.values()
    ]


def specs(raw: Iterable[FaultSpec | dict[str, Any]]) -> tuple[FaultSpec, ...]:
    return tuple(FaultSpec.parse(item) for item in raw)


def _clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def _number(value: Any, fallback: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "CATALOGUE",
    "ComponentEffect",
    "FAULT_TYPES",
    "Fault",
    "FaultKind",
    "FaultRegistry",
    "FaultSpec",
    "MAX_DEGRADATION",
    "NO_EFFECT",
    "SensorBias",
    "catalogue",
    "specs",
]
