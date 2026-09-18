"""Private, seeded scheduled disturbances for Environment V2 beta.

The beta profile is deliberately small and explicit: every 24-hour cycle owns
an independent random stream and may contain at most one finite DAC outage.
The returned :class:`DisturbanceEpisodeV1` is reset-time data.  Environments
may install its ``faults`` directly as ``Scenario.faults``; they must not make
the object, its future schedule, or its audit manifest part of an observation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np
from plant_sim import FaultSpec


DISTURBANCE_GENERATOR_VERSION = "scheduled-dac-outage-v1"
DISTURBANCE_STREAM_NAME = "disturbances"
SUPPORTED_CYCLE_H = 24.0
SECONDS_PER_HOUR = 3_600.0

SeedSource: TypeAlias = int | np.random.SeedSequence


def _json_ready(value: object) -> object:
    """Detach NumPy values into the strict JSON domain used for evidence."""
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
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a SHA-256 digest") from exc
    return value


def _finite_real(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _aligned_ticks(
    seconds: float,
    dt_s: float,
    *,
    name: str,
    allow_zero: bool = False,
) -> int:
    ticks = int(round(seconds / dt_s))
    lower_bound = 0 if allow_zero else 1
    if ticks < lower_bound or not math.isclose(
        seconds,
        ticks * dt_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{name} must align to the {dt_s:g} s physics timestep")
    # FaultSpec stores float seconds, so retain an integer lattice that can be
    # represented exactly before converting generated timestamps back to float.
    if ticks > 2**53:
        raise ValueError(f"{name} contains too many physics timesteps")
    return ticks


def _fault_to_dict(fault: FaultSpec) -> dict[str, object]:
    """Stable schedule serialization independent of numeric input subclasses."""
    return {
        "type": fault.type,
        "severity": float(fault.severity),
        "start_s": None if fault.start_s is None else float(fault.start_s),
        "duration_s": (None if fault.duration_s is None else float(fault.duration_s)),
        "channel": fault.channel,
    }


def _schedule_sha256(faults: tuple[FaultSpec, ...]) -> str:
    return _canonical_sha256([_fault_to_dict(fault) for fault in faults])


@dataclass(frozen=True, slots=True)
class _SeedStreamIdentityV1:
    """Canonical initial SeedSequence state retained only for private audit."""

    name: str
    source_kind: str
    spawn_key: tuple[int, ...]
    state_sha256: str
    initial_state_json: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("random stream name must not be empty")
        if not isinstance(self.source_kind, str) or not self.source_kind:
            raise ValueError("random stream source kind must not be empty")
        if not isinstance(self.spawn_key, tuple) or any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in self.spawn_key
        ):
            raise ValueError(
                "random stream spawn key must contain non-negative integers"
            )
        try:
            initial_state = json.loads(self.initial_state_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "random stream initial state must be canonical JSON"
            ) from exc
        if _canonical_json(initial_state) != self.initial_state_json:
            raise ValueError(
                "random stream initial state must use canonical JSON encoding"
            )
        _require_sha256("random stream state_sha256", self.state_sha256)
        if _canonical_sha256(initial_state) != self.state_sha256:
            raise ValueError("random stream digest does not match its initial state")

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
) -> _SeedStreamIdentityV1:
    state = _json_ready(deepcopy(sequence.state))
    initial_state_json = _canonical_json(state)
    return _SeedStreamIdentityV1(
        name=name,
        source_kind="numpy.random.SeedSequence",
        spawn_key=tuple(int(value) for value in sequence.spawn_key),
        state_sha256=_canonical_sha256(state),
        initial_state_json=initial_state_json,
    )


def _copy_seed_sequence(seed: SeedSource) -> np.random.SeedSequence:
    if isinstance(seed, bool):
        raise ValueError("seed must be a non-negative integer or numpy SeedSequence")
    if isinstance(seed, Integral):
        if seed < 0:
            raise ValueError("seed must be a non-negative integer")
        return np.random.SeedSequence(int(seed))
    if isinstance(seed, np.random.SeedSequence):
        # SeedSequence.spawn mutates n_children_spawned.  Clone its complete
        # descriptor so a caller can safely reuse the same stream at reset.
        return np.random.SeedSequence(**deepcopy(seed.state))
    raise TypeError("seed must be a non-negative integer or numpy SeedSequence")


@dataclass(frozen=True, slots=True)
class DisturbanceConfigV1:
    """Version-one cyclic DAC-outage profile for Environment V2 beta."""

    enabled: bool = True
    per_cycle_probability: float = 0.5
    duration_h: tuple[float, float] = (1.0, 4.0)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> DisturbanceConfigV1:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        probability = _finite_real(
            "per_cycle_probability",
            self.per_cycle_probability,
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError("per_cycle_probability must lie within [0, 1]")
        if not isinstance(self.duration_h, tuple) or len(self.duration_h) != 2:
            raise TypeError("duration_h must be a (minimum, maximum) tuple")
        low = _finite_real("duration_h minimum", self.duration_h[0], positive=True)
        high = _finite_real("duration_h maximum", self.duration_h[1], positive=True)
        if low > high:
            raise ValueError("duration_h must be an ordered positive pair")
        return self

    def validate_geometry(
        self,
        *,
        cycle_h: float,
        cycle_count: int,
        physics_dt_s: float,
    ) -> DisturbanceConfigV1:
        """Fail before reset when the configured schedule cannot fit the task grid."""

        _validated_geometry(
            self,
            cycle_h=cycle_h,
            cycle_count=cycle_count,
            physics_dt_s=physics_dt_s,
        )
        return self

    @property
    def config_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "enabled": self.enabled,
            "per_cycle_probability": float(self.per_cycle_probability),
            "duration_h": tuple(float(value) for value in self.duration_h),
        }


def _validated_geometry(
    config: DisturbanceConfigV1,
    *,
    cycle_h: object,
    cycle_count: object,
    physics_dt_s: object,
) -> tuple[float, int, float, int, int, int]:
    config.validate()
    normalized_cycle_h = _finite_real("cycle_h", cycle_h, positive=True)
    if not math.isclose(
        normalized_cycle_h,
        SUPPORTED_CYCLE_H,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"cycle_h must be exactly {SUPPORTED_CYCLE_H:g} for this generator version"
        )
    normalized_cycle_count = _positive_integer("cycle_count", cycle_count)
    normalized_dt_s = _finite_real("physics_dt_s", physics_dt_s, positive=True)
    cycle_steps = _aligned_ticks(
        normalized_cycle_h * SECONDS_PER_HOUR,
        normalized_dt_s,
        name="cycle_h",
    )
    minimum_duration_steps = _aligned_ticks(
        float(config.duration_h[0]) * SECONDS_PER_HOUR,
        normalized_dt_s,
        name="duration_h minimum",
    )
    maximum_duration_steps = _aligned_ticks(
        float(config.duration_h[1]) * SECONDS_PER_HOUR,
        normalized_dt_s,
        name="duration_h maximum",
    )
    if maximum_duration_steps >= cycle_steps:
        raise ValueError(
            "duration_h maximum must leave one physics timestep before the "
            "24 h cycle boundary"
        )
    if normalized_cycle_count * cycle_steps > 2**53:
        raise ValueError("episode contains too many physics timesteps")
    return (
        normalized_cycle_h,
        normalized_cycle_count,
        normalized_dt_s,
        cycle_steps,
        minimum_duration_steps,
        maximum_duration_steps,
    )


def _validate_fault_schedule(
    config: DisturbanceConfigV1,
    faults: tuple[FaultSpec, ...],
    *,
    cycle_count: int,
    cycle_steps: int,
    physics_dt_s: float,
) -> None:
    if not isinstance(faults, tuple):
        raise TypeError("faults must be an immutable tuple")
    if not config.enabled and faults:
        raise ValueError("a disabled disturbance profile must have an empty schedule")
    if config.per_cycle_probability == 0.0 and faults:
        raise ValueError(
            "a zero-probability disturbance profile must have an empty schedule"
        )
    if config.enabled and config.per_cycle_probability == 1.0:
        if len(faults) != cycle_count:
            raise ValueError(
                "a probability-one profile must contain one outage per cycle"
            )
    if len(faults) > cycle_count:
        raise ValueError(
            "a disturbance schedule may contain at most one outage per cycle"
        )

    occupied_cycles: set[int] = set()
    previous_start_step = -1
    for index, fault in enumerate(faults):
        if not isinstance(fault, FaultSpec):
            raise TypeError(f"faults[{index}] must be a plant_sim.FaultSpec")
        if fault.type != "dac_outage":
            raise ValueError(
                "the version-one disturbance profile permits only dac_outage"
            )
        if FaultSpec.parse(fault) != fault:
            raise ValueError(
                "disturbance FaultSpec values must use normalized semantics"
            )
        severity = _finite_real(f"faults[{index}].severity", fault.severity)
        if severity != 1.0 or fault.channel is not None:
            raise ValueError("DAC outages must have severity 1 and no sensor channel")
        if fault.start_s is None:
            raise ValueError("scheduled disturbances require an explicit start_s")
        if fault.duration_s is None:
            raise ValueError("scheduled disturbances require a finite duration_s")
        start_s = _finite_real(f"faults[{index}].start_s", fault.start_s)
        duration_s = _finite_real(
            f"faults[{index}].duration_s",
            fault.duration_s,
            positive=True,
        )
        if start_s < 0.0 or not math.isfinite(start_s + duration_s):
            raise ValueError(
                "scheduled disturbance times must be finite and non-negative"
            )
        start_step = _aligned_ticks(
            start_s,
            physics_dt_s,
            name=f"faults[{index}].start_s",
            allow_zero=True,
        )
        duration_step_count = _aligned_ticks(
            duration_s,
            physics_dt_s,
            name=f"faults[{index}].duration_s",
        )
        if start_step <= previous_start_step:
            raise ValueError("disturbance faults must be ordered by start_s")
        previous_start_step = start_step
        cycle_index = start_step // cycle_steps
        if cycle_index >= cycle_count:
            raise ValueError("disturbance start_s lies outside the episode")
        if cycle_index in occupied_cycles:
            raise ValueError("a cycle may contain at most one disturbance")
        occupied_cycles.add(cycle_index)
        cycle_end_step = (cycle_index + 1) * cycle_steps
        # PlantSimulator refreshes fault lifecycle immediately before a physics
        # dispatch, while boundary telemetry describes the interval just ended.
        # Leaving one final healthy dispatch inside the cycle prevents stale
        # end-boundary health from suppressing DAC in the following cycle.
        if start_step + duration_step_count > cycle_end_step - 1:
            raise ValueError(
                "a disturbance must end at least one physics timestep before "
                "its cycle boundary"
            )


def _episode_identity_payload(
    *,
    config_sha256: str,
    schedule_sha256: str,
    cycle_h: float,
    cycle_count: int,
    physics_dt_s: float,
    seed_stream: _SeedStreamIdentityV1,
    cycle_streams: tuple[_SeedStreamIdentityV1, ...],
) -> dict[str, object]:
    return {
        "disturbance_generator_version": DISTURBANCE_GENERATOR_VERSION,
        "config_sha256": config_sha256,
        "schedule_sha256": schedule_sha256,
        "cycle_h": cycle_h,
        "cycle_count": cycle_count,
        "physics_dt_s": physics_dt_s,
        "seed_stream": seed_stream.to_dict(),
        "cycle_streams": [stream.to_dict() for stream in cycle_streams],
    }


@dataclass(frozen=True, slots=True)
class DisturbanceEpisodeV1:
    """Immutable private schedule used while constructing a reset scenario."""

    config: DisturbanceConfigV1
    faults: tuple[FaultSpec, ...]
    cycle_h: float
    cycle_count: int
    physics_dt_s: float
    config_sha256: str
    schedule_sha256: str
    episode_sha256: str
    _seed_stream: _SeedStreamIdentityV1 = field(repr=False)
    _cycle_streams: tuple[_SeedStreamIdentityV1, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, DisturbanceConfigV1):
            raise TypeError("config must be a DisturbanceConfigV1")
        (
            cycle_h,
            cycle_count,
            physics_dt_s,
            cycle_steps,
            _,
            _,
        ) = _validated_geometry(
            self.config,
            cycle_h=self.cycle_h,
            cycle_count=self.cycle_count,
            physics_dt_s=self.physics_dt_s,
        )
        if not isinstance(self._seed_stream, _SeedStreamIdentityV1):
            raise TypeError("seed stream identity is invalid")
        if self._seed_stream.name != DISTURBANCE_STREAM_NAME:
            raise ValueError("seed stream has the wrong name")
        if (
            not isinstance(self._cycle_streams, tuple)
            or len(self._cycle_streams) != cycle_count
        ):
            raise ValueError("there must be exactly one random stream per cycle")
        for index, stream in enumerate(self._cycle_streams):
            if not isinstance(stream, _SeedStreamIdentityV1):
                raise TypeError("cycle stream identity is invalid")
            expected_name = f"{DISTURBANCE_STREAM_NAME}.cycle-{index:04d}"
            if stream.name != expected_name:
                raise ValueError("cycle random stream has the wrong name")
        _validate_fault_schedule(
            self.config,
            self.faults,
            cycle_count=cycle_count,
            cycle_steps=cycle_steps,
            physics_dt_s=physics_dt_s,
        )

        for name, digest in (
            ("config_sha256", self.config_sha256),
            ("schedule_sha256", self.schedule_sha256),
            ("episode_sha256", self.episode_sha256),
        ):
            _require_sha256(name, digest)
        if self.config_sha256 != self.config.config_sha256:
            raise ValueError("config_sha256 does not match the disturbance config")
        if self.schedule_sha256 != _schedule_sha256(self.faults):
            raise ValueError("schedule_sha256 does not match the fault schedule")
        expected_episode_sha256 = _canonical_sha256(
            _episode_identity_payload(
                config_sha256=self.config_sha256,
                schedule_sha256=self.schedule_sha256,
                cycle_h=cycle_h,
                cycle_count=cycle_count,
                physics_dt_s=physics_dt_s,
                seed_stream=self._seed_stream,
                cycle_streams=self._cycle_streams,
            )
        )
        if self.episode_sha256 != expected_episode_sha256:
            raise ValueError("episode_sha256 does not match the disturbance episode")

    def to_dict(self) -> dict[str, object]:
        """Stable non-schedule summary suitable for reset evidence."""
        return {
            "disturbance_generator_version": DISTURBANCE_GENERATOR_VERSION,
            "config": self.config.to_dict(),
            "cycle_h": float(self.cycle_h),
            "cycle_count": int(self.cycle_count),
            "physics_dt_s": float(self.physics_dt_s),
            "fault_count": len(self.faults),
            "config_sha256": self.config_sha256,
            "schedule_sha256": self.schedule_sha256,
            "episode_sha256": self.episode_sha256,
        }

    def private_audit_manifest(self) -> dict[str, object]:
        """Offline replay data; never expose this object to a controller."""
        return {
            **self.to_dict(),
            "time_convention": {
                "schedule_time_basis": "seconds since episode start",
                "fault_interval": "half-open [start_s, start_s + duration_s)",
                "cycle_interval": "half-open [cycle_start_s, cycle_end_s)",
                "physics_timestep_s": float(self.physics_dt_s),
            },
            "random_streams": {
                "root": self._seed_stream.to_dict(),
                "cycles": [stream.to_dict() for stream in self._cycle_streams],
            },
            "schedule": [_fault_to_dict(fault) for fault in self.faults],
            "private_hashes": {
                "config_sha256": self.config_sha256,
                "schedule_sha256": self.schedule_sha256,
                "episode_sha256": self.episode_sha256,
            },
        }


def generate_disturbance_episode(
    config: DisturbanceConfigV1,
    *,
    seed: SeedSource,
    cycle_h: float,
    cycle_count: int,
    physics_dt_s: float,
) -> DisturbanceEpisodeV1:
    """Draw an independent optional DAC outage for each 24-hour cycle.

    Durations and onsets are sampled uniformly on the caller's physics lattice.
    A duration's upper bound is inclusive.  Every outage leaves the final
    physics dispatch in its cycle healthy so the simulator's boundary-telemetry
    lifecycle cannot carry a stale offline indication into the next cycle.  No
    global RNG or wall clock participates in generation.
    """
    if not isinstance(config, DisturbanceConfigV1):
        raise TypeError("config must be a DisturbanceConfigV1")
    (
        normalized_cycle_h,
        normalized_cycle_count,
        normalized_dt_s,
        cycle_steps,
        minimum_duration_steps,
        maximum_duration_steps,
    ) = _validated_geometry(
        config,
        cycle_h=cycle_h,
        cycle_count=cycle_count,
        physics_dt_s=physics_dt_s,
    )

    root_sequence = _copy_seed_sequence(seed)
    seed_stream = _stream_identity(root_sequence, DISTURBANCE_STREAM_NAME)
    cycle_sequences = tuple(root_sequence.spawn(normalized_cycle_count))
    cycle_streams = tuple(
        _stream_identity(
            sequence,
            f"{DISTURBANCE_STREAM_NAME}.cycle-{index:04d}",
        )
        for index, sequence in enumerate(cycle_sequences)
    )

    faults: list[FaultSpec] = []
    if config.enabled:
        probability = float(config.per_cycle_probability)
        for cycle_index, sequence in enumerate(cycle_sequences):
            rng = np.random.default_rng(sequence)
            if float(rng.random()) >= probability:
                continue
            duration_steps = int(
                rng.integers(
                    minimum_duration_steps,
                    maximum_duration_steps + 1,
                )
            )
            onset_within_cycle = int(rng.integers(0, cycle_steps - duration_steps))
            start_steps = cycle_index * cycle_steps + onset_within_cycle
            faults.append(
                FaultSpec.parse(
                    FaultSpec(
                        type="dac_outage",
                        start_s=float(start_steps * normalized_dt_s),
                        duration_s=float(duration_steps * normalized_dt_s),
                    )
                )
            )

    frozen_faults = tuple(faults)
    config_sha256 = config.config_sha256
    schedule_sha256 = _schedule_sha256(frozen_faults)
    episode_sha256 = _canonical_sha256(
        _episode_identity_payload(
            config_sha256=config_sha256,
            schedule_sha256=schedule_sha256,
            cycle_h=normalized_cycle_h,
            cycle_count=normalized_cycle_count,
            physics_dt_s=normalized_dt_s,
            seed_stream=seed_stream,
            cycle_streams=cycle_streams,
        )
    )
    return DisturbanceEpisodeV1(
        config=config,
        faults=frozen_faults,
        cycle_h=normalized_cycle_h,
        cycle_count=normalized_cycle_count,
        physics_dt_s=normalized_dt_s,
        config_sha256=config_sha256,
        schedule_sha256=schedule_sha256,
        episode_sha256=episode_sha256,
        _seed_stream=seed_stream,
        _cycle_streams=cycle_streams,
    )


__all__ = [
    "DISTURBANCE_GENERATOR_VERSION",
    "DisturbanceConfigV1",
    "DisturbanceEpisodeV1",
    "generate_disturbance_episode",
]
