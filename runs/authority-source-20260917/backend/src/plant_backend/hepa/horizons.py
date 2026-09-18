"""Horizons: the hours an operator asks about, in the steps a model counts in.

Two clocks meet here and the bugs live in the gap between them:

* the plant runs at `PlantConfig.dt_s` (60 s by default);
* HEPA sees a *resampled* stream, one sample every `sample_period_s` (300 s by
  default), because a 24 h horizon at 60 s would be 1440 steps -- far past the
  paper's largest horizon (K = 200) and needlessly long for a signal that is
  smooth over minutes. Resampling keeps the deepest horizon at K = 288 samples,
  within the regime the architecture was studied in. This is a deliberate
  deviation and is recorded in every checkpoint.

So one grid owns the whole conversion:

    grid = HorizonGrid.of_hours((1, 3, 6, 12, 24), sample_period_s=300.0)
    grid.steps        # (12, 36, 72, 144, 288)
    grid.max_steps    # 288 -- the paper's K, and the predictor's horizon range
    grid.labels       # ('1h', '3h', '6h', '12h', '24h') -- what the API returns

The labels are the only horizon names that ever reach a user; the steps are the
only horizon numbers that ever reach a tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

#: Hours the plant is asked about by default.
DEFAULT_HOURS: tuple[float, ...] = (1.0, 3.0, 6.0, 12.0, 24.0)

#: One HEPA sample every five simulated minutes (five 60 s simulator steps).
DEFAULT_SAMPLE_PERIOD_S = 300.0

SECONDS_PER_HOUR = 3600.0


def _label(hours: float) -> str:
    """`1h`, `12h`, `30m` -- short, stable, and safe as a JSON key."""
    if hours >= 1.0 and abs(hours - round(hours)) < 1e-9:
        return f"{int(round(hours))}h"
    if hours < 1.0:
        return f"{int(round(hours * 60.0))}m"
    return f"{hours:g}h"


@dataclass(frozen=True)
class HorizonGrid:
    """The horizons a model was trained for, in hours and in samples."""

    hours: tuple[float, ...]
    sample_period_s: float = DEFAULT_SAMPLE_PERIOD_S

    def __post_init__(self) -> None:
        if not self.hours:
            raise ValueError("a horizon grid needs at least one horizon")
        if self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be positive")
        if any(hours <= 0.0 for hours in self.hours):
            raise ValueError("horizons must be positive")
        if list(self.hours) != sorted(self.hours):
            raise ValueError("horizons must be given in increasing order")
        if len(set(self.steps)) != len(self.steps):
            raise ValueError(
                "two horizons collapse onto the same step count at "
                f"sample_period_s={self.sample_period_s:g}: {self.hours}"
            )

    @classmethod
    def of_hours(
        cls,
        hours: Sequence[float] = DEFAULT_HOURS,
        sample_period_s: float = DEFAULT_SAMPLE_PERIOD_S,
    ) -> "HorizonGrid":
        return cls(hours=tuple(float(value) for value in hours), sample_period_s=sample_period_s)

    @property
    def steps(self) -> tuple[int, ...]:
        """Each horizon as a whole number of HEPA samples, at least one."""
        return tuple(
            max(1, int(round(hours * SECONDS_PER_HOUR / self.sample_period_s)))
            for hours in self.hours
        )

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(_label(hours) for hours in self.hours)

    @property
    def max_steps(self) -> int:
        """K in the paper: the deepest horizon the predictor is conditioned on."""
        return max(self.steps)

    def stride(self, dt_s: float) -> int:
        """How many simulator steps make one HEPA sample. Must divide evenly."""
        ratio = self.sample_period_s / dt_s
        stride = int(round(ratio))
        if stride < 1 or abs(ratio - stride) > 1e-9:
            raise ValueError(
                f"sample_period_s={self.sample_period_s:g} is not a whole multiple "
                f"of the simulator timestep dt_s={dt_s:g}"
            )
        return stride

    def steps_for(self, label_or_hours: str | float) -> int:
        """Resolve `'6h'` or `6.0` to a step count. Unknown horizons raise."""
        for hours, label, steps in zip(self.hours, self.labels, self.steps):
            if label_or_hours == label or (
                isinstance(label_or_hours, (int, float)) and abs(hours - label_or_hours) < 1e-9
            ):
                return steps
        raise KeyError(
            f"horizon {label_or_hours!r} is not in this grid ({', '.join(self.labels)})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hours": list(self.hours),
            "sample_period_s": self.sample_period_s,
            "labels": list(self.labels),
            "steps": list(self.steps),
            "max_steps": self.max_steps,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "HorizonGrid":
        if not raw:
            return cls.of_hours()
        return cls.of_hours(
            hours=[float(value) for value in raw.get("hours", DEFAULT_HOURS)],
            sample_period_s=float(raw.get("sample_period_s", DEFAULT_SAMPLE_PERIOD_S)),
        )


__all__ = [
    "DEFAULT_HOURS",
    "DEFAULT_SAMPLE_PERIOD_S",
    "HorizonGrid",
]
