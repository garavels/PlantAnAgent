"""Immutable, episode-relative exogenous forcing tapes.

A solar tape stores one factor at every episode boundary.  For ``N`` physics
intervals it therefore contains ``N + 1`` factors: boundary zero describes the
initial state, while interval ``[k, k + 1)`` consumes the factor at boundary
``k + 1``.  Integer indexing keeps replay independent of the scenario's
absolute clock and avoids floating-point time lookup.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

SOLAR_FORCING_VERSION = "boundary-end-factor-v1"


def _index(value: object, *, name: str) -> int:
    """Validate an episode-relative index without treating ``True`` as one."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    index = int(value)
    if index < 0:
        raise ValueError(f"{name} must be non-negative")
    return index


def _finite_real(value: object, *, name: str) -> float:
    """Return a finite real number, rejecting booleans and coercible strings."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True, slots=True)
class SolarFactorTape:
    """A finite sequence of exogenous solar-resource factors.

    ``boundary_factors`` contains boundary samples in ``[0, 1]``.  The tape is
    copied to a tuple during construction, so callers cannot mutate a running
    episode by retaining a reference to the input collection.  A single
    boundary sample is valid and represents a zero-interval tape; attempting
    to step it fails loudly as exhaustion.
    """

    dt_s: float
    boundary_factors: tuple[float, ...]

    def __post_init__(self) -> None:
        dt_s = _finite_real(self.dt_s, name="dt_s")
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")

        if isinstance(self.boundary_factors, (str, bytes)):
            raise TypeError("boundary_factors must be an iterable of real numbers")
        try:
            raw_factors = iter(self.boundary_factors)
        except TypeError as exc:
            raise TypeError(
                "boundary_factors must be an iterable of real numbers"
            ) from exc
        factors = tuple(
            _finite_real(value, name=f"boundary_factors[{index}]")
            for index, value in enumerate(raw_factors)
        )
        if not factors:
            raise ValueError("boundary_factors must contain at least boundary zero")
        for index, factor in enumerate(factors):
            if not 0.0 <= factor <= 1.0:
                raise ValueError(f"boundary_factors[{index}] must be between 0 and 1")

        object.__setattr__(self, "dt_s", dt_s)
        object.__setattr__(self, "boundary_factors", factors)

    @property
    def boundary_count(self) -> int:
        """Number of boundary samples stored by this tape."""
        return len(self.boundary_factors)

    @property
    def interval_count(self) -> int:
        """Number of complete physics intervals covered by this tape."""
        return len(self.boundary_factors) - 1

    def at_boundary(self, step_index: int) -> float:
        """Return the factor at one episode boundary."""
        boundary = _index(step_index, name="step_index")
        if boundary >= self.boundary_count:
            raise IndexError(
                f"solar factor tape exhausted at boundary {boundary}; "
                f"last available boundary is {self.boundary_count - 1}"
            )
        return self.boundary_factors[boundary]

    def for_interval(self, step_index: int) -> float:
        """Return the end-boundary factor used by interval ``[k, k + 1)``."""
        interval = _index(step_index, name="step_index")
        if interval >= self.interval_count:
            raise IndexError(
                f"solar factor tape exhausted before interval "
                f"[{interval}, {interval + 1}); it covers {self.interval_count} interval(s)"
            )
        return self.boundary_factors[interval + 1]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible replay representation."""
        return {"dt_s": self.dt_s, "boundary_factors": list(self.boundary_factors)}


__all__ = ["SOLAR_FORCING_VERSION", "SolarFactorTape"]
