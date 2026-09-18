"""Interface guard in front of the simulator. Not the full safety system yet."""

from .validation import (
    BatterySupportAvailability,
    ValidatedAction,
    battery_support_availability,
    validate,
)

__all__ = [
    "BatterySupportAvailability",
    "ValidatedAction",
    "battery_support_availability",
    "validate",
]
