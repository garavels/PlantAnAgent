"""Interface guard in front of the simulator. Not the full safety system yet."""

from .validation import ValidatedAction, validate

__all__ = ["ValidatedAction", "validate"]
