"""Evaluation: the heuristic baseline and an RL checkpoint, measured identically.

    uv run python -m plant_backend.rl.evaluation --seeds 1 2 3

No trained policy exists in this repository yet, so the command reports the
baseline and labels anything else a smoke test until a real checkpoint is supplied.
"""

from .compare import Comparison, RunResult, compare, run_controller, run_policy, write

__all__ = ["Comparison", "RunResult", "compare", "run_controller", "run_policy", "write"]
