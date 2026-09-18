"""Run app and frozen-experiment tests in separate Python interpreters."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# These suites import the immutable experimental plant. Loading them together
# with the current app would mix two different versions of plant_backend.
FROZEN = (
    "test_comparison_ppo_baselines.py", "test_comparison_ppo_environment.py",
    "test_comparison_ppo_evaluation.py", "test_forecast_ablation.py",
    "test_free_ppo_training.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("app", "research"))
    args, extra = parser.parse_known_args()
    if args.suite == "app":
        if not (ROOT / ".git").exists():
            parser.error("The full backend suite includes Git-provenance checks; use a Git clone. "
                         "The app and reproduction runner also work from a source ZIP.")
        paths = ["simulator/tests", *[
            str(p.relative_to(ROOT)) for p in sorted((ROOT / "backend/tests").glob("test_*.py"))
            if p.name not in FROZEN
        ]]
    else:
        paths = ["tests", *["backend/tests/" + name for name in FROZEN]]
    if args.suite == "research":
        bootstrap = (
            "from docs.research.tools.free_ppo_v1 import environment; "
            "import pytest, sys; raise SystemExit(pytest.main(sys.argv[1:]))"
        )
        command = [sys.executable, "-c", bootstrap]
    else:
        command = [sys.executable, "-m", "pytest"]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1"}
    subprocess.run([*command, "-q", *paths, *extra], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
