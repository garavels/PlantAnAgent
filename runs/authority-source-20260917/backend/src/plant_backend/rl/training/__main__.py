"""`python -m plant_backend.rl.training` -- the same entry point as `train.py`."""

from .train import main

if __name__ == "__main__":
    raise SystemExit(main())
