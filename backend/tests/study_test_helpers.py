"""Numerical study tests cannot construct a plant, trainer or admitted policy."""

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def forbid_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
    from plant_backend.rl.v2.training import ppo
    from plant_backend.rl import policies

    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail(
            "numerical analysis attempted environment, policy or learner access"
        )

    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "__init__", forbidden)
    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "reset", forbidden)
    monkeypatch.setattr(ppo, "make_env", forbidden)
    monkeypatch.setattr(ppo, "build_vec_env", forbidden)
    monkeypatch.setattr(ppo, "build_model", forbidden)
    monkeypatch.setattr(policies, "load_policy", forbidden)
