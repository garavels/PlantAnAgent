"""Tests never talk to a model.

The supervisor picks its engine from the environment, so a developer with
credentials exported would otherwise get network calls, cost and a different
result than CI for the same commit. Hiding the key makes every run take the
deterministic `RuleEngine` path -- which is also the fallback the plant relies
on in production, so it is the path most worth testing.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from plant_backend.agent.supervisor import API_KEY_ENV


@pytest.fixture(autouse=True, scope="session")
def _no_model_credentials() -> Iterator[None]:
    patch = pytest.MonkeyPatch()
    patch.delenv(API_KEY_ENV, raising=False)
    yield
    patch.undo()
