"""The simulator version used in evidence matches installed package metadata."""

from __future__ import annotations

from importlib.metadata import version

import plant_sim


def test_module_and_distribution_versions_match() -> None:
    assert version("plant-sim") == plant_sim.__version__
