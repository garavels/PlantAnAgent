"""The submission must remain runnable without historical local output paths."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import reproduce


def test_bundled_controllers_keep_the_published_checkpoint_identities():
    config = reproduce.configuration()
    reference = reproduce.read(reproduce.ROOT / config["reference"]["path"])
    _, evaluate, _ = reproduce.modules()
    actual = [evaluate._controller_spec(c) for c in reproduce.controllers()]
    assert {c["label"] for c in actual} == set(reference["controllers"])
    for item in actual:
        expected = reference["controllers"][item["label"]]
        if item["kind"] == "learned":
            for key, pin in item["checkpoint_files"].items():
                assert pin["sha256"] == expected["checkpoint_files"][key]["sha256"]
        else:
            assert item == expected


def test_reproduction_freezes_full_prescribed_grid_without_running_it(tmp_path):
    _, evaluate, _ = reproduce.modules()
    spec = reproduce.freeze(tmp_path, smoke=False)
    original = reproduce.read(reproduce.ROOT / "experiments/reference-evaluation.json")
    assert spec["cases"] == original["cases"]
    assert spec["primary"] == original["primary"]
    assert spec["expected_cases"] == 32 * 7 * 7
    assert evaluate.require_freeze(tmp_path / "freeze.json") == spec
    gate = tmp_path / "gate.json"
    gate.write_text(gate.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        evaluate.require_freeze(tmp_path / "freeze.json")


def test_existing_output_is_rejected_without_changing_it(tmp_path):
    sentinel = tmp_path / "keep.json"
    sentinel.write_text('{"original": true}\n')
    result = subprocess.run([sys.executable, str(Path(reproduce.__file__)), "smoke",
                             "--output", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "already exists" in result.stderr
    assert json.loads(sentinel.read_text()) == {"original": True}


def test_training_recipes_are_portable_and_preserve_seed_allocation():
    config = reproduce.configuration()
    for name in ("gaussian", "hybrid", "search"):
        for key in ("dataset", "training_dataset", "parent_checkpoint", "initial_checkpoint"):
            if config[name].get(key):
                assert not Path(config[name][key]).is_absolute()
    assert config["optimizer_seeds"] == [5901, 5902, 5903]
    assert config["selected_generation"] == 15
    assert config["search"]["generations"] == 20
