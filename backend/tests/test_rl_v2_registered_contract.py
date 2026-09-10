"""Static registered admission tests: no protected reset or optimizer update."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.training.custody_json import canonical_bytes, decode
from plant_backend.rl.v2.training.registered_contract import (
    FAMILY_SLUGS,
    TEMPLATE_SHA256,
    matrix,
    registered_config,
    task_template_digest,
)
from plant_backend.rl.v2.training.train import train


def _config(output: Path) -> ControllerNeutralTrainingConfigV1:
    return registered_config(output, "plant-beta-ppo-pi-lagrangian-v1", 101)


def _leaves(
    value: dict[str, Any], prefix: tuple[str, ...] = ()
) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    for key, child in value.items():
        if isinstance(child, dict):
            result.extend(_leaves(child, (*prefix, key)))
        else:
            result.append((*prefix, key))
    return result


TEMPLATE = _config(Path("/private/tmp/plant-contract-test-output")).to_dict()
IMMUTABLE_LEAVES = [
    path
    for path in _leaves(TEMPLATE)
    if path not in {("seed",), ("output_dir",), ("beta_rl", "algorithm")}
]


@pytest.mark.parametrize("path", IMMUTABLE_LEAVES, ids=lambda path: ".".join(path))
def test_every_frozen_leaf_rejects_mutation(path: tuple[str, ...]) -> None:
    modified = deepcopy(TEMPLATE)
    node = modified
    for key in path[:-1]:
        node = node[key]
    original = node[path[-1]]
    if isinstance(original, bool):
        replacement: Any = not original
    elif isinstance(original, (int, float)):
        replacement = original + 1
    elif isinstance(original, str):
        replacement = original + "-changed"
    elif isinstance(original, list):
        replacement = original + [None]
    else:
        replacement = "changed"
    node[path[-1]] = replacement
    with pytest.raises((ValueError, TypeError)):
        ControllerNeutralTrainingConfigV1.from_dict(modified)


def test_exact_matrix_and_complete_canonical_configs(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "campaign"
    cells = matrix(root)
    assert len(cells) == 15
    assert [(cell["family_slug"], cell["optimizer_seed"]) for cell in cells] == [
        (slug, seed) for slug in FAMILY_SLUGS for seed in (101, 211, 307, 401, 503)
    ]
    assert [cell["order"] for cell in cells] == list(range(1, 16))
    all_paths: list[str] = []
    for cell in cells:
        config = registered_config(
            Path(cell["output_path"]), cell["algorithm"], cell["optimizer_seed"]
        )
        assert task_template_digest(config) == TEMPLATE_SHA256
        raw = canonical_bytes(config.to_dict())
        assert hashlib.sha256(raw).hexdigest() == cell["config_sha256"]
        assert (
            raw
            == (json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n").encode()
        )
        for key, value in cell.items():
            if key.endswith("_path"):
                assert Path(value).is_relative_to(root)
                all_paths.append(value)
        stem = (
            f"{cell['order']:02d}-{cell['family_slug']}-seed-{cell['optimizer_seed']}"
        )
        assert cell["config_path"] == str(root / "configs" / f"{stem}.json")
    assert len(all_paths) == len(set(all_paths))
    assert not root.exists()


@pytest.mark.parametrize(
    "raw",
    [
        '{"beta_rl": null, "beta_rl": {"run_kind": "registered"}}',
        '{"beta_rl": {"run_kind": "registered", "run_kind": "fixture"}}',
        '{"plant": {"solar": {"x": 1, "x": 2}}}',
        '{"x": NaN}',
        '{"x": Infinity}',
        '{"x": -Infinity}',
        '{"x": 1e999}',
    ],
)
def test_strict_parse_precedes_run_kind_detection(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "config.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match="duplicate|non-finite"):
        ControllerNeutralTrainingConfigV1.from_file(path)


@pytest.mark.parametrize("encoding", ["compact", "no-newline", "bom", "incomplete"])
def test_registered_bytes_are_exact(tmp_path: Path, encoding: str) -> None:
    config = _config(tmp_path.resolve() / "output")
    value = config.to_dict()
    if encoding == "incomplete":
        del value["notes"]
    raw = canonical_bytes(value)
    if encoding == "compact":
        raw = json.dumps(value).encode()
    elif encoding == "no-newline":
        raw = raw[:-1]
    elif encoding == "bom":
        raw = b"\xef\xbb\xbf" + raw
    path = tmp_path / "config.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        ControllerNeutralTrainingConfigV1.from_file(path)


@pytest.mark.parametrize("plan_exists", [True, False])
def test_campaign_path_cannot_downgrade_to_legacy(
    tmp_path: Path, plan_exists: bool
) -> None:
    if plan_exists:
        (tmp_path / "plan.json").write_text("{}")
    (tmp_path / "configs").mkdir()
    value = _config(tmp_path.resolve() / "output").to_dict()
    del value["beta_rl"]
    path = tmp_path / "configs" / "01-pi-lagrangian-seed-101.json"
    path.write_bytes(canonical_bytes(value))
    with pytest.raises(ValueError, match="registered declaration"):
        ControllerNeutralTrainingConfigV1.from_file(path)


def test_legacy_strict_json_still_accepts_partial_settings(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text('{"seed": 42}')
    assert ControllerNeutralTrainingConfigV1.from_file(path).seed == 42
    assert decode('{"a": [1, 2.5, null]}') == {"a": [1, 2.5, None]}


def test_registered_training_fails_before_environment_or_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plant_backend.rl.v2.training import ppo

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("admission constructed a learner or environment")

    monkeypatch.setattr(ppo, "build_vec_env", forbidden)
    monkeypatch.setattr(ppo, "build_model", forbidden)
    monkeypatch.setattr(ppo, "build_episode_allocator", forbidden)
    config = _config(tmp_path.resolve() / "output")
    with pytest.raises(ValueError, match="admitted campaign custody"):
        train(config)
    with pytest.raises(ValueError, match="task template"):
        train(replace(config, notes="changed"))
    assert not Path(config.output_dir).exists()


def test_root_must_be_canonical_external_absolute(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    for invalid in (Path("relative"), repository / "runs" / "campaign"):
        with pytest.raises(ValueError):
            matrix(invalid)
    linked = tmp_path / "link"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="without symlinks"):
        matrix(linked / "campaign")
