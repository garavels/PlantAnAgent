"""Bounds/configuration for the fixture probe; no protected execution or training."""

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from custody_fixtures import context as training_context
from plant_backend.rl.v2.study.stage_authority import expected_latency_host
from plant_backend.rl.v2.training import custody_io as io
from plant_backend.rl.v2.training.custody_json import (
    authenticate,
    canonical_bytes,
    decode,
)
from plant_backend.rl.v2.training.registered_contract import registered_config

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docs/research/tools/study_capacity_fixture.py"
)
spec = importlib.util.spec_from_file_location("study_capacity_probe_test", SCRIPT)
assert spec is not None and spec.loader is not None
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def forbid_real_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("capacity unit fixture reached a real plant or model")

    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "__init__", forbidden)
    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "reset", forbidden)
    monkeypatch.setattr(probe, "execute_synthetic_arm", forbidden)


@pytest.fixture
def planned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    root = tmp_path.resolve() / "probe"
    ctx = training_context(tmp_path.resolve() / "training")
    ctx.identity.host["cpu_brand"] = "Fixture CPU"
    host = expected_latency_host(SimpleNamespace(identity=ctx.identity))
    host["machine"].update(torch_intraop_threads=8, torch_interop_threads=12)
    host = authenticate(host)
    identity = {"protocol_raw_sha256": "a" * 64, "source": {"git_revision": "b" * 40}}
    monkeypatch.setattr(probe, "_capacity_identity", lambda: deepcopy(identity))
    monkeypatch.setattr(probe, "expected_root", lambda _: root)
    monkeypatch.setattr(probe, "capture_latency_host", lambda: deepcopy(host))
    plan = probe.make_plan(root, identity, host)
    return SimpleNamespace(root=root, identity=identity, host=host, plan=plan)


def write_plan(value: Any) -> None:
    value.root.mkdir()
    (value.root / "cells").mkdir()
    for arm in probe.ARMS:
        (value.root / "cells" / arm).mkdir()
    io.publish_json(value.root / "plan.json", value.plan)


def test_task_preserves_entire_registered_physical_surface(tmp_path: Path) -> None:
    actual = probe.config(tmp_path)
    original = registered_config(
        tmp_path / "original", "plant-beta-ppo-pi-lagrangian-v1", 101
    )
    baseline = original.to_dict()
    baseline.update(
        beta_rl=None,
        total_timesteps=4,
        n_envs=1,
        output_dir=str(tmp_path / "fixed-policy"),
        notes=probe.NOTE,
    )
    assert actual.to_dict() == baseline
    assert actual.episode.horizon_h == 168
    assert actual.episode.decision_interval_s == 900
    assert probe.SEED == 9_100_001
    assert probe.TIMEOUT == 600
    assert probe.ROOT_FLOOR == 8 * 1024**3
    assert probe.RSS_LIMIT == 12 * 1024**3


def test_capacity_identity_binds_both_frozen_protocol_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, content in (
        (probe.PROTOCOL, b"new bounded numerical repair scope"),
        (probe.ORIGINAL_PROTOCOL, b"original negative fixture scope"),
    ):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    monkeypatch.setattr(probe, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        probe.benchmark, "_source_provenance", lambda **_: {"fixture": True}
    )
    identity = probe._capacity_identity()
    assert identity == {
        "protocol_raw_sha256": hashlib.sha256(
            b"new bounded numerical repair scope"
        ).hexdigest(),
        "original_protocol_raw_sha256": hashlib.sha256(
            b"original negative fixture scope"
        ).hexdigest(),
        "source": {"fixture": True},
    }
    assert (
        probe.SCHEMA
        == "plant-beta-constrained-rl-study-capacity-return-bounds-fixture-v1"
    )


def test_revision_has_one_destination_and_must_be_a_full_object_id() -> None:
    identity = {"source": {"git_revision": "abcdef012345" + "0" * 28}}
    path = probe.expected_root(identity)
    assert path.is_absolute() and path == path.resolve()
    assert path.name == "plantanagent-study-capacity-abcdef012345"
    for revision in (None, 123, "abc", "g" * 40, "A" * 40):
        with pytest.raises(ValueError, match="source revision"):
            probe.expected_root({"source": {"git_revision": revision}})


@pytest.mark.parametrize(
    "field",
    [
        "seed",
        "arms",
        "task",
        "identity",
        "limits",
        "timeout_seconds",
        "sampled_rss_limit_bytes",
        "log_limit_bytes",
        "minimum_free_bytes",
        "child_host",
        "parent_host",
        "fixed_action",
        "fixed_policy_trained",
        "result_claim",
        "extra",
    ],
)
def test_rehashed_plan_overrides_fail_before_any_claim(
    planned: Any, field: str
) -> None:
    write_plan(planned)
    raw = deepcopy(planned.plan)
    raw[field] = "substituted"
    (planned.root / "plan.json").write_bytes(canonical_bytes(authenticate(raw)))
    with pytest.raises((ValueError, TypeError)):
        probe.child(planned.root, "H")
    assert not list((planned.root / "cells/H").iterdir())


def test_existing_attempt_and_wrong_root_do_not_execute(planned: Any) -> None:
    write_plan(planned)
    io.publish_json(planned.root / "cells/H/claimed.json", {"consumed": True})
    with pytest.raises(ValueError, match="consumed"):
        probe.child(planned.root, "H")
    with pytest.raises(ValueError, match="destination"):
        probe.make_plan(
            planned.root.parent / "alternative", planned.identity, planned.host
        )


def fake_measurement(
    root: Path, arm: str, plan: dict[str, Any], *, full: bool
) -> dict[str, Any]:
    cell = root / "cells" / arm
    (cell / "evidence").mkdir()
    refs = []
    for filename in ("synthetic-arm.json.gz", "synthetic-latency.json.gz"):
        path = cell / "evidence" / filename
        path.write_bytes(b"explicit byte fixture; not physical evidence")
        refs.append({"file": io.file_record(path)})
    return authenticate(
        {
            "schema": probe.SCHEMA + "-measurement",
            "arm": arm,
            "seed": probe.SEED,
            "plan_sha256": plan["sha256"],
            "host": plan["child_host"],
            "classification": {
                "execution_complete": full,
                "completed_decisions": 672 if full else 3,
                "physical_steps": 10_080 if full else 45,
            },
            "full_geometry_completed": full,
            "arm_archive": refs[0],
            "timing_archive": refs[1],
        }
    )


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "failed_and_incomplete",
        "interrupted",
        "tampered_completion",
        "tampered_seed",
        "archive_path",
    ],
)
def test_parent_keeps_every_process_prefix_and_failure(
    planned: Any, monkeypatch: pytest.MonkeyPatch, defect: str | None
) -> None:
    calls = []

    def capture(command: list[str], **kwargs: Any) -> Any:
        arm = command[-1]
        assert arm == probe.ARMS[len(calls)]
        assert kwargs["timeout_seconds"] == 600 and kwargs["log_limit_bytes"] == 1024**2
        calls.append(arm)
        for name in ("stdout", "stderr"):
            io.publish_bytes(kwargs[name], b"fixture log")
        cell = planned.root / "cells" / arm
        io.publish_json(cell / "claimed.json", {"fixture_claim": True})
        failure = defect == "failed_and_incomplete" and arm == "M24"
        interrupted = defect == "interrupted" and arm == "H"
        if not failure and not interrupted:
            value = fake_measurement(
                planned.root,
                arm,
                planned.plan,
                full=not (defect == "failed_and_incomplete" and arm == "E-beta"),
            )
            if defect == "tampered_completion":
                value["classification"]["physical_steps"] = 1
            elif defect == "tampered_seed":
                value["seed"] = 2_000_000
            elif defect == "archive_path":
                value["arm_archive"] = value["timing_archive"]
            io.publish_json(cell / "measurement.json", authenticate(value))
        return {"fixture_process": True}, (
            {"root_custody_lost"}
            if interrupted
            else {"child_nonzero_exit"} if failure else set()
        )

    monkeypatch.setattr(probe, "capture_process", capture)
    if defect in {"tampered_completion", "tampered_seed", "archive_path"}:
        with pytest.raises(ValueError):
            probe.run(planned.root)
        assert calls == ["H"]
        return
    result = probe.run(planned.root)
    assert [row["arm"] for row in result["rows"]] == list(probe.ARMS)
    assert (
        result["registered_execution_authorized"] is False
        and result["result_claim"] is None
    )
    assert result["all_full_geometry_completed"] is (defect is None)
    if defect == "interrupted":
        assert calls == ["H"] and result["abort"] == "operator_interrupted"
        assert all(
            row["failure_codes"] == ["not_started"] for row in result["rows"][1:]
        )
    else:
        assert calls == list(probe.ARMS)
    if defect == "failed_and_incomplete":
        assert result["rows"][1]["failure_codes"] == ["child_nonzero_exit"]
        assert result["rows"][2]["measurement"]["full_geometry_completed"] is False
    with pytest.raises(ValueError, match="exists"):
        probe.run(planned.root)


def test_child_checks_exact_geometry_and_records_wall_rss(
    planned: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_plan(planned)

    def configure() -> None:
        monkeypatch.setattr(
            probe, "capture_latency_host", lambda: deepcopy(planned.plan["child_host"])
        )

    monkeypatch.setattr(probe, "configure_latency_threads", configure)
    monkeypatch.setattr(probe, "peak_rss_bytes", lambda: 123)

    def execute(task: Any, **kwargs: Any) -> Any:
        assert task.to_dict() == planned.plan["task"]
        assert kwargs["seed"] == 9_100_001 and kwargs["arm"] == "H"
        assert (planned.root / "cells/H/claimed.json").exists()
        measured = fake_measurement(planned.root, "H", planned.plan, full=True)
        return {
            "physical": {"classification": measured["classification"]},
            "latency": {"channels": {}},
            "arm_archive": measured["arm_archive"],
            "timing_archive": measured["timing_archive"],
        }

    monkeypatch.setattr(probe, "execute_synthetic_arm", execute)
    probe.child(planned.root, "H")
    value = decode((planned.root / "cells/H/measurement.json").read_bytes())
    assert value["full_geometry_completed"] is True
    assert value["process_peak_rss_bytes"] == 123
    assert value["producer_and_reconstruction_elapsed_ns"] > 0


def test_rss_stop_is_durable_and_exits_only_its_child(tmp_path: Path) -> None:
    # Lower the cap in this disposable subprocess only; allocate no large buffer.
    code = "import importlib.util,sys; from pathlib import Path; s=importlib.util.spec_from_file_location('probe',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.RSS_LIMIT=1; m.enforce_memory_limit(Path(sys.argv[2]))"
    completed = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT), str(tmp_path)],
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 73
    marker = decode((tmp_path / "resource-limit.json").read_bytes())
    assert marker["code"] == "sampled_peak_rss_limit"
    assert marker["observed_bytes"] > marker["limit_bytes"] == 1
