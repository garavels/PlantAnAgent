"""Launch permissions are consumable transport, never permission to open study seeds in tests."""

import os
from pathlib import Path
import sys
from typing import Any

import pytest

from plant_backend.rl.v2.training import custody_launch as launch
from plant_backend.rl.v2.training.custody_json import canonical_bytes
from plant_backend.rl.v2.training.custody_process import capture_process
from plant_backend.rl.v2.training.registered_contract import matrix
from plant_backend.rl.v2.training import registered_campaign as campaign


def _permission(parent_pid: int) -> bytes:
    return canonical_bytes(
        {
            "schema": launch.LAUNCH_SCHEMA,
            "parent_pid": parent_pid,
            "claim_sha256": "a" * 64,
            "config_sha256": "b" * 64,
        }
    )


def test_permission_consumption_is_one_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    reader, writer = os.pipe()
    os.write(writer, _permission(os.getppid()))
    os.close(writer)
    monkeypatch.setenv(launch.LAUNCH_FD_ENV, str(reader))
    assert launch._consume_permission()["claim_sha256"] == "a" * 64
    assert launch.LAUNCH_FD_ENV not in os.environ
    with pytest.raises(ValueError, match="admitted campaign custody"):
        launch._consume_permission()
    with pytest.raises(OSError):
        os.fstat(reader)


@pytest.mark.parametrize(
    "defect", ["parent", "duplicate", "empty", "noncanonical", "wrong_schema", "extra"]
)
def test_bad_launch_channel_is_closed_without_any_admission(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    raw = _permission(os.getppid())
    if defect == "parent":
        raw = _permission(os.getppid() + 1)
    elif defect == "duplicate":
        raw = b'{"schema": "x", "schema": "y"}'
    elif defect == "empty":
        raw = b""
    elif defect == "noncanonical":
        raw = raw[:-1]
    elif defect == "wrong_schema":
        raw = canonical_bytes({"schema": "fixture"})
    else:
        raw = raw.replace(b'"parent_pid"', b'"extra": true, "parent_pid"')
    reader, writer = os.pipe()
    os.write(writer, raw)
    os.close(writer)
    monkeypatch.setenv(launch.LAUNCH_FD_ENV, str(reader))
    with pytest.raises(ValueError):
        launch._consume_permission()
    assert launch.LAUNCH_FD_ENV not in os.environ
    with pytest.raises(OSError):
        os.fstat(reader)


def test_real_child_receives_and_consumes_only_its_parent_permission(
    tmp_path: Path,
) -> None:
    code = (
        "import os; from plant_backend.rl.v2.training.custody_launch import _consume_permission,LAUNCH_FD_ENV; "
        "value=_consume_permission(); assert value['parent_pid']==os.getppid(); assert LAUNCH_FD_ENV not in os.environ"
    )
    process, failures = capture_process(
        [sys.executable, "-c", code],
        stdout=tmp_path / "stdout",
        stderr=tmp_path / "stderr",
        timeout_seconds=5,
        log_limit_bytes=4096,
        launch_permission=_permission(os.getpid()),
    )
    assert process["exit_code"] == 0 and not failures
    assert (
        (tmp_path / "stdout").read_bytes() == (tmp_path / "stderr").read_bytes() == b""
    )


@pytest.mark.parametrize(
    "missing", ["assert_exclusive", "assert_first_attempt", "assert_authorized"]
)
def test_campaign_assertions_are_required_before_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail("missing assertion reached authority or execution")

    monkeypatch.setattr(campaign.capacity, "_host_identity", forbidden)
    monkeypatch.setattr(campaign, "collect_campaign", forbidden)
    flags = {
        "assert_exclusive": True,
        "assert_first_attempt": True,
        "assert_authorized": True,
    }
    flags[missing] = False
    with pytest.raises(ValueError, match="all three"):
        campaign.run_registered_campaign(tmp_path / "campaign", **flags)
    assert not (tmp_path / "campaign").exists()


def test_production_runner_has_no_injection_door(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        campaign.run_registered_campaign(tmp_path / "campaign", process=lambda: None)  # type: ignore[call-arg]
    assert not (tmp_path / "campaign").exists()


def test_valid_flags_do_not_replace_tracked_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from custody_fixtures import context

    ctx = context(tmp_path / "campaign")
    monkeypatch.setattr(campaign.capacity, "_host_identity", lambda: ctx.identity.host)

    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail("missing authority reached process collection")

    monkeypatch.setattr(campaign, "collect_campaign", forbidden)
    with pytest.raises(FileNotFoundError):
        campaign.run_registered_campaign(
            ctx.root,
            assert_exclusive=True,
            assert_first_attempt=True,
            assert_authorized=True,
        )
    assert not ctx.root.exists()


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "changed_config",
        "changed_claim",
        "terminal",
        "artifact",
        "later_claim",
        "closed",
        "prior_interrupted",
    ],
)
def test_open_claim_admission_is_read_only_and_rechecks_all_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str | None,
) -> None:
    from custody_fixtures import context
    from plant_backend.rl.v2.training import custody_documents as docs
    from plant_backend.rl.v2.training import custody_io as io
    from plant_backend.rl.v2.training.custody_state import _publish_initial, cell_config
    from plant_backend.rl.v2.training import ppo

    ctx = context(tmp_path / "campaign")
    _publish_initial(ctx)
    first = ctx.plan["cells"][0]
    claim = docs.make_claim(
        ctx.plan, first, None, timestamp="2026-09-11T12:00:00.000000Z"
    )
    io.publish_json(Path(first["claim_path"]), claim)
    cell = ctx.plan["cells"][1] if defect == "prior_interrupted" else first
    for name in ("stdout", "stderr"):
        io.publish_bytes(Path(cell[f"{name}_path"]), b"")

    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail("launch admission constructed a trainer or resettable environment")

    monkeypatch.setattr(ppo, "make_env", forbidden)
    monkeypatch.setattr(ppo, "build_vec_env", forbidden)
    monkeypatch.setattr(ppo, "build_model", forbidden)
    # Only read-only authority recapture is synthetic here. The actual launch
    # logic reopens the plan, configs and receipts, then returns without training.
    monkeypatch.setattr(launch.capacity, "_host_identity", lambda: ctx.identity.host)
    monkeypatch.setattr(
        launch.authority,
        "admit_authorization",
        lambda root, host: (ctx.authorization, ctx.identity.source),
    )
    monkeypatch.setattr(launch, "capture_identity", lambda authorization: ctx.identity)
    permission = {
        "schema": launch.LAUNCH_SCHEMA,
        "parent_pid": os.getppid(),
        "claim_sha256": claim["sha256"],
        "config_sha256": cell["config_sha256"],
    }
    if defect == "changed_config":
        Path(ctx.plan["cells"][-1]["config_path"]).write_bytes(b"{}")
    elif defect == "changed_claim":
        permission["claim_sha256"] = "0" * 64
    elif defect == "terminal":
        Path(cell["failed_receipt_path"]).write_bytes(b"{}")
    elif defect == "artifact":
        Path(cell["output_path"]).mkdir()
    elif defect == "later_claim":
        Path(ctx.plan["cells"][-1]["claim_path"]).write_bytes(b"{}")
    elif defect == "closed":
        (ctx.root / "report.json").write_bytes(b"{}")
    reader, writer = os.pipe()
    os.write(writer, canonical_bytes(permission))
    os.close(writer)
    monkeypatch.setenv(launch.LAUNCH_FD_ENV, str(reader))
    if defect is None:
        admitted_runtime = launch.admit_registered_launch(cell_config(cell))
        assert admitted_runtime.to_dict() == ctx.identity.runtime
        assert not Path(cell["output_path"]).exists()
        assert not (ctx.root / "report.json").exists()
    else:
        with pytest.raises(ValueError):
            launch.admit_registered_launch(cell_config(cell))
    assert launch.LAUNCH_FD_ENV not in os.environ
