"""Negative publication-gate tests and read-only interrupted-root inspection."""

from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from custody_fixtures import FixtureServices, context
from plant_backend.rl.v2.training import custody_documents as docs
from plant_backend.rl.v2.training import custody_io as io
from plant_backend.rl.v2.training import registered_campaign as campaign
from plant_backend.rl.v2.training.custody_state import CampaignClosedError, _stamp
from plant_backend.rl.v2.training.custody_synthetic import run_synthetic
from plant_backend.rl.v2.training.custody_verifier import inspect_root


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_post_publication_verifier_is_invoked_even_after_recapture_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[BaseException]
) -> None:
    ctx = context(tmp_path / "campaign")
    report = run_synthetic(ctx, FixtureServices(ctx).services)
    before = (ctx.root / "synthetic-report.json").read_bytes()
    invoked: list[list[str]] = []

    def fail_recapture(*args: object, **kwargs: object) -> Any:
        raise failure("synthetic recapture failure")

    def detached(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        invoked.append(command)
        assert kwargs["stdin"] == subprocess.DEVNULL
        return subprocess.CompletedProcess(
            command, 2, b"", b"synthetic verifier rejection"
        )

    monkeypatch.setattr(campaign, "capture_identity", fail_recapture)
    monkeypatch.setattr(campaign.subprocess, "run", detached)
    # This deliberately exercises only failure: no injected path is permitted
    # to issue or verify a production report or return a successful claim.
    with pytest.raises(
        CampaignClosedError, match="post_publication_verification_failed"
    ):
        campaign._post_publication_verification(ctx, report, _stamp(ctx.root))
    assert invoked == [
        [
            sys.executable,
            "-m",
            "plant_backend.rl.v2.training.registered_campaign",
            "verify",
            str(ctx.root / "report.json"),
        ]
    ]
    assert (ctx.root / "synthetic-report.json").read_bytes() == before
    assert not (ctx.root / "report.json").exists()


def test_forensic_inspection_never_confers_admission_or_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(tmp_path / "campaign")
    fixture = FixtureServices(ctx)
    original = io.inventory

    def fail(root: Path) -> dict[str, Any]:
        if root == Path(ctx.plan["cells"][1]["output_path"]):
            raise OSError("synthetic interrupted inventory")
        return original(root)

    monkeypatch.setattr(io, "inventory", fail)
    with pytest.raises(CampaignClosedError):
        run_synthetic(ctx, fixture.services)
    result = inspect_root(ctx.root)
    assert result["verified"] is False and result["result_claim"] is None
    assert result["retry_permitted"] is False
    assert result["cells"][0]["state"] == "admitted"
    assert result["cells"][1]["state"] == "claimed_interrupted"
    assert all(cell["state"] == "not_started" for cell in result["cells"][2:])
    assert not (ctx.root / "report.json").exists()
    assert not (ctx.root / "synthetic-report.json").exists()
    Path(ctx.plan["cells"][1]["claim_path"]).write_bytes(b"partial publication")
    result = inspect_root(ctx.root)
    assert result["cells"][1]["state"] == "invalid_evidence"
    assert result["retry_permitted"] is False


def test_cli_does_not_disclose_exception_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*args: object, **kwargs: object) -> Any:
        raise ValueError("SYNTHETIC_PRIVATE_PAYLOAD_MUST_NOT_BE_DISPLAYED")

    monkeypatch.setattr(campaign, "verify_report", fail)
    assert campaign.main(["verify", str(tmp_path / "report.json")]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "SYNTHETIC_PRIVATE_PAYLOAD" not in captured.err
    assert "no verified campaign claim" in captured.err
