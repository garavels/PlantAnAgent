"""One bounded, disposable rehearsal; no registered seeds, training or selection.

Run once per controller with ``python -m plant_backend.record_demo H`` or MPC.
Attempt markers prevent silently replacing a failed rehearsal with a retry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .demo_recordings import RECORDING_ROOT, seal_recording, summarize, validate
from .v2_demo import FIXTURE_SEED, StepRequest, V2DemoSession

WALL_LIMIT_SECONDS = 20 * 60


class RehearsalTimeout(BaseException):
    """Stop the rehearsal even inside a solver's ordinary exception boundary."""


def git_provenance(root: Path) -> dict[str, str | bool | None]:
    """A downloaded source ZIP has source hashes, but no Git identity."""
    if not (root / ".git").exists():
        return {"git_revision": None, "git_dirty": None}
    return {
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controller", choices=("H", "MPC"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    RECORDING_ROOT.mkdir(parents=True, exist_ok=True)
    marker = RECORDING_ROOT / f"{args.controller}.attempt.json"
    source_files = sorted(
        [*root.glob("backend/src/**/*.py"), *root.glob("simulator/src/**/*.py")]
    )
    source = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_files
    }
    manifest = {
        "purpose": "single disposable presentation rehearsal, not registered research",
        "controller": args.controller,
        "fixture_seed": FIXTURE_SEED,
        "horizon_hours": 168,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        **git_provenance(root),
        "python_source_sha256": hashlib.sha256(
            json.dumps(source, sort_keys=True).encode()
        ).hexdigest(),
        "status": "started",
    }
    with marker.open("x") as stream:
        json.dump(manifest, stream, indent=2)
    started = time.monotonic()
    demo = V2DemoSession()

    def timeout(*_: object) -> None:
        raise RehearsalTimeout("fixed rehearsal wall limit reached")

    previous = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(WALL_LIMIT_SECONDS)
    try:
        try:
            snapshot = demo.reset(args.controller)
            while snapshot["status"] == "ready":
                snapshot = demo.step(
                    StepRequest(
                        expected_run_id=snapshot["run_id"],
                        expected_revision=snapshot["revision"],
                        decisions=4,
                    )
                )
        except RehearsalTimeout:
            snapshot = demo.recording()["snapshot"]
            snapshot["status"] = "failed"
            snapshot["error"] = (
                "Rehearsal stopped (wall_limit). Last completed boundary retained; "
                "a partial interval may have executed."
            )
        finally:
            # Preserve the completed prefix outside the execution deadline.
            signal.alarm(0)
        record = validate(seal_recording(snapshot, demo.recording()["frames"]))
        with (RECORDING_ROOT / f"{snapshot['run_id']}.json").open("x") as stream:
            json.dump(record, stream, separators=(",", ":"), allow_nan=False)
        manifest.update(summarize(record, "saved"), error=snapshot["error"])
    except Exception as exc:
        manifest.update(status="recording_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        manifest["wall_seconds"] = time.monotonic() - started
        marker.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
