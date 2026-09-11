"""One bounded three-seed fixture for the authorized uncertainty repair."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import threading
from typing import Any

from plant_backend.rl.v2 import beta_controller_benchmark as benchmark
from plant_backend.rl.v2.study.arm_execution import execute_synthetic_arm
from plant_backend.rl.v2.study.compressed_evidence import verify_synthetic_execution
from plant_backend.rl.v2.study.latency_evidence import configure_latency_threads
from plant_backend.rl.v2.training import custody_io as io
from plant_backend.rl.v2.training.custody_authorization import repository_root
from plant_backend.rl.v2.training.custody_json import authenticate, decode
from plant_backend.rl.v2.training.custody_process import capture_process
from plant_backend.rl.v2.training.custody_validation import exact
from study_capacity_fixture import (
    config,
    LIMITS,
    ROOT_FLOOR,
    TIMEOUT,
    LOG_LIMIT,
    watch_memory,
)

SEEDS = (9_100_001, 9_100_002, 9_100_003)
SCHEMA = "plant-beta-mpc-uncertainty-fixture-v1"
PROTOCOL = "docs/research/BETA_MPC_UNCERTAINTY_REPAIR.md"


def identity() -> dict[str, Any]:
    return {
        "source": benchmark._source_provenance(require_clean=True),
        "scope_raw_sha256": hashlib.sha256(
            (repository_root() / PROTOCOL).read_bytes()
        ).hexdigest(),
    }


def expected_root(value: dict[str, Any]) -> Path:
    revision = value["source"]["git_revision"]
    if not isinstance(revision, str) or len(revision) != 40:
        raise ValueError("fixture requires a complete Git revision")
    return Path("/private/tmp") / ("plantanagent-beta-uncertainty-" + revision[:12])


def plan(root: Path) -> dict[str, Any]:
    current = identity()
    if root != expected_root(current):
        raise ValueError("fixture root must be the unused source-specific destination")
    return authenticate(
        {
            "schema": SCHEMA,
            "identity": current,
            "root": str(root),
            "seeds": list(SEEDS),
            "arm": "E-beta",
            "timeout_seconds": TIMEOUT,
            "log_limit_bytes": LOG_LIMIT,
            "limits": LIMITS.to_dict(),
            "registered_execution_authorized": False,
        }
    )


def check(root: Path) -> dict[str, Any]:
    saved = decode(io.read_regular(root / "plan.json"))
    exact("unchanged fixture plan/source", saved, plan(root))
    return saved


def child(root: Path, seed: int) -> None:
    if seed not in SEEDS:
        raise ValueError("unknown fixture seed")
    saved = check(root)
    cell = root / str(seed)
    io.mkdir_exclusive(cell)
    io.publish_json(
        cell / "claimed.json", {"seed": seed, "plan_sha256": saved["sha256"]}
    )
    stop = threading.Event()
    watcher = threading.Thread(target=watch_memory, args=(cell, stop), daemon=True)
    watcher.start()
    try:
        configure_latency_threads()
        result = execute_synthetic_arm(
            config(root),
            arm="E-beta",
            seed=seed,
            directory=cell / "evidence",
            limits=LIMITS,
        )
        check(root)
        io.publish_json(cell / "verification.json", result)
    finally:
        stop.set()
        watcher.join(timeout=1)


def run(root: Path) -> dict[str, Any]:
    saved = plan(root)
    io.require_new_root(root, repository_root(), ROOT_FLOOR)
    io.mkdir_exclusive(root)
    io.mkdir_exclusive(root / "logs")
    io.publish_json(root / "plan.json", saved)
    rows = []
    for seed in SEEDS:
        check(root)
        process, failures = capture_process(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "child",
                str(root),
                "--seed",
                str(seed),
            ],
            stdout=root / "logs" / f"{seed}.stdout.log",
            stderr=root / "logs" / f"{seed}.stderr.log",
            timeout_seconds=TIMEOUT,
            log_limit_bytes=LOG_LIMIT,
        )
        check(root)
        result = None
        verification_error = None
        try:
            if not failures:
                result = verify_synthetic_execution(
                    root / str(seed) / "evidence", limits=LIMITS
                )
                exact(
                    "child and detached reconstruction",
                    result,
                    decode(io.read_regular(root / str(seed) / "verification.json")),
                )
        except Exception as reason:
            verification_error = {"type": type(reason).__name__, "message": str(reason)}
        rows.append(
            {
                "seed": seed,
                "process": process,
                "failure_codes": sorted(failures),
                "verification_error": verification_error,
                "verification": result,
                "inventory": io.inventory(root / str(seed)),
            }
        )
    result = authenticate(
        {
            "schema": SCHEMA,
            "plan": saved,
            "rows": rows,
            "registered_execution_authorized": False,
            "result_claim": None,
        }
    )
    io.publish_json(root / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "child"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.operation == "child":
        child(args.root, args.seed)
    else:
        report = run(args.root)
        print(report["sha256"])


if __name__ == "__main__":
    main()
