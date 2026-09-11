"""One full-geometry fixture capacity probe; no training or protected seed access."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time
from typing import Any

from plant_backend.rl import checkpoints
from plant_backend.rl.v2 import beta_controller_benchmark as benchmark
from plant_backend.rl.v2.policy_setpoint_controller import (
    policy_contract_for_environment,
)
from plant_backend.rl.v2.study.arm_execution import execute_synthetic_arm
from plant_backend.rl.v2.study.execution_budget import EvidenceLimits
from plant_backend.rl.v2.study.latency_evidence import (
    capture_latency_host,
    configure_latency_threads,
    validate_latency_host,
)
from plant_backend.rl.v2.training import custody_io as io
from plant_backend.rl.v2.training.custody_authorization import repository_root
from plant_backend.rl.v2.training.custody_json import (
    authenticate,
    canonical_bytes,
    decode,
)
from plant_backend.rl.v2.training.custody_process import capture_process
from plant_backend.rl.v2.training.custody_validation import exact
from plant_backend.rl.v2.training.registered_contract import registered_config
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1

SCHEMA = "plant-beta-constrained-rl-study-capacity-fixture-v1"
PROTOCOL = "docs/research/CONSTRAINED_RL_STUDY_CAPACITY_FIXTURE.md"
ARMS = ("H", "M24", "E-beta", "PPO")
SEED = 9_100_001
TIMEOUT = 600
LOG_LIMIT = 1024**2
ROOT_FLOOR = 8 * 1024**3
RSS_LIMIT = 12 * 1024**3
LIMITS = EvidenceLimits(1024**3, 4 * 1024**3, 8 * 1024**2, 64 * 1024**2)
NOTE = "full-geometry study capacity fixture; fixed untrained policy; no optimizer"


def config(root: Path) -> ControllerNeutralTrainingConfigV1:
    base = registered_config(
        root / "unused-training-destination", "plant-beta-ppo-pi-lagrangian-v1", 101
    )
    return replace(
        base,
        beta_rl=None,
        total_timesteps=4,
        n_envs=1,
        output_dir=str(root / "fixed-policy"),
        notes=NOTE,
    ).validate()


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _capacity_identity() -> dict[str, Any]:
    return {
        "protocol_raw_sha256": hashlib.sha256(
            (repository_root() / PROTOCOL).read_bytes()
        ).hexdigest(),
        "source": benchmark._source_provenance(require_clean=True),
    }


def expected_root(identity: dict[str, Any]) -> Path:
    parent = Path("/private/tmp" if platform.system() == "Darwin" else "/tmp").resolve()
    revision = identity["source"]["git_revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(c not in "0123456789abcdef" for c in revision)
    ):
        raise ValueError("capacity source revision is invalid")
    return parent / ("plantanagent-study-capacity-" + revision[:12])


def make_plan(
    root: Path, identity: dict[str, Any], parent_host: dict[str, Any]
) -> dict[str, Any]:
    exact("fixed capacity destination", str(root), str(expected_root(identity)))
    child_host = deepcopy(validate_latency_host(parent_host))
    child_host["machine"].update(torch_intraop_threads=1, torch_interop_threads=1)
    child_host = validate_latency_host(authenticate(child_host))
    return authenticate(
        {
            "schema": SCHEMA + "-plan",
            "root": str(root),
            "identity": identity,
            "task": config(root).to_dict(),
            "arms": list(ARMS),
            "seed": SEED,
            "limits": LIMITS.to_dict(),
            "child_host": child_host,
            "parent_host": parent_host,
            "timeout_seconds": TIMEOUT,
            "log_limit_bytes": LOG_LIMIT,
            "sampled_rss_limit_bytes": RSS_LIMIT,
            "minimum_free_bytes": ROOT_FLOOR,
            "fixed_action": [0.0, 0.0, 0.0, 0.0],
            "fixed_policy_trained": False,
            "result_claim": None,
        }
    )


def _read_plan(root: Path) -> dict[str, Any]:
    io.require_directory(root)
    plan = decode(io.read_regular(root / "plan.json"))
    exact(
        "complete fixture plan",
        plan,
        make_plan(root, _capacity_identity(), capture_latency_host()),
    )
    return plan


def enforce_memory_limit(cell: Path) -> None:
    peak = peak_rss_bytes()
    if peak >= RSS_LIMIT:
        try:
            io.publish_json(
                cell / "resource-limit.json",
                {
                    "code": "sampled_peak_rss_limit",
                    "observed_bytes": peak,
                    "limit_bytes": RSS_LIMIT,
                },
            )
        finally:
            os._exit(73)


def watch_memory(cell: Path, stop: threading.Event) -> None:
    while not stop.wait(0.1):
        enforce_memory_limit(cell)


def child(root: Path, arm: str) -> None:
    if arm not in ARMS:
        raise ValueError("unknown fixture arm")
    plan = _read_plan(root)
    cell = root / "cells" / arm
    io.require_directory(cell)
    if list(cell.iterdir()):
        raise ValueError("fixture attempt was already consumed")
    io.publish_json(
        cell / "claimed.json", {"arm": arm, "seed": SEED, "plan_sha256": plan["sha256"]}
    )
    stop = threading.Event()

    watcher = threading.Thread(target=watch_memory, args=(cell, stop), daemon=True)
    watcher.start()
    try:
        configure_latency_threads()
        host = validate_latency_host(capture_latency_host())
        exact("fixture child host", host, plan["child_host"])
        task = config(root)
        checkpoint = None
        started = time.perf_counter_ns()
        if arm == "PPO":
            environment = benchmark._new_beta_environment(task)
            try:
                checkpoint = checkpoints.write_fixed_checkpoint(
                    cell / "fixed-policy",
                    (0.0, 0.0, 0.0, 0.0),
                    contract=policy_contract_for_environment(environment),
                )
                metadata = checkpoints.load_metadata(checkpoint, check=False)
                checkpoints.save_metadata(
                    checkpoint, replace(metadata, training_config=task.to_dict())
                )
            finally:
                environment.close()
        result = execute_synthetic_arm(
            task,
            arm=arm,
            seed=SEED,
            directory=cell / "evidence",
            limits=LIMITS,
            checkpoint=checkpoint,
        )
        elapsed = time.perf_counter_ns() - started
        stop.set()
        watcher.join(timeout=1)
        enforce_memory_limit(cell)
        exact("fixture identity after", _capacity_identity(), plan["identity"])
        exact("fixture host after", capture_latency_host(), host)
        classification = result["physical"]["classification"]
        measurement = authenticate(
            {
                "schema": SCHEMA + "-measurement",
                "arm": arm,
                "seed": SEED,
                "plan_sha256": plan["sha256"],
                "host": host,
                "classification": classification,
                "full_geometry_completed": classification["execution_complete"] is True
                and classification["completed_decisions"] == 672
                and classification["physical_steps"] == 10_080,
                "arm_archive": result["arm_archive"],
                "timing_archive": result["timing_archive"],
                "channel_totals": result["latency"]["channels"],
                "producer_and_reconstruction_elapsed_ns": elapsed,
                "process_peak_rss_bytes": peak_rss_bytes(),
                "registered_checkpoint_verified": False,
                "result_claim": None,
            }
        )
        io.publish_json(cell / "measurement.json", measurement)
    except Exception as error:
        io.publish_json(
            cell / "failure.json", {"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        stop.set()
        watcher.join(timeout=1)


def run(root: Path) -> dict[str, Any]:
    identity = _capacity_identity()
    parent_host = capture_latency_host()
    plan = make_plan(root, identity, parent_host)
    child_host = plan["child_host"]
    io.require_new_root(root, repository_root(), ROOT_FLOOR)
    io.mkdir_exclusive(root)
    for directory in (root / "cells", root / "logs"):
        io.mkdir_exclusive(directory)
    for arm in ARMS:
        io.mkdir_exclusive(root / "cells" / arm)
    io.publish_json(root / "plan.json", plan)
    rows: list[dict[str, Any]] = []
    abort = None
    for arm in ARMS:
        _read_plan(root)
        exact("fixture parent host", capture_latency_host(), parent_host)
        fs = os.statvfs(root)
        if (
            fs.f_bavail * fs.f_frsize + io.inventory(root)["allocated_bytes"]
            < ROOT_FLOOR
        ):
            abort = "storage_budget_unavailable"
            break
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "child",
            str(root),
            "--arm",
            arm,
        ]
        process, failures = capture_process(
            command,
            stdout=root / "logs" / (arm + ".stdout.log"),
            stderr=root / "logs" / (arm + ".stderr.log"),
            timeout_seconds=TIMEOUT,
            log_limit_bytes=LOG_LIMIT,
        )
        _read_plan(root)
        exact("fixture parent host after", capture_latency_host(), parent_host)
        cell = root / "cells" / arm
        measurement = None
        path = cell / "measurement.json"
        if not failures and path.exists():
            measurement = decode(io.read_regular(path))
            exact("fixture measurement digest", measurement, authenticate(measurement))
            exact(
                "fixture measurement plan", measurement["plan_sha256"], plan["sha256"]
            )
            exact("fixture measurement host", measurement["host"], child_host)
            exact("fixture measurement actor", measurement["arm"], arm)
            exact("fixture measurement seed", measurement["seed"], SEED)
            classification = measurement["classification"]
            exact(
                "fixture completion",
                measurement["full_geometry_completed"],
                classification["execution_complete"] is True
                and classification["completed_decisions"] == 672
                and classification["physical_steps"] == 10_080,
            )
            for key in ("arm_archive", "timing_archive"):
                descriptor = measurement[key]["file"]
                filename = (
                    "synthetic-arm.json.gz"
                    if key == "arm_archive"
                    else "synthetic-latency.json.gz"
                )
                exact(
                    "fixture archive path",
                    descriptor["path"],
                    str(cell / "evidence" / filename),
                )
                exact(
                    "retained fixture archive",
                    io.file_record(Path(descriptor["path"])),
                    descriptor,
                )
        elif not failures:
            failures.add("missing_measurement")
        rows.append(
            {
                "arm": arm,
                "process": process,
                "failure_codes": sorted(failures),
                "logs": {
                    name: io.file_record(root / "logs" / (arm + "." + name + ".log"))
                    for name in ("stdout", "stderr")
                },
                "measurement": measurement,
                "inventory": io.inventory(cell),
            }
        )
        if "root_custody_lost" in failures:
            abort = "operator_interrupted"
            break
    for arm in ARMS[len(rows) :]:
        rows.append(
            {
                "arm": arm,
                "process": None,
                "failure_codes": ["not_started"],
                "logs": None,
                "measurement": None,
                "inventory": io.inventory(root / "cells" / arm),
            }
        )
    report = authenticate(
        {
            "schema": SCHEMA + "-report",
            "plan_sha256": plan["sha256"],
            "identity": identity,
            "rows": rows,
            "abort": abort,
            "all_full_geometry_completed": all(
                row["measurement"] is not None
                and row["measurement"]["full_geometry_completed"]
                for row in rows
            ),
            "result_claim": None,
            "registered_execution_authorized": False,
        }
    )
    io.publish_json(root / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "child"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    arguments = parser.parse_args()
    if arguments.command == "child":
        child(arguments.root, arguments.arm)
    else:
        if arguments.arm is not None:
            parser.error("parent arm override is forbidden")
        report = run(arguments.root)
        print(
            canonical_bytes(
                {
                    "report": str(arguments.root / "report.json"),
                    "sha256": report["sha256"],
                    "all_full_geometry_completed": report[
                        "all_full_geometry_completed"
                    ],
                    "registered_execution_authorized": False,
                }
            ).decode(),
            end="",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
