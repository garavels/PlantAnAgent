"""One-shot handoff from existing training to frozen evaluation.

Default invocation only prints the proposed continuation. --execute admits one
exclusive attempt, waits for existing training without starting or retrying it,
freezes the six endpoints once, then invokes the pinned evaluation driver once.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SCHEMA = "authority-ppo-v2-one-shot-continuation"
WAIT_STOP_UTC = "2026-09-17T05:27:30Z"
EVALUATION_STOP_UTC = "2026-09-17T10:26:17Z"
POLL_SECONDS = 15
FREEZE_TIMEOUT_SECONDS = 300
EXPECTED = {(arm, seed) for arm in ("narrow", "broad") for seed in (101, 211, 307)}


class TrainingFailed(RuntimeError):
    pass


class StopRequested(RuntimeError):
    pass


def utc():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Deadline must include a timezone")
    return parsed.timestamp()


def file_pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(path, value, *, exclusive=False):
    path = Path(path)
    if exclusive:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temporary = path.with_suffix(path.suffix + ".part")
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    sync_directory(path.parent)


def invocation(campaign, source, driver, python):
    campaign, source, driver = (Path(value).resolve() for value in (campaign, source, driver))
    # Preserve the virtual-environment executable symlink rather than resolving it.
    python = Path(os.path.abspath(python))
    training_root = campaign / "training"
    trainer = source / "docs/research/tools/authority_ppo_v2/train.py"
    return {
        "schema": SCHEMA + "-invocation", "created_at": utc(), "pid": os.getpid(),
        "campaign": str(campaign), "source": str(source), "training_root": str(training_root),
        "driver": str(driver), "python": str(python),
        "supervision": str(campaign / "process-logs/supervision.json"),
        "attempt_root": str(campaign / "continuation"),
        "wait_stop_utc": WAIT_STOP_UTC, "evaluation_stop_utc": EVALUATION_STOP_UTC,
        "poll_seconds": POLL_SECONDS, "freeze_timeout_seconds": FREEZE_TIMEOUT_SECONDS,
        "expected_training_attempts": [{"arm": arm, "optimizer_seed": seed} for arm, seed in sorted(EXPECTED)],
        "pinned_sources": {"helper": file_pin(Path(__file__)), "driver": file_pin(driver), "snapshot_trainer": file_pin(trainer)},
        "freeze_command": [str(python), str(trainer), "freeze-endpoints", "--root", str(training_root)],
        "evaluation_command": [str(python), str(driver), "--campaign", str(campaign), "--source", str(source),
                               "--python", str(python), "--training-root", str(training_root),
                               "--stop-utc", EVALUATION_STOP_UTC, "--execute"],
        "selection": "One existing training campaign, all six fixed final endpoints, no retraining, no retries, no alternate checkpoints, no resumed evaluation attempt.",
    }


def verify_sources(request):
    for name, expected in request["pinned_sources"].items():
        if file_pin(Path(expected["path"])) != expected:
            raise ValueError(f"Pinned {name} changed after continuation invocation")


def supervised_status(payload, training_root):
    """Admit exactly six closed successes; partial valid waves remain waiting."""
    if payload.get("schema") != "authority-training-supervision-v1":
        raise ValueError("Unknown supervision schema")
    records = payload.get("processes")
    if not isinstance(records, list):
        raise ValueError("Training supervision has no process list")
    seen, finished = set(), []
    for row in records:
        key = (row.get("arm"), row.get("optimizer_seed"))
        if key not in EXPECTED or key in seen:
            raise ValueError("Unexpected or duplicate supervised training attempt")
        seen.add(key)
        code = row.get("exit_code")
        if code is not None and type(code) is not int:
            raise ValueError("Training exit code must be an integer or null")
        if code not in (None, 0):
            raise TrainingFailed(f"Training {key[0]}-{key[1]} failed with exit code {code}; no retry")
        if row.get("supervisor_terminated_at"):
            raise TrainingFailed(f"Training {key[0]}-{key[1]} was terminated by its supervisor")
        if row.get("finished_at") and code != 0:
            raise ValueError("Finished training lacks a successful exit code")
        if row.get("finished_at"):
            timestamp(row["finished_at"])
            finished.append(key)
    # Failure artifacts can precede the supervisor's next fifteen-second read.
    for arm, seed in sorted(EXPECTED):
        failure = Path(training_root) / f"{arm}-seed-{seed}" / "failure.json"
        if failure.exists():
            content = json.loads(failure.read_text())
            raise TrainingFailed(f"Training failure artifact for {arm}-{seed}: {content.get('type')}: {content.get('message')}")
    ready = len(records) == 6 and seen == EXPECTED and set(finished) == EXPECTED
    return {"ready": ready, "attempts_observed": len(records), "successful_closed_attempts": len(finished),
            "closed": [{"arm": arm, "optimizer_seed": seed} for arm, seed in sorted(finished)],
            "updated_at": payload.get("updated_at")}


def environment(source):
    source = Path(source)
    return {**os.environ, "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(map(str, (source, source / "backend/src", source / "simulator/src"))),
            **{key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def descendant_pids(parent):
    result = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
    by_parent = {}
    for line in result.splitlines():
        pid, ppid = map(int, line.split())
        by_parent.setdefault(ppid, []).append(pid)
    pending, descendants = [parent], set()
    while pending:
        for pid in by_parent.get(pending.pop(), []):
            if pid not in descendants:
                descendants.add(pid)
                pending.append(pid)
    return descendants


def send_signal(pid, sig):
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False


def alive(pid):
    return send_signal(pid, 0)


def stop_child_tree(proc):
    """Only this helper's active child tree is touched, never training processes."""
    known = descendant_pids(proc.pid) | {proc.pid}
    # Freeze the discovered tree while collecting late-spawned children. Sending
    # SIGTERM before SIGCONT prevents resumed launchers from scheduling new work.
    for pid in known:
        send_signal(pid, signal.SIGSTOP)
    known |= descendant_pids(proc.pid)
    for pid in known:
        send_signal(pid, signal.SIGSTOP)
    for pid in known:
        send_signal(pid, signal.SIGTERM)
    for pid in sorted(known, reverse=True):
        send_signal(pid, signal.SIGCONT)
    limit = time.monotonic() + 65
    while time.monotonic() < limit:
        proc.poll()  # Reap our immediate child when it exits.
        if not any(alive(pid) for pid in known):
            break
        time.sleep(1)
    killed = []
    for pid in known:
        if send_signal(pid, signal.SIGKILL):
            killed.append(pid)
    proc.wait(timeout=10)
    return {"descendant_pids_signalled": sorted(known), "pids_requiring_kill": sorted(killed), "exit_code": proc.returncode}


def run(request):
    attempt = Path(request["attempt_root"])
    # This non-reusable directory is the atomic admission guard. No alternate
    # attempt path or resume mode exists, so two helpers cannot admit duplicates.
    attempt.mkdir(parents=False, exist_ok=False)
    sync_directory(attempt.parent)
    publish(attempt / "invocation.json", {**request, "sha256": canonical_sha(request)}, exclusive=True)
    state = {"schema": SCHEMA + "-state", "pid": os.getpid(), "started_at": utc(),
             "status": "waiting_for_training", "invocation_sha256": canonical_sha(request), "children": []}
    status_path = attempt / "status.json"
    publish(status_path, state)
    active = None
    old_sigterm = signal.getsignal(signal.SIGTERM)
    old_sigint = signal.getsignal(signal.SIGINT)

    def requested_stop(signum, frame):
        raise StopRequested(f"Continuation received signal {signum}")

    signal.signal(signal.SIGTERM, requested_stop)
    signal.signal(signal.SIGINT, requested_stop)
    try:
        while True:
            verify_sources(request)
            if time.time() >= timestamp(request["wait_stop_utc"]):
                raise TimeoutError("Training did not close successfully before the 05:27:30 UTC continuation cutoff")
            supervision_path = Path(request["supervision"])
            payload = json.loads(supervision_path.read_text())
            summary = supervised_status(payload, request["training_root"])
            state.update(last_checked_at=utc(), training_status=summary)
            publish(status_path, state)
            if summary["ready"]:
                state["closed_supervision"] = file_pin(supervision_path)
                publish(attempt / "closed-supervision.json", payload, exclusive=True)
                break
            time.sleep(min(request["poll_seconds"], max(0, timestamp(request["wait_stop_utc"]) - time.time())))
        endpoint_path = Path(request["training_root"]) / "endpoints-frozen.json"
        if endpoint_path.exists():
            raise FileExistsError("Endpoints were already frozen outside this continuation; refusing a duplicate freeze or evaluation launch")
        if (Path(request["campaign"]) / "evaluation-orchestration/plan.json").exists():
            raise FileExistsError("An evaluation attempt already exists; continuation cannot launch a duplicate")
        for stage, command, timeout_s in (
            ("freeze_endpoints", request["freeze_command"], request["freeze_timeout_seconds"]),
            ("evaluate", request["evaluation_command"], None),
        ):
            verify_sources(request)
            # Recheck the exact closed-success condition immediately before each
            # mutation, in case a foreign writer changed the supervision record.
            current = json.loads(Path(request["supervision"]).read_text())
            if not supervised_status(current, request["training_root"])["ready"]:
                raise ValueError("Training closure changed before continuation stage")
            if time.time() >= timestamp(request["evaluation_stop_utc"]):
                raise TimeoutError("Evaluation cutoff already passed")
            if stage == "evaluate":
                if not endpoint_path.exists():
                    raise ValueError("Successful freeze process did not publish frozen endpoints")
                state["frozen_endpoints"] = file_pin(endpoint_path)
            log_path = attempt / f"{stage}.log"
            started = time.monotonic()
            with log_path.open("xb") as log:
                active = subprocess.Popen(command, cwd=request["source"], env=environment(request["source"]),
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row = {"stage": stage, "pid": active.pid, "command": command, "log": str(log_path),
                       "started_at": utc(), "exit_code": None}
                state["children"].append(row)
                state["status"] = stage
                publish(status_path, state)
                print(json.dumps({"event": "launched", **row}), flush=True)
                last_flush = time.monotonic()
                while active.poll() is None:
                    if timeout_s is not None and time.monotonic() - started >= timeout_s:
                        raise TimeoutError(f"{stage} exceeded its fixed {timeout_s}-second limit")
                    if time.time() >= timestamp(request["evaluation_stop_utc"]):
                        raise TimeoutError("Continuation reached the absolute evaluation cutoff")
                    if time.monotonic() - last_flush >= request["poll_seconds"]:
                        state["last_checked_at"] = utc()
                        publish(status_path, state)
                        os.fsync(log.fileno())
                        last_flush = time.monotonic()
                    time.sleep(1)
                row.update(exit_code=active.wait(), finished_at=utc(), elapsed_s=time.monotonic() - started)
                active = None
                log.flush()
                os.fsync(log.fileno())
            row["log_pin"] = file_pin(log_path)
            publish(status_path, state)
            print(json.dumps({"event": "closed", **row}), flush=True)
            if row["exit_code"] != 0:
                raise RuntimeError(f"{stage} failed with exit code {row['exit_code']}; continuation stopped without retry")
        state["status"] = "complete"
    except BaseException as exc:
        # Ignore repeated cancellation while allowing the active child auditor to
        # retain traces; descendants still face the bounded kill fallback.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if active is not None and active.poll() is None:
            try:
                state["child_cleanup"] = stop_child_tree(active)
            except BaseException as cleanup:
                state["cleanup_failure"] = {"type": type(cleanup).__name__, "message": str(cleanup)}
                send_signal(active.pid, signal.SIGKILL)
        state["status"] = "stopped"
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        state["finished_at"] = utc()
        publish(status_path, state)
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--driver", type=Path, default=Path(__file__).with_name("campaign-evaluate.py"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--execute", action="store_true", help="Admit the exclusive one-shot continuation; otherwise print a read-only plan")
    args = parser.parse_args()
    request = invocation(args.campaign, args.source, args.driver, args.python)
    if not args.execute:
        print(json.dumps({"mode": "read_only_plan", "invocation": request}, indent=2))
        return
    result = run(request)
    print(json.dumps({"status": result["status"], "status_path": str(Path(request["attempt_root"]) / "status.json")}))


if __name__ == "__main__":
    main()
