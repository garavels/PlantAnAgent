"""One-attempt campaign orchestration, explicitly launched with --execute.

This helper never trains or writes the training directory. It requires all six
closed training processes and their frozen endpoints. Evaluation executes only
from the specified immutable source snapshot. Existing complete panels are
verified before reuse; incomplete or source-inconsistent panels are never skipped.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

SCHEMA = "authority-ppo-v2-campaign-evaluation-driver"
FINAL_STOP_UTC = "2026-09-17T10:26:17Z"
MODULE = "docs.research.tools.authority_ppo_v2.evaluate"
ANALYSIS_MODULE = "docs.research.tools.authority_ppo_v2.analyze"
ARMS = ("narrow", "broad")
OPTIMIZERS = (101, 211, 307)
INTERMEDIATES = (65536, 131072, 196608)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def parse_deadline(value):
    date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if date.tzinfo is None:
        raise ValueError("Stop time must have a timezone")
    if date.timestamp() > datetime.fromisoformat(FINAL_STOP_UTC.replace("Z", "+00:00")).timestamp():
        raise ValueError("Stop cannot exceed the absolute research cutoff")
    return date.timestamp()


def load_snapshot(source):
    """Resolve imports only after the caller names the source snapshot."""
    source = Path(source).resolve()
    for directory in (source / "simulator/src", source / "backend/src", source):
        sys.path.insert(0, str(directory))
    from docs.research.tools.authority_ppo_v2 import evaluate
    if Path(evaluate.__file__).resolve() != source / "docs/research/tools/authority_ppo_v2/evaluate.py":
        raise ValueError("Evaluator resolved outside the explicitly selected source snapshot")
    return evaluate


def training_is_closed(campaign, training_root, evaluator):
    supervision_path = campaign / "process-logs/supervision.json"
    supervision = json.loads(supervision_path.read_text())
    records = supervision.get("processes", [])
    expected = {(arm, seed) for arm in ARMS for seed in OPTIMIZERS}
    observed = {(row["arm"], row["optimizer_seed"]) for row in records}
    if len(records) != 6 or observed != expected:
        raise ValueError("Evaluation requires all six supervised training attempts")
    if any(row.get("exit_code") != 0 or not row.get("finished_at") for row in records):
        raise ValueError("Evaluation requires all six training processes to have finished successfully")
    # Do not create endpoints-frozen.json here: the helper cannot write training.
    endpoints = evaluator.training().require_frozen_endpoints(training_root)
    return {"supervision": file_pin(supervision_path), "endpoints": endpoints}


def panel_step(campaign, split, label, kind, *, arm=None, optimizer=None, checkpoint_steps=None, workers=3):
    if checkpoint_steps is not None:
        output = campaign / "evaluations/learning-curves" / f"{label}-step-{checkpoint_steps}"
        ident = f"learning-{label}-{checkpoint_steps}"
    else:
        output = campaign / "evaluations" / split / label
        ident = f"{split}-{label}"
    return {"id": ident, "kind": "panel", "controller": kind, "label": label, "arm": arm,
            "optimizer_seed": optimizer, "split": split, "checkpoint_steps": checkpoint_steps,
            "output": str(output), "workers": workers, "case_timeout_s": 300,
            "priority": "optional_diagnostic" if checkpoint_steps else "required"}


def make_plan(campaign, source, python, training_root, stop_utc, min_free_gib):
    steps = []
    for split in ("development", "held_out"):
        steps.append(panel_step(campaign, split, "coverage", "coverage"))
        for arm in ARMS:
            for seed in OPTIMIZERS:
                steps.append(panel_step(campaign, split, f"{arm}-{seed}", "ppo", arm=arm, optimizer=seed))
    steps.append(panel_step(campaign, "held_out", "mpc", "mpc", workers=2))
    for arm, seed in (("narrow", 9960000), ("broad", 9960000), ("broad", 9960001)):
        steps.append({"id": f"replay-{arm}-101-{seed}", "kind": "replay", "priority": "required",
                      "original": str(campaign / f"evaluations/held_out/{arm}-101/{seed}.json.gz"),
                      "output": str(campaign / f"replay-audits/{arm}-101-{seed}")})
    steps.append({"id": "analysis-primary", "kind": "analysis", "priority": "required",
                  "output": str(campaign / "analysis-primary")})
    for transitions in INTERMEDIATES:
        for arm in ARMS:
            for seed in OPTIMIZERS:
                steps.append(panel_step(campaign, "development", f"{arm}-{seed}", "ppo", arm=arm,
                                        optimizer=seed, checkpoint_steps=transitions))
    steps.append({"id": "analysis-final", "kind": "analysis", "priority": "required",
                  "output": str(campaign / "analysis")})
    return {"schema": SCHEMA + "-plan", "campaign": str(campaign), "source": str(source),
            "python": str(python), "training_root": str(training_root), "stop_utc": stop_utc,
            "minimum_free_gib": min_free_gib, "maximum_case_seconds": 300,
            "hard_stop_preparation_seconds": 90, "optional_cutoff_reserve_seconds": 900,
            "attempts_per_case": 1, "driver_source": file_pin(Path(__file__)),
            "analysis_source": file_pin(source / "docs/research/tools/authority_ppo_v2/analyze.py"),
            "freeze_path": str(campaign / "evaluation-freeze.json"), "steps": steps,
            "selection": "All six final PPO policies and coverage on all 8 development and 32 held-out scenarios. Strict MPC only first eight test scenarios. Intermediate endpoints only on development, no final-selection changes. All failures retained. Replay audits are not new replications."}


def child_environment(source):
    return {**os.environ,
            "PYTHONPATH": os.pathsep.join(map(str, (source, source / "backend/src", source / "simulator/src"))),
            "PYTHONUNBUFFERED": "1",
            **{key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def command_for_step(step, plan):
    common = [plan["python"], "-m", MODULE, "--training-root", plan["training_root"],
              "--freeze-path", plan["freeze_path"], "--output", step["output"],
              "--min-free-gib", str(plan["minimum_free_gib"])]
    if step["kind"] == "replay":
        return common + ["--replay-of", step["original"], "--case-timeout-s", "300"]
    if step["kind"] == "analysis":
        return [plan["python"], "-m", ANALYSIS_MODULE, "--root", str(Path(plan["campaign"]) / "evaluations"),
                "--output", step["output"], "--require-complete-panel"]
    result = common + ["--controller", step["controller"], "--split", step["split"],
                       "--workers", str(step["workers"]), "--case-timeout-s", str(step["case_timeout_s"])]
    if step["arm"] is not None:
        result += ["--arm", step["arm"], "--optimizer-seed", str(step["optimizer_seed"])]
    if step["checkpoint_steps"] is not None:
        result += ["--checkpoint-steps", str(step["checkpoint_steps"])]
    return result


def verify_panel(step, plan, evaluator, freeze_sha):
    path = Path(step["output"]) / "summary.json"
    report = json.loads(path.read_text())
    expected_label = evaluator.controller_label(step["controller"], step["arm"], step["optimizer_seed"])
    if step["checkpoint_steps"] is not None:
        expected_label += f'-step-{step["checkpoint_steps"]}'
    expected_seeds = evaluator.select_seeds(step["controller"], step["split"], None)
    if (report.get("schema") != evaluator.SCHEMA + "-panel" or report["controller"] != expected_label
        or report["split"] != step["split"] or report["requested_seeds"] != expected_seeds
        or report.get("checkpoint_steps") != step["checkpoint_steps"]
        or report["benchmark_contract_sha256"] != evaluator.benchmark().load_benchmark_contract()["contract_sha256"]
        or report["source_files"] != evaluator.comparison_sources()
        or report.get("source_files_unchanged_during_panel") is not True or report.get("validity_failure")
        or sorted(row["seed"] for row in report["runs"]) != sorted(expected_seeds)):
        raise ValueError(f"Existing panel is incomplete or differs from frozen source/selection: {path}")
    if step["split"] == "held_out" and report["evaluation_freeze_sha256"] != freeze_sha:
        raise ValueError("Held-out panel belongs to a different evaluation freeze")
    for row in report["runs"]:
        if row.get("recording"):
            if evaluator.file_pin(Path(row["recording"]["path"])) != row["recording"]:
                raise ValueError("Existing panel recording differs from its published checksum")
        elif any(row.get(key) for key in ("execution_complete", "mission_passed", "hard_safety_passed", "every_delivery_passed", "all_daily_returns_passed")):
            raise ValueError("Unrecorded runner failure received a full-week pass")
    return {"summary": file_pin(path), "cases": len(report["runs"]),
            "mission_passes": sum(row["mission_passed"] for row in report["runs"]),
            "runner_failures": sum(row.get("recording") is None for row in report["runs"])}


def verify_existing(step, plan, evaluator, freeze_sha):
    output = Path(step["output"])
    if not output.exists():
        return None
    if step["kind"] == "panel":
        return verify_panel(step, plan, evaluator, freeze_sha)
    if step["kind"] == "replay":
        path = output / "replay-audit.json"
        result = json.loads(path.read_text())
        if (result["evaluation_freeze_sha256"] != freeze_sha or result["original"] != evaluator.file_pin(Path(step["original"]))):
            raise ValueError("Existing replay belongs to a different source recording or freeze")
        replay_pin = result.get("replay", {}).get("recording")
        if replay_pin and evaluator.file_pin(Path(replay_pin["path"])) != replay_pin:
            raise ValueError("Replay recording checksum differs")
        return {"audit": file_pin(path), "deterministic_replay_passed": result["deterministic_replay_passed"]}
    path = output / "analysis.json"
    result = json.loads(path.read_text())
    if result["evaluation_freeze_sha256"] != freeze_sha or result["analysis_source"] != plan["analysis_source"]:
        raise ValueError("Existing analysis source or freeze differs")
    for pin in result["summaries"]:
        if evaluator.file_pin(Path(pin["path"])) != pin:
            raise ValueError("Existing analysis summarizes an altered panel")
    return {"analysis": file_pin(path), "outcomes": len(result["rows"])}


def descendant_pids(parent):
    text = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
    children = {}
    for line in text.splitlines():
        pid, ppid = map(int, line.split())
        children.setdefault(ppid, []).append(pid)
    descendants, pending = [], [parent]
    while pending:
        current = pending.pop()
        for pid in children.get(current, []):
            descendants.append(pid)
            pending.append(pid)
    return descendants


def signal_if_alive(pid, sig):
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def terminate_step(proc):
    """Freeze launcher, let active case auditors serialize, then kill leftovers."""
    signal_if_alive(proc.pid, signal.SIGSTOP)
    children = descendant_pids(proc.pid)
    for pid in children:
        signal_if_alive(pid, signal.SIGTERM)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        live = []
        for pid in children:
            try:
                os.kill(pid, 0)
                live.append(pid)
            except ProcessLookupError:
                pass
        if not live:
            break
        time.sleep(1)
    for pid in children:
        signal_if_alive(pid, signal.SIGKILL)
    signal_if_alive(proc.pid, signal.SIGKILL)
    return proc.wait(timeout=10)


def retain_interrupted_panel(step, evaluator, *, deadline=True):
    """Account explicitly for every requested case without replaying any case."""
    output = Path(step["output"])
    request = output / "panel-request.json"
    if not request.exists():
        return None
    report = json.loads(request.read_text())
    if report["source_files"] != evaluator.comparison_sources():
        raise ValueError("Cannot qualify interrupted panel after source changed")
    rows = []
    for seed in report["requested_seeds"]:
        provenance = output / f"{seed}-provenance.json"
        failure = output / f"{seed}-runner-failure.json"
        if provenance.exists():
            row = json.loads(provenance.read_text())
        elif failure.exists():
            row = json.loads(failure.read_text())
        else:
            progress = output / f"{seed}-progress.json"
            cause = ("CampaignDeadlineInterrupted" if progress.exists() else "CampaignDeadlineNotRun") if deadline else "CampaignLauncherFailure"
            row = evaluator.failure_row(seed, report["controller"], report["split"], report["evaluation_freeze_sha256"],
                cause, "Campaign execution stopped; missing measurements retained, not retried and not imputed.", 0)
            if progress.exists():
                row["last_progress"] = json.loads(progress.read_text())
            evaluator.publish(failure, row)
        rows.append(row)
    report.update(runs=rows, source_files_unchanged_during_panel=True, completed_at_unix_s=time.time(),
                  campaign_deadline_interrupted=deadline, campaign_launcher_failure=not deadline,
                  interpretation="Some cases were interrupted or not run. These are infrastructure failures, not evidence of controller infeasibility; all full-week flags remain false.")
    evaluator.publish(output / "summary.json", report, exclusive=False)
    return file_pin(output / "summary.json")


def execute(plan, evaluator, *, resume=False):
    campaign = Path(plan["campaign"])
    training_root = Path(plan["training_root"])
    freeze_path = Path(plan["freeze_path"])
    stop_s = parse_deadline(plan["stop_utc"])
    if time.time() >= stop_s - 90:
        raise TimeoutError("Insufficient time before the absolute campaign cutoff")
    closed = training_is_closed(campaign, training_root, evaluator)
    evaluator.check_disk(campaign, plan["minimum_free_gib"])
    orchestration = campaign / "evaluation-orchestration"
    plan_path = orchestration / "plan.json"
    state_path = orchestration / "state.json"
    previous_state_pin = None
    if plan_path.exists():
        if not resume:
            raise FileExistsError("Driver already has a frozen plan; use explicit --resume after verifying the previous driver stopped")
        saved = json.loads(plan_path.read_text())
        if saved["plan"] != plan or saved["sha256"] != digest(saved["plan"]):
            raise ValueError("Driver plan/source changed; refusing to resume")
        previous = json.loads(state_path.read_text())
        oldpid = previous.get("pid")
        if oldpid:
            try:
                os.kill(oldpid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("Previous orchestration PID is still alive; no duplicate execution")
        archive = orchestration / "prior-attempts" / f'state-{digest(previous)[:16]}.json'
        if archive.exists():
            if json.loads(archive.read_text()) != previous:
                raise ValueError("Prior attempt archive identity differs")
        else:
            write(archive, previous, exclusive=True)
        previous_state_pin = file_pin(archive)
    else:
        orchestration.mkdir(parents=True, exist_ok=True)
        write(plan_path, {"schema": SCHEMA + "-frozen-plan", "frozen_at": utc(), "plan": plan,
                          "sha256": digest(plan), "training_closed": closed}, exclusive=True)
    if freeze_path.exists():
        freeze = evaluator.require_evaluation_freeze(training_root, freeze_path)
    else:
        freeze = evaluator.freeze_evaluation(training_root, freeze_path)
    state = {"schema": SCHEMA + "-state", "pid": os.getpid(), "started_at": utc(),
             "stop_utc": plan["stop_utc"], "plan_sha256": digest(plan),
             "evaluation_freeze_sha256": freeze["sha256"], "steps": [], "status": "running",
             "previous_attempt": previous_state_pin}
    write(state_path, state)
    failures = []
    optional_budget_exhausted = False
    active_process = None
    active_step = None
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def stop_requested(signum, frame):
        raise RuntimeError("Campaign driver received a termination request; retaining current case outcomes")

    signal.signal(signal.SIGTERM, stop_requested)
    try:
        for index, step in enumerate(plan["steps"]):
            if file_pin(Path(__file__)) != plan["driver_source"] or file_pin(Path(plan["analysis_source"]["path"])) != plan["analysis_source"]:
                raise ValueError("Driver or analysis source changed after step-plan freeze")
            evaluator.require_evaluation_freeze(training_root, freeze_path)
            existing = verify_existing(step, plan, evaluator, freeze["sha256"])
            if existing is not None:
                row = {"step": step["id"], "status": "verified_existing", "verification": existing}
                if step["kind"] == "replay" and not existing["deterministic_replay_passed"]:
                    failures.append(step["id"])
                    row["status"] = "verified_existing_replay_mismatch"
                state["steps"].append(row)
                write(state_path, state)
                print(json.dumps(row), flush=True)
                continue
            remaining = stop_s - time.time()
            if step["priority"] == "optional_diagnostic" and (optional_budget_exhausted or remaining <= plan["optional_cutoff_reserve_seconds"] + 90):
                state["steps"].append({"step": step["id"], "status": "omitted_optional_for_deadline", "remaining_seconds": remaining})
                write(state_path, state)
                continue
            if remaining <= 90:
                state["status"] = "deadline_reached"
                state["unstarted_steps"] = [item["id"] for item in plan["steps"][index:]]
                break
            if step["kind"] == "panel" and step["controller"] == "mpc":
                training_is_closed(campaign, training_root, evaluator)
            command = command_for_step(step, plan)
            log_path = orchestration / f'{step["id"]}.log'
            # One attempt per step. A failed prior attempt requires review, never overwrite.
            with log_path.open("xb") as log:
                proc = subprocess.Popen(command, cwd=plan["source"], env=child_environment(Path(plan["source"])),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active_process, active_step = proc, step
                row = {"step": step["id"], "status": "running", "pid": proc.pid, "started_at": utc(),
                       "command": command, "log": str(log_path)}
                state["steps"].append(row)
                write(state_path, state)
                print(json.dumps(row), flush=True)
                deadline_hit = False
                step_stop_s = stop_s - (plan["optional_cutoff_reserve_seconds"] if step["priority"] == "optional_diagnostic" else 0)
                while proc.poll() is None:
                    if time.time() >= step_stop_s - 90:
                        deadline_hit = True
                        terminate_step(proc)
                        break
                    time.sleep(1)
                row.update(exit_code=proc.wait(), finished_at=utc())
                active_process = None
            if deadline_hit:
                optional = step["priority"] == "optional_diagnostic"
                row["status"] = "optional_budget_interrupted" if optional else "campaign_deadline_interrupted"
                if step["kind"] == "panel":
                    row["retained_summary"] = retain_interrupted_panel(step, evaluator)
                if optional:
                    optional_budget_exhausted = True
                    state["optional_work_stopped_for_deadline"] = True
                    write(state_path, state)
                    print(json.dumps(row), flush=True)
                    continue
                state["status"] = "deadline_reached"
                state["unstarted_steps"] = [item["id"] for item in plan["steps"][index + 1:]]
                write(state_path, state)
                break
            if row["exit_code"] != 0:
                row["status"] = "process_failed"
                failures.append(step["id"])
                if step["kind"] == "panel":
                    # Child failures are usually returned as rows with process exit 0.
                    # An unexpected launcher failure is preserved for explicit review.
                    row["retained_summary"] = retain_interrupted_panel(step, evaluator, deadline=False)
                write(state_path, state)
                print(json.dumps(row), flush=True)
                continue
            row["status"] = "completed"
            row["verification"] = verify_existing(step, plan, evaluator, freeze["sha256"])
            if step["kind"] == "replay" and not row["verification"]["deterministic_replay_passed"]:
                failures.append(step["id"])
                row["status"] = "replay_mismatch_retained"
            write(state_path, state)
            print(json.dumps({"step": step["id"], "status": row["status"], "verification": row["verification"]}), flush=True)
        if state["status"] == "running":
            state["status"] = "completed_with_failures" if failures else "completed"
    except BaseException as exc:
        if active_process is not None and active_process.poll() is None:
            terminate_step(active_process)
            if active_step["kind"] == "panel":
                try:
                    retain_interrupted_panel(active_step, evaluator, deadline=False)
                except Exception as cleanup_error:
                    state["cleanup_error"] = {"type": type(cleanup_error).__name__, "message": str(cleanup_error)}
        state["status"] = "blocked_error"
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        signal.signal(signal.SIGTERM, original_sigterm)
        state["finished_at"] = utc()
        state["failed_steps"] = failures
        write(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--stop-utc", default=FINAL_STOP_UTC)
    parser.add_argument("--min-free-gib", type=float, default=2)
    parser.add_argument("--execute", action="store_true", help="Required to freeze and start evaluation; omitted means read-only plan output")
    parser.add_argument("--resume", action="store_true", help="Resume the identical plan after its previous PID has stopped; never repeats a case")
    args = parser.parse_args()
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        parser.error("Disk reserve must be nonnegative and finite")
    campaign, source = args.campaign.resolve(), args.source.resolve()
    training_root = (args.training_root or campaign / "training").resolve()
    # Keep the supplied virtual-environment executable path; resolve() would turn
    # a venv symlink into its base interpreter and lose installed dependencies.
    python = Path(os.path.abspath(args.python))
    parse_deadline(args.stop_utc)
    evaluator = load_snapshot(source)
    plan = make_plan(campaign, source, python, training_root, args.stop_utc, args.min_free_gib)
    if not args.execute:
        print(json.dumps({"mode": "read_only_plan", "plan": plan, "sha256": digest(plan)}, indent=2))
        return
    result = execute(plan, evaluator, resume=args.resume)
    print(json.dumps({"status": result["status"], "state": str(campaign / "evaluation-orchestration/state.json")}))
    if result["status"] != "completed":
        sys.exit(2)


if __name__ == "__main__":
    main()
