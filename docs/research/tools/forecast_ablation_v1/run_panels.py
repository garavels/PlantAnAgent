"""Run a frozen list of panels once with a global three-process ceiling."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from docs.research.tools.forecast_ablation_v1 import evaluate as e


def run_panels(*, freeze_path, output, gate_path=None, workers=3, case_timeout=300,
               deadline_utc=None, min_free_gib=2, max_decisions=None, split=None, main_completion_path=None):
    freeze = e.require_freeze(freeze_path)
    panels = [p for p in freeze["panels"] if split is None or p["split"] == split]
    if not panels or not 1 <= workers <= 3 or case_timeout <= 0:
        raise ValueError("Empty selection or invalid worker/timeout bound")
    if max_decisions is not None and any(p["split"] != "train" for p in panels):
        raise ValueError("Short fixtures require training panels")
    deadline = datetime.fromisoformat((deadline_utc or freeze["hard_stop_utc"]).replace("Z", "+00:00"))
    if deadline and deadline.tzinfo is None:
        raise ValueError("Deadline must be timezone aware")
    deadline = min(deadline, datetime.fromisoformat(freeze["hard_stop_utc"]))
    deadline_utc = deadline.isoformat()
    for panel in panels:
        e.admit(freeze, controller=panel["controller"], condition=panel["condition"], split=panel["split"],
            seed=panel["seeds"][0], gate_path=gate_path, max_decisions=max_decisions, main_completion_path=main_completion_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output).free < min_free_gib * 1024**3:
        raise OSError("Insufficient disk reserve")
    e.publish(output / "attempt.json", {"schema": e.SCHEMA + "-wave-attempt", "pid": os.getpid(),
        "evaluation_freeze": e.file_pin(freeze_path), "evaluation_freeze_sha256": freeze["sha256"], "panels": panels,
        "workers": workers, "case_timeout_s": case_timeout, "deadline_utc": deadline_utc, "created_at": datetime.now(timezone.utc).isoformat()})
    active, lock, cancelled = {}, threading.Lock(), threading.Event()
    def stop_active(signum=None, frame=None):
        cancelled.set()
        with lock:
            for process in active.values():
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
    previous = signal.getsignal(signal.SIGTERM)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, stop_active)
    def expired():
        return deadline is not None and datetime.now(timezone.utc) >= deadline
    def run_one(panel, seed):
        folder = output / panel["split"] / panel["condition"] / panel["controller"]
        folder.mkdir(parents=True, exist_ok=True)
        resultpath, recording, logfile = (folder / f"case-{seed}{suffix}" for suffix in (".result.json", ".json.gz", ".log"))
        began = time.monotonic()
        if cancelled.is_set() or expired() or shutil.disk_usage(output).free < min_free_gib * 1024**3:
            row = e.empty_case(seed=seed, controller=panel["controller"], condition=panel["condition"], split=panel["split"], freeze=freeze,
                failure={"type": "NotStarted", "message": "Campaign cancelled, UTC deadline reached, or disk reserve exhausted; no retry"})
            e.publish(resultpath, row)
            return panel, row
        command = [sys.executable, "-m", "docs.research.tools.forecast_ablation_v1.evaluate", "--freeze", str(Path(freeze_path).resolve()),
            "--controller", panel["controller"], "--condition", panel["condition"], "--split", panel["split"], "--seed", str(seed),
            "--output", str(recording.resolve()), "--result", str(resultpath.resolve())]
        for name, value in (("gate", gate_path), ("main-completion", main_completion_path), ("deadline-utc", deadline_utc), ("max-decisions", max_decisions)):
            if value is not None:
                command.extend(["--" + name, str(value)])
        if seed == panel["seeds"][0]:
            command.append("--retain-minutes")
        with logfile.open("x") as stream:
            envvars = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            with lock:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=envvars, cwd=ROOT, start_new_session=True)
                active[process.pid] = process
            interrupted_at = None
            try:
                while process.poll() is None:
                    if cancelled.is_set() or expired() or time.monotonic() - began >= case_timeout:
                        if interrupted_at is None:
                            interrupted_at = time.monotonic()
                            try:
                                os.killpg(process.pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                        elif time.monotonic() - interrupted_at > 15:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                    try:
                        process.wait(timeout=.25)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                with lock:
                    active.pop(process.pid, None)
        if resultpath.exists():
            row = json.loads(resultpath.read_text())
        else:
            row = e.empty_case(seed=seed, controller=panel["controller"], condition=panel["condition"], split=panel["split"], freeze=freeze,
                failure={"type": "WorkerFailed", "returncode": process.returncode, "log": e.file_pin(logfile)}, elapsed=time.monotonic() - began)
            e.publish(resultpath, row)
        print(json.dumps({"controller": panel["controller"], "condition": panel["condition"], "seed": seed,
            "complete": row["execution_complete"], "all_deliveries": row["every_delivery_passed"], "seconds": round(time.monotonic() - began, 2)}), flush=True)
        return panel, row
    by_panel = {(p["split"], p["condition"], p["controller"]): [] for p in panels}
    summary_pins = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_one, panel, seed) for panel in panels for seed in panel["seeds"]]
            for future in as_completed(futures):
                panel, row = future.result()
                key = panel["split"], panel["condition"], panel["controller"]
                by_panel[key].append(row)
                if len(by_panel[key]) == len(panel["seeds"]):
                    unchanged = freeze["sources"] == e.sources()
                    summary = {"schema": e.SCHEMA + "-panel", "controller": panel["controller"], "condition": panel["condition"], "split": panel["split"],
                        "requested_seeds": panel["seeds"], "runs": sorted(by_panel[key], key=lambda r: r["seed"]),
                        "evaluation_freeze": e.file_pin(freeze_path), "evaluation_freeze_sha256": freeze["sha256"],
                        "source_files_unchanged_during_panel": unchanged, "completed_at": datetime.now(timezone.utc).isoformat()}
                    path = output.joinpath(*key) / "summary.json"
                    e.publish(path, summary)
                    summary_pins.append(e.file_pin(path))
    except BaseException:
        stop_active()
        raise
    finally:
        stop_active()
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, previous)
    summary = {"schema": e.SCHEMA + "-wave", "evaluation_freeze": e.file_pin(freeze_path), "panels": summary_pins,
        "all_requested_cases_retained": all(len(by_panel[p["split"], p["condition"], p["controller"]]) == len(p["seeds"]) for p in panels),
        "all_execution_complete": all(r["execution_complete"] for rows in by_panel.values() for r in rows),
        "source_files_unchanged": freeze["sources"] == e.sources(), "completed_at": datetime.now(timezone.utc).isoformat()}
    e.publish(output / "wave.json", summary)
    if not summary["source_files_unchanged"]:
        raise ValueError("Source changed during panel wave")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gate", type=Path)
    p.add_argument("--main-completion", type=Path)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--case-timeout", type=float, default=300)
    p.add_argument("--deadline-utc")
    p.add_argument("--min-free-gib", type=float, default=2)
    p.add_argument("--max-decisions", type=int)
    p.add_argument("--split", choices=("train", "held_out"))
    args = p.parse_args()
    result = run_panels(freeze_path=args.freeze, output=args.output, gate_path=args.gate, workers=args.workers,
        case_timeout=args.case_timeout, deadline_utc=args.deadline_utc, min_free_gib=args.min_free_gib,
        max_decisions=args.max_decisions, split=args.split, main_completion_path=args.main_completion)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
