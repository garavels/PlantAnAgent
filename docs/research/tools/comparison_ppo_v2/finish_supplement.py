"""Queue the declared forecast supplement after the main evaluations close."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "docs/research/results/comparison_ppo_20260917"
SUPPLEMENT = RESULTS / "forecast-ablation"
OUTPUT = SUPPLEMENT / "supervision"
DEADLINE = datetime.fromisoformat("2026-09-17T18:42:48+00:00")


def now():
    return datetime.now(timezone.utc).isoformat()


def pin(path):
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def publish(name, value):
    with (OUTPUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def read_published(path):
    for attempt in range(5):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(.2)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source = pin(__file__)
    publish("launch.json", {"schema": "forecast-ablation-finish-launch", "at": now(),
        "source": source, "freeze": pin(SUPPLEMENT / "freeze/freeze.json"),
        "deadline_utc": DEADLINE.isoformat(), "pid": os.getpid(),
        "dependency": str(RESULTS / "supervision/completion.json")})
    child = None
    stage = "waiting_for_main_completion"
    try:
        complete = RESULTS / "supervision/completion.json"
        while not complete.exists():
            if (RESULTS / "supervision/failure.json").exists():
                raise RuntimeError("Main campaign stopped; supplementary cases were not launched")
            if datetime.now(timezone.utc) >= DEADLINE:
                raise TimeoutError("Deadline reached before main completion; supplementary cases were not launched")
            time.sleep(5)
        main_receipt = read_published(complete)
        if main_receipt.get("status") != "completed" or main_receipt.get("outcomes") != 1160:
            raise ValueError("Main campaign completion is not the prescribed 1160-case closure")
        gate = RESULTS / "held-out-gate.json"
        commands = [
            ("held-out", [sys.executable, "-m", "docs.research.tools.forecast_ablation_v1.run_panels",
                "--freeze", str(SUPPLEMENT / "freeze/freeze.json"), "--gate", str(gate),
                "--main-completion", str(complete), "--output", str(SUPPLEMENT / "held-out"),
                "--workers", "3", "--deadline-utc", DEADLINE.isoformat()]),
            ("analysis", [sys.executable, "-m", "docs.research.tools.forecast_ablation_v1.analyze",
                "--root", str(SUPPLEMENT / "held-out"), "--main-analysis", str(RESULTS / "analysis/analysis.json"),
                "--gate", str(gate), "--main-completion", str(complete), "--output", str(SUPPLEMENT / "analysis")]),
        ]
        receipts = []
        for stage, command in commands:
            if pin(__file__) != source:
                raise ValueError("Supplement queue source changed")
            if datetime.now(timezone.utc) >= DEADLINE:
                raise TimeoutError("Deadline reached before next supplementary stage")
            publish(stage + "-launch.json", {"at": now(), "stage": stage, "command": command})
            print(json.dumps({"at": now(), "stage": stage, "event": "started"}), flush=True)
            logfile = OUTPUT / (stage + ".log")
            with logfile.open("x") as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    start_new_session=True)
                while child.poll() is None:
                    if datetime.now(timezone.utc) >= DEADLINE:
                        raise TimeoutError("Supplementary stage reached the authorized deadline")
                    time.sleep(1)
                receipt = {"at": now(), "stage": stage, "exit_code": child.returncode, "log": pin(logfile)}
                child = None
            publish(stage + "-receipt.json", receipt)
            receipts.append(receipt)
            if receipt["exit_code"] != 0:
                raise RuntimeError(f"Supplementary {stage} failed; attempts retained without retry")
        analysis_path = SUPPLEMENT / "analysis/analysis.json"
        result = read_published(analysis_path)
        if len(result["rows"]) != 32:
            raise ValueError("Supplement did not retain all 32 prescribed cases")
        if not result["criterion"]["eligible_complete_audited_pairs"]:
            raise ValueError("Supplement contains incomplete, unaudited, or unpaired cases")
        if not all(row["execution_complete"] and row["physical_invariants_passed"]
                   and row["economic_audit"]["passed"] for row in result["rows"]):
            raise ValueError("Supplement did not complete all 32 cases with passing audits")
        completion = {"schema": "forecast-ablation-finish-completion", "status": "completed", "at": now(),
            "rows": 32, "analysis": pin(analysis_path), "main_completion": pin(complete), "stages": receipts}
        publish("completion.json", completion)
        print(json.dumps(completion), flush=True)
    except BaseException as exc:
        publish("failure.json", {"schema": "forecast-ablation-finish-failure", "status": "stopped",
            "at": now(), "stage": stage, "type": type(exc).__name__, "reason": str(exc),
            "retry_policy": "No case retries or outcome omission"})
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)


if __name__ == "__main__":
    main()
