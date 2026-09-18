"""One-shot final evaluation supervisor; never trains, retries, or replaces endpoints.

The midpoint operation must close first. Held-out access is created only after
the frozen evaluator authenticates every final policy and the training closure.
This orchestration source is pinned at invocation, separate from frozen science.
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

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "free-ppo-v1-final-evaluation-operation"
SEEDS = (101, 211, 307)
CUTOFF = datetime(2026, 9, 17, 14, 0, tzinfo=timezone.utc).timestamp()


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def publish(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checked_pin(record):
    if pin(record["path"]) != record:
        raise ValueError(f"Pinned file changed: {record['path']}")


def closed_training(training, plan):
    """None means not ready; any observed failure or malformed closure is fatal."""
    for seed in SEEDS:
        folder = training / f"seed-{seed}"
        if (folder / "failure.json").exists():
            raise ValueError(f"Training seed{seed} retained a failure; no retry")
    path = training / "campaign-closure.json"
    if not path.exists():
        return None
    try:
        closure = read(path)
    except json.JSONDecodeError:
        # Publication uses exclusive creation followed by a write. A reader
        # may see an unfinished JSON document before the next15second poll.
        return None
    runs = closure.get("runs", [])
    if (closure.get("schema") != "free-direct-scalar-ppo-v1"
            or closure.get("all_fixed_endpoints_completed") is not True
            or closure.get("plan_sha256") != digest(plan)
            or sorted(r.get("optimizer_seed", -1) for r in runs) != list(SEEDS)):
        raise ValueError("Training closure does not contain exactly the three successful fixed endpoints")
    for run in runs:
        expected = training / f"seed-{run['optimizer_seed']}/result.json"
        if (run.get("exit_code") != 0 or run.get("status") != "completed"
                or run.get("num_timesteps") != plan["total_timesteps"]
                or run.get("result") != pin(expected)):
            raise ValueError("Training closure has a failed or unbound endpoint")
        result = read(expected)  # The closure follows wait() on every writer.
        if (result.get("status") != "completed" or result.get("optimizer_seed") != run["optimizer_seed"]
                or result.get("num_timesteps") != plan["total_timesteps"]
                or result.get("requested_timesteps") != plan["total_timesteps"]
                or result.get("plan_sha256") != digest(plan)
                or result.get("actor_parameters_changed") is not True):
            raise ValueError("Incomplete or mismatched final training result")
    return pin(path)


def midpoint_closed(campaign, freeze_sha):
    path = campaign / "preflight/midpoint-operation/status.json"
    if not path.exists():
        return None
    state = read(path)
    if state.get("status") == "failed":
        raise ValueError("Midpoint supervisor failed; no final evaluation overlap or retry")
    if state.get("status") != "completed":
        return None
    panels = state.get("panels", [])
    if (state.get("schema") != "free-ppo-v1-midpoint-operation"
            or state.get("optimizer_seeds") != list(SEEDS)
            or state.get("checkpoint_steps") != 393216
            or [p.get("optimizer_seed") for p in panels] != list(SEEDS)
            or any(p.get("exit_code") != 0 for p in panels)):
        raise ValueError("Midpoint operation does not match the fixed three-panel sequence")
    pins = [pin(path)]
    for seed in SEEDS:
        p = campaign / f"evaluations/learning-curves/free-ppo-v1-{seed}-step-393216/summary.json"
        panel = read(p)
        if (panel.get("evaluation_freeze_sha256") != freeze_sha
                or panel.get("source_files_unchanged_during_panel") is not True
                or panel.get("split") != "development"
                or panel.get("controller") != f"free-ppo-v1-{seed}-step-393216"
                or panel.get("requested_seeds") != list(range(10100000, 10100008))
                or [r["seed"] for r in panel["runs"]] != list(range(10100000, 10100008))):
            raise ValueError("Midpoint source binding or case denominator differs")
        pins.append(pin(p))
    return pins


def descendants(pid):
    result = subprocess.run(["ps", "-axo", "pid=,ppid="], capture_output=True, text=True, check=True)
    pairs = [tuple(map(int, line.split())) for line in result.stdout.splitlines() if line.strip()]
    found = {pid}
    while True:
        expanded = found | {child for child, parent in pairs if parent in found}
        if expanded == found:
            return found
        found = expanded


class Operation:
    def __init__(self, campaign, analysis_python):
        self.campaign = campaign.resolve()
        self.directory = self.campaign / "preflight/final-evaluation-operation"
        self.freeze_path = self.campaign / "evaluation-freeze.json"
        self.freeze = read(self.freeze_path)
        self.training = self.campaign / "training"
        self.plan = read(self.training / "plan.json")
        self.analysis_python = analysis_python.absolute()
        self.evaluator = Path(__file__).with_name("evaluate.py")
        self.analyzer = Path(__file__).with_name("analyze.py")
        self.endpoints = self.campaign / "endpoints-frozen.json"
        self.held_gate = self.campaign / "gate-held-out.json"
        self.dev_gate = self.campaign / "gate-development.json"
        self.inputs = [pin(p) for p in (Path(__file__), self.freeze_path, self.training / "plan.json",
                                        self.dev_gate, self.analysis_python, self.evaluator, self.analyzer)]
        self.active = None
        self.cancelled = False
        self.state = {"schema": SCHEMA, "pid": os.getpid(), "started_at": now(), "status": "waiting",
                      "cutoff_utc": datetime.fromtimestamp(CUTOFF, timezone.utc).isoformat(),
                      "inputs": self.inputs, "commands": [], "case_failures": [], "workers_per_panel": 3,
                      "sequence": [*[f"development/free-ppo-v1-{s}" for s in SEEDS],
                                   "held_out/coverage-direct", "held_out/untrained-101",
                                   *[f"held_out/free-ppo-v1-{s}" for s in SEEDS], "analysis"]}

    def check_sources(self):
        for record in self.inputs + self.freeze["sources"]:
            checked_pin(record)
        if self.freeze["sha256"] != digest({k: v for k, v in self.freeze.items() if k != "sha256"}):
            raise ValueError("Invalid evaluation freeze digest")
        if (self.freeze["training_plan"] != pin(self.training / "plan.json")
                or self.plan["optimizer_seeds"] != list(SEEDS)
                or self.plan["total_timesteps"] != 786432
                or self.freeze["selection"]["development_checkpoint_timesteps"] != [0, 65536, 393216, 786432]):
            raise ValueError("Supervisor does not match the frozen campaign")

    def write(self):
        self.state["updated_at"] = now()
        temporary = self.directory / "status.json.tmp"
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.directory / "status.json")

    def interrupt(self, signum, frame):
        self.cancelled = True

    def bound(self):
        if self.cancelled:
            raise InterruptedError("Supervisor cancelled; active descendants will stop")
        if time.time() >= CUTOFF:
            raise TimeoutError("14:00UTC evaluation cutoff reached")

    def stop_child(self):
        if self.active is None:
            return
        targets = {self.active.pid}
        errors = self.state.setdefault("cleanup_notes", [])
        try:
            targets |= descendants(self.active.pid)
        except Exception as exc:
            errors.append({"at": now(), "operation": "optional_descendant_discovery", "error": str(exc)})
        # The known evaluator group can always be signalled without ps. Its
        # own SIGTERM handler stops its separately grouped case workers.
        try: os.killpg(self.active.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        except OSError as exc: errors.append({"at": now(), "operation": "group_sigterm", "error": str(exc)})
        for pid in targets:
            try: os.kill(pid, signal.SIGTERM)
            except ProcessLookupError: pass
            except OSError as exc: errors.append({"at": now(), "operation": "child_sigterm", "error": str(exc)})
        try: self.active.wait(timeout=20)
        except subprocess.TimeoutExpired: pass
        try: os.killpg(self.active.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        except OSError as exc: errors.append({"at": now(), "operation": "group_sigkill", "error": str(exc)})
        for pid in targets:
            try: os.kill(pid, signal.SIGKILL)
            except ProcessLookupError: pass
            except OSError as exc: errors.append({"at": now(), "operation": "child_sigkill", "error": str(exc)})
        try: self.active.wait(timeout=5)
        except subprocess.TimeoutExpired:
            errors.append({"at": now(), "operation": "wait_after_kill", "error": "Known child did not close"})

    def run_command(self, name, command):
        self.bound()
        self.check_sources()
        log = self.directory / f"{name}.log"
        record = {"name": name, "command": command, "started_at": now(), "log": str(log)}
        self.state["status"] = name
        self.state["commands"].append(record)
        with log.open("x") as stream:
            self.active = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True, env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
            record["pid"] = self.active.pid
            self.write()
            try:
                while self.active.poll() is None:
                    self.bound()
                    try: self.active.wait(timeout=1)
                    except subprocess.TimeoutExpired: pass
            except BaseException:
                self.stop_child()
                raise
            finally:
                record.update(exit_code=self.active.poll(), closed_at=now(), log_pin=pin(log))
                self.write()
        if record["exit_code"] != 0:
            self.stop_child()
            self.active = None
            raise RuntimeError(f"{name} exited{record['exit_code']}; retained output, no retry")
        self.active = None
        self.check_sources()
        print(json.dumps({"closed": name, "exit_code": record["exit_code"]}), flush=True)

    def panel(self, split, kind, seed=None):
        label = f"free-ppo-v1-{seed}" if kind == "ppo" else "coverage-direct" if kind == "coverage" else "untrained-101"
        output = self.campaign / f"evaluations/{split}/{label}"
        command = [sys.executable, str(self.evaluator), "panel", "--freeze", str(self.freeze_path),
                   "--gate", str(self.held_gate if split == "held_out" else self.dev_gate), "--controller", kind,
                   "--split", split, "--output", str(output), "--workers", "3", "--case-timeout", "300"]
        if seed is not None: command += ["--optimizer-seed", str(seed)]
        if split == "held_out": command += ["--endpoints", str(self.endpoints)]
        self.run_command(f"{split}-{label}", command)
        summary = read(output / "summary.json")
        for row in summary["runs"]:
            if not row["execution_complete"]:
                self.state["case_failures"].append({"split": split, "controller": label, "seed": row["seed"], "failure": row["failure"]})
        self.state["commands"][-1]["summary"] = pin(output / "summary.json")
        self.write()

    def execute(self):
        self.check_sources()
        self.directory.mkdir()
        self.write()
        signal.signal(signal.SIGTERM, self.interrupt)
        signal.signal(signal.SIGINT, self.interrupt)
        try:
            if self.endpoints.exists() or self.held_gate.exists() or (self.campaign / "analysis").exists():
                raise FileExistsError("A downstream attempt already exists; never resume or overwrite")
            while True:
                self.bound()
                self.check_sources()
                closure = closed_training(self.training, self.plan)
                midpoints = midpoint_closed(self.campaign, self.freeze["sha256"])
                if closure and midpoints:
                    self.state.update(training_closure=closure, midpoint_inputs=midpoints)
                    self.write()
                    break
                time.sleep(15)
            self.run_command("freeze-endpoints", [sys.executable, str(self.evaluator), "freeze-endpoints", "--freeze", str(self.freeze_path), "--output", str(self.endpoints)])
            endpoint_record = read(self.endpoints)
            self.bound()
            self.check_sources()
            checked_pin(self.state["training_closure"])
            publish(self.held_gate, {"schema": "free-ppo-v1-evaluation-gate", "created_at": now(),
                                    "environment_identity_sha256": self.freeze["environment_identity_sha256"],
                                    "source_identity_sha256": self.freeze["source_identity_sha256"],
                                    "allowed_splits": ["held_out"], "endpoints_frozen": True,
                                    "evaluation_freeze_sha256": self.freeze["sha256"],
                                    "endpoint_freeze_sha256": endpoint_record["sha256"], "endpoint_freeze": pin(self.endpoints)})
            self.state.update(endpoints=pin(self.endpoints), held_out_gate=pin(self.held_gate))
            self.inputs += [pin(self.endpoints), pin(self.held_gate)]
            self.write()
            for seed in SEEDS: self.panel("development", "ppo", seed)
            self.panel("held_out", "coverage")
            self.panel("held_out", "untrained")
            for seed in SEEDS: self.panel("held_out", "ppo", seed)
            self.run_command("analysis", [str(self.analysis_python), str(self.analyzer), "--root", str(self.campaign / "evaluations"),
                                          "--freeze", str(self.freeze_path), "--output", str(self.campaign / "analysis"), "--require-complete-panel", "--plots"])
            self.state.update(status="completed" if not self.state["case_failures"] else "completed_with_retained_case_failures",
                              completed_at=now(), analysis=pin(self.campaign / "analysis/analysis.json"))
            self.write()
        except BaseException as exc:
            self.stop_child()
            self.state.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)}, closed_at=now())
            self.write()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--analysis-python", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    operation = Operation(args.campaign, args.analysis_python)
    operation.check_sources()
    if args.execute: operation.execute()
    print(json.dumps(operation.state, indent=2))


if __name__ == "__main__":
    main()
