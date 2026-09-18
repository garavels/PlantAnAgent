"""Finish the already authorized comparison after its existing training closes.

This supervisor never starts, resumes, stops or changes the training campaign.
Every evaluation stage is attempted once; outputs and failures remain immutable.
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
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "docs/research/results/comparison_ppo_20260917"
SCHEMA = "comparison-ppo-v2-finish-supervisor"
SPECS = ("development-learning", "held-out-primary", "held-out-battery")
EXISTING_WAVES = ("development-primary", "development-battery")
EXPECTED_CASES = {"development-learning": 48, "held-out-primary": 832, "held-out-battery": 96}


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    # The existing immutable publisher creates a file then writes JSON. Allow
    # only its brief publication window; this is never a retried experiment.
    for attempt in range(5):
        try:
            return json.loads(Path(path).read_text())
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(.2)


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def deadline(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Deadline must include a timezone")
    return parsed.astimezone(timezone.utc)


def preflight(results=RESULTS):
    from docs.research.tools.comparison_ppo_v2 import environment, evaluate, training
    results = Path(results).resolve()
    authorization = read_json(results / "authorization.json")
    if authorization["deadline_at"] != "2026-09-17T18:42:48+00:00" or authorization["training_cutoff_at"] != "2026-09-17T17:27:48+00:00":
        raise ValueError("Supervisor is bound to the existing four-hour authorization")
    decision = read_json(results / "training-decision.json")
    if decision.get("train_new_policies") is not True:
        raise ValueError("The declared campaign did not authorize new training")
    plan = training.require_plan(results / "training")
    if (plan["optimizer_seeds"] != [101, 211, 307] or plan["total_timesteps"] != 786432
            or plan["training_variant"] != {"delivery_rate_kg_h": 30.0, "opening_battery": "sampled", "battery_mode": "enabled"}
            or deadline(plan["deadline_utc"]) != deadline(authorization["training_cutoff_at"])):
        raise ValueError("Unexpected fixed training plan")
    environment._admit("development", results / "development-gate.json")
    fixed = [results / name for name in ("authorization.json", "training-decision.json", "development-gate.json")]
    fixed.extend((results / "training/plan.json", results / "training/campaign-launch.json"))
    for name in EXISTING_WAVES:
        wave_path = results / name / "wave.json"
        wave = read_json(wave_path)
        if not all(wave.get(k) is True for k in ("all_requested_cases_retained", "all_execution_complete", "source_files_unchanged")):
            raise ValueError(f"Prerequisite wave did not finish cleanly: {name}")
        if pin(wave["evaluation_freeze"]["path"]) != wave["evaluation_freeze"]:
            raise ValueError("Existing wave freeze content changed")
        evaluate.require_freeze(wave["evaluation_freeze"]["path"])
        fixed.extend((wave_path, Path(wave["evaluation_freeze"]["path"])))
    for name in SPECS:
        path = results / (name + "-spec.json")
        spec = read_json(path)
        split = "development" if name == "development-learning" else "held_out"
        count = len(spec["panels"]) * (8 if split == "development" else 32)
        if count != EXPECTED_CASES[name] or any(p["split"] != split for p in spec["panels"]):
            raise ValueError(f"Prewritten panel scope differs: {name}")
        fixed.append(path)
    return {"schema": SCHEMA + "-preflight", "checked_at": now(), "results": str(results),
            "authorization": authorization, "training_plan_sha256": training.sha(plan),
            "fixed_inputs": [pin(p) for p in fixed], "scientific_sources": evaluate.sources(),
            "supervisor_source": pin(__file__), "expected_cases": EXPECTED_CASES,
            "training_closure_present": (results / "training/campaign-closure.json").exists()}


def verify_endpoints(results, expected_plan_sha):
    from docs.research.tools.comparison_ppo_v2 import evaluate, training
    root = results / "training"
    plan = training.require_plan(root)
    if training.sha(plan) != expected_plan_sha:
        raise ValueError("Training plan differs from supervisor launch")
    closure_path = root / "campaign-closure.json"
    closure = read_json(closure_path)
    runs = closure.get("runs", [])
    if (closure.get("schema") != training.SCHEMA or closure.get("all_fixed_endpoints_completed") is not True
            or closure.get("plan_sha256") != expected_plan_sha
            or sorted(r["optimizer_seed"] for r in runs) != [101, 211, 307]
            or any(r.get("status") != "completed" or r.get("exit_code") != 0 or r.get("num_timesteps") != 786432 for r in runs)):
        raise ValueError("Training closure lacks all three fixed 786432-transition endpoints")
    training.configure_torch()
    records = {}
    for run in runs:
        seed = run["optimizer_seed"]
        result_path = root / f"seed-{seed}/result.json"
        if run["result"] != pin(result_path):
            raise ValueError("Training result differs from immutable closure pin")
        result = read_json(result_path)
        if (result.get("status") != "completed" or result.get("num_timesteps") != 786432
                or result.get("requested_timesteps") != 786432 or result.get("plan_sha256") != expected_plan_sha
                or result.get("actor_parameters_changed") is not True or result.get("actor_mean_parameters_changed") is not True):
            raise ValueError("Fixed final endpoint is incomplete or lacks actual actor learning")
        entries = {"result": pin(result_path)}
        for name, expected_steps in (("initial-policy", 0), ("final-policy", 786432)):
            path = result_path.parent / (name + ".zip")
            metadata = read_json(path.with_suffix(".json"))
            if any(metadata.get(key) is not True for key in ("checkpoint_reopen_verified", "normalizer_reopen_verified")):
                raise ValueError("Checkpoint reopen evidence is incomplete")
            if expected_steps and metadata.get("deterministic_prediction_reopen_verified") is not True:
                raise ValueError("Final checkpoint lacks prediction reopen evidence")
            model = training.load_checkpoint(path, root / "plan.json")
            try:
                if model.seed != seed or model.num_timesteps != expected_steps or model.research_identity != plan["environment_identity"]:
                    raise ValueError("Reopened checkpoint has the wrong identity")
            finally:
                model.close()
            entries[name] = evaluate.checkpoint_record(path)
        records[str(seed)] = entries
    payload = {"schema": SCHEMA + "-endpoints", "created_at": now(), "training_plan": pin(root / "plan.json"),
               "training_plan_sha256": expected_plan_sha, "campaign_closure": pin(closure_path), "endpoints": records}
    return {"sha256": digest(payload), **payload}


class Supervisor:
    def __init__(self, results, output, *, poll_seconds=5):
        self.results, self.output = Path(results).resolve(), Path(output).resolve()
        self.poll_seconds = poll_seconds
        self.child = None
        self.cancelled = False
        self.current_stage = "preflight"
        self.started = time.monotonic()
        self.frozen = None
        self.stage_receipts = []
        self.output.mkdir(parents=True, exist_ok=False)

    def event(self, event, **details):
        value = {"at": now(), "event": event, "stage": self.current_stage, **details}
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(canonical(value).decode() + "\n")
        temporary = self.output / "status.writing.json"
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.output / "status.json")
        print(json.dumps(value), flush=True)

    def interrupt(self, signum, frame):
        self.cancelled = True
        if self.child is not None and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def check_time(self):
        if self.cancelled:
            raise InterruptedError("Supervisor interrupted; existing training was not stopped")
        if datetime.now(timezone.utc) >= self.finish_deadline:
            raise TimeoutError("Authorized campaign deadline reached; no next stage started")

    def check_inputs(self):
        from docs.research.tools.comparison_ppo_v2 import evaluate, training
        self.check_time()
        for item in self.frozen["fixed_inputs"]:
            if pin(item["path"]) != item:
                raise ValueError(f"Frozen supervisor input changed: {item['path']}")
        if pin(__file__) != self.frozen["supervisor_source"] or evaluate.sources() != self.frozen["scientific_sources"]:
            raise ValueError("Scientific or supervisor source changed after launch")
        if training.sha(training.require_plan(self.results / "training")) != self.frozen["training_plan_sha256"]:
            raise ValueError("Training plan changed after launch")

    def wait_for_training(self):
        self.current_stage = "waiting_for_existing_training"
        self.event("stage_started", training_started_here=False)
        closure = self.results / "training/campaign-closure.json"
        last_update = 0.0
        training_deadline = deadline(self.frozen["authorization"]["training_cutoff_at"])
        while not closure.exists():
            self.check_time()
            for seed in (101, 211, 307):
                folder = self.results / f"training/seed-{seed}"
                if (folder / "failure.json").exists():
                    raise RuntimeError(f"Existing training seed {seed} failed; attempt retained")
                if (folder / "result.json").exists() and read_json(folder / "result.json").get("status") != "completed":
                    raise RuntimeError(f"Existing training seed {seed} is incomplete; no final endpoint invented")
            if datetime.now(timezone.utc).timestamp() > training_deadline.timestamp() + 120:
                raise TimeoutError("Existing training did not publish closure within its bounded shutdown window")
            if time.monotonic() - last_update >= 60:
                self.event("waiting", closure_present=False)
                last_update = time.monotonic()
            time.sleep(self.poll_seconds)
        self.check_inputs()
        self.event("training_closure_available", closure=pin(closure))

    def run_command(self, name, command, *, expected_output, case_directory=None):
        self.check_inputs()
        self.current_stage = name
        log = self.output / (name + ".log")
        self.event("stage_started", command=command, expected_output=str(expected_output))
        began, termination_requested = time.monotonic(), None
        last_count, last_report = None, 0.0
        with log.open("x") as stream:
            envvars = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            self.child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                          env=envvars, start_new_session=True)
            self.event("child_started", pid=self.child.pid, log=str(log))
            while self.child.poll() is None:
                expired = self.cancelled or datetime.now(timezone.utc) >= self.finish_deadline
                if expired:
                    if termination_requested is None:
                        termination_requested = time.monotonic()
                        try:
                            os.killpg(self.child.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    elif time.monotonic() - termination_requested > 30:
                        try:
                            os.killpg(self.child.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                if case_directory and time.monotonic() - last_report >= 15:
                    count = sum(1 for _ in Path(case_directory).rglob("*.result.json"))
                    if count != last_count:
                        self.event("stage_progress", retained_cases=count, expected_cases=EXPECTED_CASES.get(name),
                                   elapsed_s=round(time.monotonic() - began, 3))
                        last_count = count
                    last_report = time.monotonic()
                time.sleep(1)
            code = self.child.returncode
            self.child = None
        receipt = {"stage": name, "finished_at": now(), "exit_code": code,
                   "elapsed_s": time.monotonic() - began, "log": pin(log),
                   "expected_output": pin(expected_output) if Path(expected_output).exists() else None,
                   "interrupted_or_deadline": termination_requested is not None or self.cancelled}
        publish(self.output / (name + "-receipt.json"), receipt)
        self.stage_receipts.append(receipt)
        self.event("stage_finished", **{k: v for k, v in receipt.items() if k != "stage"})
        if code != 0 or not Path(expected_output).exists() or receipt["interrupted_or_deadline"]:
            raise RuntimeError(f"Stage {name} did not complete; existing cases retained and never retried")
        self.check_inputs()

    def freeze_panel(self, name, gate_path):
        from docs.research.tools.comparison_ppo_v2 import evaluate
        self.check_inputs()
        spec = read_json(self.results / (name + "-spec.json"))
        freeze_path = self.results / (name + "-freeze.json")
        folder = self.results / name
        if freeze_path.exists() or folder.exists():
            raise FileExistsError(f"Stage already has an attempt: {name}; supervisor will not retry or overwrite")
        freeze = evaluate.freeze_evaluation(freeze_path, **spec, gate_path=gate_path,
                                            description="Prewritten comparison panel; source and endpoints frozen before access")
        self.event("evaluation_frozen", panel=name, freeze=pin(freeze_path), canonical_sha256=freeze["sha256"])
        return freeze_path

    def run_frozen_panel(self, name, gate_path):
        from docs.research.tools.comparison_ppo_v2 import evaluate
        self.check_inputs()
        freeze_path = self.results / (name + "-freeze.json")
        folder = self.results / name
        if folder.exists():
            raise FileExistsError(f"Wave already has an attempt: {name}; no retry")
        evaluate.require_freeze(freeze_path)
        command = [sys.executable, "-m", "docs.research.tools.comparison_ppo_v2.run_panels",
                   "--freeze", str(freeze_path), "--output", str(folder), "--gate", str(gate_path),
                   "--workers", "3", "--case-timeout", "300", "--deadline-utc", self.frozen["authorization"]["deadline_at"]]
        self.run_command(name, command, expected_output=folder / "wave.json", case_directory=folder)
        wave = read_json(folder / "wave.json")
        if not all(wave.get(k) is True for k in ("all_requested_cases_retained", "all_execution_complete", "source_files_unchanged")):
            raise RuntimeError(f"Wave {name} contains incomplete execution or changed source; no subsequent stage started")
        count = 0
        for item in wave["panels"]:
            if pin(item["path"]) != item:
                raise ValueError("Published panel changed during stage closure")
            runs = read_json(item["path"])["runs"]
            count += len(runs)
            if any(r.get("physical_invariants_passed") is not True or r.get("economic_audit", {}).get("passed") is not True for r in runs):
                raise ValueError(f"Wave {name} contains a physical or economic audit failure")
        if count != EXPECTED_CASES[name]:
            raise ValueError(f"Wave {name} retained {count} cases, expected {EXPECTED_CASES[name]}")
        evaluate.require_freeze(freeze_path)
        self.event("stage_validated", retained_cases=count)

    def execute(self):
        from docs.research.tools.comparison_ppo_v2 import environment
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        for sig in previous:
            signal.signal(sig, self.interrupt)
        try:
            self.frozen = preflight(self.results)
            self.finish_deadline = deadline(self.frozen["authorization"]["deadline_at"])
            for name in (*SPECS, "analysis"):
                if (self.results / name).exists():
                    raise FileExistsError(f"Future stage already exists: {name}; no automatic restart")
            if (self.results / "held-out-gate.json").exists():
                raise FileExistsError("Held-out gate already exists; manual review required before any new supervisor")
            publish(self.output / "launch.json", {**self.frozen, "schema": SCHEMA + "-launch", "created_at": now(), "pid": os.getpid()})
            self.wait_for_training()
            self.current_stage = "verify_fixed_endpoints"
            endpoints = verify_endpoints(self.results, self.frozen["training_plan_sha256"])
            endpoint_path = self.output / "endpoints-frozen.json"
            publish(endpoint_path, endpoints)
            self.event("endpoints_verified", endpoints=pin(endpoint_path), optimizer_seeds=[101, 211, 307], transitions_each=786432)
            self.freeze_panel("development-learning", self.results / "development-gate.json")
            self.run_frozen_panel("development-learning", self.results / "development-gate.json")
            self.check_inputs()
            # Reopen all endpoints immediately before the held-out gate. No
            # development score selects, replaces or changes any endpoint.
            confirmed = verify_endpoints(self.results, self.frozen["training_plan_sha256"])
            if confirmed["endpoints"] != endpoints["endpoints"] or confirmed["campaign_closure"] != endpoints["campaign_closure"]:
                raise ValueError("Endpoints changed before held-out access")
            gate_path = self.results / "held-out-gate.json"
            publish(gate_path, {"schema": "comparison-ppo-v2-evaluation-gate",
                "environment_identity_sha256": environment.environment_identity_sha256(),
                "source_identity_sha256": environment.source_identity_sha256(), "allowed_splits": ["held_out"],
                "endpoints_frozen": True, "created_at": now(), "endpoints": pin(endpoint_path),
                "training_closure": pin(self.results / "training/campaign-closure.json")})
            self.event("held_out_gate_published", gate=pin(gate_path))
            # Both held-out specifications and all checkpoint identities are
            # frozen before the first held-out episode is opened.
            self.freeze_panel("held-out-primary", gate_path)
            self.freeze_panel("held-out-battery", gate_path)
            self.run_frozen_panel("held-out-primary", gate_path)
            self.run_frozen_panel("held-out-battery", gate_path)
            output = self.results / "analysis"
            roots = [self.results / name for name in (*EXISTING_WAVES, *SPECS)]
            command = [sys.executable, "-m", "docs.research.tools.comparison_ppo_v2.analyze",
                       "--root", *map(str, roots), "--output", str(output)]
            self.run_command("analysis", command, expected_output=output / "analysis.json")
            analysis = read_json(output / "analysis.json")
            if len(analysis["rows"]) != 1160:
                raise ValueError("Final analysis must retain184 prior development +48 learning +832 primarytest +96 diagnostic cases")
            receipt = {"schema": SCHEMA + "-completion", "status": "completed", "completed_at": now(),
                "elapsed_s": time.monotonic() - self.started, "analysis": pin(output / "analysis.json"),
                "outcomes": len(analysis["rows"]), "groups": len(analysis["groups"]),
                "endpoints": pin(endpoint_path), "stages": self.stage_receipts,
                "report_and_export_completed_here": False, "existing_training_started_or_stopped_here": False}
            publish(self.output / "completion.json", receipt)
            self.current_stage = "complete"
            self.event("supervision_completed", outcomes=receipt["outcomes"], analysis=receipt["analysis"])
            return receipt
        except BaseException as exc:
            failure = {"schema": SCHEMA + "-failure", "status": "stopped", "at": now(),
                "stage": self.current_stage, "type": type(exc).__name__, "message": str(exc),
                "elapsed_s": time.monotonic() - self.started, "stages": self.stage_receipts,
                "existing_training_started_or_stopped_here": False,
                "case_retry_policy": "None: retained partial or failed attempts require explicit review; no invented endpoints."}
            publish(self.output / "failure.json", failure)
            self.event("supervision_stopped", failure_type=type(exc).__name__, message=str(exc))
            raise
        finally:
            if self.child is not None and self.child.poll() is None:
                self.interrupt(signal.SIGTERM, None)
                try:
                    self.child.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.child.wait(timeout=5)
            for sig, previous_handler in previous.items():
                signal.signal(sig, previous_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=5)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        raise ValueError("Polling interval must be between1and60seconds")
    if args.check_only:
        result = preflight(args.results)
        print(json.dumps({"verified": True, "training_closure_present": result["training_closure_present"],
                          "fixed_inputs": len(result["fixed_inputs"]), "scientific_sources": len(result["scientific_sources"]),
                          "expected_cases": result["expected_cases"], "source": result["supervisor_source"]}, indent=2))
    else:
        supervisor = Supervisor(args.results, args.output or args.results / "supervision", poll_seconds=args.poll_seconds)
        print(json.dumps(supervisor.execute(), indent=2))


if __name__ == "__main__":
    main()
