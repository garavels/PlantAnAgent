"""Frozen, isolated, exact-minute evaluation for the twelve-hour authority study.

Every requested case has one attempt. Held-out access is gated before environment
construction in both parent and child. A parent watchdog retains timeouts as
failures; SIGTERM first allows the auditor to serialize the partial trajectory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
from typing import Any
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
for folder in (ROOT, ROOT / "backend/src", ROOT / "simulator/src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from docs.research.tools.feasible_mission_replay import canonical_sha, file_pin, run_controller

SCHEMA = "authority-ppo-v2-evaluation"
ARMS = ("narrow", "broad")
OPTIMIZER_SEEDS = (101, 211, 307)
INTERMEDIATE_STEPS = (65536, 131072, 196608)
RESOURCES = ("battery", "h2", "co2", "ch4")
COVERAGE_LABEL = "daily-coverage-offgrid-v1"
MPC_LABEL = "daily-return-mission-mpc-v1"


def benchmark():
    from docs.research.tools.authority_ppo_v2 import benchmark as module
    return module


def training():
    from docs.research.tools.authority_ppo_v2 import train
    return train


def publish(path: Path, value: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        temporary = path.with_suffix(path.suffix + ".part")
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)


def comparison_sources() -> list[dict[str, Any]]:
    """Bind controller/physics/evaluation code; selection freezes analytic rules."""
    paths = {ROOT / "uv.lock", benchmark().CONTRACT_PATH,
             ROOT / "docs/research/benchmarks/feasible25_offgrid_v1.json",
             ROOT / "docs/research/AUTHORITY_PPO_V2_PROTOCOL.md"}
    paths.update((Path(__file__).parent / name) for name in
                 ("benchmark.py", "policy.py", "train.py", "evaluate.py"))
    paths.update(ROOT / "docs/research/tools" / name for name in
                 ("feasible_benchmark_v1.py", "feasible_mission_controller.py",
                  "feasible_mission_replay.py", "offgrid_coverage_v1.py", "coverage_ppo_v1.py"))
    for directory in (ROOT / "backend/src/plant_backend", ROOT / "simulator/src/plant_sim"):
        paths.update(directory.rglob("*.py"))
    return [file_pin(path.resolve()) for path in sorted(paths)]


def frozen_selection(contract: dict) -> dict:
    return {
        "ppo_arms": list(ARMS), "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "checkpoint": "All six fixed final endpoints; no held-out checkpoint selection.",
        "held_out_seeds": contract["splits"]["held_out"],
        "development_seeds": contract["splits"]["development"],
        "strict_mpc_held_out_seeds": contract["splits"]["held_out"][:8],
        "primary_outcome": "Complete mission: all deliveries, all seven daily stock returns, exact-minute hard safety, full execution.",
        "incomplete_outcome": "Incomplete execution is failure for every full-week pass flag; partial economics never imputed as full-week economics.",
        "secondary_outcomes": ["each daily stock deficit", "delivery shortfall", "economic contribution", "equipment wear", "curtailment", "controller latency"],
        "bootstrap": {"unit": "paired environmental scenario; optimizer seeds nested within arm", "seed": 20260917, "replicates": 10000, "interval": "percentile 95%"},
        "analysis_rules": {
            "stock_deficit": "Daily max(0,target-actual), in each resource's physical unit; report sum and maximum across observed daily returns, plus excess over fixed tolerance. Full-week means require complete execution.",
            "all_case_mean": "Unavailable if any requested scenario lacks a complete measurement; partial money and stock deficits are never imputed as full-week values. All binary full-week failures remain in denominator.",
            "paired_effect": "Treatment minus reference on the identical scenario. Each optimizer seed reported separately; arm effect averages three paired optimizer differences within scenario before resampling scenarios.",
            "conditional_economics": "Per-policy comparison requires both missions to pass. Arm comparison requires every treatment/reference optimizer pair to pass in that scenario; report its conditional denominator.",
            "interval_caveat": "Descriptive percentile intervals; not multiplicity-adjusted and not evidence of generalization beyond sampled simulator distribution.",
        },
        "replay_cases": [{"controller": controller_label("ppo", "narrow", 101), "seed": 9960000}, {"controller": controller_label("ppo", "broad", 101), "seed": 9960000}, {"controller": controller_label("ppo", "broad", 101), "seed": 9960001}],
        "attempts_per_case": 1, "mpc_solver_seconds_per_stage": 5.0,
        "maximum_mpc_concurrent_cases": 2,
    }


def freeze_evaluation(training_root: Path, freeze_path: Path) -> dict:
    contract = benchmark().load_benchmark_contract()
    payload = {
        "schema": SCHEMA + "-freeze", "created_at_unix_s": time.time(),
        "benchmark_contract_sha256": contract["contract_sha256"],
        "sources": comparison_sources(),
        "endpoints": training().require_frozen_endpoints(training_root),
        "selection": frozen_selection(contract),
    }
    payload["sha256"] = canonical_sha(payload)
    publish(freeze_path, payload)
    return payload


def require_evaluation_freeze(training_root: Path, freeze_path: Path) -> dict:
    payload = json.loads(freeze_path.read_text())
    if payload.get("schema") != SCHEMA + "-freeze":
        raise ValueError("Unexpected evaluation freeze schema")
    if payload.get("sha256") != canonical_sha({k: v for k, v in payload.items() if k != "sha256"}):
        raise ValueError("Evaluation freeze checksum differs")
    contract = benchmark().load_benchmark_contract()
    if payload["benchmark_contract_sha256"] != contract["contract_sha256"] or payload["selection"] != frozen_selection(contract):
        raise ValueError("Evaluation task or selection differs from freeze")
    if payload["sources"] != comparison_sources():
        raise ValueError("Comparison source changed after evaluation freeze")
    if payload["endpoints"] != training().require_frozen_endpoints(training_root):
        raise ValueError("Comparison endpoints changed after evaluation freeze")
    return payload


def controller_label(kind: str, arm: str | None, optimizer_seed: int | None) -> str:
    if kind == "ppo":
        if arm not in ARMS or optimizer_seed not in OPTIMIZER_SEEDS:
            raise ValueError("PPO requires one frozen arm and optimizer seed")
        return f"authority-ppo-v2-{arm}-{optimizer_seed}"
    if arm is not None or optimizer_seed is not None:
        raise ValueError("Only PPO has an arm or optimizer seed")
    if kind == "coverage":
        return COVERAGE_LABEL
    if kind == "mpc":
        return MPC_LABEL
    raise ValueError("Unknown controller")


def validate_checkpoint_request(kind, split, checkpoint_steps):
    if checkpoint_steps is not None and (kind != "ppo" or split != "development" or checkpoint_steps not in INTERMEDIATE_STEPS):
        raise ValueError("Only the three prospective intermediate PPO endpoints may be evaluated, on development only")


def load_intermediate(root: Path, arm: str, optimizer_seed: int, steps: int):
    """Verify a published diagnostic endpoint without changing final selection."""
    from docs.research.tools.authority_ppo_v2.policy import interface_identity
    from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPO
    from plant_backend.rl.v2.training.research_effort_batch import preflight_checkpoint
    if steps not in INTERMEDIATE_STEPS:
        raise ValueError("Not a prospective intermediate endpoint")
    trainer = training()
    trainer.require_plan(root)
    directory = trainer.endpoint_directory(root, arm, optimizer_seed)
    artifact = directory / "checkpoints" / f"policy-{steps:06d}.zip"
    metadata = json.loads(artifact.with_suffix(".json").read_text())
    if metadata["num_timesteps"] != steps or metadata["checkpoint"] != file_pin(artifact):
        raise ValueError("Intermediate checkpoint bytes or step metadata differ")
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if metadata not in result["intermediate_checkpoints"]:
            raise ValueError("Intermediate checkpoint differs from final training provenance")
    model = ConstrainedPPO.load(artifact, device="cpu")
    base = benchmark().BenchmarkEnv(split="train")  # Construction only; no episode is opened.
    try:
        expected_identity = interface_identity(base, arm)
    finally:
        base.close()
    state = model.embedded_constraint_state
    preflight_checkpoint(artifact, expected_identity, state)
    attestation = model.loaded_constraint_state_attestation()
    if (model.num_timesteps != steps or model.seed != optimizer_seed
        or canonical_sha(model.research_identity) != canonical_sha(expected_identity)
        or trainer.actor_mean_parameter_sha256(model) != metadata["actor_mean_sha256"]
        or attestation.artifact_sha256 != metadata["checkpoint"]["sha256"]
        or attestation.num_timesteps != steps or attestation.constraint_state != state
        or len(model.rollout_ledgers) != steps // 8192
        or any(item.transition_count != 8192 or item.optimizer_steps != 320 for item in model.rollout_ledgers)):
        raise ValueError("Intermediate checkpoint identity, weights, or training ledger differs")
    return model, metadata


def select_seeds(kind: str, split: str, seeds: list[int] | None) -> list[int]:
    contract = benchmark().load_benchmark_contract()
    if split not in ("train", "development", "held_out"):
        raise ValueError("Unsupported evaluation split")
    expected = list(contract["splits"][split])
    if split == "held_out" and kind == "mpc":
        expected = expected[:8]
    selected = expected if seeds is None else list(seeds)
    if not selected or len(set(selected)) != len(selected) or any(seed not in expected for seed in selected):
        raise ValueError("Empty, duplicate, or out-of-split seeds")
    if split == "held_out" and selected != expected:
        raise ValueError("Held-out panels must contain the entire frozen panel in order")
    if split == "train" and seeds is None:
        raise ValueError("Engineering evaluation must explicitly name training seeds")
    return selected


def check_disk(path: Path, min_free_gib: float) -> dict:
    if not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError("Minimum free disk must be nonnegative and finite")
    existing = path
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < min_free_gib * 1024**3:
        raise OSError(f"Disk reserve reached: {free / 1024**3:.3f} GiB available, {min_free_gib:.3f} GiB required")
    return {"free_bytes": free, "required_free_bytes": int(min_free_gib * 1024**3)}


def instrumented_ppo_controller(env, model, arm):
    """Verify canonical identity and the raw deterministic mean behind Box clipping."""
    import numpy as np
    import torch
    from docs.research.tools.authority_ppo_v2.policy import AuthorityPPOController, interface_identity

    expected = interface_identity(env, arm)
    serialized_sha = canonical_sha(model.research_identity)
    if serialized_sha != canonical_sha(expected):
        raise ValueError("Loaded policy has a different canonical task identity")
    model.research_identity = expected  # JSON tuple/list normalization, after canonical equality.

    class InstrumentedController(AuthorityPPOController):
        def compute_command(self, observation, **kwargs):
            start = time.perf_counter()
            decision = super().compute_command(observation, **kwargs)
            compute_s = time.perf_counter() - start
            start = time.perf_counter()
            vector = np.asarray(decision.evidence["policy_observation"], dtype=np.float32)
            if not np.array_equal(vector[:135], env.current_policy_observation):
                raise ValueError("Policy vector does not contain the actual current observation")
            tensor, _ = self.model.policy.obs_to_tensor(vector)
            if self.model.policy.squash_output:
                raise ValueError("Unexpected squashed policy changes action interpretation")
            with torch.no_grad():
                mean = self.model.policy.get_distribution(tensor).get_actions(deterministic=True).cpu().numpy().reshape(-1)
            clipped = np.clip(mean, self.model.action_space.low, self.model.action_space.high)
            if not np.array_equal(clipped, np.asarray(decision.evidence["raw_residual"], dtype=np.float32)):
                raise ValueError("Recorded adapter input differs from deterministic policy inference")
            return replace(decision, evidence={**decision.evidence, "inference_witness": {
                "unclipped_deterministic_action": mean.tolist(), "sb3_clipped_action": clipped.tolist(),
                "upstream_action_clipped": bool(np.any(mean != clipped)), "adapter_input_verified": True,
                "actual_current_observation_verified": True, "canonical_task_identity_sha256": serialized_sha,
                "controller_compute_s": compute_s, "witness_verification_s": time.perf_counter() - start,
            }})
    return InstrumentedController(env, model, arm=arm)


class CaseTimeLimitExceeded(RuntimeError):
    pass


def _timeout_signal(signum, frame):
    raise CaseTimeLimitExceeded("Parent per-case wall-clock limit reached; partial execution retained")


def summarize_record(result: dict) -> dict:
    """Compact sufficient outcome data; trajectories stay in the compressed artifact."""
    p, e = result["physical_metrics"], result["economic_metrics"]
    cycles = result["cycle_settlements"]
    deficits = {resource: [max(0.0, cycle["targets"][resource] - cycle["actual"][resource]) for cycle in cycles] for resource in RESOURCES}
    excess = {resource: [max(0.0, value - cycle["tolerances"][resource]) for value, cycle in zip(deficits[resource], cycles)] for resource in RESOURCES}
    commands = result["commands"]
    evidence_rows = [item["evidence"] for item in commands if isinstance(item.get("evidence"), dict)]
    timings = [item["evidence"]["evaluation_compute_s"] for item in commands if isinstance(item.get("evidence"), dict) and "evaluation_compute_s" in item["evidence"]]
    witnesses = [item["evidence"]["inference_witness"] for item in commands if isinstance(item.get("evidence"), dict) and "inference_witness" in item["evidence"]]
    row = {key: result[key] for key in ("seed", "controller", "split", "execution_complete", "mission_passed", "hard_safety_passed", "every_delivery_passed", "all_daily_returns_passed", "failure", "decisions", "physical_steps", "wall_time_s", "benchmark_contract_sha256", "evaluation_freeze_sha256", "minute_truth_sha256")}
    row.update({
        "physical_metrics": p, "economic_metrics": e, "cycle_settlements": cycles,
        "delivery_settlements": result["delivery"]["settlements"],
        "daily_returns_met": sum(cycle["feasible"] for cycle in cycles),
        "stock_deficits": {resource: {"observed_daily_deficits": deficits[resource], "observed_daily_excess_over_tolerance": excess[resource], "sum": sum(deficits[resource]), "max": max(deficits[resource], default=0), "failed_daily_returns": sum(value > 1e-12 for value in excess[resource])} for resource in RESOURCES},
        "minute_checks": result["minute_checks"], "manifest_sha256": result["manifest"]["manifest_sha256"],
        "beta_exogenous_episode_sha256": result["manifest"]["beta_exogenous_episode_sha256"],
        "rejected_plans": result.get("rejected_plans", []),
        "controller_latency_s": {"count": len(timings), "mean": statistics.mean(timings) if timings else None, "median": statistics.median(timings) if timings else None, "max": max(timings, default=None), "total": sum(timings), "includes": "public observation to bound command; excludes audit witness recomputation"},
        "inference": {"verified_decisions": len(witnesses), "all_inputs_verified": all(w["adapter_input_verified"] and w["actual_current_observation_verified"] for w in witnesses), "upstream_clipped_decisions": sum(w["upstream_action_clipped"] for w in witnesses), "max_absolute_raw_action": max((abs(v) for w in witnesses for v in w["unclipped_deterministic_action"]), default=0), "commands_changed_from_coverage": sum(item["evidence"].get("nominal_command", {}).get("targets") != item["evidence"].get("mapped_targets") for item in commands if isinstance(item.get("evidence"), dict) and "nominal_command" in item["evidence"])},
    })
    row["inference"].update({
        "mean_absolute_raw_action_by_actuator": [statistics.mean(abs(w["unclipped_deterministic_action"][i]) for w in witnesses) for i in range(4)] if witnesses else None,
        "prior_zero_target_enabled_decisions_by_actuator": [sum(bool(e.get("prior_zero_target_enabled", [False] * 4)[i]) for e in evidence_rows) for i in range(4)],
        "authority_projection_decisions": sum(any(abs(x) > 1e-9 for x in e.get("authority_projection_kw", [])) for e in evidence_rows),
        "commitment_projection_decisions": sum(any(abs(x) > 1e-9 for x in e.get("commitment_projection_kw", [])) for e in evidence_rows),
        "commitment_authority_conflicts": sum(bool(e.get("commitment_authority_conflict")) for e in evidence_rows),
    })
    row["fault_windows"] = fault_windows(result)
    return row


def fault_windows(result: dict) -> list[dict]:
    """Descriptive six-hour pre/post DAC windows; no causal identification claim."""
    minutes = result["minute_truth_states"]
    windows = []
    for fault in result["manifest"]["disturbances"]["schedule"]:
        start, end = fault["start_s"], fault["start_s"] + fault["duration_s"]
        item = {"fault": fault, "windows": {}}
        for label, a, b in (("pre_6h", max(0, start - 21600), start), ("during", start, end), ("post_6h", end, end + 21600)):
            # State index i records the interval that ended at elapsed i*60 s.
            selected = minutes[max(1, int(a / 60) + 1):min(len(minutes), int(b / 60) + 1)]
            if selected:
                item["windows"][label] = {"observed_minutes": len(selected), "mean_dac_power_kw": statistics.mean(state["dac_power_kw"] for state in selected),
                    "mean_electrolyser_power_kw": statistics.mean(state["electrolyser_power_kw"] for state in selected),
                    "minimum_co2_kg": min(state["co2_level_kg"] for state in selected),
                    "minimum_ch4_kg": min(state["ch4_level_kg"] for state in selected),
                    "minimum_battery_soc": min(state["battery_soc"] for state in selected)}
            else:
                item["windows"][label] = None
        windows.append(item)
    return windows


def run_case(*, kind, arm, optimizer_seed, split, seed, output, training_root, freeze_path, expected_freeze_sha256, min_free_gib, max_decisions=None, checkpoint_steps=None) -> dict:
    """Only invoked in an isolated child; gate precedes model/env construction."""
    validate_checkpoint_request(kind, split, checkpoint_steps)
    if split == "held_out":
        freeze = require_evaluation_freeze(training_root, freeze_path)
        if freeze["sha256"] != expected_freeze_sha256:
            raise ValueError("Evaluation freeze changed before child evaluation")
        admitted = freeze["selection"]["strict_mpc_held_out_seeds" if kind == "mpc" else "held_out_seeds"]
        if seed not in admitted or max_decisions is not None:
            raise ValueError("Nonfrozen held-out case or partial-horizon request")
    contract = benchmark().load_benchmark_contract()
    if seed not in contract["splits"][split]:
        raise ValueError("Case seed is outside selected split")
    label = controller_label(kind, arm, optimizer_seed)
    if checkpoint_steps is not None:
        label += f"-step-{checkpoint_steps}"
    source_pins = comparison_sources()
    disk_start = check_disk(output, min_free_gib)
    started = time.monotonic()
    model = None
    intermediate_metadata = None
    if kind == "ppo":
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if checkpoint_steps is None:
            model = training().load_final(training_root, arm, optimizer_seed)
        else:
            model, intermediate_metadata = load_intermediate(training_root, arm, optimizer_seed, checkpoint_steps)
    env = benchmark().BenchmarkEnv(split=split)
    rejected_plans = []
    original_sigterm_handler = signal.getsignal(signal.SIGTERM)
    try:
        env.reset(seed=seed)
        if kind == "ppo":
            controller = instrumented_ppo_controller(env, model, arm)
        elif kind == "coverage":
            from docs.research.tools.offgrid_coverage_v1 import OffgridCoverageController
            controller = OffgridCoverageController()
        else:
            from plant_backend.mission_mpc import MissionMPCControllerV1
            controller = MissionMPCControllerV1(horizon_h=env.episode_config.horizon_h,
                interval_s=env.episode_config.decision_interval_s, assumptions=env.economic_assumptions,
                beta_task=env.beta_task, electricity_model=env.electricity_model.to_dict())
            if controller.mission_settings.solver_time_limit_s != 5.0:
                raise ValueError("Strict MPC solver time limit changed")

        class TimedController:
            def reset(self):
                if hasattr(controller, "reset"):
                    controller.reset()
            def set_observation(self, vector):
                if hasattr(controller, "set_observation"):
                    controller.set_observation(vector)
            def compute_command(self, observation, **kwargs):
                start = time.perf_counter()
                try:
                    decision = controller.compute_command(observation, **kwargs)
                except Exception as exc:
                    if kind == "mpc":
                        rejected_plans.append({"decision": env.decisions_taken, "exception_type": type(exc).__name__, "message": str(exc), "stages": [stage.to_dict() for stage in getattr(exc, "stages", ())], "fallback": None})
                    raise
                evidence = decision.evidence
                if hasattr(evidence, "to_dict"):
                    evidence = evidence.to_dict()
                elapsed = evidence.get("inference_witness", {}).get("controller_compute_s", time.perf_counter() - start)
                if env.decisions_taken % 24 == 0:
                    publish(output / f"{seed}-progress.json", {"seed": seed, "decisions": env.decisions_taken, "wall_time_s": time.monotonic() - started}, exclusive=False)
                return SimpleNamespace(command=decision.command, invalid=getattr(decision, "invalid", False), evidence={**evidence, "evaluation_compute_s": elapsed})

        signal.signal(signal.SIGTERM, _timeout_signal)
        result = run_controller(env, TimedController(), max_decisions=max_decisions, controller_id=label)
        # Do not interrupt trace serialization after the run has returned.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        result.update(schema=SCHEMA + "-recording", split=split, arm=arm, optimizer_seed=optimizer_seed,
            checkpoint_steps=checkpoint_steps, intermediate_checkpoint=intermediate_metadata,
            benchmark_contract_sha256=contract["contract_sha256"], evaluation_freeze_sha256=expected_freeze_sha256,
            controller_configuration=controller.to_dict(), electricity_model=env.electricity_model.to_dict(),
            additional_source_files=source_pins, rejected_plans=rejected_plans,
            wall_time_s=time.monotonic() - started,
            change="Versioned 25 kg/h off-grid authority study; unchanged physical distribution, all delivery/stock/safety obligations.",
            disk_before=disk_start, disk_before_serialization=check_disk(output, min_free_gib))
        if comparison_sources() != source_pins:
            raise ValueError("Comparison source changed during case")
        if split == "held_out":
            require_evaluation_freeze(training_root, freeze_path)
        recording = output / f"{seed}.json.gz"
        with recording.open("xb") as binary:
            with gzip.GzipFile(fileobj=binary, mode="wb", compresslevel=1, mtime=0) as compressed:
                import io
                with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                    json.dump(result, stream, separators=(",", ":"), allow_nan=False)
        row = summarize_record(result)
        row["recording"] = file_pin(recording)
        row["elapsed_with_serialization_s"] = time.monotonic() - started
        row["source_files_unchanged_during_run"] = True
        publish(output / f"{seed}-provenance.json", row)
        return row
    finally:
        signal.signal(signal.SIGTERM, original_sigterm_handler)
        env.close()


def failure_row(seed, label, split, freeze_sha, exc_type, message, elapsed_s) -> dict:
    return {"seed": seed, "controller": label, "split": split, "evaluation_freeze_sha256": freeze_sha,
            "execution_complete": False, "mission_passed": False, "hard_safety_passed": False,
            "every_delivery_passed": False, "all_daily_returns_passed": False,
            "failure": {"type": exc_type, "message": message}, "wall_time_s": elapsed_s,
            "recording": None, "metrics_status": "Unavailable; no full-week values imputed"}


def launch_case(*, kind, arm, optimizer_seed, split, seed, output, training_root, freeze_path, freeze_sha, timeout_s, min_free_gib, max_decisions, checkpoint_steps=None):
    started = time.monotonic()
    label = controller_label(kind, arm, optimizer_seed)
    if checkpoint_steps is not None:
        label += f"-step-{checkpoint_steps}"
    child = [sys.executable, "-m", "docs.research.tools.authority_ppo_v2.evaluate", "--one-case", "--controller", kind,
             "--split", split, "--seeds", str(seed), "--output", str(output), "--training-root", str(training_root),
             "--freeze-path", str(freeze_path), "--min-free-gib", str(min_free_gib)]
    if arm is not None:
        child += ["--arm", arm, "--optimizer-seed", str(optimizer_seed)]
    if freeze_sha:
        child += ["--expected-freeze-sha256", freeze_sha]
    if max_decisions is not None:
        child += ["--max-decisions", str(max_decisions)]
    if checkpoint_steps is not None:
        child += ["--checkpoint-steps", str(checkpoint_steps)]
    envvars = {**os.environ, **{key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    timed_out = False
    try:
        check_disk(output, min_free_gib)
        with (output / f"{seed}.log").open("x") as log:
            proc = subprocess.Popen(child, cwd=ROOT, env=envvars, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    returncode = proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    returncode = proc.wait(timeout=10)
        path = output / f"{seed}-provenance.json"
        if path.exists():
            row = json.loads(path.read_text())
            row["parent_watchdog_triggered"] = timed_out
            row["parent_wall_time_s"] = time.monotonic() - started
            row["child_returncode"] = returncode
            if timed_out and row["execution_complete"]:
                # A child may finish simulation while serialization crosses the limit.
                row["watchdog_note"] = "Full audited trajectory returned during serialization grace; watchdog still reported."
            return row
        kind_error = "CaseWallTimeout" if timed_out else "ChildProcessFailure"
        message = f"Child returned {returncode}; inspect {seed}.log. No retry."
    except Exception as exc:
        kind_error, message = type(exc).__name__, str(exc)
    row = failure_row(seed, label, split, freeze_sha, kind_error, message, time.monotonic() - started)
    progress = output / f"{seed}-progress.json"
    if progress.exists():
        row["last_progress"] = json.loads(progress.read_text())
    publish(output / f"{seed}-runner-failure.json", row)
    return row


def run_panel(*, kind, split, output, training_root, freeze_path, arm=None, optimizer_seed=None, seeds=None, workers=1, case_timeout_s=300, min_free_gib=2, max_decisions=None, checkpoint_steps=None) -> dict:
    validate_checkpoint_request(kind, split, checkpoint_steps)
    label = controller_label(kind, arm, optimizer_seed)
    if checkpoint_steps is not None:
        label += f"-step-{checkpoint_steps}"
    selected = select_seeds(kind, split, seeds)
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= (2 if kind == "mpc" else 3):
        raise ValueError("Use 1–3 isolated processes (strict MPC maximum 2)")
    if not math.isfinite(case_timeout_s) or case_timeout_s <= 0:
        raise ValueError("Per-case wall time must be positive and finite")
    if max_decisions is not None and (split != "train" or not 1 <= max_decisions <= 672):
        raise ValueError("Partial-horizon engineering checks use train split only")
    freeze = require_evaluation_freeze(training_root, freeze_path) if split == "held_out" else None
    source_pins = comparison_sources()
    check_disk(output, min_free_gib)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"schema": SCHEMA + "-panel", "controller": label, "arm": arm, "optimizer_seed": optimizer_seed,
        "checkpoint_steps": checkpoint_steps, "final_selection_eligible": checkpoint_steps is None,
        "split": split, "requested_seeds": selected, "evaluation_freeze_sha256": freeze["sha256"] if freeze else None,
        "benchmark_contract_sha256": benchmark().load_benchmark_contract()["contract_sha256"], "source_files": source_pins,
        "workers": workers, "case_timeout_s": case_timeout_s, "minimum_free_gib": min_free_gib,
        "timeout_grace_s": 60, "attempts_per_case": 1, "started_at_unix_s": time.time(), "runs": [],
        "runtime_context": "One isolated process per case, numerical library threads=1. Strict MPC unchanged five-second per-stage limit; no retries or fallback."}
    publish(output / "panel-request.json", report)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(launch_case, kind=kind, arm=arm, optimizer_seed=optimizer_seed, split=split, seed=seed,
            output=output, training_root=training_root, freeze_path=freeze_path, freeze_sha=report["evaluation_freeze_sha256"],
            timeout_s=case_timeout_s, min_free_gib=min_free_gib, max_decisions=max_decisions, checkpoint_steps=checkpoint_steps): seed for seed in selected}
        for future in as_completed(pending):
            row = future.result()
            report["runs"].append(row)
            report["runs"].sort(key=lambda row: row["seed"])
            report["wall_time_s"] = time.monotonic() - started
            publish(output / "summary.json", report, exclusive=False)
            print(json.dumps({"controller": label, "seed": row["seed"], "complete": row["execution_complete"], "mission": row["mission_passed"], "seconds": round(row["wall_time_s"], 2), "failure": row["failure"]}), flush=True)
    report["source_files_unchanged_during_panel"] = comparison_sources() == source_pins
    if split == "held_out":
        require_evaluation_freeze(training_root, freeze_path)
    if not report["source_files_unchanged_during_panel"]:
        report["validity_failure"] = "Comparison source changed during panel; do not combine as frozen results"
    report["wall_time_s"] = time.monotonic() - started
    report["completed_at_unix_s"] = time.time()
    publish(output / "summary.json", report, exclusive=False)
    return report


def deterministic_replay_view(record: dict) -> dict:
    def without_timing(value):
        if isinstance(value, dict):
            return {key: without_timing(item) for key, item in value.items()
                    if key not in {"evaluation_compute_s", "controller_compute_s", "witness_verification_s"}}
        if isinstance(value, list):
            return [without_timing(item) for item in value]
        return value
    return {"commands_with_policy_inputs_and_raw_outputs": without_timing(record["commands"]),
            **{key: record[key] for key in ("minute_truth_sha256", "physical_metrics", "economic_metrics",
                                           "cycle_settlements", "delivery", "mission_passed", "failure")}}


def run_replay(*, original: Path, output: Path, training_root: Path, freeze_path: Path,
               case_timeout_s=300, min_free_gib=2) -> dict:
    """Repeat only three prospectively selected cases; never add replications."""
    freeze = require_evaluation_freeze(training_root, freeze_path)
    with gzip.open(original, "rt") as stream:
        source = json.load(stream)
    selected = {"controller": source["controller"], "seed": source["seed"]}
    if selected not in freeze["selection"]["replay_cases"] or source["split"] != "held_out":
        raise ValueError("Replay is outside the prospectively selected audit cases")
    if source["evaluation_freeze_sha256"] != freeze["sha256"]:
        raise ValueError("Replay source belongs to a different freeze")
    provenance = json.loads((original.parent / f'{source["seed"]}-provenance.json').read_text())
    if provenance["recording"] != file_pin(original):
        raise ValueError("Replay source differs from original published recording")
    output.mkdir(parents=True, exist_ok=False)
    row = launch_case(kind="ppo", arm=source["arm"], optimizer_seed=source["optimizer_seed"],
        split="held_out", seed=source["seed"], output=output, training_root=training_root,
        freeze_path=freeze_path, freeze_sha=freeze["sha256"], timeout_s=case_timeout_s,
        min_free_gib=min_free_gib, max_decisions=None)
    comparisons = {}
    if row.get("recording"):
        with gzip.open(row["recording"]["path"], "rt") as stream:
            replay = json.load(stream)
        expected, actual = deterministic_replay_view(source), deterministic_replay_view(replay)
        comparisons = {key: {"equal": expected[key] == actual[key], "original_sha256": canonical_sha(expected[key]),
                             "replay_sha256": canonical_sha(actual[key])} for key in expected}
    report = {"schema": SCHEMA + "-replay-audit", "evaluation_freeze_sha256": freeze["sha256"],
              "original": file_pin(original), "replay": row, "comparisons": comparisons,
              "deterministic_replay_passed": bool(comparisons) and all(item["equal"] for item in comparisons.values()),
              "interpretation": "Repeatability audit of an existing case; not an additional environmental replication. Latency measurements excluded from deterministic equality."}
    publish(output / "replay-audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--replay-of", type=Path)
    parser.add_argument("--controller", choices=("coverage", "ppo", "mpc"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--optimizer-seed", type=int, choices=OPTIMIZER_SEEDS)
    parser.add_argument("--checkpoint-steps", type=int, choices=INTERMEDIATE_STEPS)
    parser.add_argument("--split", choices=("train", "development", "held_out"), default="development")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-root", required=True, type=Path)
    parser.add_argument("--freeze-path", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--case-timeout-s", type=float, default=300)
    parser.add_argument("--min-free-gib", type=float, default=2)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--one-case", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-freeze-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.freeze:
        payload = freeze_evaluation(args.training_root.resolve(), args.freeze_path.resolve())
        print(json.dumps({"freeze": str(args.freeze_path), "sha256": payload["sha256"]}))
        return
    if args.replay_of:
        if args.output is None:
            parser.error("Replay requires --output")
        payload = run_replay(original=args.replay_of.resolve(), output=args.output.resolve(),
            training_root=args.training_root.resolve(), freeze_path=args.freeze_path.resolve(),
            case_timeout_s=args.case_timeout_s, min_free_gib=args.min_free_gib)
        print(json.dumps({"replay_passed": payload["deterministic_replay_passed"], "output": str(args.output)}))
        return
    if args.controller is None or args.output is None:
        parser.error("Evaluation requires --controller and --output")
    common = dict(kind=args.controller, arm=args.arm, optimizer_seed=args.optimizer_seed, split=args.split,
                  output=args.output.resolve(), training_root=args.training_root.resolve(), freeze_path=args.freeze_path.resolve(),
                  min_free_gib=args.min_free_gib, max_decisions=args.max_decisions, checkpoint_steps=args.checkpoint_steps)
    if args.one_case:
        if not args.seeds or len(args.seeds) != 1:
            parser.error("Child requires exactly one seed")
        run_case(**common, seed=args.seeds[0], expected_freeze_sha256=args.expected_freeze_sha256)
    else:
        run_panel(**common, seeds=args.seeds, workers=args.workers, case_timeout_s=args.case_timeout_s)


if __name__ == "__main__":
    main()
