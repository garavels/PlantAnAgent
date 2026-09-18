"""Bounded diagnostic replays of two previously opened feasible25 failures.

This module cannot access fresh study seeds. Settings tried here are diagnostic
and are never written into the production controller or new evaluation protocol.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
for folder in (ROOT, ROOT / "backend/src", ROOT / "simulator/src"):
    sys.path.insert(0, str(folder))

from docs.research.tools.feasible_benchmark_v1 import BenchmarkEnv
from docs.research.tools.feasible_mission_controller import CoverageSettings, DailyCoverageController
from docs.research.tools.feasible_mission_replay import canonical_sha, file_pin, run_controller
from plant_backend.rl.v2.setpoint_control import ProcessSetpointTargetsV1

ALLOWED = (9930001, 9930011)
OUT = ROOT / "docs/research/results/authority_ppo_20260917/diagnosis"
OLD = ROOT / "docs/research/results/feasible25_offgrid_20260915"


class TimedBatchDiagnostic:
    """Diagnostic day1 schedule: bank early sun, concentrate stack production.

    The start clock and load are selected on the already-opened failures. This is
    a constructive feasibility experiment, not an untouched test or new baseline.
    From day2 onward, the unchanged production coverage controller takes over.
    """
    def __init__(self, start_hour=10., stack_kw=3000., running_downstream=0., methane_kw=180., dac_kw=1200.):
        self.parameters = dict(start_hour=start_hour, stack_kw=stack_kw, running_downstream=running_downstream, methane_kw=methane_kw, dac_kw=dac_kw)
        self.id = "diagnostic-timed-batch"
        self.baseline = DailyCoverageController(CoverageSettings(co2_buffer_kg=100., ch4_buffer_kg=50.))
        self.reset()

    def reset(self):
        self.baseline.reset()
        self.finished_stack = False

    def compute_command(self, observation, *, constraints, config, commitment_profile):
        base = self.baseline.compute_command(observation, constraints=constraints, config=config, commitment_profile=commitment_profile)
        if self.baseline.decisions > 96:
            return base
        plant, cycle = observation.plant, observation.cycle
        state = plant.state.boundary
        remaining_h = (cycle.cycle_decisions - cycle.decision_in_cycle) * plant.decision_interval_s / 3600
        remaining_steps = remaining_h * 3600 / config.dt_s
        delivery = plant.delivery
        count = 0 if delivery.steps_to_next is None or delivery.steps_to_next > remaining_steps else 1 + int((remaining_steps - delivery.steps_to_next + 1e-8) // delivery.delivery_period_steps)
        methane = max(0., count * delivery.next_requested_kg - cycle.ch4_margin_kg)
        hydrogen = max(0., methane * .5 - cycle.h2_margin_kg)
        carbon = max(0., methane * 2.75 - cycle.co2_margin_kg)
        snap = plant.state.electrolyser_commitment
        mode = snap.mode.value
        p = self.parameters
        hour = state.time_s / 3600 % 24
        e = 0.
        if hydrogen < .1:
            self.finished_stack = True
        if not self.finished_stack and p["start_hour"] <= hour < 19.:
            if mode in {"off", "cooldown", "cold_starting", "hot_starting"}:
                e = commitment_profile.start_load_frac * config.electrolyser.rated_kw
            else:
                e = min(p["stack_kw"], max(1200., hydrogen * 55. / .25))
        if mode in {"cold_starting", "hot_starting"}:
            e = commitment_profile.start_load_frac * config.electrolyser.rated_kw
        elif snap.min_up_remaining_s > 0:
            e = max(e, 1200.)
        downstream = p["running_downstream"] if e >= 1200. else 1.
        d = min(p["dac_kw"] * downstream, carbon * 2. / .25)
        m = min(p["methane_kw"] * downstream, methane * 1.2 / .25)
        if state.health["dac"].offline:
            d = 0.
        return ProcessSetpointTargetsV1(e, d, m, config.battery.max_charge_kw)


def source_path(seed):
    return OLD / ("coverage-held-out-a" if seed == 9930001 else "coverage-held-out-b") / f"{seed}.json.gz"


def load(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def summarize(result):
    minutes = result["minute_truth_states"]
    first_day = minutes[:1441]
    first, last = first_day[0], first_day[-1]
    metrics = result["physical_metrics"]
    settlement = result["cycle_settlements"][0] if result["cycle_settlements"] else None
    energies = {key: sum(s[key] for s in first_day[1:]) / 60 for key in (
        "solar_available_kw", "solar_curtailed_kw", "solar_to_battery_kw",
        "battery_to_electrolyser_kw", "electrolyser_power_kw", "dac_power_kw", "methanator_power_kw",
    )}
    return {
        "seed": result["seed"], "controller": result["controller"],
        "diagnostic_only": True, "execution_complete": result["execution_complete"],
        "failure": result["failure"], "mission_passed": result["mission_passed"],
        "day1_settlement": settlement, "day1_energies_kwh": energies,
        "day1_net_production_kg": {r: last[f"{r}_level_kg"] - first[f"{r}_level_kg"] for r in ("h2", "co2", "ch4")},
        "day1_delivery_unmet_kg": sum(s["unmet_kg"] for s in result["delivery"]["settlements"] if s["due_step"] <= 1440),
        "day1_guard_or_forced_stop_scope": "metrics cover all executed minutes; day1 only in 96-decision runs",
        "metrics": {k: metrics[k] for k in (
            "electrolyser_forced_stops", "electrolyser_aborted_starts", "guard_interventions",
            "guard_intervention_kwh", "out_of_range_decisions", "electrolyser_startup_energy_kwh",
            "electrolyser_low_load_time_s", "electrolyser_operating_time_s")},
        "minute_checks": result["minute_checks"],
        "minute_truth_sha256": result["minute_truth_sha256"],
        "all_daily_returns_passed": result["all_daily_returns_passed"],
        "every_delivery_passed": result["every_delivery_passed"],
        "hard_safety_passed": result["hard_safety_passed"],
        "note": "A partial day1 run cannot pass the seven-day mission gate; day1 settlement is assessed separately.",
    }


def candidate_settings():
    original = dict(co2_buffer_kg=100., ch4_buffer_kg=50.)
    yield "original", CoverageSettings(**original)
    for solar in (.95, 1.0):
        for hydrogen_weight in (40., 48., 52., 56.):
            yield f"zero-buffers-solar{solar}-hweight{hydrogen_weight}", CoverageSettings(
                solar_fraction=solar, h2_buffer_kg=0., co2_buffer_kg=0., ch4_buffer_kg=0.,
                battery_buffer_kwh=20., h2_energy_kwh_kg=hydrogen_weight,
            )
    for hydrogen_weight in (40., 48., 52.):
        yield f"small-buffers-hweight{hydrogen_weight}", CoverageSettings(
            solar_fraction=1., h2_buffer_kg=1., co2_buffer_kg=5., ch4_buffer_kg=5.,
            battery_buffer_kwh=20., h2_energy_kwh_kg=hydrogen_weight,
        )


class AuditedDiagnosticEnv(BenchmarkEnv):
    def __init__(self):
        super().__init__("held_out")
        self.minute_application_checks = []

    def step_process_setpoints(self, *args, **kwargs):
        result = super().step_process_setpoints(*args, **kwargs)
        execution = self.last_process_setpoint_execution
        assert execution is not None and len(execution.applications) == 15
        for app in execution.applications:
            self.minute_application_checks.append({
                "step": app.next_boundary_step,
                "guard_adjustments": list(app.adjustments),
                "guard_changed_action": app.guarded_action != app.dispatch.action,
                "applied_matches_guarded": app.applied_action == app.guarded_action,
            })
        return result


def run_one(seed, name, settings=None, *, full_week=False, retain=False, controller=None):
    if seed not in ALLOWED:
        raise ValueError("Diagnostic runner admits only the two already-opened failures")
    env = AuditedDiagnosticEnv()
    try:
        env.reset(seed=seed)
        baseline = load(source_path(seed))
        original_states = baseline["minute_truth_states"]
        assert env.simulator.state.to_dict() == original_states[0]
        assert all(abs(a - b["solar_available_kw"]) <= 1e-9 for a, b in zip(env.frozen_source.expected_solar_kw, original_states[1:], strict=True))
        assert canonical_sha(env.audit_manifest()["disturbances"]) == canonical_sha(baseline["manifest"]["disturbances"])
        controller = controller or DailyCoverageController(settings)
        controller.id = "diagnostic-" + name
        result = run_controller(env, controller, max_decisions=None if full_week else 96)
        result["minute_application_checks"] = env.minute_application_checks
        if name == "original":
            assert result["minute_truth_states"] == original_states[:len(result["minute_truth_states"])], "Baseline physical replay is not exact"
        row = summarize(result)
        row["settings"] = asdict(settings) if settings else controller.parameters
        changed = [m for m in env.minute_application_checks if m["guard_adjustments"] or m["guard_changed_action"]]
        row["exact_minute_application_audit"] = {
            "minutes_checked": len(env.minute_application_checks),
            "applied_matches_guarded_every_minute": all(m["applied_matches_guarded"] for m in env.minute_application_checks),
            "guard_changed_minutes": changed,
            "forced_stop_count_monotone_cumulative": result["physical_metrics"]["electrolyser_forced_stops"],
        }
        row["original_source"] = file_pin(source_path(seed))
        row["original_forcing_initial_and_faults_exact"] = True
        if retain:
            result["diagnostic_only"] = True
            result["diagnostic_settings"] = row["settings"]
            result["registered_evaluation_eligible"] = False
            path = OUT / f"{name}-{seed}-{'week' if full_week else 'day1'}.json.gz"
            path.write_bytes(gzip.compress(json.dumps(result, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
            row["recording"] = file_pin(path)
        return row
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("sweep", "week", "batch", "selected-week"), default="sweep")
    parser.add_argument("--candidate")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.monotonic()
    if args.mode == "selected-week":
        for seed in ALLOWED:
            row = run_one(seed, "batch-start10.0-stack3000.0", controller=TimedBatchDiagnostic(), full_week=True, retain=True)
            rows.append(row)
            payload = {
                "schema": "known-failure-constructive-feasibility-v1",
                "selection": "First tested batch candidate passed both day1 gates; replayed same configuration for full week with unchanged coverage from day2. Exploratory diagnostic only.",
                "source": file_pin(Path(__file__)), "rows": rows,
            }
            (OUT / "selected-full-weeks.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"seed": seed, "mission_passed": row["mission_passed"], "elapsed_s": time.monotonic() - start}), flush=True)
        return
    if args.mode == "batch":
        for start_hour in (10., 9., 11., 12.):
            for stack_kw in (3000., 2700., 2400.):
                name = f"batch-start{start_hour}-stack{stack_kw}"
                if args.candidate and name != args.candidate:
                    continue
                for seed in ALLOWED:
                    controller = TimedBatchDiagnostic(start_hour=start_hour, stack_kw=stack_kw)
                    row = run_one(seed, name, controller=controller, retain=True)
                    rows.append(row)
                    print(json.dumps({"candidate": name, "seed": seed, "day1": row["day1_settlement"], "metrics": row["metrics"], "elapsed_s": time.monotonic() - start}), flush=True)
                    (OUT / "batch-sweep.json").write_text(json.dumps({"schema": "known-failure-diagnostic-batch-sweep-v1", "source": file_pin(Path(__file__)), "rows": rows}, indent=2, allow_nan=False) + "\n")
        return
    for name, settings in candidate_settings():
        if args.candidate and name != args.candidate:
            continue
        for seed in ALLOWED:
            row = run_one(seed, name, settings, full_week=args.mode == "week", retain=args.mode == "week" or name == "original")
            rows.append(row)
            print(json.dumps({"candidate": name, "seed": seed, "day1": row["day1_settlement"], "metrics": row["metrics"], "elapsed_s": time.monotonic() - start}), flush=True)
            payload = {"schema": "known-failure-diagnostic-settings-sweep-v1", "created_at": datetime.now(timezone.utc).isoformat(), "source": file_pin(Path(__file__)), "scope": "Exploratory diagnostic only on previously opened scenarios; no generalization claim or new held-out scenarios", "rows": rows}
            path = OUT / ("settings-sweep.json" if args.mode == "sweep" else f"week-{args.candidate or 'all'}.json")
            path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
