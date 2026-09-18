"""Hash-verified comparison and compact inspection data for the off-grid task."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import statistics

from docs.research.tools.feasible_benchmark_v1 import ROOT, load_benchmark_contract
from docs.research.tools.feasible_mission_replay import file_pin

BASE = ROOT / "docs/research/results/feasible25_offgrid_20260915"
LABELS = {
    "daily-coverage-feasibility-v2": "Original coverage (development diagnostic)",
    "daily-coverage-offgrid-v1": "Coverage reference",
    "daily-return-mission-mpc-v1": "Strict MPC",
    "mission-mpc-coverage-recovery-v1": "MPC with coverage recovery",
    "coverage-ppo-101": "PPO · seed101",
    "coverage-ppo-211": "PPO · seed211",
    "coverage-ppo-307": "PPO · seed307",
}


def collect():
    contract = load_benchmark_contract()
    expected_pins = {}
    for summary in BASE.rglob("summary.json"):
        for row in json.loads(summary.read_text()).get("runs", []):
            pin = row.get("recording")
            if pin and "path" in pin:
                path = Path(pin["path"])
                expected_pins[path.resolve()] = pin
    # The frozen development MPC panel publishes its hashes in a separate
    # index, while the newer evaluators include them in their summaries.
    for index in BASE.rglob("recording-index.json"):
        payload = json.loads(index.read_text())
        for field in ("summary", "freeze"):
            expected = payload.get(field)
            if expected:
                path = Path(expected["path"])
                actual = file_pin(path if path.is_absolute() else ROOT / path)
                if any(actual[key] != expected[key] for key in ("sha256", "bytes")):
                    raise ValueError(f"Recording index has changed provenance: {index}")
        for row in payload.get("records", []):
            pin = row["recording"]
            path = Path(pin["path"])
            path = path if path.is_absolute() else ROOT / path
            existing = expected_pins.get(path.resolve())
            if existing and any(
                existing[key] != pin[key] for key in ("sha256", "bytes")
            ):
                raise ValueError(f"Conflicting published recording pins: {path}")
            expected_pins[path.resolve()] = pin
    rows = []
    identities = {}
    seen = set()
    for path in sorted(BASE.rglob("*.json.gz")):
        # Training files and supplemental replay audits are not evaluation runs.
        if "coverage-ppo-v1" in path.parts:
            continue
        pin = file_pin(path)
        expected = expected_pins.get(path.resolve())
        if expected and (
            expected["sha256"] != pin["sha256"] or expected["bytes"] != pin["bytes"]
        ):
            raise ValueError(f"Recording differs from its published pin: {path}")
        record = json.loads(gzip.decompress(path.read_bytes()))
        if "mission_passed" not in record or "minute_truth_states" not in record:
            continue
        if record.get("benchmark_contract_sha256") != contract["contract_sha256"]:
            raise ValueError(f"Foreign task in result directory: {path}")
        split, seed, controller = record["split"], record["seed"], record["controller"]
        if seed not in contract["splits"][split]:
            raise ValueError(f"Scenario is outside recorded split: {path}")
        key = (split, seed, controller)
        if key in seen:
            raise ValueError(f"Duplicate controller/scenario recording: {key}")
        seen.add(key)
        manifest = record["manifest"]
        identity = (
            manifest["manifest_sha256"],
            manifest["beta_exogenous_episode_sha256"],
        )
        scenario_key = (split, seed)
        if scenario_key in identities and identities[scenario_key] != identity:
            raise ValueError(
                f"Controllers evaluated different scenario contracts: {scenario_key}"
            )
        identities[scenario_key] = identity
        e, p = record["economic_metrics"], record["physical_metrics"]
        if e["components_eur"].get("solar_opportunity_cost", 0) != 0:
            raise ValueError(f"Nonzero electricity charge in off-grid task: {path}")
        if abs(e["net_operating_value_eur"] - sum(e["components_eur"].values())) > 1e-7:
            raise ValueError(f"Economic components do not sum: {path}")
        full = record["execution_complete"]
        cycles = record["cycle_settlements"]
        deliveries = record["delivery"]["settlements"]
        expected_pass = (
            full
            and record["hard_safety_passed"]
            and len(cycles) == 7
            and all(x["feasible"] for x in cycles)
            and len(deliveries) == 28
            and all(x["unmet_kg"] <= 1e-9 for x in deliveries)
        )
        if record["mission_passed"] != expected_pass:
            raise ValueError(f"Incorrect complete-mission classification: {path}")
        modes = {}
        for command in record["commands"]:
            evidence = command.get("evidence")
            if isinstance(evidence, dict) and (
                "decision_mode" in evidence or "mode" in evidence
            ):
                mode = evidence.get("decision_mode", evidence.get("mode"))
                modes[mode] = modes.get(mode, 0) + 1
        rows.append(
            {
                "controller": controller,
                "label": LABELS.get(controller, controller),
                "split": split,
                "seed": seed,
                "execution_complete": full,
                "mission_passed": record["mission_passed"],
                "hard_safety_passed": record["hard_safety_passed"],
                "unmet_kg": p["unmet_kg"],
                "delivered_kg": p["delivered_kg"],
                "deliveries_met": p["commitments_met"],
                "modeled_value_eur": e["net_operating_value_eur"],
                "components_eur": e["components_eur"],
                "cycle_settlements": cycles,
                "decisions": record["decisions"],
                "failure": record["failure"],
                "decision_modes": modes,
                "recording": pin,
                "published_recording_pin_verified": expected is not None,
                "manifest_sha256": identity[0],
                "minute_truth_sha256": record["minute_truth_sha256"],
            }
        )
    groups = []
    for split, controller in sorted({(r["split"], r["controller"]) for r in rows}):
        selected = [
            r for r in rows if r["split"] == split and r["controller"] == controller
        ]
        complete = [r for r in selected if r["execution_complete"]]
        groups.append(
            {
                "split": split,
                "controller": controller,
                "label": selected[0]["label"],
                "cases": len(selected),
                "complete_weeks": len(complete),
                "full_missions": sum(r["mission_passed"] for r in selected),
                "safe_full_weeks": sum(r["hard_safety_passed"] for r in complete),
                "passing_daily_returns": sum(
                    sum(x["feasible"] for x in r["cycle_settlements"]) for r in selected
                ),
                "mean_unmet_kg_complete_weeks": (
                    statistics.mean(r["unmet_kg"] for r in complete)
                    if complete
                    else None
                ),
                "mean_contribution_eur_complete_weeks": (
                    statistics.mean(r["modeled_value_eur"] for r in complete)
                    if complete
                    else None
                ),
            }
        )
    return {
        "schema": "feasible25-inspection-data-v1",
        "benchmark_contract_sha256": contract["contract_sha256"],
        "contribution_boundary": "Dispatch contribution with benchmark wear/liability reserves; solar capital and fixed O&M unestimated and excluded; not full business profit.",
        "groups": groups,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-held-out", action="store_true")
    args = parser.parse_args()
    result = collect()
    if args.require_held_out:
        expected = set(load_benchmark_contract()["splits"]["held_out"])
        for controller in (
            "daily-coverage-offgrid-v1",
            "coverage-ppo-101",
            "coverage-ppo-211",
            "coverage-ppo-307",
            "daily-return-mission-mpc-v1",
        ):
            actual = {
                r["seed"]
                for r in result["rows"]
                if r["split"] == "held_out" and r["controller"] == controller
            }
            if actual != expected:
                raise ValueError(f"Incomplete held-out panel: {controller}")
    (BASE / "inspection-data.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result["groups"], indent=2))


if __name__ == "__main__":
    main()
