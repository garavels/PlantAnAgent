"""Supplementary paired forecast-minus-persistence effects, with all32 cases."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path

from docs.research.tools.comparison_ppo_v2 import analyze as main_a
from docs.research.tools.comparison_ppo_v2 import evaluate as main_e
from docs.research.tools.free_ppo_v1 import analyze as v1
from docs.research.tools.forecast_ablation_v1.freeze import SCHEMA, LABEL, CONDITION, require_freeze, main_admission
from docs.research.tools.forecast_ablation_v1.policy import PersistencePolicy, verify_witness
from docs.research.tools.comparison_ppo_v2.baselines import BaselineConfig


def validate_record(row, *, persistence=False):
    main_a.validate_row(row)
    pin = row.get("recording")
    if not pin:
        if row["execution_complete"]:
            raise ValueError("Complete outcome lacks its recording")
        return
    if v1.file_pin(pin["path"]) != pin:
        raise ValueError("Recording changed")
    with gzip.open(pin["path"], "rt") as stream:
        record = json.load(stream)
    if record["summary"] != {k: val for k, val in row.items() if k != "recording"}:
        raise ValueError("Recording and row summary differ")
    states = record.get("minute_truth_states")
    if states and v1.canonical_sha(states) != row["retained_minute_truth_sha256"]:
        raise ValueError("Full minute trace changed")
    if row["execution_complete"] and row["seed"] == 20200000 and (not states or len(states) != 10081):
        raise ValueError("Prescribed first full trajectory is missing")
    if persistence and record["decision_trace"]:
        policy = PersistencePolicy(BaselineConfig(**row["baseline_metadata"]["baseline"]["config"]))
        import hashlib
        digest = hashlib.sha256()
        for decision in record["decision_trace"]:
            evidence = decision["evidence"]
            verify_witness(evidence, policy)
            if evidence["action"] != decision["action"]:
                raise ValueError("Recorded inference/action differ")
            digest.update(json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        if digest.hexdigest() != row["inference"]["witness_sha256"]:
            raise ValueError("Full inference witness sequence differs")
        if row["inference"]["transformation_verified_decisions"] != len(record["decision_trace"]):
            raise ValueError("Transformation count differs")


def criterion(paired, reference, persistence, *, pairing_verified):
    estimate = paired["metrics"]["net_operating_value_eur"]["all_case"]
    eligible = (len(reference) == len(persistence) == 32 and pairing_verified
                and all(r["execution_complete"] and r["physical_invariants_passed"] and r["economic_audit"]["passed"]
                        for r in [*reference, *persistence]))
    reliability = len(paired["rescued_cases"]) >= 4 and not paired["regressed_cases"]
    no_lower_deliveries = sum(r["every_delivery_passed"] for r in reference) >= sum(r["every_delivery_passed"] for r in persistence)
    contribution = (no_lower_deliveries and estimate["mean"] is not None and estimate["mean"] >= 250
                    and estimate["ci95"] is not None and estimate["ci95"][0] > 0)
    return {"eligible_complete_audited_pairs": eligible, "forecast_rescues": len(paired["rescued_cases"]),
        "forecast_regressions": len(paired["regressed_cases"]), "forecast_no_lower_delivery_count": no_lower_deliveries,
        "reliability_bar_met": bool(eligible and reliability), "contribution_bar_met": bool(eligible and contribution),
        "practical_evidence_bar_met": bool(eligible and (reliability or contribution)),
        "forecast_minus_persistence_eur": estimate,
        "interpretation": "Incremental forecast-input value for this fixed heuristic; means and p10 ablated jointly. No claim of optimal planning, RL necessity or battery necessity."}


def collect(root, *, main_analysis_path, gate_path, main_completion_path):
    root = Path(root)
    wave = json.loads((root / "wave.json").read_text())
    freeze_pin = wave["evaluation_freeze"]
    if v1.file_pin(freeze_pin["path"]) != freeze_pin:
        raise ValueError("Supplementary freeze file changed")
    freeze = require_freeze(freeze_pin["path"])
    admission = main_admission(freeze, gate_path, main_completion_path)
    if v1.file_pin(main_analysis_path) != admission["main_analysis"]:
        raise ValueError("Analysis is not the completed main campaign analysis")
    if wave.get("schema") != SCHEMA + "-wave" or wave["all_requested_cases_retained"] is not True or wave["source_files_unchanged"] is not True:
        raise ValueError("Supplementary wave did not retain its frozen panel")
    if len(wave["panels"]) != 1:
        raise ValueError("Exactly one supplementary panel is prescribed")
    panel_pin = wave["panels"][0]
    if v1.file_pin(panel_pin["path"]) != panel_pin:
        raise ValueError("Panel changed")
    panel = json.loads(Path(panel_pin["path"]).read_text())
    expected = freeze["panels"][0]
    if (panel["schema"] != SCHEMA + "-panel" or panel["split"] != "held_out"
            or panel["controller"] != LABEL or panel["condition"] != CONDITION
            or panel["evaluation_freeze_sha256"] != freeze["sha256"] or not panel["source_files_unchanged_during_panel"]
            or panel["requested_seeds"] != expected["seeds"]):
        raise ValueError("Supplementary panel identity differs")
    rows = panel["runs"]
    if sorted(r["seed"] for r in rows) != list(range(20200000, 20200032)):
        raise ValueError("Every supplementary success and failure must be retained exactly once")
    main = json.loads(Path(main_analysis_path).read_text())
    if main.get("schema") != main_e.SCHEMA + "-analysis" or len(main["rows"]) != 1160:
        raise ValueError("Main analysis schema/count differs")
    for pin in main["summaries"]:
        if v1.file_pin(pin["path"]) != pin:
            raise ValueError("Main source panel changed")
    for item in main["freezes"]:
        main_e.require_freeze(item["file"]["path"])
    reference = [r for r in main["rows"] if (r["split"], r["condition"], r["controller"]) == ("held_out", CONDITION, "daylight")]
    if sorted(r["seed"] for r in reference) != list(range(20200000, 20200032)):
        raise ValueError("Main daylight reference must contain all32 prescribed cases")
    for row in rows:
        if (row["controller"], row["condition"], row["split"], row["evaluation_freeze_sha256"]) != (LABEL, CONDITION, "held_out", freeze["sha256"]):
            raise ValueError("Supplementary outcome identity differs")
        validate_record(row, persistence=True)
        if row["execution_complete"] and row["main_campaign_admission"] != admission:
            raise ValueError("Outcome ran against a different main closure")
    for row in reference:
        validate_record(row)
    by_ref = {r["seed"]: r for r in reference}
    by_persistence = {r["seed"]: r for r in rows}
    pairing = []
    for seed in expected["seeds"]:
        a, b = by_ref[seed], by_persistence[seed]
        keys = ("paired_source_sha256", "exogenous_episode_sha256", "manifest_sha256", "initial_state", "variant")
        available = all(k in a and k in b for k in keys)
        if available and any(a[k] != b[k] for k in keys):
            raise ValueError("Reference differs in weather, faults, opening stock, or delivery schedule")
        pairing.append({"seed": seed, "verified": available})
    paired = main_a.paired_comparison(by_persistence, by_ref,
        label="Forecast daylight minus matched persistence (supplementary)", split="held_out", condition=CONDITION)
    return {"schema": SCHEMA + "-analysis", "supplementary": True, "shared_environmental_scenarios": 32,
        "rows": rows, "groups": [main_a.summarize_group(rows)],
        "reference_rows": reference, "reference_groups": [main_a.summarize_group(reference)],
        "paired_comparisons": [paired], "criterion": criterion(paired, reference, rows, pairing_verified=all(p["verified"] for p in pairing)),
        "pairing": pairing, "main_campaign_admission": admission, "reference_main_analysis": admission["main_analysis"], "summaries": [panel_pin],
        "freezes": [{"sha256": freeze["sha256"], "description": freeze["description"], "file": freeze_pin}],
        "analysis_source": v1.file_pin(__file__), "bootstrap": freeze["bootstrap"],
        "interpretation": "Supplement declared after development but frozen before fresh outcomes; not32 additional independent weather scenarios. All outcomes retained."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "main-analysis", "gate", "main-completion", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = collect(a.root, main_analysis_path=a.main_analysis, gate_path=a.gate, main_completion_path=a.main_completion)
    a.output.mkdir(parents=True, exist_ok=False)
    v1.publish(a.output / "analysis.json", result)
    main_a.write_csv(result, a.output / "outcomes.csv")
    print(json.dumps({"outcomes": len(result["rows"]), "criterion": result["criterion"], "output": str(a.output)}))


if __name__ == "__main__":
    main()
