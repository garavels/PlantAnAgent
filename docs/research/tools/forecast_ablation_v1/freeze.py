"""Freeze supplementary rules before the main held-out gate exists."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from docs.research.tools.comparison_ppo_v2 import evaluate as main_e
from docs.research.tools.free_ppo_v1.analyze import canonical_sha, file_pin, publish
from docs.research.tools.forecast_ablation_v1.policy import PersistencePolicy

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "forecast-ablation-v1"
LABEL = "daylight-persistence"
CONDITION = "d30-sampled"
VARIANT = {"delivery_rate_kg_h": 30, "opening_battery": "sampled", "battery_mode": "enabled"}
HARD_STOP = "2026-09-17T18:42:48+00:00"


def sources():
    paths = {Path(pin["path"]) for pin in main_e.sources()}
    paths.update(Path(__file__).parent.glob("*.py"))
    paths.add(ROOT / "docs/research/FORECAST_ABLATION_V1_PROTOCOL.md")
    return [file_pin(path) for path in sorted(paths)]


def freeze_evaluation(output, *, split="held_out", training_seed=20000019):
    if split not in ("train", "held_out"):
        raise ValueError("Supplement admits training fixtures or prescribed held-out only")
    if not 20000000 <= training_seed < 20100000:
        raise ValueError("Engineering fixture seed must be in training namespace")
    env = main_e.environment()
    seeds = list(env.HELD_OUT_SEEDS) if split == "held_out" else [training_seed]
    payload = {"schema": SCHEMA + "-freeze", "created_at": datetime.now(timezone.utc).isoformat(),
        "description": "Supplement declared after development; no new performance tuning; shared32 main scenarios",
        "sources": sources(), "environment_identity_sha256": env.environment_identity_sha256(),
        "source_identity_sha256": env.source_identity_sha256(),
        "controllers": {LABEL: {"label": LABEL, "kind": "baseline", "baseline": "daylight-persistence", "policy_metadata": PersistencePolicy().metadata()}},
        "conditions": {CONDITION: VARIANT}, "panels": [{"split": split, "controller": LABEL, "condition": CONDITION, "seeds": seeds}],
        "attempts_per_case": 1, "gate": None,
        "late_gate_contract": {"schema": "comparison-ppo-v2-evaluation-gate", "endpoints_frozen": True,
            "completion_schema": "comparison-ppo-v2-finish-supervisor-completion", "main_outcomes": 1160,
            "require_gate_created_after_this_freeze": True, "require_completion_before_any_supplementary_heldout": True},
        "full_trace_rule": "First prescribed scenario; all inputs/actions retained for every case",
        "bootstrap": {"seed": 20260917, "replicates": 10000, "unit": "32 paired environmental scenarios, shared with main study"},
        "criteria": {"rescues_at_least": 4, "regressions_at_most": 0, "mean_contribution_at_least_eur": 250,
            "paired_ci_lower_above": 0, "contribution_requires_no_lower_delivery_count": True},
        "hard_stop_utc": HARD_STOP}
    payload["sha256"] = canonical_sha(payload)
    publish(output, payload)
    return payload


def require_freeze(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA + "-freeze" or payload.get("sha256") != canonical_sha({k: v for k, v in payload.items() if k != "sha256"}):
        raise ValueError("Supplementary freeze identity differs")
    env = main_e.environment()
    if payload["sources"] != sources() or payload["environment_identity_sha256"] != env.environment_identity_sha256() or payload["source_identity_sha256"] != env.source_identity_sha256():
        raise ValueError("Frozen supplementary source or environment changed")
    return payload


def _verified_pin(pin):
    if file_pin(pin["path"]) != pin:
        raise ValueError("Main campaign evidence pin changed")
    return pin


def main_admission(freeze, gate_path, main_completion_path):
    if gate_path is None or main_completion_path is None:
        raise ValueError("Main held-out gate and completed supervisor receipt are required")
    main_e.environment()._admit("held_out", Path(gate_path))
    gate = json.loads(Path(gate_path).read_text())
    closure = json.loads(Path(main_completion_path).read_text())
    contract = freeze["late_gate_contract"]
    if (closure.get("schema") != contract["completion_schema"] or closure.get("status") != "completed"
            or closure.get("outcomes") != contract["main_outcomes"] or gate.get("schema") != contract["schema"]
            or gate.get("endpoints_frozen") is not True):
        raise ValueError("Main campaign has not closed all prescribed evaluations")
    if datetime.fromisoformat(gate["created_at"]) <= datetime.fromisoformat(freeze["created_at"]):
        raise ValueError("Supplementary rules were not frozen before the main held-out gate")
    if closure.get("endpoints") != gate.get("endpoints"):
        raise ValueError("Main gate and completion identify different fixed endpoints")
    for pin in (gate["endpoints"], gate["training_closure"], closure["analysis"]):
        _verified_pin(pin)
    stages = closure.get("stages", [])
    if not {"development-learning", "held-out-primary", "held-out-battery", "analysis"}.issubset({s.get("stage") for s in stages}) or any(stage.get("exit_code") != 0 for stage in stages):
        raise ValueError("Main campaign stage did not close successfully")
    return {"gate": file_pin(gate_path), "main_completion": file_pin(main_completion_path),
            "main_analysis": closure["analysis"], "endpoints": gate["endpoints"], "training_closure": gate["training_closure"]}


def admit(freeze, *, controller, condition, split, seed, gate_path=None, max_decisions=None, main_completion_path=None):
    matching = [p for p in freeze["panels"] if (p["controller"], p["condition"], p["split"]) == (controller, condition, split)]
    if len(matching) != 1 or seed not in matching[0]["seeds"]:
        raise ValueError("Case is outside the frozen supplementary panel")
    if max_decisions is not None and (split != "train" or not 1 <= max_decisions <= 672):
        raise ValueError("Short engineering fixtures require training seeds")
    if split == "held_out":
        main_admission(freeze, gate_path, main_completion_path)
    return matching[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("train", "held_out"), default="held_out")
    p.add_argument("--training-seed", type=int, default=20000019)
    a = p.parse_args()
    result = freeze_evaluation(a.output, split=a.split, training_seed=a.training_seed)
    print(json.dumps({"path": str(a.output), "sha256": result["sha256"]}))


if __name__ == "__main__":
    main()
