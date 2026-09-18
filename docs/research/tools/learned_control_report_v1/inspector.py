"""Build an offline inspector from one completed learned-control evaluation.

Usage: python -m docs.research.tools.learned_control_report_v1.inspector
       --results PANEL_DIRECTORY --output NEW_HTML

Scientific inputs are read-only. This module never imports the evaluator, constructs
an environment, loads a model, chooses a policy, or modifies scientific outputs.
The same builder accepts development and final panels and derives all labels,
groups, seeds and conditions from the supplied frozen evaluation manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

CONDITIONS = ("ordinary", "control", "pv6", "pv18", "pv36", "meth24", "meth48")
METRICS = ("net_operating_value_eur", "unmet_kg", "all_delivery_success", "missed_deliveries")
FIELDS = ("solar_available_kw", "battery_soc", "battery_energy_kwh", "h2_level_kg", "co2_level_kg", "ch4_level_kg",
          "dac_readiness_fraction", "methanator_readiness_fraction", "electrolyser_power_kw", "dac_power_kw", "methanator_power_kw",
          "battery_to_electrolyser_kw", "battery_to_dac_kw", "battery_to_methanator_kw", "ch4_production_kg_h",
          "h2_production_kg_h", "co2_capture_kg_h", "solar_to_battery_kw", "solar_curtailed_kw")
ACTION_LABELS = ("Solar charging share", "Solar electrolysis share", "Solar capture share", "Solar methanation share",
                 "Battery electrolysis support", "Battery capture support", "Battery methanation support")
SCHEMA = "learned-control-offline-inspector-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_pin(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def verify_pin(pin, path=None):
    actual = file_pin(path or pin["path"])
    if actual["sha256"] != pin["sha256"] or actual["bytes"] != pin["bytes"]:
        raise ValueError("Pinned artifact changed: " + actual["path"])
    return actual


def audited(row):
    return bool(row.get("execution_complete") and row.get("sources_unchanged") and row.get("failure") is None
                and row.get("minute_audit", {}).get("all_physical_invariants_passed")
                and row.get("economic_audit", {}).get("passed"))


def outcome(row, name):
    if not audited(row):
        return None
    if name == "all_delivery_success":
        return float(row["every_delivery_passed"])
    if name == "missed_deliveries":
        return float(sum(d["unmet_kg"] > 1e-9 for d in row["delivery"]["settlements"]))
    value = row["economic_metrics"].get(name)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _mean(values):
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def _same_number(actual, expected, description):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError(description + " availability differs")
    elif not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError(description + " differs from retained outcomes")


def validate_panel(spec, summary, analysis):
    """Cross-check every compact outcome and the displayed analysis quantities."""
    if spec.get("schema") != "learned-control-evaluation-v1" or spec.get("sha256") != digest({k: v for k, v in spec.items() if k != "sha256"}):
        raise ValueError("Evaluation manifest schema or digest differs")
    if spec.get("max_decisions") is not None:
        raise ValueError("This seven-day inspector does not present truncated training prefixes as weeks")
    if spec.get("split") not in ("development", "final", "train"):
        raise ValueError("Unknown evaluation split")
    if not spec["seeds"] or len(set(spec["seeds"])) != len(spec["seeds"]):
        raise ValueError("Scenario list is empty or repeated")
    if not spec["conditions"] or any(c not in CONDITIONS for c in spec["conditions"]):
        raise ValueError("Unknown or absent condition")
    if spec["split"] == "final" and (spec["seeds"] != list(range(52_000_000, 52_000_032)) or tuple(spec["conditions"]) != CONDITIONS or not spec.get("gate") or not spec.get("selected_recipe")):
        raise ValueError("Final label requires the full 32-seed seven-condition grid and selection pins")
    expected = set(itertools.product(spec["seeds"], spec["conditions"], spec["controllers"]))
    prescribed = {(c["seed"], c["condition"], c["controller"]) for c in spec["cases"]}
    if expected != prescribed or len(spec["cases"]) != len(expected) or spec["expected_cases"] != len(expected):
        raise ValueError("Frozen case grid is not the complete declared product")
    index = {}
    for row in summary["rows"]:
        key = row["seed"], row["condition"], row["controller"]
        if key in index or key not in expected:
            raise ValueError("Duplicate or unprescribed retained outcome")
        if row.get("freeze_sha256") != spec["sha256"] or row.get("split") != spec["split"]:
            raise ValueError("Outcome belongs to a different manifest or split")
        if row.get("controller_kind") != spec["controllers"][row["controller"]]["kind"]:
            raise ValueError("Outcome controller kind differs from the frozen controller")
        if row.get("execution_complete"):
            if row["decisions"] != 672 or row["physical_states"] != 10081 or len(row["delivery"]["settlements"]) != 28:
                raise ValueError("Completed case lacks a full week")
            settlements = row["delivery"]["settlements"]
            if row["every_delivery_passed"] != all(d["unmet_kg"] <= 1e-9 for d in settlements):
                raise ValueError("Full-week success disagrees with delivery settlements")
            _same_number(row["economic_metrics"]["unmet_kg"], sum(d["unmet_kg"] for d in settlements), "Unmet mass")
        index[key] = row
    if set(index) != expected or summary["case_count"] != len(expected):
        raise ValueError("Not every prescribed outcome is retained")
    if analysis.get("schema") != "learned-control-analysis-v1" or analysis.get("freeze_sha256") != spec["sha256"] or analysis.get("split") != spec["split"] or analysis.get("case_count") != len(expected):
        raise ValueError("Analysis identity differs from the frozen outcome panel")
    if analysis["all_cases_audited"] != all(audited(row) for row in index.values()):
        raise ValueError("Analysis audit status differs from retained cases")
    if summary["all_executions_and_audits_passed"] != analysis["all_cases_audited"]:
        raise ValueError("Summary audit status differs from the analysis")
    for seed, condition in itertools.product(spec["seeds"], spec["conditions"]):
        paired = [index[(seed, condition, label)] for label in spec["controllers"]]
        for field in ("physical_solar_forcing_sha256", "initial_state"):
            known = [digest(row[field]) for row in paired if row.get(field) is not None]
            if len(set(known)) > 1:
                raise ValueError("Controller forcing or opening states are not paired")
    groups = {}
    for label, controller in spec["controllers"].items():
        group = controller.get("group", controller.get("recipe"))
        if group:
            groups.setdefault(group, []).append(label)
    if analysis["groups"] != groups:
        raise ValueError("Recipe groups differ from the frozen controller manifest")
    populations = {label: [label] for label in spec["controllers"]}
    populations.update({"recipe:"+group: members for group, members in groups.items()})
    panels = {item["panel"]: item["weights"] for item in analysis["summaries"]}
    expected_panels = {c: {c: 1.0} for c in spec["conditions"]}
    stress = CONDITIONS[2:]
    if all(c in spec["conditions"] for c in stress):
        expected_panels["stress"] = {c: .2 for c in stress}
    if "ordinary" in spec["conditions"] and all(c in spec["conditions"] for c in stress):
        expected_panels["mixture"] = {"ordinary": .5, **{c: .1 for c in stress}}
    if panels != expected_panels:
        raise ValueError("Displayed panels/weights differ from the declared analysis design")
    expected_summaries = set(itertools.product(populations, panels))
    actual_summaries = set()
    for item in analysis["summaries"]:
        key = item["population"], item["panel"]
        if key in actual_summaries or key not in expected_summaries:
            raise ValueError("Duplicate or unprescribed displayed summary")
        actual_summaries.add(key)
        if item["members"] != populations[item["population"]] or item["n_base_scenarios"] != len(spec["seeds"]):
            raise ValueError("Summary population or scenario count differs")
        weights = item["weights"]
        if weights != expected_panels[item["panel"]]:
            raise ValueError("Population has different weights within one panel")
        if not weights or any(c not in spec["conditions"] or not isinstance(w, (int, float)) or w <= 0 for c, w in weights.items()):
            raise ValueError("Summary weights are not positive weights on declared conditions")
        if item["n_policy_condition_cells"] != len(spec["seeds"]) * len(item["members"]) * len(weights):
            raise ValueError("Summary correlated cell count differs")
        for name in METRICS:
            values = []
            for seed in spec["seeds"]:
                terms = [(weight, _mean([outcome(index[(seed, condition, member)], name) for member in item["members"]])) for condition, weight in weights.items()]
                values.append(sum(w*v for w, v in terms) / sum(weights.values()) if all(v is not None for _, v in terms) else None)
            reported = item["metrics"][name]
            if reported["n"] != len(values) or reported["n_observed"] != sum(v is not None for v in values) or reported["complete"] != all(v is not None for v in values):
                raise ValueError("Summary completeness or denominator differs")
            if len(reported["values"]) != len(values):
                raise ValueError("Summary scenario values differ in length")
            for actual, expected_value in zip(reported["values"], values):
                _same_number(actual, expected_value, "Per-scenario summary")
            _same_number(reported["mean"], _mean(values), "Summary mean")
    if actual_summaries != expected_summaries:
        raise ValueError("Some declared population summaries are missing")
    validate_analysis_statistics(spec, analysis, populations, expected_panels)
    return index


def validate_analysis_statistics(spec, analysis, populations, panels):
    """Independently reconcile displayed paired estimates and registered flags.

    The bootstrap is recomputed from the already verified per-seed outcomes.
    This is arithmetic over retained evidence, not another policy evaluation.
    """
    import numpy as np
    summaries = {(s["population"], s["panel"]): s for s in analysis["summaries"]}
    bootstrap_cache = {}

    def expected_estimate(values, confidence):
        key = tuple(values), confidence
        if key not in bootstrap_cache:
            if any(v is None for v in values):
                bootstrap_cache[key] = (None, [None, None])
            else:
                data = np.asarray(values, dtype=float)
                rng = np.random.default_rng(91_818)
                means = data[rng.integers(0, len(data), size=(10_000, len(data)))].mean(axis=1)
                tail = (1-confidence)/2
                bootstrap_cache[key] = (float(data.mean()), np.quantile(means, [tail, 1-tail]).tolist())
        return bootstrap_cache[key]

    def comparison(item, treatment, reference, panel, confidence):
        if (item["treatment_members"] != populations[treatment] or item["reference_members"] != populations[reference]
                or item["weights"] != panels[panel] or item["direction"] != "treatment minus reference"):
            raise ValueError("Paired comparison identity or weighting differs")
        if set(item["metrics"]) != set(METRICS):
            raise ValueError("Paired comparison metric set differs")
        for name in METRICS:
            left = summaries[(reference, panel)]["metrics"][name]["values"]
            right = summaries[(treatment, panel)]["metrics"][name]["values"]
            differences = [r-l if r is not None and l is not None else None for l, r in zip(left, right)]
            estimate = item["metrics"][name]
            if (estimate["n_base_scenarios"] != len(spec["seeds"]) or estimate["confidence"] != confidence
                    or estimate["replicates"] != 10_000 or estimate["bootstrap_seed"] != 91_818
                    or estimate["complete"] != all(v is not None for v in differences)
                    or len(estimate["per_seed_differences"]) != len(differences)):
                raise ValueError("Paired bootstrap design or completeness differs")
            for actual, expected in zip(estimate["per_seed_differences"], differences):
                _same_number(actual, expected, "Paired scenario difference")
            _same_number(estimate["reference_mean"], _mean(left), "Reference mean")
            _same_number(estimate["treatment_mean"], _mean(right), "Treatment mean")
            expected_mean, expected_interval = expected_estimate(differences, confidence)
            _same_number(estimate["mean"], expected_mean, "Paired mean")
            if len(estimate["interval"]) != 2:
                raise ValueError("Paired interval must have two bounds")
            for actual, expected in zip(estimate["interval"], expected_interval):
                _same_number(actual, expected, "Paired bootstrap interval")

    baselines = [label for label, c in spec["controllers"].items() if c["kind"] == "baseline"]
    expected_pairs = {(treatment, reference, panel) for treatment in populations for reference in baselines for panel in panels if populations[treatment] != [reference]}
    comparisons = {}
    for item in analysis["paired_comparisons"]:
        key = item["treatment"], item["reference"], item["panel"]
        if key in comparisons or key not in expected_pairs:
            raise ValueError("Duplicate or unprescribed paired comparison")
        comparison(item, *key, .95)
        comparisons[key] = item
    if comparisons.keys() != expected_pairs:
        raise ValueError("Some prescribed paired comparisons are missing")
    primary = analysis.get("primary_claim")
    if bool(primary) != bool(spec.get("primary")):
        raise ValueError("A primary claim is missing or was not prospectively registered")
    if not primary:
        return
    recipe = spec["primary"].get("recipe", spec["primary"].get("group"))
    reference = spec["primary"]["reference"]
    members = analysis["groups"].get(recipe)
    if primary["recipe"] != recipe or primary["reference"] != reference or primary["members"] != members:
        raise ValueError("Registered primary recipe/reference differs")
    population = "recipe:" + recipe
    for panel in ("mixture", "stress", "ordinary"):
        comparison(primary["comparisons"][panel], population, reference, panel, .975)
    get = lambda panel, name: primary["comparisons"][panel]["metrics"][name]
    mix, unmet, success, ordinary = get("mixture", "net_operating_value_eur"), get("stress", "unmet_kg"), get("stress", "all_delivery_success"), get("ordinary", "net_operating_value_eur")
    complete = all(m["complete"] for m in (mix, unmet, success, ordinary))
    optimizer_seeds = [spec["controllers"][m].get("optimizer_seed") for m in members]
    independent = len(members) == 3 and None not in optimizer_seeds and len(set(optimizer_seeds)) == 3
    reduction = (unmet["reference_mean"]-unmet["treatment_mean"]) / unmet["reference_mean"] if complete and unmet["reference_mean"] > 0 else None
    checks = {
        "three_independent_optimizer_seeds": independent,
        "all_cases_complete_and_audited": complete,
        "mixture_contribution_gain_at_least_500_eur": bool(complete and mix["mean"] >= 500),
        "mixture_contribution_975ci_positive": bool(complete and mix["interval"][0] > 0),
        "stress_unmet_reduction_at_least_20_percent": bool(reduction is not None and reduction >= .2),
        "stress_unmet_delta_975ci_negative": bool(complete and unmet["interval"][1] < 0),
        "stress_all_delivery_success_not_lower": bool(complete and success["mean"] >= 0),
        "ordinary_contribution_loss_at_most_500_eur": bool(complete and ordinary["mean"] >= -500),
    }
    if primary["checks"] != checks or primary["strong_headline_bar_met"] != all(checks.values()):
        raise ValueError("Primary headline flags differ from the prescribed checks")
    _same_number(primary["relative_stress_unmet_reduction"], reduction, "Relative stress unmet reduction")
    individuals = primary["individual_policies"]
    if [p["controller"] for p in individuals] != members:
        raise ValueError("Primary claim omits or selects optimizer policies")
    for individual in individuals:
        member = individual["controller"]
        contribution = comparisons[(member, reference, "mixture")]["metrics"]["net_operating_value_eur"]["mean"]
        loss = comparisons[(member, reference, "stress")]["metrics"]["unmet_kg"]["mean"]
        _same_number(individual["mixture_contribution_delta"], contribution, "Individual contribution delta")
        _same_number(individual["stress_unmet_delta"], loss, "Individual unmet delta")
        expected_direction = contribution is not None and loss is not None and contribution > 0 and loss < 0
        if individual["optimizer_seed"] != spec["controllers"][member].get("optimizer_seed") or individual["both_directions_improved"] != expected_direction:
            raise ValueError("Individual policy identity or improvement flag differs")
    if primary["consistent_across_optimizer_seeds"] != (independent and all(i["both_directions_improved"] for i in individuals)):
        raise ValueError("Cross-optimizer consistency flag differs")


def controller_metadata(controller):
    result = {k: controller[k] for k in ("label", "kind", "baseline", "config", "group", "recipe", "optimizer_seed", "checkpoint", "checkpoint_files") if k in controller}
    if controller["kind"] == "baseline":
        result["display_kind"] = "Configured baseline" if controller.get("config") else "Original baseline"
        result["configuration_status"] = "Explicit parameter overrides" if controller.get("config") else "Original configuration; no parameter overrides"
    elif controller["kind"] == "legacy-ppo":
        result["display_kind"] = "Legacy PPO"
    else:
        result["display_kind"] = "Learned controller"
    for checkpoint_pin in controller.get("checkpoint_files", {}).values():
        verify_pin(checkpoint_pin)
    pin = controller.get("checkpoint_files", {}).get("metadata")
    if pin:
        verify_pin(pin)
        metadata = json.loads(Path(pin["path"]).read_text())
        result["checkpoint_training"] = {k: metadata[k] for k in ("schema", "kind", "num_timesteps", "additional_timesteps", "ppo_update_count", "optimizer_seed", "created_at", "plan_sha256") if k in metadata}
    return result


def trace_payload(recording, row, controller=None):
    if recording["summary"] != {k: v for k, v in row.items() if k != "recording"}:
        raise ValueError("Retained recording and compact outcome differ")
    states, commands = recording["minute_truth_states"], recording["decision_trace"]
    if len(commands) != row["decisions"] or (states and len(states) != 15 * len(commands) + 1):
        raise ValueError("Retained minute/decision lengths do not reconcile")
    if row["execution_complete"] and (len(states) != 10081 or len(commands) != 672):
        raise ValueError("First-seed recording lacks its full minute trace")
    if row.get("retained_minute_truth_sha256") != (digest(states) if states else None):
        raise ValueError("Full physical minute-trace digest differs")
    if states:
        if states[0] != row["initial_state"] or states[-1] != row["final_state"]:
            raise ValueError("Recording opening/final state differs from its summary")
        if any(state["step"] != i or not math.isclose(state["time_s"], states[0]["time_s"] + 60*i, abs_tol=1e-7)
               for i, state in enumerate(states)):
            raise ValueError("Physical time is not a contiguous minute sequence")
    witness = hashlib.sha256()
    for index, command in enumerate(commands):
        evidence = command["evidence"]
        if command["decision_index"] != index or len(evidence["observation"]) != 149 or len(command["action"]) != 7:
            raise ValueError("Actual observation/action witness has the wrong dimensions")
        if not all(math.isfinite(v) for v in [*evidence["observation"], *command["action"]]):
            raise ValueError("Non-finite policy input/action witness")
        if any(not -1 <= v <= 1 for v in command["action"]):
            raise ValueError("Policy action is outside its physical action interface")
        if evidence["observation_sha256"] != digest(evidence["observation"]) or command["action"] != evidence["action"]:
            raise ValueError("Actual observation/action witness digest differs")
        if row["controller_kind"] != "baseline":
            if len(evidence["normalized_observation"]) != 149 or len(evidence["raw_mean"]) != 7:
                raise ValueError("Learned inference witness dimensions differ")
            if [min(1.0, max(-1.0, x)) for x in evidence["raw_mean"]] != command["action"]:
                raise ValueError("Saved learned command differs from its clipped actor mean")
            if not all(math.isfinite(v) for v in [*evidence["normalized_observation"], *evidence["raw_mean"]]):
                raise ValueError("Non-finite learned inference witness")
        else:
            details = evidence["baseline_inference"]
            expected_input = evidence["observation"].copy()
            persistence = bool(controller and controller.get("baseline") == "storage-persistence")
            if persistence:
                for coordinate in (34, 35, 36):
                    expected_input[coordinate] = expected_input[2]
            if (details["actual_observation"] != evidence["observation"]
                    or details["actual_observation_sha256"] != evidence["observation_sha256"]
                    or details["controller_input"] != expected_input
                    or details["controller_input_sha256"] != digest(expected_input)
                    or details["action"] != command["action"]):
                raise ValueError("Baseline input/transformation witness differs")
        witness.update(canonical(evidence))
    if witness.hexdigest() != row["input_action_witness_sha256"]:
        raise ValueError("Complete inference witness digest differs")
    sampled = states[::15]
    return {"condition": row["condition"], "controller": row["controller"], "seed": row["seed"],
            "execution_complete": row["execution_complete"], "audited": audited(row), "failure": row.get("failure"),
            "states": [[state[field] for field in FIELDS] for state in sampled],
            "elapsed_h": [state["step"] / 60 for state in sampled],
            "modes": [[state["dac_mode"], state["methanator_mode"]] for state in sampled],
            "equipment_issues": [[{"component": name, **health} for name, health in state["health"].items()
                                   if health.get("offline") or health.get("degradation", 0) > 0 or health.get("status") != "nominal"] for state in sampled],
            "actions": [c["action"] for c in commands], "delivery": (row.get("delivery") or {}).get("settlements", []),
            "event": row["event_metrics"], "economic": row.get("economic_metrics"), "operating": row.get("operating_metrics"),
            "all_delivery_success": row["every_delivery_passed"], "recording": row["recording"],
            "full_minute_states": len(states), "verified_decisions": len(commands),
            "input_action_witness_sha256": row["input_action_witness_sha256"]}


def strict_headline(primary, all_cases_audited):
    """Report-level protocol conjunction; preserves every frozen analysis flag."""
    if primary is None:
        return None
    components = {"strong_headline_bar_met": bool(primary["strong_headline_bar_met"]),
                  "consistent_across_optimizer_seeds": bool(primary["consistent_across_optimizer_seeds"]),
                  "all_cases_audited": bool(all_cases_audited)}
    return {"met": all(components.values()), "components": components,
            "definition": "Strict joint headline requires the component performance bar, cross-optimizer directional consistency, and every case audited; no frozen analysis flag is overwritten"}


def collect(results):
    results = Path(results).resolve()
    spec_path, summary_path, analysis_path = (results / name for name in ("freeze.json", "summary.json", "analysis.json"))
    spec, summary, analysis = (json.loads(path.read_text()) for path in (spec_path, summary_path, analysis_path))
    verify_pin(summary["freeze"], spec_path)
    verify_pin(analysis["summary"], summary_path)
    index = validate_panel(spec, summary, analysis)
    if spec["split"] == "final":
        verify_pin(spec["gate"])
        verify_pin(spec["selected_recipe"])
        gate = json.loads(Path(spec["gate"]["path"]).read_text())
        chosen = json.loads(Path(spec["selected_recipe"]["path"]).read_text())
        if (gate.get("schema") != "learned-control-v1-evaluation-gate" or not gate.get("selected_recipe_frozen")
                or gate["selected_recipe"]["sha256"] != spec["selected_recipe"]["sha256"]
                or chosen["seeds"] != spec["seeds"] or chosen["conditions"] != spec["conditions"]
                or chosen["primary"] != spec["primary"]):
            raise ValueError("Final selection gate or selected recipe does not match this panel")
        identity_fields = ("label", "kind", "baseline", "config", "checkpoint", "group", "recipe", "optimizer_seed")
        identity = lambda c: {k: c[k] for k in identity_fields if k in c}
        learned = {label: c for label, c in spec["controllers"].items() if c["kind"] == "learned"}
        selected = {c["label"]: c for c in chosen["controllers"]}
        if learned.keys() != selected.keys() or any(identity(learned[label]) != identity(selected[label]) for label in learned):
            raise ValueError("Final learned controllers differ from their frozen selection")
        for label in learned:
            if selected[label].get("checkpoint_sha256") not in (None, learned[label]["checkpoint_files"]["checkpoint"]["sha256"]):
                raise ValueError("Final checkpoint digest differs from its frozen selection")
        champion = chosen["baseline_champion"]
        if champion["label"] not in spec["controllers"] or identity(spec["controllers"][champion["label"]]) != identity(champion):
            raise ValueError("Final reference differs from the selected baseline champion")
        if (not spec["primary"] or spec["primary"].get("recipe", spec["primary"].get("group")) != chosen["recipe_id"]
                or spec["primary"].get("reference") != champion["label"]):
            raise ValueError("Final primary comparison is not bound to the selected recipe and champion")
    metadata = {label: controller_metadata(controller) for label, controller in spec["controllers"].items()}
    records, outcomes, case_pins = {}, [], []
    for seed, condition, controller in itertools.product(spec["seeds"], spec["conditions"], spec["controllers"]):
        row = index[(seed, condition, controller)]
        path = results / "cases" / f"{condition}--{controller}--{seed}" / "result.json"
        if json.loads(path.read_text()) != row:
            raise ValueError("Case file differs from the all-case summary")
        case_pins.append(file_pin(path))
        outcomes.append({"seed": seed, "condition": condition, "controller": controller,
                         "audited": audited(row), "execution_complete": row["execution_complete"],
                         "every_delivery_passed": row["every_delivery_passed"],
                         **{name: outcome(row, name) for name in METRICS}})
        if seed != spec["seeds"][0]:
            continue
        # Prefer the local case recording and verify its bytes against the
        # original pin; original provenance stays visible in the output.
        recording_path = path.with_name("recording.json.gz")
        verify_pin(row["recording"], recording_path)
        with gzip.open(recording_path, "rt") as stream:
            records[f"{condition}|{controller}"] = trace_payload(json.load(stream), row, spec["controllers"][controller])
    return {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(), "split": spec["split"],
            "scenario_count": len(spec["seeds"]), "seeds": spec["seeds"], "seed": spec["seeds"][0],
            "case_count": len(index), "all_cases_audited": analysis["all_cases_audited"],
            "fields": FIELDS, "action_labels": ACTION_LABELS, "sample_minutes": 15,
            "conditions": spec["conditions"], "controllers": list(spec["controllers"]), "controller_metadata": metadata,
            "groups": analysis["groups"], "summaries": analysis["summaries"], "comparisons": analysis["paired_comparisons"],
            "primary_claim": analysis.get("primary_claim"), "distribution_caveat": analysis["distribution_caveat"],
            "strict_headline": strict_headline(analysis.get("primary_claim"), analysis["all_cases_audited"]),
            "outcomes": outcomes, "records": records,
            "provenance": {"freeze": file_pin(spec_path), "summary": file_pin(summary_path), "analysis": file_pin(analysis_path),
                           "case_results": case_pins, "selection_gate": spec.get("gate"), "selected_recipe": spec.get("selected_recipe")}}


def render(payload):
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return HTML.replace("__PAYLOAD__", encoded)


def build(results, output):
    payload = collect(results)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(render(payload))
    return {"html": file_pin(output), "split": payload["split"], "scenarios": payload["scenario_count"],
            "cases": payload["case_count"], "traces": len(payload["records"]),
            "all_cases_audited": payload["all_cases_audited"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New HTML path; existing artifacts are never overwritten")
    args = parser.parse_args()
    print(json.dumps(build(args.results, args.output)))


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src data:">
<title>PlantAnAgent · Learned-control evidence</title>
<style>
:root{--ink:#182c2b;--muted:#536a68;--paper:#f4f6f2;--line:#cad6cf;--green:#17654f;--amber:#9d5b15;--bad:#973344;--card:#fff}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,-apple-system,sans-serif}main{max-width:1300px;margin:auto;padding:34px 32px 70px}h1{font-size:clamp(30px,5vw,52px);font-weight:650;line-height:1.06;letter-spacing:-1.8px;margin:14px 0 20px}h2{font-size:23px;margin:0 0 10px;letter-spacing:-.5px}h3{font-size:16px;margin:0}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:12px;color:var(--green);font-weight:750}.intro{max-width:900px;font-size:17px;color:var(--muted)}.ribbon{display:flex;flex-wrap:wrap;gap:9px;margin:20px 0}.pill{border:1px solid var(--line);padding:5px 11px;border-radius:30px;font-size:12px}.section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px;margin-top:24px}.lede{color:var(--muted);margin:0 0 16px}.controls{display:flex;gap:15px;flex-wrap:wrap;align-items:end;margin:18px 0}.controls>div{max-width:100%}label{display:block;font-size:12px;text-transform:uppercase;letter-spacing:1px;font-weight:650;color:var(--muted)}select,button{font:inherit;padding:9px 12px;border:1px solid #a7b9aa;border-radius:7px;background:#fff;color:var(--ink);min-height:42px;max-width:100%}select{margin-top:5px;min-width:170px;max-width:min(100%,440px)}button{cursor:pointer}button:hover{background:#edf5ee}button:disabled{opacity:.45;cursor:default}.notice{background:#eaf2eb;border-left:4px solid var(--green);padding:14px 16px;margin:18px 0;font-size:14px}.notice.development{background:#fff0d9;border-color:#ad711f;color:#70480e}.notice.failure{background:#fae7e7;border-color:var(--bad);color:#6d2731}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.stat{border-top:2px solid var(--line);padding:10px 0}.stat span{font-size:12px;color:var(--muted)}.stat strong{display:block;font-size:24px;line-height:1.3;font-weight:600}.timeline{background:#f2f6f1;border-radius:8px;padding:16px;margin:20px 0}.timeline input{width:100%;accent-color:var(--green);cursor:pointer}.timehead{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}.timehead strong{font-size:18px}.plotgrid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.plot{border:1px solid #e0e6dc;border-radius:9px;padding:14px 14px 7px;min-width:0}.plot p{font-size:12px;color:var(--muted);margin:0 0 2px}.plot svg{width:100%;display:block;overflow:visible}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:6px}.legend i{display:inline-block;width:17px;height:3px;margin-right:5px;vertical-align:middle}.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #e5e9e1}th:first-child,td:first-child{text-align:left}th{font-weight:600;background:#f5f7f2;color:var(--muted)}tr.selected{background:#eaf3e9}.small{font-size:12px;color:var(--muted)}.mono{font-family:ui-monospace,monospace;font-size:12px;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600;padding:8px 0}details{margin:15px 0}pre{font-size:11px;background:#f5f7f2;white-space:pre-wrap;overflow-wrap:anywhere;padding:15px;max-height:330px;overflow:auto}.good{color:var(--green)}.bad{color:var(--bad)}.split{display:grid;grid-template-columns:1fr 1fr;gap:20px}.subtitle{font-size:12px;color:var(--muted);display:block}.conditions{font-size:13px;color:var(--muted)}footer{color:var(--muted);font-size:12px;margin-top:24px}a{color:var(--green)}[hidden]{display:none!important}@media(max-width:720px){main{padding:22px 15px 45px}.section{padding:17px}.plotgrid,.split{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}select{min-width:135px}.controls{gap:10px}h1{letter-spacing:-1px}.tablewrap:before{content:'Swipe horizontally to inspect every column';display:block;font-size:11px;color:var(--muted);padding:5px 0}th,td{padding:8px}.plot{padding:12px}}
</style></head><body><main>
<div class="eyebrow">PlantAnAgent · Reproducible control research</div><h1>Learned-control evidence</h1>
<p class="intro">Inspect the complete controller comparison, then follow a prescribed first-seed week through power allocation, stored gas, equipment readiness and delivery commitments.</p>
<div id="split-notice" class="notice"></div><div class="ribbon" id="ribbon"></div><div id="audit-notice" class="notice failure" hidden></div>
<section class="section"><h2>All retained outcomes</h2><p class="lede">Every controller and scenario in the frozen panel is represented. Group means average separately evaluated policies; they are not ensemble controllers.</p>
<div class="controls"><div><label for="summary-panel">Condition or objective</label><select id="summary-panel"></select></div></div><p id="panel-description" class="conditions"></p>
<div class="tablewrap"><table><thead><tr><th>Population</th><th>Type</th><th id="success-heading">Every-delivery success</th><th>Mean unmet kg</th><th>Mean missed deliveries</th><th>Mean modeled €/week</th><th>Correlated cells</th></tr></thead><tbody id="summary-rows"></tbody></table></div>
<p class="small">The independent uncertainty unit is the base environment seed. Multiple conditions and policies for that seed remain together. The declared mixture is 50% ordinary operation and 10% for each of five stress conditions; healthy control has zero mixture weight. These weights are an experimental objective, not estimated event frequencies.</p>
<details open><summary>Paired comparison with a baseline</summary><div class="controls"><div><label for="treatment">Treatment population</label><select id="treatment"></select></div><div><label for="reference">Reference baseline</label><select id="reference"></select></div></div><p id="comparison-description" class="small"></p><div class="tablewrap"><table><thead><tr><th>Metric</th><th>Reference mean</th><th>Treatment mean</th><th>Difference [interval]</th></tr></thead><tbody id="comparison-rows"></tbody></table></div><p class="small">Every difference is treatment minus reference. Higher contribution and success are favorable; lower unmet mass and missed-delivery counts are favorable. These displayed intervals come directly from the frozen analysis and describe uncertainty over environment seeds conditional on the policies evaluated.</p></details>
<details id="primary-section"><summary>Registered headline criteria</summary><div id="primary-intro" class="small"></div><div class="tablewrap"><table><thead><tr><th>Predeclared check</th><th>Status</th></tr></thead><tbody id="primary-rows"></tbody></table></div><p id="primary-limits" class="small"></p></details>
<details><summary>Controller identities and baseline configurations</summary><p class="small">Original baselines have no parameter overrides. Configured or tuned baselines disclose their overrides below; these parameters change the controller, not the plant. Learned-controller labels alone do not establish whether PPO updates have occurred.</p><div class="tablewrap"><table><thead><tr><th>Controller</th><th>Type</th><th>Algorithm or group</th><th>PPO transitions</th><th>PPO update count</th></tr></thead><tbody id="metadata-rows"></tbody></table></div><pre id="metadata-detail"></pre></details>
</section>
<section class="section"><h2>A prescribed first-seed week</h2><div class="notice" id="trace-notice"></div>
<div class="controls"><div><label for="condition">Condition</label><select id="condition"></select></div><div><label for="controller">Controller</label><select id="controller"></select></div><button id="jump-start">Event start</button><button id="jump-end">Event end</button><button id="download">Download selected trace</button></div>
<p id="event-description" class="lede"></p><p id="controller-description" class="small"></p><div id="trace-failure" class="notice failure" hidden></div>
<div id="trace-content"><div class="stats" id="week-stats"></div><div class="timeline"><div class="timehead"><strong id="time-label"></strong><span class="small">Elapsed hours from D1 06:00 · 15-minute display samples</span></div><input id="timeline" type="range" min="0" max="672" step="1" value="192" aria-label="Elapsed time in 15-minute intervals"></div><div class="stats" id="state-stats"></div><p id="equipment-status" class="small"></p><div class="plotgrid" id="plots"></div>
<p class="small">Shading marks the prescribed disruption; the vertical line marks the selected time. Charts connect samples of the physical state every 15 minutes. Full original minute states and all 149-coordinate policy inputs remain in the source recordings. At a boundary, physical power describes the last completed minute; the command below is issued for the following interval.</p>
<div class="split" id="event-tables"><div><h3>Event-boundary stocks</h3><div class="tablewrap"><table><thead><tr><th>Stock</th><th>Before event</th><th>After event</th></tr></thead><tbody id="stock-rows"></tbody></table></div></div><div><h3>Delivery losses by period</h3><div class="tablewrap"><table><thead><tr><th>Period</th><th>Unmet kg</th></tr></thead><tbody id="event-rows"></tbody></table></div><p class="small">A delivery at a boundary belongs to the period ending at that boundary. Healthy control uses hour 48 as an accounting reference, with no imposed outage.</p></div></div>
<details><summary>Delivery settlements</summary><div class="tablewrap"><table><thead><tr><th>Elapsed hour</th><th>Demand kg</th><th>Delivered kg</th><th>Unmet kg</th></tr></thead><tbody id="delivery-rows"></tbody></table></div></details>
<details><summary>Selected state and next raw command</summary><p class="small">All seven raw commands lie in [−1,1]. The simulator applies resource and equipment limits. There is no next command after the final decision boundary. The download contains display samples and all retained seven-coordinate commands, not the full 149-coordinate observation arrays.</p><pre id="detail"></pre></details></div>
</section><footer id="provenance"></footer></main><script id="payload" type="application/json">__PAYLOAD__</script><script>
'use strict';
const D=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id),F=Object.fromEntries(D.fields.map((f,i)=>[f,i]));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,d=1)=>Number.isFinite(v)?new Intl.NumberFormat('en-GB',{minimumFractionDigits:d,maximumFractionDigits:d}).format(v):'Unavailable';
const labels={ordinary:'Ordinary · random faults',control:'Healthy control · no imposed event',pv6:'Solar loss · 6 h',pv18:'Solar loss · 18 h',pv36:'Solar loss · 36 h',meth24:'Methanator outage · 24 h',meth48:'Methanator outage · 48 h',stress:'Five-condition stress average',mixture:'Declared 50 / 50 objective mixture'};
const metricLabels={net_operating_value_eur:'Contribution / € per week',unmet_kg:'Unmet methane / kg',all_delivery_success:'Every-delivery success / percentage points',missed_deliveries:'Missed deliveries / week'};
const panels=[...new Set(D.summaries.map(s=>s.panel))],populations=[...new Set(D.summaries.map(s=>s.population))],baselines=D.controllers.filter(c=>D.controller_metadata[c].kind==='baseline');
const populationLabel=p=>p.startsWith('recipe:')?'Group mean · '+p.slice(7):p;
function options(el,values,display){el.replaceChildren();for(const v of values){let o=document.createElement('option');o.value=v;o.textContent=display(v);el.append(o)}}
options($('summary-panel'),panels,p=>labels[p]||p);options($('treatment'),populations,populationLabel);options($('reference'),baselines,p=>p);options($('condition'),D.conditions,c=>labels[c]||c);options($('controller'),D.controllers,c=>c);
if(panels.includes('mixture'))$('summary-panel').value='mixture';
const development=D.split!=='final';$('split-notice').className='notice'+(development?' development':'');$('split-notice').innerHTML=development?`<strong>${esc(D.split.toUpperCase())} / MODEL-SELECTION EVIDENCE · ${D.scenario_count} base scenarios.</strong> This panel is an engineering or development check. It is not final held-out evidence and does not establish a final-policy advantage.`:`<strong>FINAL EVALUATION · ${D.scenario_count} fresh base scenarios.</strong> Controller identities and the comparator were fixed by the validated selection gate before these outcomes.`;
$('ribbon').innerHTML=[`${D.case_count} retained rollouts`,`${D.scenario_count} base scenarios`,`${D.controllers.length} controllers`,`${D.conditions.length} conditions`,'7-day episodes','Entirely offline'].map(s=>`<span class="pill">${esc(s)}</span>`).join('');
if(!D.all_cases_audited){$('audit-notice').hidden=false;$('audit-notice').textContent='One or more cases is incomplete or fails an audit. All retained cases remain visible; unavailable aggregate metrics are not replaced with zero and no full-panel claim is established.'}
function weightsDescription(weights){return Object.entries(weights).map(([c,w])=>`${fmt(100*w/Object.values(weights).reduce((a,b)=>a+b,0),0)}% ${labels[c]||c}`).join(' + ')}
function successText(g){const value=g.metrics.all_delivery_success.mean;if(!Number.isFinite(value))return 'Unavailable';const uniform=new Set(Object.values(g.weights)).size===1;return `${fmt(value*100)}%`+(uniform?` · ${fmt(value*g.n_policy_condition_cells,0)}/${g.n_policy_condition_cells} cells`:' · weighted rate')}
function summary(){const panel=$('summary-panel').value,groups=D.summaries.filter(g=>g.panel===panel);$('panel-description').textContent=weightsDescription(groups[0].weights)+`. Each summary averages within its ${D.scenario_count} base seeds; policy/condition cells are correlated.`;$('success-heading').textContent=panel==='mixture'?'Weighted every-delivery success':'Every-delivery success';$('summary-rows').innerHTML=groups.map(g=>{const group=g.population.startsWith('recipe:'),type=group?`${g.members.length} separately evaluated policies`:D.controller_metadata[g.population].display_kind;return `<tr><td>${esc(populationLabel(g.population))}${group?`<span class="subtitle">${esc(g.members.join(', '))}</span>`:''}</td><td>${esc(type)}</td><td>${successText(g)}</td><td>${fmt(g.metrics.unmet_kg.mean)}</td><td>${fmt(g.metrics.missed_deliveries.mean,2)}</td><td>${fmt(g.metrics.net_operating_value_eur.mean,0)}</td><td>${g.n_policy_condition_cells}</td></tr>`}).join('');comparison()}
function comparison(){const p=D.comparisons.find(x=>x.panel===$('summary-panel').value&&x.treatment===$('treatment').value&&x.reference===$('reference').value);if(!p){$('comparison-description').textContent=baselines.length?'No comparison is required when treatment and reference are the same controller.':'This panel contains no baseline comparator.';$('comparison-rows').innerHTML='';return}const first=Object.values(p.metrics)[0];$('comparison-description').textContent=`${populationLabel(p.treatment)} minus ${p.reference}. ${fmt(first.confidence*100,1)}% paired bootstrap intervals; ${first.replicates.toLocaleString()} resamples of ${first.n_base_scenarios} base seeds. All conditions and group policies remain together.`;$('comparison-rows').innerHTML=Object.entries(p.metrics).map(([name,m])=>{const scale=name==='all_delivery_success'?100:1,d=name==='net_operating_value_eur'?0:name==='missed_deliveries'?2:1;const f=v=>Number.isFinite(v)?fmt(v*scale,d):'Unavailable';return `<tr><td>${esc(metricLabels[name]||name)}</td><td>${f(m.reference_mean)}</td><td>${f(m.treatment_mean)}</td><td>${f(m.mean)} [${f(m.interval[0])}, ${f(m.interval[1])}]</td></tr>`}).join('')}
function primary(){const p=D.primary_claim;if(!p){$('primary-intro').textContent='No primary final-claim comparison is registered in this panel. Development summaries can guide a later selection; they are not a held-out claim.';$('primary-rows').innerHTML='';return}const strict=D.strict_headline;$('primary-intro').textContent=`${development?'Development screening only. ':''}Recipe ${p.recipe} versus ${p.reference}. Strict joint headline: ${strict.met?'met':'not met'}. This requires the component performance bar, cross-optimizer directional consistency, and every case audited. All three must hold.`;const rows=[['Strict joint headline — all components required',strict.met],['Component performance bar',p.strong_headline_bar_met],['Cross-optimizer directional consistency',p.consistent_across_optimizer_seeds],['Every retained case audited',D.all_cases_audited],...Object.entries(p.checks).map(([k,v])=>[k.replaceAll('_',' '),v])];$('primary-rows').innerHTML=rows.map(([k,v])=>`<tr><td>${esc(k)}</td><td class="${v?'good':'bad'}">${v?'Met':'Not met'}</td></tr>`).join('');$('primary-limits').textContent=strict.definition+'. '+p.multiplicity+'. '+p.limitation}
$('metadata-rows').innerHTML=D.controllers.map(label=>{const m=D.controller_metadata[label],t=m.checkpoint_training||{};return `<tr><td>${esc(label)}</td><td>${esc(m.display_kind)}</td><td>${esc(m.baseline||m.group||m.recipe||'—')}</td><td>${Number.isFinite(t.num_timesteps)?fmt(t.num_timesteps,0):'Not applicable'}</td><td>${Number.isFinite(t.ppo_update_count)?fmt(t.ppo_update_count,0):'Not recorded'}</td></tr>`}).join('');$('metadata-detail').textContent=JSON.stringify(D.controller_metadata,null,2);
$('trace-notice').innerHTML=`<strong>Prescribed first-seed example: ${fmt(D.seed,0)}.</strong> The same first manifest seed supplies every controller/condition trace. It was not chosen for a favorable outcome. ${development?`This development panel contains only ${D.scenario_count} base scenarios. `:''}Comparative conclusions use all retained outcomes above.`;
const chartSpecs=[['Solar available','MW',[['solar_available_kw','Solar','#b77919',.001]],null],['Battery charge','% of capacity',[['battery_soc','Charge','#17684a',100]],100],['Methane inventory','kg',[['ch4_level_kg','Methane','#356396',1]],null],['Process readiness','% of full readiness',[['dac_readiness_fraction','Capture','#7c568f',100],['methanator_readiness_fraction','Methanation','#bc572e',100]],100],['Hydrogen inventory','kg',[['h2_level_kg','Hydrogen','#268678',1]],null],['Carbon dioxide inventory','kg',[['co2_level_kg','Carbon dioxide','#7c568f',1]],null],['Process electrical power','MW',[['electrolyser_power_kw','Electrolysis','#268678',.001],['dac_power_kw','Capture','#7c568f',.001],['methanator_power_kw','Methanation','#bc572e',.001]],null],['Battery power to processes','kW',[['battery_to_electrolyser_kw','Electrolysis','#268678',1],['battery_to_dac_kw','Capture','#7c568f',1],['battery_to_methanator_kw','Methanation','#bc572e',1]],null]];
function selected(){return D.records[$('condition').value+'|'+$('controller').value]}
function chart(spec,r,tick){const [title,unit,series,fixed]=spec,W=530,H=190,L=44,R=13,T=12,B=30,PW=W-L-R,PH=H-T-B,max=fixed||Math.max(1,...series.map(s=>Math.max(...r.states.map(row=>row[F[s[0]]]*s[3]))))*1.06;const x=h=>L+h/168*PW,y=v=>T+PH-v/max*PH;let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title} across seven days">`;if(r.event.is_intervention)svg+=`<rect class="event-band" x="${x(r.event.onset_h)}" y="${T}" width="${x(r.event.end_h)-x(r.event.onset_h)}" height="${PH}" fill="#f8e5c9"/>`;for(let i=0;i<=4;i++){const val=max*i/4,Y=y(val);svg+=`<line x1="${L}" y1="${Y}" x2="${W-R}" y2="${Y}" stroke="#dce4db"/><text x="${L-6}" y="${Y+3}" text-anchor="end" font-size="10" fill="#607568">${fmt(val,max>10?0:1)}</text>`}for(let hour=0;hour<=168;hour+=24)svg+=`<text x="${x(hour)}" y="${H-8}" text-anchor="middle" font-size="10" fill="#607568">${hour}h</text>`;for(const [field,label,color,scale] of series){const path=r.states.map((row,i)=>(i?'L':'M')+x(r.elapsed_h[i]).toFixed(2)+','+y(row[F[field]]*scale).toFixed(2)).join(' ');svg+=`<path d="${path}" stroke="${color}" stroke-width="1.6" fill="none"/>`}svg+=`<line x1="${x(r.elapsed_h[tick])}" y1="${T}" x2="${x(r.elapsed_h[tick])}" y2="${T+PH}" stroke="#253d2e" stroke-width="1.2" stroke-dasharray="4 3"/></svg>`;return `<div class="plot"><h3>${title}</h3><p>${unit}</p><div class="legend">${series.map(s=>`<span><i style="background:${s[2]}"></i>${s[1]}</span>`).join('')}</div>${svg}</div>`}
function stat(label,value){return `<div class="stat"><span>${esc(label)}</span><strong>${value}</strong></div>`}
function timeLabel(hours){const minutes=Math.round((6+hours)*60),day=Math.floor(minutes/1440)+1,h=Math.floor(minutes%1440/60),m=minutes%60;return `${fmt(hours,2)} h · D${day} ${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}`}
function renderTrace(){const r=selected(),e=r.event,m=D.controller_metadata[r.controller],last=r.states.length-1,tick=Math.max(0,Math.min(+$('timeline').value,last));$('timeline').max=Math.max(0,last);$('timeline').value=tick;$('jump-start').disabled=e.onset_h===null||last<0;$('jump-end').disabled=e.end_h===null||last<0;
$('event-description').textContent=r.condition==='ordinary'?'Ordinary sampled weather and the original random component faults. There is no imposed single stress interval; inspect current equipment issues below.':e.is_intervention?`${labels[r.condition]} from elapsed hour ${e.onset_h} to ${e.end_h}. ${r.condition.startsWith('meth')?'Repair occurs at D4 20:00.':'The original sampled solar resource returns after this interval.'} Unrelated random component faults are disabled.`:'Healthy control: sampled weather with random component faults disabled. Hour 48 is only an accounting reference; no event is imposed.';
const training=m.checkpoint_training;$('controller-description').textContent=`${m.label} · ${m.display_kind}. `+(m.kind==='baseline'?`${m.baseline}; ${m.configuration_status}.`:training?`Checkpoint stage: ${training.kind||'not recorded'}; ${fmt(training.num_timesteps,0)} PPO transitions; ${Number.isFinite(training.ppo_update_count)?fmt(training.ppo_update_count,0):'unrecorded'} PPO update count.`:'Checkpoint provenance is listed in controller metadata.');
$('trace-failure').hidden=r.execution_complete&&r.audited;$('trace-failure').textContent='This case is incomplete or failed an audit. The retained prefix is displayed without treating it as a valid full-week outcome. '+(r.failure?JSON.stringify(r.failure):'');$('trace-content').hidden=last<0;if(last<0){window.__learnedInspectorQA.selection={condition:r.condition,controller:r.controller,tick:null};return}
const s=r.states[tick],passed=r.delivery.filter(d=>d.unmet_kg<=1e-9).length;$('week-stats').innerHTML=stat('Deliveries met',passed+' / '+r.delivery.length)+stat('Recorded unmet methane',fmt(r.economic?.unmet_kg)+' kg')+stat(r.execution_complete?'Modeled week contribution':'Partial contribution','€'+fmt(r.economic?.net_operating_value_eur,0))+stat('Battery electricity used',fmt(r.operating?.battery_discharge_kwh,0)+' kWh');$('time-label').textContent=timeLabel(r.elapsed_h[tick]);$('state-stats').innerHTML=stat('Battery charge',fmt(s[F.battery_soc]*100)+'%')+stat('Methane in store',fmt(s[F.ch4_level_kg])+' kg')+stat('Capture readiness',fmt(s[F.dac_readiness_fraction]*100)+'%')+stat('Methanator readiness',fmt(s[F.methanator_readiness_fraction]*100)+'%');
const issues=r.equipment_issues[tick];$('equipment-status').textContent=issues.length?'Recorded equipment issues: '+issues.map(x=>`${x.component}: ${x.status}${x.offline?' (offline)':''}${x.degradation?' (degradation '+fmt(x.degradation*100)+'%)':''}`).join('; '):'Recorded equipment health: no current issue.';$('plots').innerHTML=chartSpecs.map(spec=>chart(spec,r,tick)).join('');
const hasEventStocks=!!(e.pre_event_state&&e.post_event_state);$('event-tables').hidden=!hasEventStocks;if(hasEventStocks){$('stock-rows').innerHTML=[['Battery / kWh','battery_energy_kwh'],['Hydrogen / kg','h2_level_kg'],['Carbon dioxide / kg','co2_level_kg'],['Methane / kg','ch4_level_kg']].map(([label,key])=>`<tr><td>${label}</td><td>${fmt(e.pre_event_state[key])}</td><td>${fmt(e.post_event_state[key])}</td></tr>`).join('');$('event-rows').innerHTML=[['Before / at onset','pre_event_unmet_kg'],['During event','event_unmet_kg'],['After event','post_event_unmet_kg']].map(([label,key])=>`<tr><td>${label}</td><td>${fmt(e[key])}</td></tr>`).join('')}
$('delivery-rows').innerHTML=r.delivery.map(d=>`<tr class="${Math.abs(d.due_step/60-r.elapsed_h[tick])<1e-9?'selected':''}"><td>${d.due_step/60}</td><td>${fmt(d.requested_kg)}</td><td>${fmt(d.delivered_kg)}</td><td class="${d.unmet_kg>1e-9?'bad':''}">${fmt(d.unmet_kg)}</td></tr>`).join('');
const detail={seed:r.seed,condition:r.condition,controller:r.controller,elapsed_h:r.elapsed_h[tick],physical_state:Object.fromEntries(D.fields.map((f,i)=>[f,s[i]])),dac_mode:r.modes[tick][0],methanator_mode:r.modes[tick][1],equipment_issues:issues,next_decision_index:tick<r.actions.length?tick:null,next_raw_action:tick<r.actions.length?Object.fromEntries(D.action_labels.map((label,i)=>[label,r.actions[tick][i]])):null};$('detail').textContent=JSON.stringify(detail,null,2);window.__learnedInspectorQA.selection={condition:r.condition,controller:r.controller,tick}}
$('summary-panel').addEventListener('change',summary);$('treatment').addEventListener('change',comparison);$('reference').addEventListener('change',comparison);$('condition').addEventListener('change',renderTrace);$('controller').addEventListener('change',renderTrace);$('timeline').addEventListener('input',renderTrace);$('jump-start').addEventListener('click',()=>{$('timeline').value=selected().event.onset_h*4;renderTrace()});$('jump-end').addEventListener('click',()=>{$('timeline').value=selected().event.end_h*4;renderTrace()});
$('download').addEventListener('click',()=>{const r=selected(),payload={schema:D.schema,split:D.split,scenario_count:D.scenario_count,fields:D.fields,action_labels:D.action_labels,sample_minutes:D.sample_minutes,controller_metadata:D.controller_metadata[r.controller],trace:r},blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`${D.split}-${r.condition}-${r.controller}-${r.seed}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)});
$('provenance').innerHTML=`Analysis SHA-256 <span class="mono">${D.provenance.analysis.sha256}</span><br>${Object.keys(D.records).length} prescribed first-seed traces · Display samples every ${D.sample_minutes} minutes · Full source records are preserved · No remote assets or network requests.`;
window.__learnedInspectorQA={ready:false,split:D.split,cases:D.case_count,scenarios:D.scenario_count,traces:Object.keys(D.records).length,seed:D.seed,selection:null};primary();summary();renderTrace();window.__learnedInspectorQA.ready=true;
</script></body></html>'''


if __name__ == "__main__":
    main()
