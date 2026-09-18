"""Portable entry points for the published learned-control experiment.

All new artifacts go into a new output directory. Historical checkpoints and
source files are read-only. Re-executions are not new held-out evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def configuration():
    config = read(ROOT / "experiments/reproduction.json")
    reference = ROOT / config["reference"]["path"]
    if hashlib.sha256(reference.read_bytes()).hexdigest() != config["reference"]["sha256"]:
        raise ValueError("The reference evaluation was changed")
    return config


def modules():
    # Import the frozen physics before any installed application modules.
    from docs.research.tools.learned_control_v1 import environment, evaluate
    from docs.research.tools.wholeplant_ppo_v1 import training
    training.configure_torch()
    return environment, evaluate, training


def controllers(training_root=None):
    values = read(ROOT / "experiments/controllers.json")
    for item in values:
        if item["kind"] == "baseline":
            continue
        if training_root is None:
            path = ROOT / item["checkpoint"]
        elif item["label"] == "imitation-parent-903":
            path = training_root / "initialization/hybrid-solar-ridge/training/initial-policy.zip"
        else:
            path = training_root / f"runs/cem-ridge-bias-cohort/seed-{item['optimizer_seed']}/training/checkpoints/generation-015.zip"
        item["checkpoint"] = str(path.resolve())
    return values


def freeze(output, *, smoke, training_root=None, deadline=None):
    """Bind this reproduction to its source, policies and predetermined panel."""
    environment, evaluate, common = modules()
    config = configuration()
    specs = [evaluate._controller_spec(c) for c in controllers(training_root)]
    seeds = [environment.TRAIN_BASE] if smoke else config["seeds"]
    conditions = ["ordinary"] if smoke else config["conditions"]
    split = "train" if smoke else "final"
    gate_pin = selection_pin = None
    if not smoke:
        selected = {
            "recipe_id": config["primary"]["recipe"], "primary": config["primary"],
            "seeds": seeds, "conditions": conditions,
            "controllers": [c for c in specs if c["kind"] == "learned"],
            "baseline_champion": next(c for c in specs if c["label"] == config["primary"]["reference"]),
            "interpretation": config["reference"]["interpretation"],
        }
        selection = output / "selected-recipe.json"
        write(selection, selected)
        gate = output / "gate.json"
        write(gate, {
            "schema": "learned-control-v1-evaluation-gate",
            "environment_identity_sha256": environment.environment_identity_sha256(),
            "source_identity_sha256": environment.source_identity_sha256(),
            "allowed_splits": ["final"], "selected_recipe_frozen": True,
            "selected_recipe": evaluate.file_pin(selection),
            "interpretation": config["reference"]["interpretation"],
        })
        gate_pin = evaluate.file_pin(gate)
        selection_pin = evaluate._validate_selection(gate, specs, seeds, conditions, config["primary"])
    paths = {Path(p["path"]) for p in environment.source_records()}
    # Include the actual reproduction code, model implementations and analysis.
    for folder in (ROOT / "docs/research/tools", ROOT / "tools"):
        paths.update(folder.rglob("*.py"))
    paths.update(ROOT / "experiments" / n for n in ("controllers.json", "reproduction.json"))
    paths.add(ROOT / "uv.lock")
    spec = {
        "schema": evaluate.SCHEMA, "created_at": common.now(), "split": split,
        "seeds": seeds, "conditions": conditions,
        "controllers": {c["label"]: c for c in specs},
        "cases": [{"seed": s, "condition": c, "controller": p["label"]}
                  for s in seeds for c in conditions for p in specs],
        "expected_cases": len(seeds) * len(conditions) * len(specs),
        "attempts_per_case": 1, "gate": gate_pin, "selected_recipe": selection_pin,
        "deadline_utc": deadline, "max_decisions": 4 if smoke else None,
        "episode_hours": 168, "sources": [evaluate.file_pin(p) for p in sorted(paths)],
        "original_evaluation": evaluate.file_pin(ROOT / config["reference"]["path"]),
        "primary": config["primary"],
        "trace_rule": "Original evaluator: physical minute audits and deterministic action witnesses",
        "interpretation": ("Truncated engineering smoke on a training seed; not a research result"
                           if smoke else config["reference"]["interpretation"]),
    }
    spec["sha256"] = evaluate.digest(spec)
    write(output / "freeze.json", spec)
    evaluate.require_freeze(output / "freeze.json")
    return spec


def run_evaluation(args):
    _, evaluate, _ = modules()
    spec = freeze(args.output, smoke=args.command == "smoke",
                  training_root=args.training_root, deadline=args.deadline)
    if args.prepare_only:
        print(json.dumps({"prepared_cases": spec["expected_cases"], "output": str(args.output)}))
        return
    evaluate.campaign(args.output, args.workers)
    rows = [read(p) for p in (args.output / "cases").glob("*/result.json")]
    if len(rows) != spec["expected_cases"] or any(r.get("failure") for r in rows):
        raise RuntimeError("Evaluation incomplete or failed; inspect retained case results")
    if args.command == "evaluate":
        from docs.research.tools.learned_control_v1 import analyze
        result = analyze.analyze_rows(rows, spec)
        result.update(created_at=datetime.now(timezone.utc).isoformat(),
                      summary=evaluate.file_pin(args.output / "summary.json"))
        write(args.output / "analysis.json", result)
    print(json.dumps({"cases": len(rows), "output": str(args.output),
                      "interpretation": spec["interpretation"]}))


def train(args):
    modules()
    from docs.research.tools.learned_control_v1 import imitation, training, aggregate_datasets
    from docs.research.tools.learned_control_hybrid_v1 import training as hybrid
    from docs.research.tools.learned_control_refit_v1 import refit
    from docs.research.tools.learned_control_search_v1 import training as search
    config = configuration()
    out = args.output

    def recipe(name):
        value = deepcopy(config[name])
        value["deadline_utc"] = args.deadline
        for key in ("dataset", "training_dataset", "parent_checkpoint", "initial_checkpoint"):
            if value.get(key):
                value[key] = str(out / value[key])
        if name == "search":
            value["workers"] = args.workers
        return value

    def completed(result):
        if result.get("status") != "completed":
            raise RuntimeError("Training did not complete; retained artifacts identify the partial attempt")

    first = out / "demonstrations/original-storage"
    imitation.collect_dataset(first, **config["original-storage"], deadline_utc=args.deadline)
    gaussian = out / "runs/original-storage-bc-pilot/seed-901"
    training.freeze_plan(gaussian, recipe("gaussian"))
    completed(training.train(gaussian))
    dagger = out / "demonstrations/dagger-original-storage"
    imitation.collect_dataset(dagger, **config["dagger-original-storage"],
                              learner_checkpoint=gaussian / "training/initial-policy.zip",
                              deadline_utc=args.deadline)
    merged = out / "demonstrations/original-plus-dagger"
    aggregate_datasets.aggregate(merged, [first / "demonstrations.npz", dagger / "demonstrations.npz"])
    hybrid_root = out / "runs/hybrid-dagger-fidelity/seed-903"
    hybrid.freeze_plan(hybrid_root, recipe("hybrid"))
    completed(hybrid.train(hybrid_root))
    refit_root = out / "initialization/hybrid-solar-ridge"
    refit.freeze_plan(refit_root, hybrid_root / "training/initial-policy.zip",
                      merged / "demonstrations.npz", optimizer_seed=903, deadline_utc=args.deadline)
    refit.train(refit_root)
    for seed in config["optimizer_seeds"]:
        value = recipe("search")
        value["search_seed"] = value["optimizer_seed"] = seed
        directory = out / f"runs/cem-ridge-bias-cohort/seed-{seed}"
        search.freeze_plan(directory, value)
        completed(search.train(directory))
    write(out / "reproduction-complete.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
          "optimizer_seeds": config["optimizer_seeds"], "selected_generation": config["selected_generation"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "evaluate", "train", "plan"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-root", type=Path, help="Evaluate regenerated policies instead of bundled checkpoints")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--deadline", help="ISO UTC deadline; default is seven days from launch")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze an evaluation without running simulations")
    args = parser.parse_args()
    if args.command == "plan":
        print(json.dumps(configuration(), indent=2))
        return
    if args.output is None:
        parser.error("--output must name a new directory")
    if args.workers < 1:
        parser.error("--workers must be positive")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("--output already exists; preserve it and choose a new directory")
    if args.training_root:
        args.training_root = args.training_root.resolve()
    args.deadline = args.deadline or (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc):
        parser.error("--deadline must be a future timestamp with a timezone")
    if args.command == "train" and args.prepare_only:
        parser.error("Use plan to inspect training without executing it")
    train(args) if args.command == "train" else run_evaluation(args)


if __name__ == "__main__":
    main()
