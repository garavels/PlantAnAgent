"""Export comparison evidence and content-bound, relocatable replay inputs.

This helper never rewrites original plans, gates, checkpoints or result records.
Canonical export requires every primary test case, including failed attempts.
An explicit engineering mode supports train-only integration fixtures.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "docs/research/results/comparison_ppo_20260917"
OLD_RESULTS = ROOT / "docs/research/results/free_ppo_20260917"
SCHEMA = "comparison-ppo-v2-portable-bundle"
EXCLUDED_PARTS = {"__pycache__", ".build", ".git", ".pytest_cache", "source-snapshot", "source_snapshot", "portable-relocation-fixture"}
EXCLUDED_NAMES = {"bundle-build.json", "portable-verification.json", "completion.json", ".DS_Store"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Bundle input escaped the project: {path}")
    return path.relative_to(ROOT).as_posix()


def verify_pin(pin):
    path = Path(pin["path"])
    relative(path)
    if path.is_symlink() or path.stat().st_size != pin["bytes"] or sha(path) != pin["sha256"]:
        raise ValueError(f"Pinned scientific content changed: {path}")
    return path.resolve()


def _include_tree(paths, directory, output):
    for path in directory.rglob("*"):
        if (path.is_file() and not path.is_symlink()
                and not any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts)
                and path.name not in EXCLUDED_NAMES and path.resolve() != output.resolve()
                and not (path.suffix == ".zip" and ("research-package" in path.name or "portable-fixture" in path.name))):
            paths.add(path.resolve())


def _validate_primary(rows):
    primary = ("daylight", "storage", "ppo25-101", "ppo25-211", "ppo25-307")
    conditions = ("d25-sampled", "d25-floor", "d30-sampled", "d30-floor")
    expected_seeds = list(range(20_200_000, 20_200_032))
    for controller in primary:
        for condition in conditions:
            actual = sorted(r["seed"] for r in rows if (r["split"], r["controller"], r["condition"]) == ("held_out", controller, condition))
            if actual != expected_seeds:
                raise ValueError(f"Complete primary held-out panel required: {controller}/{condition}")
    fresh = [f"ppo30-{seed}" for seed in (101, 211, 307)]
    if any(r["split"] == "held_out" and r["controller"] in fresh for r in rows):
        for controller in fresh:
            for condition in ("d30-sampled", "d30-floor"):
                actual = sorted(r["seed"] for r in rows if (r["split"], r["controller"], r["condition"]) == ("held_out", controller, condition))
                if actual != expected_seeds:
                    raise ValueError(f"Every final PPO30 panel required: {controller}/{condition}")
        return list(primary) + fresh
    return list(primary)


def _require_evaluation_freeze(path):
    schema = json.loads(Path(path).read_text()).get("schema")
    if schema == "forecast-ablation-v1-freeze":
        from docs.research.tools.forecast_ablation_v1.freeze import require_freeze
    else:
        from docs.research.tools.comparison_ppo_v2.evaluate import require_freeze
    return require_freeze(path)


def _supplement(analysis_path, gate_path, *, engineering_fixture, deadline_receipt=None):
    """Keep the separately declared supplement separate from the main 1,160 rows."""
    folder = RESULTS / "forecast-ablation"
    freeze_path = folder / "freeze/freeze.json"
    if engineering_fixture:
        return {"status": "engineering_fixture_only"}, [], set(), set(), None
    if not freeze_path.exists():
        if deadline_receipt is not None:
            raise ValueError("A deadline receipt requires an existing supplementary freeze")
        return {"status": "not_declared"}, [], set(), set(), None
    from docs.research.tools.forecast_ablation_v1 import analyze as supplement_a
    freeze = _require_evaluation_freeze(freeze_path)
    paths = {freeze_path}
    metadata = {"freeze": relative(freeze_path), "evaluation_freeze_sha256": freeze["sha256"],
                "expected_rows": 32, "shared_main_scenarios": True, "main_analysis_unchanged": True}
    if deadline_receipt is not None:
        receipt_path = Path(deadline_receipt).resolve()
        relative(receipt_path)
        if receipt_path != (folder / "supervision/failure.json").resolve():
            raise ValueError("Deadline receipt must be the supplementary supervisor's recorded failure")
        receipt = json.loads(receipt_path.read_text())
        reason = receipt.get("reason", "")
        if (receipt.get("schema") != "forecast-ablation-finish-failure"
                or receipt.get("status") != "stopped"
                or "deadline" not in json.dumps(reason).lower()):
            raise ValueError("Only an explicit recorded deadline failure can label the supplement incomplete")
        if (folder / "supervision/completion.json").exists():
            raise ValueError("A completed supplement cannot be relabeled as deadline-incomplete")
        launch_path = folder / "supervision/launch.json"
        launch = json.loads(launch_path.read_text())
        if (launch.get("schema") != "forecast-ablation-finish-launch"
                or verify_pin(launch["freeze"]) != freeze_path.resolve()):
            raise ValueError("Deadline receipt is not bound to the prospective supplementary freeze")
        paths.update((receipt_path, launch_path))
        result_paths = sorted((folder / "held-out").rglob("case-*.result.json"))
        partial_rows = [json.loads(path.read_text()) for path in result_paths]
        paths.update(result_paths)
        if len({row["seed"] for row in partial_rows}) != len(partial_rows):
            raise ValueError("Supplementary deadline evidence contains repeated case attempts")
        for row in partial_rows:
            if (row["schema"] != "forecast-ablation-v1-case"
                    or row["evaluation_freeze_sha256"] != freeze["sha256"]
                    or (row["controller"], row["condition"], row["split"]) != ("daylight-persistence", "d30-sampled", "held_out")):
                raise ValueError("Deadline evidence is outside the frozen supplementary panel")
            if row.get("main_campaign_admission"):
                admission = row["main_campaign_admission"]
                paths.update(verify_pin(pin) for pin in admission.values())
                if (verify_pin(admission["main_analysis"]) != analysis_path
                        or verify_pin(admission["gate"]) != gate_path):
                    raise ValueError("Partial supplementary case references another main campaign")
        metadata.update(status="incomplete_due_to_deadline", deadline_receipt=relative(receipt_path),
                        reason=reason, retained_case_results=len(partial_rows),
                        completed_case_results=sum(row["execution_complete"] for row in partial_rows))
        return metadata, partial_rows, {freeze_path}, paths, None
    completion_path = folder / "supervision/completion.json"
    supplement_analysis_path = folder / "analysis/analysis.json"
    if not completion_path.is_file() or not supplement_analysis_path.is_file():
        raise ValueError("Frozen forecast supplement lacks completion; finish it or supply an explicit deadline receipt")
    completion = json.loads(completion_path.read_text())
    if (completion.get("schema") != "forecast-ablation-finish-completion"
            or completion.get("status") != "completed" or completion.get("rows") != 32
            or verify_pin(completion["analysis"]) != supplement_analysis_path.resolve()):
        raise ValueError("Supplementary completion does not bind all 32 outcomes")
    if (not {"held-out", "analysis"}.issubset({stage["stage"] for stage in completion["stages"]})
            or any(stage["exit_code"] != 0 for stage in completion["stages"])):
        raise ValueError("Supplementary supervisor did not complete every stage")
    supplemental = json.loads(supplement_analysis_path.read_text())
    admission = supplemental["main_campaign_admission"]
    if verify_pin(supplemental["reference_main_analysis"]) != analysis_path:
        raise ValueError("Supplement is paired with another main analysis")
    if verify_pin(admission["gate"]) != gate_path:
        raise ValueError("Supplement is bound to another held-out gate")
    if completion["main_completion"] != admission["main_completion"]:
        raise ValueError("Supplement completion and recorded cases reference different main closures")
    paths.update(verify_pin(pin) for pin in admission.values())
    recomputed = supplement_a.collect(folder / "held-out", main_analysis_path=analysis_path,
        gate_path=gate_path, main_completion_path=admission["main_completion"]["path"])
    if supplemental != recomputed or len(supplemental["rows"]) != 32:
        raise ValueError("Supplementary analysis differs from its fixed paired cases")
    if any(not row["execution_complete"] for row in supplemental["rows"]):
        raise ValueError("Supplement has incomplete executions; an explicit deadline receipt is required")
    paths.update((completion_path, supplement_analysis_path))
    paths.update(verify_pin(pin) for pin in supplemental["summaries"])
    freezes = {verify_pin(item["file"]) for item in supplemental["freezes"]}
    if freezes != {freeze_path.resolve()}:
        raise ValueError("Supplement uses another prospective freeze")
    metadata.update(status="completed", analysis=relative(supplement_analysis_path),
                    analysis_sha256=sha(supplement_analysis_path), completion=relative(completion_path), rows=32)
    return metadata, supplemental["rows"], freezes, paths, supplemental


def export(*, analysis_path, gate_path, output, engineering_fixture=False, freeze_paths=(),
           supplement_deadline_receipt=None):
    from docs.research.tools.comparison_ppo_v2 import evaluate
    from docs.research.tools.free_ppo_v1 import train

    analysis_path, output = Path(analysis_path).resolve(), Path(output).resolve()
    gate_path = Path(gate_path).resolve() if gate_path else None
    analysis = json.loads(analysis_path.read_text())
    rows = analysis["rows"]
    if not rows:
        raise ValueError("No recorded outcomes to export")
    if engineering_fixture:
        if any(r["split"] != "train" for r in rows):
            raise ValueError("Engineering fixture export admits training records only")
        required_controllers = sorted({r["controller"] for r in rows})
    else:
        if analysis.get("schema") != evaluate.SCHEMA + "-analysis" or gate_path is None:
            raise ValueError("Canonical export requires the comparison analysis and held-out gate")
        required_controllers = _validate_primary(rows)
        evaluate.environment()._admit("held_out", gate_path)
    supplement, supplement_rows, supplement_freezes, supplement_paths, supplement_analysis = _supplement(
        analysis_path, gate_path, engineering_fixture=engineering_fixture, deadline_receipt=supplement_deadline_receipt)
    paths = {analysis_path, *supplement_paths}
    exported_rows = [*rows, *supplement_rows]
    if supplement_rows:
        required_controllers.append("daylight-persistence")
    if gate_path:
        paths.add(gate_path)
    freeze_candidates = {Path(path).resolve() for path in freeze_paths} | supplement_freezes
    for item in analysis.get("freezes", []):
        if item.get("file"):
            freeze_candidates.add(verify_pin(item["file"]))
    for pin in analysis.get("summaries", []):
        path = verify_pin(pin)
        paths.add(path)
        summary = json.loads(path.read_text())
        freeze_candidates.add(verify_pin(summary["evaluation_freeze"]))
    freezes, verified_sources = {}, []
    for path in sorted(freeze_candidates):
        freeze = _require_evaluation_freeze(path)
        freezes[freeze["sha256"]] = (path, freeze)
        paths.add(path)
        if freeze.get("gate"):
            paths.add(verify_pin(freeze["gate"]))
        for pin in freeze["sources"]:
            paths.add(verify_pin(pin))
        for spec in freeze["controllers"].values():
            if spec["kind"] == "ppo":
                paths.update(verify_pin(pin) for pin in spec["checkpoint"].values())
        verified_sources.append({"record": relative(path), "kind": "evaluation_freeze", "files": len(freeze["sources"])})
    if not freezes:
        raise ValueError("No complete scientific evaluation freeze available")
    original_plan_path = OLD_RESULTS / "training/plan.json"
    original_plan = json.loads(original_plan_path.read_text())
    plans = [(original_plan_path, original_plan)]
    fresh_plan_path = RESULTS / "training/plan.json"
    if fresh_plan_path.exists() and not engineering_fixture:
        plans.append((fresh_plan_path, json.loads(fresh_plan_path.read_text())))
    for plan_path, plan in plans:
        paths.add(plan_path)
        for pin in plan["source_files"]:
            paths.add(verify_pin(pin))
        verified_sources.append({"record": relative(plan_path), "kind": "training_plan", "files": len(plan["source_files"])})
    # The original three PPO25 endpoints and exact normalizers are always
    # included as fixed transfer policies, alongside their original plan.
    train.configure_torch()
    for seed in (101, 211, 307):
        checkpoint = OLD_RESULTS / f"training/seed-{seed}/final-policy.zip"
        for path in (checkpoint, checkpoint.with_suffix(".json"), checkpoint.with_suffix(".vecnormalize.pkl")):
            paths.add(path)
        model = train.load_checkpoint(checkpoint)
        try:
            if (model.seed != seed or model.num_timesteps != original_plan["total_timesteps"]
                    or model.research_plan_sha256 != hashlib.sha256(canonical(original_plan)).hexdigest()
                    or model.research_identity != original_plan["environment_identity"]):
                raise ValueError("Original PPO25 final endpoint differs from its training plan")
        finally:
            model.close()
    if any(label.startswith("ppo30-") for label in required_controllers) and not engineering_fixture:
        if not fresh_plan_path.exists():
            raise ValueError("New final policies lack their training plan")
        fresh_plan = plans[-1][1]
        closure = json.loads((fresh_plan_path.parent / "campaign-closure.json").read_text())
        if (not closure["all_fixed_endpoints_completed"]
                or closure["plan_sha256"] != hashlib.sha256(canonical(fresh_plan)).hexdigest()
                or sorted(r["optimizer_seed"] for r in closure["runs"]) != [101, 211, 307]
                or any(r["exit_code"] != 0 or r["status"] != "completed" or r["num_timesteps"] != fresh_plan["total_timesteps"] for r in closure["runs"])):
            raise ValueError("PPO30 final panels require every fixed endpoint completed")
        for seed in (101, 211, 307):
            checkpoint = fresh_plan_path.parent / f"seed-{seed}/final-policy.zip"
            result = json.loads(checkpoint.with_name("result.json").read_text())
            if result["status"] != "completed" or result["num_timesteps"] != fresh_plan["total_timesteps"]:
                raise ValueError("Incomplete PPO30 run was labeled final")
            model = train.load_checkpoint(checkpoint)
            try:
                if model.seed != seed or model.num_timesteps != fresh_plan["total_timesteps"] or model.research_plan_sha256 != hashlib.sha256(canonical(fresh_plan)).hexdigest() or model.research_identity != fresh_plan["environment_identity"]:
                    raise ValueError("PPO30 checkpoint differs from frozen training plan")
            finally:
                model.close()
    for directory in (ROOT / "runs/authority-source-20260917", Path(__file__).parent, Path(__file__).parent.parent / "free_ppo_v1"):
        _include_tree(paths, directory, output)
    if supplement["status"] != "not_declared" or any(r["schema"] == "forecast-ablation-v1-case" for r in rows):
        _include_tree(paths, Path(__file__).parent.parent / "forecast_ablation_v1", output)
        paths.add(ROOT / "docs/research/FORECAST_ABLATION_V1_PROTOCOL.md")
    if not engineering_fixture:
        _include_tree(paths, RESULTS, output)
    paths.update(ROOT / name for name in ("uv.lock", "docs/research/FREE_PPO_V1_PROTOCOL.md",
        "docs/research/FREE_CONTROL_GUIDE.md", "docs/research/COMPARISON_PPO_V2_PROTOCOL.md"))
    replays = []
    verified_recordings = 0
    from docs.research.tools.comparison_ppo_v2.analyze import validate_row
    for row in exported_rows:
        validate_row(row)
        if row["evaluation_freeze_sha256"] not in freezes:
            raise ValueError("Outcome refers to an unverified evaluation freeze")
        freeze_path, freeze = freezes[row["evaluation_freeze_sha256"]]
        supplementary = freeze.get("schema") == "forecast-ablation-v1-freeze"
        if supplementary:
            from docs.research.tools.forecast_ablation_v1.analyze import validate_record
            validate_record(row, persistence=True)
        spec = freeze["controllers"][row["controller"]]
        selected = [p for p in freeze["panels"] if (p["controller"], p["condition"], p["split"]) == (row["controller"], row["condition"], row["split"])]
        if len(selected) != 1 or row["seed"] not in selected[0]["seeds"]:
            raise ValueError("Outcome is outside its frozen panel")
        record = None
        if row.get("recording"):
            record_path = verify_pin(row["recording"])
            paths.add(record_path)
            record = json.loads(gzip.decompress(record_path.read_bytes()))
            if record["summary"] != {k: v for k, v in row.items() if k != "recording"}:
                raise ValueError("Recorded case differs from compact outcome")
            verified_recordings += 1
        if row["seed"] != selected[0]["seeds"][0] or not row["execution_complete"] or not row.get("retained_minute_truth_sha256"):
            continue
        if record is None:
            raise ValueError("A complete retained trajectory has no recording")
        record_path = Path(row["recording"]["path"])
        if len(record["minute_truth_states"] or []) != 10081 or len(record["decision_trace"]) != 672:
            raise ValueError("Prescribed trajectory is not a complete week")
        if hashlib.sha256(canonical(record["minute_truth_states"])).hexdigest() != row["retained_minute_truth_sha256"]:
            raise ValueError("Prescribed physical trace checksum differs")
        replays.append({"controller": row["controller"], "condition": row["condition"], "split": row["split"],
            "evaluation_family": freeze["schema"], "supplementary": supplementary,
            "scenario_seed": row["seed"], "variant": row["variant"], "controller_kind": spec["kind"],
            "training_family": spec.get("training_family"), "recording": relative(record_path),
            "freeze": relative(freeze_path), "gate": (relative(row["main_campaign_admission"]["gate"]["path"])
                if supplementary and row.get("main_campaign_admission") else relative(freeze["gate"]["path"]) if freeze.get("gate") else None),
            "checkpoint": relative(spec["checkpoint"]["checkpoint"]["path"]) if spec["kind"] == "ppo" else None,
            "trace_sha256": row["retained_minute_truth_sha256"]})
    default_availability = [{"controller": label, "condition": "d30-sampled", "scenario_seed": 20200000,
        "full_trace_available": any((r["controller"], r["condition"], r["split"], r["scenario_seed"]) == (label, "d30-sampled", "held_out", 20200000) for r in replays)} for label in required_controllers]
    files = [{"path": relative(path), "bytes": path.stat().st_size, "sha256": sha(path)} for path in sorted(paths)]
    requirements = "\n".join(f"{name}=={version}" for name, version in original_plan["runtime"]["packages"].items()) + "\n"
    readme = """# PlantAnAgent: comparison-first PPO study

Extract the package and use Python 3.12 with the versions in `requirements-replay.txt`.
From the extracted root, verification and a complete prescribed replay are:

```sh
python -m docs.research.tools.comparison_ppo_v2.portable verify
python -m docs.research.tools.comparison_ppo_v2.portable replay --controller ppo25-101 --condition d30-sampled
python -m docs.research.tools.comparison_ppo_v2.portable replay --controller daylight --condition d30-sampled
python -m docs.research.tools.comparison_ppo_v2.portable replay --controller daylight-persistence --condition d30-sampled
```

The replay verifies every public input, matching frozen normalization and actor mean where applicable, all 672 requested actions, the requested/applied experimental intervention, interval rewards, final economic ledger and all 10,081 minute states. Controllers 211/307 and all other available first-scenario conditions are listed by `verify`; new PPO30 controllers are present only if their actual training campaign completed. `--split development` selects an available development trajectory. Exact equality is checked for the current execution, not promised across hardware or library versions.

PPO25 policies retain their original training identity when transferred to 30 kg/hour or different battery conditions. No observation statistics are refitted. `free_ppo_v1.train.load_checkpoint` loads each model and its matching observation normalizer; the sidecar records and content manifest bind both. Baselines use only the same raw 135-value causal observation and public plant constants.

The separately declared forecast supplement is identified in the manifest under `forecast_ablation`. If completed, its 32 outcomes remain in their own analysis and are paired with the main study's existing 32 daylight scenarios; they do not increase the independent weather sample or alter the main analysis. Its persistence replay also verifies all three forecast-coordinate replacements and the transformed 135-value input. If deadline-incomplete, the explicit receipt and all available attempts are retained and the manifest states that status. The persistence command requires an available complete prescribed trace.

The protocol is `docs/research/COMPARISON_PPO_V2_PROTOCOL.md`; the report, inspector and all new outcomes are in `docs/research/results/comparison_ppo_20260917`. The frozen physical source is under `runs/authority-source-20260917`. The three original final PPO25 checkpoint triplets and their plan are retained under `docs/research/results/free_ppo_20260917/training`. Original absolute paths are provenance: the portable helper resolves content using relative manifest paths and does not rewrite the records or call absolute-path campaign verification after relocation.

All runs last seven days. Battery electricity supplies electrolysis only; DAC and methanation lack thermal/startup dynamics. Contributions use declared synthetic cost assumptions and are not calibrated plant profit. Complete and failed cases are retained. This archive supports inspection and replay, not a claim of industrial optimality or indefinite operation.

The manifest declares whether this is a train-only engineering fixture. Such fixtures are integration evidence and never canonical experiment results. This exporter uses exclusive archive creation and leaves original evidence unchanged.
"""
    if engineering_fixture:
        readme = "# ENGINEERING FIXTURE — TRAINING SEEDS ONLY\n\n" + readme
    generated = {"README-comparison-ppo.md": readme.encode(), "requirements-replay.txt": requirements.encode()}
    files.extend({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in generated.items())
    manifest = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "engineering_fixture": bool(engineering_fixture), "original_root": str(ROOT),
        "analysis": relative(analysis_path), "analysis_sha256": sha(analysis_path),
        "main_analysis_rows": len(rows), "forecast_ablation": supplement,
        "held_out_gate": relative(gate_path) if gate_path else None,
        "recorded_runtime": original_plan["runtime"], "verified_scientific_source_sets": verified_sources,
        "verified_recordings": verified_recordings,
        "prescribed_replays": sorted(replays, key=lambda r: (r["split"], r["controller"], r["condition"])),
        "default_test_trace_availability": default_availability,
        "files": sorted(files, key=lambda r: r["path"]),
        "original_paths": "Absolute paths remain execution provenance; relative names and content hashes bind portable replay.",
        "selection": "First scenario per prospectively frozen panel, all final optimizer seeds, no outcome-based selection; incomplete traces are retained but cannot be full-week replays."}
    manifest["sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            archive.write(path, relative(path))
        for name, data in generated.items():
            archive.writestr(name, data)
        archive.writestr("bundle-manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
        for row in manifest["files"]:
            data = archive.read(row["path"])
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"Archive member changed while packaging: {row['path']}")
    return {"schema": "comparison-ppo-v2-bundle-build", "archive": str(output), "bytes": output.stat().st_size,
            "sha256": sha(output), "manifest_sha256": manifest["sha256"], "files": len(files),
            "engineering_fixture": bool(engineering_fixture), "prescribed_replays": len(replays),
            "main_analysis_rows": len(rows), "forecast_ablation": supplement,
            "default_test_trace_availability": default_availability}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--gate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engineering-fixture", action="store_true")
    parser.add_argument("--freeze", type=Path, action="append", default=[])
    parser.add_argument("--supplement-deadline-receipt", type=Path)
    args = parser.parse_args()
    result = export(analysis_path=args.analysis, gate_path=args.gate, output=args.output,
                    engineering_fixture=args.engineering_fixture, freeze_paths=args.freeze,
                    supplement_deadline_receipt=args.supplement_deadline_receipt)
    if not args.engineering_fixture:
        with (RESULTS / "bundle-build.json").open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
