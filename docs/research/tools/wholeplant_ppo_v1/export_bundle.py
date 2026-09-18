"""Exclusive final archive, admitted only by a pinned final artifact release.

Release JSON: {"schema":"wholeplant-ppo-v1-artifact-release","status":"final",
"analysis":PIN,"report":PIN,"inspector":PIN,"completion":PIN,"artifacts":[PIN]}.
Each PIN contains the original absolute path, byte count and SHA256. Additional
artifacts can include the main LaTeX document and its separate source archive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from docs.research.tools.wholeplant_ppo_v1.portable import SCHEMA, canonical, digest, file_pin, file_sha

RESULTS = ROOT / "docs/research/results/wholeplant_ppo_20260917"
RELEASE_SCHEMA = "wholeplant-ppo-v1-artifact-release"
ARMS = ("restricted-instant", "shared-instant", "restricted-dynamic", "shared-dynamic")
VALIDATION_TESTS = tuple(f"tests/test_wholeplant_{name}.py" for name in
                         ("physics", "environment", "baselines", "evaluation"))


def read(path):
    return json.loads(Path(path).read_text())


def checked(pin):
    if file_pin(pin["path"]) != pin:
        raise ValueError(f"Released content changed: {pin['path']}")
    return Path(pin["path"]).resolve()


def expected_cases():
    cases = set()
    for split, seeds in (("development", range(30100000, 30100008)), ("held_out", range(30200000, 30200032))):
        for arm in ARMS:
            for label in ("daylight", "storage"):
                cases.update((split, arm, label, seed) for seed in seeds)
        labels = [f"ppo-{s}" for s in (101, 211, 307)]
        if split == "development":
            labels += [f"ppo-initial-{s}" for s in (101, 211, 307)]
        else:
            labels += ["storage-persistence"]
        for label in labels:
            cases.update((split, "shared-dynamic", label, seed) for seed in seeds)
    return cases


def export(*, release_path, output):
    release_path, output = Path(release_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    release = read(release_path)
    if release.get("schema") != RELEASE_SCHEMA or release.get("status") != "final":
        raise ValueError("Canonical export requires the final artifact release")
    bindings = [release[key] for key in ("analysis", "report", "inspector", "completion")]
    bindings += release.get("artifacts", [])
    for record in bindings:
        checked(record)
    analysis_path = checked(release["analysis"])
    analysis, completion = read(analysis_path), read(checked(release["completion"]))
    if (analysis.get("schema") != "wholeplant-ppo-v1-evaluation-analysis"
            or completion.get("status") != "completed" or completion.get("rows") != 496
            or completion.get("scientific_sources_unchanged") is not True
            or completion.get("analysis") != release["analysis"]):
        raise ValueError("Main campaign completion does not bind the released 496-case analysis")
    rows = analysis["rows"]
    keys = [(r["split"], r["condition"], r["controller"], r["seed"]) for r in rows]
    if len(keys) != 496 or set(keys) != expected_cases() or len(analysis["groups"]) != 26:
        raise ValueError("Every prescribed development and held-out outcome must be retained exactly once")
    plan_path = RESULTS / "training/plan.json"
    plan = read(plan_path)
    if plan["optimizer_seeds"] != [101, 211, 307]:
        raise ValueError("Final training seeds differ from protocol")
    closure_path = RESULTS / "training/campaign-closure.json"
    closure = read(closure_path)
    if (closure.get("all_fixed_endpoints_completed") is not True
            or closure.get("plan_sha256") != digest(plan)
            or sorted(r["optimizer_seed"] for r in closure["runs"]) != [101, 211, 307]):
        raise ValueError("Training has not closed all three fixed endpoints")
    from docs.research.tools.wholeplant_ppo_v1 import evaluate
    freeze_path = RESULTS / "held-out-freeze.json"
    freeze = evaluate.require_freeze(freeze_path)
    bindings += plan["source_files"] + freeze["sources"]
    bindings += [file_pin(plan_path), file_pin(closure_path), file_pin(freeze_path), freeze["gate"], completion["endpoints"]]
    for run in closure["runs"]:
        if run["exit_code"] != 0 or run["status"] != "completed" or run["num_timesteps"] != plan["total_timesteps"]:
            raise ValueError("A training attempt did not finish its fixed budget")
        result = read(checked(run["result"]))
        if (result["status"] != "completed" or result["num_timesteps"] != plan["total_timesteps"]
                or result["plan_sha256"] != digest(plan) or not result["actor_mean_parameters_changed"]):
            raise ValueError("Final result does not prove the fixed learned endpoint")
        bindings.append(run["result"])
    replays = {}
    for panel in freeze["panels"]:
        if panel["split"] != "held_out":
            raise ValueError("Held-out freeze contains another split")
        first = next(r for r in rows if (r["split"], r["condition"], r["controller"], r["seed"]) ==
                     ("held_out", panel["condition"], panel["controller"], panel["seeds"][0]))
        available = bool(first["execution_complete"] and first.get("retained_minute_truth_sha256") and first.get("recording"))
        if first.get("recording"):
            bindings.append(first["recording"])
        replays[panel["controller"]+"/"+panel["condition"]] = {
            "scenario_seed": first["seed"], "recording": first["recording"],
            "available": available, "unavailable_reason": None if available else "First prescribed case did not retain a complete week; no replacement selected",
            "trace_sha256": first.get("retained_minute_truth_sha256")}
    for controller in freeze["controllers"].values():
        if controller["kind"] == "ppo":
            if controller["checkpoint_timesteps"] != plan["total_timesteps"]:
                raise ValueError("Held-out policy is not the fixed final checkpoint")
            bindings += list(controller["checkpoint"].values())
    for row in rows:
        if row.get("recording"):
            bindings.append(row["recording"])
    for entry in analysis["freezes"]:
        bindings.append(entry["file"])
    bindings += analysis["summaries"]
    for record in bindings:
        checked(record)
    unique = {record["path"]: record for record in bindings}
    if any(record != unique[record["path"]] for record in bindings):
        raise ValueError("Inconsistent scientific file pins")
    paths = {release_path, analysis_path, ROOT / "uv.lock", ROOT / "docs/research/WHOLEPLANT_PPO_V1_PROTOCOL.md"}
    paths.update(ROOT / name for name in VALIDATION_TESTS)
    paths.update(checked(record) for record in unique.values())
    # Include all captured physical source, transitive research helpers and all
    # outcomes/attempts; only generated caches and this archive's receipts omit.
    folders = [ROOT / "runs/authority-source-20260917", Path(__file__).parent, RESULTS]
    folders += [Path(__file__).parent.parent / name for name in ("free_ppo_v1", "comparison_ppo_v2", "forecast_ablation_v1")]
    for directory in folders:
        for path in directory.rglob("*"):
            if (path.is_file() and not path.is_symlink() and path.resolve() != output
                    and not any(part in {"__pycache__", ".git", ".build"} for part in path.relative_to(ROOT).parts)
                    and path.name not in {"bundle-build.json", "portable-verification.json"}):
                paths.add(path.resolve())
    files = []
    for path in sorted(paths):
        if path.is_symlink() or not path.is_relative_to(ROOT) or path == output:
            raise ValueError("Bundle input escaped source root or is the output archive")
        pin = file_pin(path)
        files.append({"path": path.relative_to(ROOT).as_posix(), "original_path": pin["path"],
                      "bytes": pin["bytes"], "sha256": pin["sha256"]})
    requirements = "\n".join(f"{name}=={version}" for name, version in sorted(plan["runtime"]["packages"].items()))+"\n"
    test_requirements = "-r requirements-replay.txt\npytest==" + importlib.metadata.version("pytest") + "\n"
    readme = f"""# Whole-plant battery access and paid downstream readiness

This bundle retains 496 prescribed outcomes, all four physical arms, the three
fresh PPO endpoints ({plan['total_timesteps']:,} transitions each), initial and
intermediate policies, observation normalizers, audits, traces, report and inspector.
Readiness coefficients are assumptions, not measured thermochemical dynamics.
Training completion and seven-day performance do not establish indefinite operation
or reliability outside the sampled conditions.

Use Python 3.12 with requirements-replay.txt installed. Dependencies must already
be installed or locally available for an offline installation; wheels are not
bundled. From this extracted directory, using the same runtime as the study:

```sh
python -m docs.research.tools.wholeplant_ppo_v1.portable verify
python -m docs.research.tools.wholeplant_ppo_v1.portable replay --controller storage --arm shared-dynamic
python -m docs.research.tools.wholeplant_ppo_v1.portable replay --controller storage-persistence --arm shared-dynamic
python -m docs.research.tools.wholeplant_ppo_v1.portable replay --controller ppo-101 --arm shared-dynamic
```

Other choices are daylight/storage in every arm and ppo-211/ppo-307 in shared-dynamic.
Each replay uses the first prospectively prescribed held-out scenario, 30200000;
it verifies 149 public inputs, transformed baseline inputs or normalized PPO inputs
and actor means, seven requested/applied controls, interval rewards, final ledgers,
and all 10081 minute states. Replays are integrity checks, not additional outcomes.
Exact equality is tested against the captured runtime; other hardware/software may
introduce numerical differences. Original absolute paths remain unmodified
provenance and are rebound through the content-hashed relative manifest.

The four included validation files substantiate the retained 53-check receipt.
Pytest is a separate test dependency, pinned in requirements-test.txt alongside
the replay requirements. After installing those dependencies, run from this
extracted directory:

```sh
python -m pytest -q tests/test_wholeplant_physics.py tests/test_wholeplant_environment.py tests/test_wholeplant_baselines.py tests/test_wholeplant_evaluation.py
```

The study is under docs/research/results/wholeplant_ppo_20260917. The protocol is
docs/research/WHOLEPLANT_PPO_V1_PROTOCOL.md and the assumptions review is in the
study's assumptions directory. Engineering fixtures are explicitly labeled and
excluded from the 496 canonical outcomes. The trainer cannot exactly resume an
interrupted simulator episode from its policy checkpoint alone. This bundle does
not overwrite or reclassify any earlier study.
"""
    generated = {"README-wholeplant-ppo.md": readme.encode(), "requirements-replay.txt": requirements.encode(),
                 "requirements-test.txt": test_requirements.encode()}
    for name, data in generated.items():
        files.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
                "release": release_path.relative_to(ROOT).as_posix(), "analysis": analysis_path.relative_to(ROOT).as_posix(),
                "training_plan": plan_path.relative_to(ROOT).as_posix(), "held_out_freeze": freeze_path.relative_to(ROOT).as_posix(),
                "files": sorted(files, key=lambda r: r["path"]), "scientific_bindings": sorted(unique.values(), key=lambda r: r["path"]),
                "prescribed_replays": replays, "canonical_outcomes": 496,
                "selection": "First prescribed held-out scenario in every panel; no outcome-based replay selection"}
    manifest["sha256"] = digest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(ROOT).as_posix())
        for name, data in generated.items():
            archive.writestr(name, data)
        archive.writestr("bundle-manifest.json", json.dumps(manifest, indent=2)+"\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC failed")
        for row in files:
            data = archive.read(row["path"])
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"Archive member changed during export: {row['path']}")
    return {"schema": "wholeplant-ppo-v1-bundle-build", "archive": file_pin(output),
            "release": file_pin(release_path), "manifest_sha256": manifest["sha256"],
            "files": len(files), "canonical_outcomes": 496, "prescribed_replays": len(replays)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(release_path=args.release, output=args.output)
    with (RESULTS / "bundle-build.json").open("x") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
