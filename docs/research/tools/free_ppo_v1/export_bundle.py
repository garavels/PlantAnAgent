"""Export the completed direct-control study without altering its source evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[4]
RESULTS = ROOT / "docs/research/results/free_ppo_20260917"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(*, analysis_path, gate_path, output):
    analysis = json.loads(analysis_path.read_text())
    plan_path = RESULTS / "training/plan.json"
    plan = json.loads(plan_path.read_text())
    groups = {g["controller"]: g for g in analysis["groups"] if g["split"] == "held_out"}
    labels = ["coverage-direct", "untrained-101", *[f"free-ppo-v1-{s}" for s in plan["optimizer_seeds"]]]
    if any(groups.get(label, {}).get("n_scenarios") != 32 for label in labels):
        raise ValueError("The complete prespecified held-out panel is required")
    for record in plan["source_files"]:
        path = Path(record["path"])
        if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError(f"Training source changed: {path}")
    paths = set()
    for directory in (ROOT / "runs/authority-source-20260917", Path(__file__).parent, RESULTS):
        for path in directory.rglob("*"):
            relative = path.relative_to(ROOT)
            if path.is_file() and not path.is_symlink() and not any(
                part in {"__pycache__", ".build", ".git", "source-snapshot", "pilots", "engineering"}
                for part in relative.parts
            ) and path.resolve() != output.resolve() and path.name not in {"bundle-build.json", "portable-verification.json"}:
                paths.add(path)
    paths.update(ROOT / name for name in (
        "uv.lock", "docs/research/FREE_PPO_V1_PROTOCOL.md", "docs/research/FREE_CONTROL_GUIDE.md"))
    paths.add(gate_path)
    paths.add(analysis_path)
    files = []
    for path in sorted(paths):
        if not path.resolve().is_relative_to(ROOT) or path.is_symlink():
            raise ValueError("Bundle input escaped the project")
        files.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    replay_rows = {}
    for seed in plan["optimizer_seeds"]:
        label = f"free-ppo-v1-{seed}"
        row = next(r for r in analysis["rows"] if r["split"] == "held_out" and r["controller"] == label and r["seed"] == 10200000)
        if not row["execution_complete"] or not row["retained_minute_truth_sha256"]:
            raise ValueError("Prescribed first-scenario replay is incomplete")
        replay_rows[str(seed)] = {
            "checkpoint": (RESULTS / f"training/seed-{seed}/final-policy.zip").relative_to(ROOT).as_posix(),
            "recording": Path(row["recording"]["path"]).relative_to(ROOT).as_posix(),
            "scenario_seed": row["seed"], "trace_sha256": row["retained_minute_truth_sha256"],
        }
    manifest = {"schema": "free-ppo-v1-portable-bundle", "created_at": datetime.now(timezone.utc).isoformat(),
                "analysis": analysis_path.relative_to(ROOT).as_posix(), "analysis_sha256": sha(analysis_path),
                "held_out_gate": gate_path.relative_to(ROOT).as_posix(), "prescribed_replays": replay_rows,
                "files": files, "original_paths": "Absolute record paths are original execution provenance; relative bundle paths bind portable content.",
                "selection": "Every final training seed is included; default replay is optimizer101 on the prospectively selected first test scenario, without outcome selection."}
    requirements = "\n".join(f"{name}=={version}" for name, version in plan["runtime"]["packages"].items()) + "\n"
    readme = f"""# PlantAnAgent: direct-control reinforcement learning

This package contains all {len(plan['optimizer_seeds'])} final policies, their initial and intermediate checkpoints, observation normalization statistics, the executed simulator, comparison code, and the completed evidence. Each final policy received {plan['total_timesteps']:,} transitions. The report records performance, including failed deliveries; training completion is not a claim of reliable operation.

Use Python 3.12 and install the versions in `requirements-replay.txt`. From this extracted directory:

```sh
python -m docs.research.tools.free_ppo_v1.portable verify
python -m docs.research.tools.free_ppo_v1.portable replay --optimizer-seed 101
```

The replay checks observations, normalized inputs, actor means, every action, interval reward and minute state against the retained first test scenario. Optimizer seeds 211 and 307 are also available. The policy and its observation normalizer must always be used together; `train.load_checkpoint` handles both. Exact replay is a check against the recorded runtime, not a promise of bitwise equality on different hardware or software.

The main evidence is in `docs/research/results/free_ppo_20260917`. The binding design is `docs/research/FREE_PPO_V1_PROTOCOL.md`; the readable editing guide is `docs/research/FREE_CONTROL_GUIDE.md`. Checkpoint and source records preserve their original absolute paths as provenance, so the original campaign `require_plan` command is not a portable re-training launcher. The portable replay verifies relative names and content without rewriting those records.

To reproduce training in a new directory on the current host, first create a new plan with its actual paths and runtime. Replace the deadline placeholder below with a future UTC date/time that leaves enough uninterrupted training time; do not reuse the completed campaign directory:

```sh
python -m docs.research.tools.free_ppo_v1.train freeze --root replication-training --total-timesteps {plan['total_timesteps']} --optimizer-seeds 101 211 307 --deadline-utc "YYYY-MM-DDTHH:MM:SS+00:00"
python -m docs.research.tools.free_ppo_v1.train campaign --root replication-training
```

This launches a new replication from random weights. Checkpoints retain optimizer state but this trainer does not implement exact restart of interrupted simulator episodes and their random state. A new scientific comparison also needs its own declared evaluation design; reusing these inspected test weeks does not constitute a fresh test.

All results are synthetic. The direct-control experiment changes the task and action interface from the earlier hybrid-controller campaign; its scores should not be compared as though the tasks were identical.
"""
    generated = {"README-direct-ppo.md": readme.encode(), "requirements-replay.txt": requirements.encode()}
    for name, data in generated.items():
        files.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    manifest["files"] = sorted(files, key=lambda r: r["path"])
    manifest["sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(ROOT).as_posix())
        for name, data in generated.items():
            archive.writestr(name, data)
        archive.writestr("bundle-manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
        for row in manifest["files"]:
            data = archive.read(row["path"])
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError(f"Archive member differs: {row['path']}")
    return {"schema": "free-ppo-v1-bundle-build", "archive": str(output), "bytes": output.stat().st_size,
            "sha256": sha(output), "manifest_sha256": manifest["sha256"], "files": len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(analysis_path=args.analysis.resolve(), gate_path=args.gate.resolve(), output=args.output.resolve())
    with (RESULTS / "bundle-build.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
