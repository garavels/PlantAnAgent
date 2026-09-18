"""Bounded, discarded train-only cProfile fixture for environment bottlenecks."""
from __future__ import annotations

import argparse
import cProfile
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from docs.research.tools.free_ppo_v1 import environment


def functions(profiler, *, by, limit=40):
    rows = []
    for (path, line, name), (primitive, calls, self_seconds, cumulative, _) in pstats.Stats(profiler).stats.items():
        rows.append({"path": path, "line": line, "function": name,
                     "primitive_calls": primitive, "calls": calls,
                     "self_seconds": self_seconds, "cumulative_seconds": cumulative})
    return sorted(rows, key=lambda row: row[by], reverse=True)[:limit]


def profile(output_dir: Path):
    output_json = output_dir / "environment-profile.json"
    output_text = output_dir / "environment-profile.txt"
    if output_json.exists() or output_text.exists():
        raise FileExistsError("Retain the existing engineering fixture")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_identity = environment.source_identity_sha256()
    env = environment.DirectPlantEnv(split="train", record_minutes=False)
    reset_profiler, step_profiler = cProfile.Profile(), cProfile.Profile()
    started = time.perf_counter()
    deadline = started + 110.0
    started_at = datetime.now(timezone.utc).isoformat()
    decisions = resets = 0
    reset_wall = step_wall = 0.0
    seeds = []
    action = np.zeros(5, dtype=np.float32)
    live = False
    try:
        while decisions < 1024 and time.perf_counter() < deadline:
            if not live:
                seed = 10_099_980 + resets
                seeds.append(seed)
                tick = time.perf_counter()
                reset_profiler.runcall(env.reset, seed=seed)
                reset_wall += time.perf_counter() - tick
                resets += 1
                live = True
            tick = time.perf_counter()
            _, _, terminated, truncated, _ = step_profiler.runcall(env.step, action)
            step_wall += time.perf_counter() - tick
            decisions += 1
            live = not (terminated or truncated)
        elapsed = time.perf_counter() - started
        if environment.source_identity_sha256() != source_identity:
            raise ValueError("Frozen environment source changed during fixture")
        sections = []
        for title, profiler in (("SCENARIO PREPARATION / RESET", reset_profiler),
                                ("REPEATED CONTROL DECISIONS / STEP", step_profiler)):
            for ordering in ("cumulative", "tottime"):
                buffer = io.StringIO()
                pstats.Stats(profiler, stream=buffer).sort_stats(ordering).print_stats(40)
                sections.append(f"{title}: sorted by {ordering}\n{buffer.getvalue()}")
        record = {
            "schema": "free-ppo-environment-profile-v1", "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Discarded train-only environment fixture; no learner, no model training, no weights or outcome selection.",
            "seed_sequence": seeds, "action": action.tolist(),
            "action_description": "Fixed normalized zero vector: conservation-normalized equal solar shares and half-rated battery support.",
            "requested_decisions": 1024, "decisions": decisions, "physical_minutes": decisions * 15,
            "resets": resets, "wall_seconds": elapsed, "reset_wall_seconds": reset_wall,
            "step_wall_seconds": step_wall, "stopped_at_runtime_limit": decisions < 1024,
            "environment_identity_sha256": environment.environment_identity_sha256(),
            "source_identity_sha256": source_identity,
            "profiler_source": {"path": str(Path(__file__).resolve()),
                                "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            "reset_top_self": functions(reset_profiler, by="self_seconds"),
            "reset_top_cumulative": functions(reset_profiler, by="cumulative_seconds"),
            "step_top_self": functions(step_profiler, by="self_seconds"),
            "step_top_cumulative": functions(step_profiler, by="cumulative_seconds"),
            "limitations": [
                "cProfile instrumentation changes timings; use rankings to locate work, not these absolute timings to predict production throughput.",
                "One fixed-action trajectory on two admitted training scenarios is not representative of every learned operating mode.",
                "Production training and evaluation continue on the same local host; this is not a Bocconi or HPC hardware measurement.",
                "Scenario resets and step calls are collected separately; imports and environment construction precede timing.",
                "The helper changes no frozen file, imported production module instance or production process.",
            ],
        }
        with output_text.open("x") as stream:
            stream.write("Discarded train-only cProfile fixture; shared local host; no optimization performed.\n\n")
            stream.write("\n\n".join(sections))
        with output_json.open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return record
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = profile(args.output_dir.resolve())
    print(json.dumps({key: result[key] for key in
                      ("decisions", "resets", "wall_seconds", "reset_wall_seconds", "step_wall_seconds", "stopped_at_runtime_limit")}, indent=2))


if __name__ == "__main__":
    main()
