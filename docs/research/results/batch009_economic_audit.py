"""Frozen batch-008 saved-trajectory economics. No simulator/model imports."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import time
from typing import Any, Iterator, BinaryIO

SOURCE = Path(
    "/Users/filippogombac/Developer/PlantAnAgentResearch/adaptive-rl-batch-008-analysis/report/analysis.json"
)
SOURCE_SHA = "9b4e86afe39148a9ad5f9877c8a2ddcdb5e135bbb104ba6043d51a76da56934c"
EPISODE_ROOT = Path(
    "/Users/filippogombac/Developer/PlantAnAgentResearch/adaptive-rl-batch-008/evaluations"
)
SEEDS = list(range(9700300, 9700308))
ACTORS = [
    "H",
    "OperatingMPC",
    "M24",
    "E-beta",
    "initial-mpc-residual",
    "A-mpc-residual-scalar",
    "B-mpc-residual-pi",
]
CANDIDATES = ACTORS[-2:]
COMPONENTS = [
    "battery_endowment_liability_change",
    "battery_throughput_reserve",
    "delivered_service_value",
    "electrolyser_life_reserve",
    "methane_endowment_liability_change",
    "solar_opportunity_cost",
    "unmet_delivery_penalty",
]
FACTORS = [0.0, 0.5, 1.0, 1.5, 2.0]
SENSITIVITY_COMPONENTS = [
    "battery_throughput_reserve",
    "solar_opportunity_cost",
    "electrolyser_life_reserve",
    "unmet_delivery_penalty",
]
GAMMAS = [0.99, 0.995, 0.999, 1.0]
MAX_COMPRESSED = 256 * 1024 * 1024
MAX_DECOMPRESSED = 1024 * 1024 * 1024
MAX_ROW = 8 * 1024 * 1024


def finite(value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"Not a finite number: {value!r}")
    return float(value)


def close(a: float, b: float) -> None:
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError(f"Reconciliation failed: {a!r} != {b!r}")


def stable_identity(s: os.stat_result) -> tuple[int, ...]:
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)


@contextmanager
def verified_file(record: dict[str, Any], maximum: int) -> Iterator[BinaryIO]:
    path = Path(record["path"])
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError("File path must be absolute and contain no symlinks")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size > maximum
    ):
        raise ValueError("Invalid regular file or file bound")
    if before.st_size != record["bytes"]:
        raise ValueError("File size differs")
    with path.open("rb") as stream:
        if stable_identity(os.fstat(stream.fileno())) != stable_identity(before):
            raise ValueError("Opened descriptor differs")
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != record["sha256"]:
            raise ValueError("File digest differs")
        stream.seek(0)
        yield stream
        if stable_identity(os.fstat(stream.fileno())) != stable_identity(
            before
        ) or stable_identity(path.lstat()) != stable_identity(before):
            raise ValueError("File changed during analysis")


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def read_report() -> tuple[dict[str, Any], dict[str, Any]]:
    record = {"path": str(SOURCE), "bytes": SOURCE.stat().st_size, "sha256": SOURCE_SHA}
    with verified_file(record, 64 * 1024 * 1024) as stream:
        data = json.load(stream)
    if data["fixed_development_seeds"] != SEEDS or list(data["actors"]) != ACTORS:
        raise ValueError("Frozen actor/seed panel differs")
    return data, record


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "minimum": None,
            "maximum": None,
            "positive_count": 0,
            "negative_count": 0,
            "values": [],
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "minimum": min(values),
        "maximum": max(values),
        "positive_count": sum(v > 0 for v in values),
        "negative_count": sum(v < 0 for v in values),
        "values": values,
    }


def rescore(components: dict[str, float], factors: dict[str, float]) -> float:
    if set(factors) - set(COMPONENTS):
        raise ValueError("Unknown component")
    return math.fsum(
        finite(components[k]) * finite(factors.get(k, 1.0)) for k in COMPONENTS
    )


def break_even(base_delta: float, component_delta: float) -> dict[str, Any]:
    if component_delta == 0:
        return {
            "multiplier": None,
            "status": "unchanged_slope",
            "baseline_delta_eur": base_delta,
            "slope_eur_per_multiplier": component_delta,
        }
    factor = 1 - base_delta / component_delta
    return {
        "multiplier": factor,
        "status": "nonnegative" if factor >= 0 else "negative_factor_only",
        "baseline_delta_eur": base_delta,
        "slope_eur_per_multiplier": component_delta,
    }


def ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def extract_actor(actor: str, data: dict[str, Any]) -> dict[str, Any]:
    slots = data["actors"][actor]["slots"]
    if [s["seed"] for s in slots] != SEEDS:
        raise ValueError("Slot seed order differs")
    rows = []
    for s in slots:
        summary = s.get("summary")
        row: dict[str, Any] = {
            "seed": s["seed"],
            "eligible_complete_episode": s["eligible_for_descriptive_comparison"],
            "failures": s["failures"],
        }
        if summary is None:
            row["recorded_prefix"] = None
            rows.append(row)
            continue
        economic = summary["economic_metrics"]
        empty_unstarted = (
            summary["decisions"] == 0
            and economic["components_eur"] == {}
            and economic["net_operating_value_eur"] == 0
        )
        components = {
            k: (0.0 if empty_unstarted else finite(economic["components_eur"][k]))
            for k in COMPONENTS
        }
        if not empty_unstarted and set(economic["components_eur"]) != set(COMPONENTS):
            raise ValueError("Economic component surface differs")
        close(
            math.fsum(components.values()), finite(economic["net_operating_value_eur"])
        )
        p = summary["physical_metrics"]
        row.update(
            recorded_prefix={
                "decisions": summary["decisions"],
                "execution_complete": summary["execution_complete"],
                "failure": summary["failure"],
                "economic_metrics": economic,
                "physical_metrics": p,
            },
            components_eur=components,
            net_value_eur=finite(economic["net_operating_value_eur"]),
        )
        if row["eligible_complete_episode"] and (
            summary["decisions"] != 672 or not summary["execution_complete"]
        ):
            raise ValueError("Eligible episode is incomplete")
        rows.append(row)
    eligible = all(r["eligible_complete_episode"] for r in rows)
    result: dict[str, Any] = {
        "actor": actor,
        "complete_panel": eligible,
        "rows": rows,
        "aggregate": None,
    }
    if eligible:
        means = {k: stats([r["components_eur"][k] for r in rows]) for k in COMPONENTS}
        normalized = {}
        for numerator in [
            "battery_throughput_kwh",
            "electrolyser_nonproductive_energy_kwh",
            "accepted_solar_kwh",
            "accepted_process_energy_kwh",
        ]:
            for denominator in ["ch4_produced_kg", "delivered_kg"]:
                nums, dens = [], []
                for r in rows:
                    e, p = (
                        r["recorded_prefix"]["economic_metrics"],
                        r["recorded_prefix"]["physical_metrics"],
                    )
                    nums.append(
                        finite(e[numerator] if numerator in e else p[numerator])
                    )
                    dens.append(
                        finite(e[denominator] if denominator in e else p[denominator])
                    )
                normalized[f"{numerator}_per_{denominator}"] = {
                    "ratio_of_totals": ratio(math.fsum(nums), math.fsum(dens)),
                    "per_seed_ratios": [ratio(n, d) for n, d in zip(nums, dens)],
                    "numerator": stats(nums),
                    "denominator": stats(dens),
                }
        result["aggregate"] = {
            "components_eur": means,
            "net_value_eur": stats([r["net_value_eur"] for r in rows]),
            "physical_intensity_proxies": normalized,
        }
    return result


def comparison(
    candidate: dict[str, Any], baseline: dict[str, Any], factors: dict[str, float]
) -> dict[str, Any]:
    if not candidate["complete_panel"] or not baseline["complete_panel"]:
        return {"available": False, "reason": "Requires eight complete paired cases"}
    values = [
        rescore(c["components_eur"], factors) - rescore(b["components_eur"], factors)
        for c, b in zip(candidate["rows"], baseline["rows"], strict=True)
    ]
    return {"available": True, "factors": factors, "delta_eur": stats(values)}


def iter_history(stream: BinaryIO) -> Iterator[dict[str, Any]]:
    """Stream the frozen pretty-JSON history, one bounded decision object at a time.

    The admitted writer uses indent=2. Exact structural indentation is checked;
    nested JSON and arbitrary string contents are decoded by the standard parser.
    We still consume the remaining gzip stream to verify CRC and size bounds.
    """
    total, rows, stage = 0, 0, "before"
    current: list[bytes] = []
    current_size = 0
    with gzip.GzipFile(fileobj=stream) as zipped:
        while True:
            line = zipped.readline(MAX_ROW + 1)
            if not line:
                break
            total += len(line)
            if total > MAX_DECOMPRESSED or len(line) > MAX_ROW:
                raise ValueError("Decompressed bound exceeded")
            if stage == "before":
                if line == b'  "history": [\n':
                    stage = "history"
                elif line == b'  "history": [],\n':
                    stage = "after"
                continue
            if stage == "after":
                continue
            if not current:
                if line == b"  ],\n":
                    stage = "after"
                    continue
                if line != b"    {\n":
                    raise ValueError("Unexpected history entry boundary")
            current.append(line)
            current_size += len(line)
            if current_size > MAX_ROW:
                raise ValueError("History row bound exceeded")
            if line in (b"    },\n", b"    }\n"):
                raw = b"".join(current).strip().removesuffix(b",")
                row = json.loads(raw)
                if not isinstance(row, dict) or row.get("decision_index") != rows:
                    raise ValueError("Decision order differs")
                rows += 1
                if rows > 672:
                    raise ValueError("Too many decisions")
                yield row
                current, current_size = [], 0
    if stage != "after" or current:
        raise ValueError("Incomplete history structure")


def extract_reward(row: dict[str, Any]) -> tuple[float, dict[str, float]]:
    reward = finite(row["reward_eur"])
    breakdown = row["info"]["reward"]
    if set(breakdown["components_eur"]) != set(COMPONENTS):
        raise ValueError("Reward component keys differ")
    components = {k: finite(breakdown["components_eur"][k]) for k in COMPONENTS}
    close(math.fsum(components.values()), reward)
    close(finite(breakdown["total_eur"]), reward)
    return reward, components


def summarize_history(
    rows: Iterator[dict[str, Any]], gammas: list[float]
) -> dict[str, Any]:
    discounted: dict[float, dict[str, list[float]]] = {
        g: defaultdict(list) for g in gammas
    }
    days: list[dict[str, list[float]]] = [defaultdict(list) for _ in range(7)]
    discounted_days: dict[float, list[dict[str, list[float]]]] = {
        g: [defaultdict(list) for _ in range(7)] for g in gammas
    }
    reward_values = []
    for index, row in enumerate(rows):
        if row["decision_index"] != index:
            raise ValueError("Noncontiguous decision index")
        reward, components = extract_reward(row)
        reward_values.append(reward)
        for g in gammas:
            weight = g**index
            discounted[g]["total_eur"].append(weight * reward)
            discounted_days[g][index // 96]["total_eur"].append(weight * reward)
            for name, value in components.items():
                discounted[g][name].append(weight * value)
                discounted_days[g][index // 96][name].append(weight * value)
        days[index // 96]["total_eur"].append(reward)
        for name, value in components.items():
            days[index // 96][name].append(value)
    return {
        "decisions": len(reward_values),
        "reward_sum_eur": math.fsum(reward_values),
        "discounted": {
            str(g): {k: math.fsum(v) for k, v in groups.items()}
            for g, groups in discounted.items()
        },
        "daily_undiscounted_components_eur": [
            {k: math.fsum(v) for k, v in groups.items()} for groups in days
        ],
        "daily_discounted_components_eur": {
            str(g): [
                {k: math.fsum(v) for k, v in groups.items()} for groups in gamma_days
            ]
            for g, gamma_days in discounted_days.items()
        },
    }


def analyze(include_histories: bool) -> dict[str, Any]:
    started = time.monotonic()
    data, source_record = read_report()
    actors = {name: extract_actor(name, data) for name in ACTORS}
    baseline = actors["OperatingMPC"]
    result: dict[str, Any] = {
        "schema": "plant-batch009-fixed-trajectory-economic-audit-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source_record,
        "analysis_source": file_record(Path(__file__)),
        "analysis_plan": file_record(
            Path(__file__).with_name("batch009_economic_audit_plan.md")
        ),
        "seeds": SEEDS,
        "actors": actors,
        "sensitivity": {},
        "timing": None,
        "limits": [
            "No simulation reset, inference, training, policy selection, or protected seed access.",
            "All sensitivity coefficients are illustrative multipliers, not calibration ranges.",
            "Fixed-trajectory rescoring cannot predict policy adaptation to changed prices.",
            "Physical ratios are throughput/intensity proxies, not thermodynamic efficiency or measured equipment lifetime.",
            "Safety, individual delivery and daily resource regressions remain binding at every rescore.",
            "Discounted realized returns do not reconstruct PPO advantages, diagnose causality, or prove retraining benefit.",
        ],
    }
    for name in CANDIDATES:
        candidate = actors[name]
        component_deltas = {
            k: stats(
                [
                    c["components_eur"][k] - b["components_eur"][k]
                    for c, b in zip(candidate["rows"], baseline["rows"], strict=True)
                ]
            )
            for k in COMPONENTS
        }
        base = comparison(candidate, baseline, {})
        sensitivity: dict[str, Any] = {
            "baseline": base,
            "component_differences_eur": component_deltas,
            "one_factor": {},
            "break_even": {},
            "battery_penalty_grid": [],
        }
        for component in SENSITIVITY_COMPONENTS:
            sensitivity["one_factor"][component] = [
                comparison(candidate, baseline, {component: factor})
                for factor in FACTORS
            ]
            sensitivity["break_even"][component] = break_even(
                base["delta_eur"]["mean"], component_deltas[component]["mean"]
            )
        for battery_factor in FACTORS:
            for penalty_factor in FACTORS:
                sensitivity["battery_penalty_grid"].append(
                    comparison(
                        candidate,
                        baseline,
                        {
                            "battery_throughput_reserve": battery_factor,
                            "unmet_delivery_penalty": penalty_factor,
                        },
                    )
                )
        result["sensitivity"][name] = sensitivity
    if include_histories:
        templates = [
            data["training"][name]["raw_training_evidence"]["result"][
                "environment_template"
            ]
            for name in CANDIDATES
        ]
        gammas = sorted(set(GAMMAS + [finite(t["ppo"]["gamma"]) for t in templates]))
        records = {r["path"]: r for r in data["audit_provenance"]["artifact_records"]}
        history_results: dict[str, Any] = {}
        used_records = []
        for name, actor in actors.items():
            actor_timing = []
            for slot in actor["rows"]:
                path = EPISODE_ROOT / f"{slot['seed']}-{name}" / "episode.json.gz"
                record = records[str(path)]
                with verified_file(record, MAX_COMPRESSED) as stream:
                    timing = summarize_history(iter_history(stream), gammas)
                if timing["decisions"] != slot["recorded_prefix"]["decisions"]:
                    raise ValueError("History count differs from frozen summary")
                close(timing["reward_sum_eur"], slot["net_value_eur"])
                for k in COMPONENTS:
                    close(
                        timing["discounted"]["1.0"].get(k, 0), slot["components_eur"][k]
                    )
                actor_timing.append(
                    {
                        "seed": slot["seed"],
                        "eligible_complete_episode": slot["eligible_complete_episode"],
                        **timing,
                    }
                )
                used_records.append(record)
            history_results[name] = {
                "complete_panel": actor["complete_panel"],
                "rows": actor_timing,
            }
            print(
                json.dumps(
                    {
                        "actor_completed": name,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        gamma_summary = {}
        for gamma in gammas:
            key = str(gamma)
            complete_means = {
                a: stats([r["discounted"][key]["total_eur"] for r in v["rows"]])
                for a, v in history_results.items()
                if v["complete_panel"]
            }
            pairs = {}
            for name in CANDIDATES:
                pairs[name] = {
                    k: stats(
                        [
                            c["discounted"][key][k] - b["discounted"][key][k]
                            for c, b in zip(
                                history_results[name]["rows"],
                                history_results["OperatingMPC"]["rows"],
                                strict=True,
                            )
                        ]
                    )
                    for k in ["total_eur", *COMPONENTS]
                }
            gamma_summary[key] = {
                "complete_actor_returns_eur": complete_means,
                "ranking": sorted(
                    complete_means,
                    key=lambda a: complete_means[a]["mean"],
                    reverse=True,
                ),
                "candidate_minus_mpc": pairs,
                "candidate_minus_mpc_by_day": {
                    name: [
                        {
                            k: stats(
                                [
                                    c["daily_discounted_components_eur"][key][day][k]
                                    - b["daily_discounted_components_eur"][key][day][k]
                                    for c, b in zip(
                                        history_results[name]["rows"],
                                        history_results["OperatingMPC"]["rows"],
                                        strict=True,
                                    )
                                ]
                            )
                            for k in ["total_eur", *COMPONENTS]
                        }
                        for day in range(7)
                    ]
                    for name in CANDIDATES
                },
                "continuation_weights": {
                    f"{hours}h": gamma ** (hours * 4) for hours in [6, 24, 168]
                },
                "effective_horizon_hours": None if gamma == 1 else 0.25 / (1 - gamma),
            }
        result["timing"] = {
            "first_reward_exponent": 0,
            "training_gamma": [t["ppo"]["gamma"] for t in templates],
            "gammas": gammas,
            "consumed_artifacts": used_records,
            "actors": history_results,
            "summary": gamma_summary,
        }
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-histories", action="store_true")
    args = parser.parse_args()
    result = analyze(args.include_histories)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": file_record(args.output),
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
