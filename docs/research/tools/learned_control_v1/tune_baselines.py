"""Prospectively specified, development-only tuning of the existing storage rule."""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import statistics

from docs.research.tools.learned_control_v1 import evaluate
from docs.research.tools.learned_control_v1.analyze import MIXTURE, STRESS, audited
from docs.research.tools.wholeplant_ppo_v1.portable import file_pin


def configurations():
    return [{"label": f"storage-m{m}-b{round(b*100):02d}-r{r}", "kind": "baseline",
             "baseline": "storage", "config": {"methane_buffer_h": m,
                 "battery_target_soc": b, "replenish_h": r}}
            for m, b, r in product((48, 72, 96), (.35, .60, .85), (3, 6))]


def ranking(summary):
    rows = summary["rows"]
    if not all(audited(r) for r in rows):
        raise ValueError("Cannot select a baseline from invalid development cases")
    labels = sorted({r["controller"] for r in rows})
    results = []
    for label in labels:
        by_condition = {c: [r for r in rows if r["controller"] == label and r["condition"] == c]
                        for c in evaluate.CONDITIONS}
        mean = lambda c, k: statistics.mean(r["economic_metrics"][k] for r in by_condition[c])
        results.append({"controller": label,
            "mixture_contribution_eur": sum(w*mean(c, "net_operating_value_eur") for c,w in MIXTURE.items()),
            "mixture_unmet_kg": sum(w*mean(c, "unmet_kg") for c,w in MIXTURE.items()),
            "stress_unmet_kg": statistics.mean(mean(c,"unmet_kg") for c in STRESS),
            "ordinary_contribution_eur": mean("ordinary", "net_operating_value_eur"),
            "condition_metrics": {c: {"contribution_eur": mean(c,"net_operating_value_eur"),
                "unmet_kg": mean(c,"unmet_kg"), "successes": sum(r["every_delivery_passed"] for r in group),
                "n": len(group)} for c,group in by_condition.items()}})
    return sorted(results, key=lambda r: (-r["mixture_contribution_eur"], r["mixture_unmet_kg"], r["controller"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("screen", "confirm"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = args.root / args.command
    if args.command == "screen":
        controllers = configurations()
        seeds = evaluate.DEVELOPMENT_SEEDS[:4]
        rationale = "All18 preregistered storage configurations on first4 development seeds"
    else:
        prior = json.loads((args.root / "screen" / "selection-ranking.json").read_text())
        selected = [r["controller"] for r in prior["ranking"][:3]]
        controllers = [c for c in configurations() if c["label"] in selected]
        controllers += [{"label": "daylight", "kind": "baseline", "baseline": "daylight"},
                        {"label": "storage", "kind": "baseline", "baseline": "storage"}]
        seeds = evaluate.DEVELOPMENT_SEEDS
        rationale = "Top3 screening configurations plus both original baselines on all16 development seeds"
    evaluate.freeze_panel(output / "freeze.json", controllers, seeds,
                          deadline_utc="2026-09-18T05:45:00+00:00", source_paths=(Path(__file__),))
    summary = evaluate.campaign(output, workers=args.workers)
    order = ranking(summary)
    result = {"rule": "Highest .5ordinary+.1eachstress contribution, then lowest mixture unmet",
              "rationale": rationale, "summary": file_pin(output / "summary.json"), "ranking": order,
              "champion": next(c for c in controllers if c["label"] == order[0]["controller"])}
    evaluate.publish(output / "selection-ranking.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
