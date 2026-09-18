"""Inspect saved pilot trajectories and ledgers; performs no environment resets."""

import gzip, json
from pathlib import Path
import numpy as np

P = Path("/Volumes/Lexar/PlantAnAgentResearch/pilot-pi-seed101-v1")
O = Path("/private/tmp/plant-pilot-behavior-summary.json")
actors = {k: [] for k in ["initial", "final", "H"]}
physkeys = [
    "delivery_fill_rate",
    "commitments_met",
    "delivered_kg",
    "ch4_produced_kg",
    "terminal_h2_kg",
    "terminal_co2_kg",
    "terminal_ch4_kg",
    "h2_inventory_change_kg",
    "co2_inventory_change_kg",
    "ch4_inventory_change_kg",
    "terminal_battery_soc",
    "electrolyser_applied_energy_kwh",
    "dac_applied_energy_kwh",
    "methanator_applied_energy_kwh",
    "electrolyser_nonproductive_energy_kwh",
    "electrolyser_cold_starts",
    "electrolyser_hot_starts",
    "electrolyser_aborted_starts",
    "electrolyser_forced_stops",
    "guard_interventions",
    "curtailed_kwh",
    "solar_kwh",
    "dispatch_gap_kwh",
    "request_application_gap_kwh",
]
actionnames = ["electrolyser", "dac", "methanator", "battery_charge_ceiling"]


def dist(arr):
    a = np.asarray(arr)
    return {
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "q10": np.quantile(a, 0.1, axis=0).tolist(),
        "q50": np.quantile(a, 0.5, axis=0).tolist(),
        "q90": np.quantile(a, 0.9, axis=0).tolist(),
        "fraction_le_minus_999": (a <= -0.999).mean(axis=0).tolist(),
        "fraction_ge_999": (a >= 0.999).mean(axis=0).tolist(),
    }


for f in sorted((P / "evaluation").glob("[0-9]*.json.gz")):
    d = json.loads(gzip.decompress(f.read_bytes()))
    hist = d["history"]
    actor = d["actor"]
    a = np.array([r["action"] for r in hist]) if actor != "H" else None
    targets = np.array(
        [
            [r["info"]["requested"]["targets"]["setpoints_kw"][k] for k in actionnames]
            for r in hist
        ]
    )
    applied = np.array(
        [
            [
                r["info"]["interval_power"]["applied_mean_kw"].get(k, 0)
                for k in actionnames[:-1] + ["battery"]
            ]
            for r in hist
        ]
    )
    rewards = np.array([r["reward_eur"] for r in hist])
    settle = np.array([r["info"]["delivery"]["commitments_settled"] > 0 for r in hist])
    summary = {
        "seed": d["seed"],
        "physical": {k: d["physical_metrics"][k] for k in physkeys},
        "net_eur": d["economic_metrics"]["net_operating_value_eur"],
        "components_eur": d["economic_metrics"]["components_eur"],
        "action": dist(a) if a is not None else None,
        "targets_kw": dist(targets),
        "applied_kw": dist(applied),
        "reward": {
            "settlement_sum": float(rewards[settle].sum()),
            "ordinary_sum": float(rewards[~settle].sum()),
            "settlement_mean": float(rewards[settle].mean()),
            "ordinary_mean": float(rewards[~settle].mean()),
            "step_std": float(rewards.std()),
        },
        "cycle_deficit_count": {
            k: sum(
                s["deficits"][k] > s["tolerances"][k] for s in d["cycle_settlements"]
            )
            for k in ["battery", "h2", "co2", "ch4"]
        },
        "cycles": d["cycle_settlements"],
        "daily": [],
    }
    for day in range(7):
        hh = hist[day * 96 : (day + 1) * 96]
        summary["daily"].append(
            {
                "day": day + 1,
                "mean_actions": (
                    a[day * 96 : (day + 1) * 96].mean(axis=0).tolist()
                    if a is not None
                    else None
                ),
                "targets_kw": targets[day * 96 : (day + 1) * 96].mean(axis=0).tolist(),
                "delivered_kg": sum(r["info"]["delivery"]["delivered_kg"] for r in hh),
                "reward_eur": sum(r["reward_eur"] for r in hh),
                "applied_energy_kwh": {
                    k: sum(r["info"]["interval_power"]["applied_kwh"][k] for r in hh)
                    for k in ["electrolyser", "dac", "methanator", "battery"]
                },
            }
        )
    actors[actor].append(summary)
rollouts = [
    json.loads(p.read_text()) for p in sorted((P / "rollouts").glob("[0-9]*.json"))
]
rs = {}
for first, last in [
    (1, 16),
    (17, 32),
    (33, 48),
    (49, 64),
    (65, 80),
    (81, 96),
    (97, 112),
    (113, 128),
]:
    r = rollouts[first - 1 : last]
    rs[f"{first}-{last}"] = {
        "mean_reward_per_transition": float(
            np.mean([o["reward_return"] / o["transition_count"] for o in r])
        ),
        "mean_episode_cost": np.mean(
            [o["mean_completed_episode_cost"] for o in r], axis=0
        ).tolist(),
        "mean_kl": float(np.mean([o["approximate_kl"] for o in r])),
        "mean_clipfrac": float(np.mean([o["clip_fraction"] for o in r])),
        "mean_entropy": float(-np.mean([o["entropy_loss"] for o in r])),
        "end_multipliers": r[-1]["multipliers"],
        "mean_cost_value_losses": np.mean(
            [o["cost_value_losses"] for o in r], axis=0
        ).tolist(),
    }
mean = {}
for actor, aa in actors.items():
    mean[actor] = {
        "physical": {
            k: float(np.mean([d["physical"][k] for d in aa])) for k in physkeys
        },
        "net_eur": float(np.mean([d["net_eur"] for d in aa])),
        "components_eur": {
            k: float(np.mean([d["components_eur"][k] for d in aa]))
            for k in aa[0]["components_eur"]
        },
        "mean_targets_kw": np.mean(
            [d["targets_kw"]["mean"] for d in aa], axis=0
        ).tolist(),
        "mean_actions": (
            np.mean([d["action"]["mean"] for d in aa], axis=0).tolist()
            if actor != "H"
            else None
        ),
        "min_actions": (
            np.min([d["action"]["min"] for d in aa], axis=0).tolist()
            if actor != "H"
            else None
        ),
        "max_actions": (
            np.max([d["action"]["max"] for d in aa], axis=0).tolist()
            if actor != "H"
            else None
        ),
        "fraction_ge_999": (
            np.mean([d["action"]["fraction_ge_999"] for d in aa], axis=0).tolist()
            if actor != "H"
            else None
        ),
        "total_cycle_deficit_count": {
            k: sum(d["cycle_deficit_count"][k] for d in aa)
            for k in aa[0]["cycle_deficit_count"]
        },
        "mean_reward_diagnostics": {
            k: float(np.mean([d["reward"][k] for d in aa])) for k in aa[0]["reward"]
        },
    }
result = {
    "input_root": str(P),
    "action_order": actionnames,
    "all_episodes": actors,
    "actor_means": mean,
    "rollout_blocks": rs,
    "rollout_extrema": {
        "min_kl": min(r["approximate_kl"] for r in rollouts),
        "max_kl": max(r["approximate_kl"] for r in rollouts),
        "max_clip_fraction": max(r["clip_fraction"] for r in rollouts),
        "sum_transitions": sum(r["transition_count"] for r in rollouts),
        "sum_completed_episodes": sum(r["completed_episode_count"] for r in rollouts),
        "sum_completed_cycles": sum(r["complete_cycle_count"] for r in rollouts),
    },
}
O.write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            "actor_means": mean,
            "rollout_blocks": rs,
            "rollout_extrema": result["rollout_extrema"],
        },
        indent=2,
    )
)
print("SAVED", O)
