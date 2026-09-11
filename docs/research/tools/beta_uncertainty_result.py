"""Summarize the retained delivery-repair fixture; no reset or solver call."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
import sys
from typing import Any

from plant_backend.rl.v2.beta_controller_benchmark_verifier import (
    _decode_beta_observation,
    _parse_beta_settings,
)
from plant_backend.rl.v2.beta_mpc_model import derive_beta_mpc_cycle_return
from plant_backend.rl.v2.beta_mpc_uncertainty import project_beta_mpc_solar
from plant_backend.rl.v2.mpc_forecast import project_issued_mean_solar
from plant_backend.rl.v2.mpc_model import infer_delivery_targets
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.training.custody_json import (
    authenticate,
    canonical_bytes,
    decode,
)


def summarize(root: Path) -> dict[str, Any]:
    raw = (root / "report.json").read_bytes()
    report = decode(raw)
    if report != authenticate(report):
        raise ValueError("fixture report digest differs")
    if [row["seed"] for row in report["rows"]] != [9100001, 9100002, 9100003]:
        raise ValueError("unexpected fixture seeds")
    rows = []
    for row in report["rows"]:
        verified = row["verification"]
        archive = verified["arm_archive"]
        compressed = (
            root / str(row["seed"]) / "evidence/synthetic-arm.json.gz"
        ).read_bytes()
        if hashlib.sha256(compressed).hexdigest() != archive["file"]["sha256"]:
            raise ValueError("retained archive digest differs")
        decoded = gzip.decompress(compressed)
        if hashlib.sha256(decoded).hexdigest() != archive["uncompressed_sha256"]:
            raise ValueError("decoded archive digest differs")
        arm = decode(decoded)
        cfg = ControllerNeutralTrainingConfigV1.from_dict(arm["training_task"])
        boundary = arm["run"]["evidence"]["boundary_ledger"]["decision_boundaries"][0]
        obs = _decode_beta_observation(boundary["typed_observation"], name="saved")
        if obs.plant.forecast is None:
            raise ValueError("saved public solar forecast is required")
        settings = _parse_beta_settings(arm["run"]["controller"]["settings"])
        if obs.plant.state.boundary.step != 0 or cfg.beta_task is None:
            raise ValueError("bound is scoped to the opening beta boundary")
        plant = cfg.plant
        if (
            obs.plant.state.boundary.time_s,
            plant.solar.sunrise_h,
            plant.solar.sunset_h,
        ) != (21600.0, 6.0, 19.5):
            raise ValueError("bound requires the disclosed daylight geometry")
        targets = infer_delivery_targets(
            obs.plant, config=plant, interval_s=900.0, horizon_intervals=96
        )
        if [(t.due_boundary, t.requested_kg) for t in targets] != [
            (24, 240.0),
            (48, 240.0),
            (72, 240.0),
            (96, 240.0),
        ]:
            raise ValueError("bound requires the disclosed delivery geometry")
        cycle = derive_beta_mpc_cycle_return(
            obs, settings=settings, beta_task=cfg.beta_task, config=plant
        )
        methane = cycle.resource("ch4")
        # Positive final inventory forces both nighttime shipments to be full.
        # A partial 18:00 shipment empties the tank; even full-rated production
        # until sunset cannot then supply those shipments plus the return.
        later_need = 480.0 + methane.required_minimum
        later_cap = (
            1.5
            * plant.methanator.rated_kw
            / plant.methanator.specific_energy_kwh_per_kg
        )
        if methane.required_minimum <= 0 or later_cap >= later_need:
            raise ValueError("mandatory 18:00 shipment has not been established")
        minimum_delivery = 720.0 + min(methane.current, 240.0)
        minimum_production = (
            minimum_delivery + methane.required_minimum - methane.current
        )
        h2 = cycle.resource("h2")
        co2 = cycle.resource("co2")
        h2_production = max(
            0.0,
            plant.methanator.h2_kg_per_kg_ch4 * minimum_production
            + h2.required_minimum
            - h2.current,
        )
        co2_production = max(
            0.0,
            plant.methanator.co2_kg_per_kg_ch4 * minimum_production
            + co2.required_minimum
            - co2.current,
        )
        profile = cfg.episode.electrolyser_commitment
        if profile is None:
            raise ValueError("bound requires the declared electrolyser profile")
        specific = max(
            plant.electrolyser.specific_energy_kwh_per_kg,
            profile.nominal_specific_energy_kwh_per_kg,
        ) * max(m for _, m in profile.specific_energy_multiplier_anchors)
        electricity = (
            minimum_production * plant.methanator.specific_energy_kwh_per_kg
            + h2_production * specific
            + co2_production * plant.dac.specific_energy_kwh_per_kg
        )
        battery = cycle.resource("battery")
        allowed_battery = (
            max(0.0, battery.current - battery.required_minimum)
            * plant.battery.discharge_efficiency
        )
        p10 = project_beta_mpc_solar(
            obs.plant.forecast, boundary_time_s=21600.0, config=plant, settings=settings
        )
        mean = project_issued_mean_solar(
            obs.plant.forecast, boundary_time_s=21600.0, config=plant, interval_s=900.0
        )
        p10_energy = sum(p10.available_kw) * 0.25
        mean_energy = sum(mean.available_kw) * 0.25
        rows.append(
            {
                "seed": row["seed"],
                "process": row["process"],
                "failure_codes": row["failure_codes"],
                "verification_error": row["verification_error"],
                "arm_archive": archive,
                "timing_archive": verified["timing_archive"],
                "classification": verified["physical"]["classification"],
                "public_energy_bound": {
                    "typed_observation": boundary["typed_observation"],
                    "cycle_return": cycle.to_dict(),
                    "maximum_post_18h_methane_kg": later_cap,
                    "required_for_night_shipments_and_return_kg": later_need,
                    "minimum_delivery_kg": minimum_delivery,
                    "minimum_ch4_production_kg": minimum_production,
                    "minimum_h2_production_kg": h2_production,
                    "minimum_co2_production_kg": co2_production,
                    "model_h2_specific_energy_kwh_per_kg": specific,
                    "minimum_process_energy_kwh": electricity,
                    "allowed_battery_endowment_use_kwh": allowed_battery,
                    "issued_p10_energy_kwh": p10_energy,
                    "issued_mean_energy_kwh": mean_energy,
                    "p10_energy_deficit_lower_bound_kwh": electricity
                    - allowed_battery
                    - p10_energy,
                    "mean_energy_deficit_lower_bound_kwh": electricity
                    - allowed_battery
                    - mean_energy,
                    "scope": "Necessary bound for issued scenarios; not future-weather truth or a physical-task infeasibility proof. Ignores added buffers, constant-power constraints, startup costs and losses.",
                },
            }
        )
    return {
        "schema": "plant-beta-mpc-delivery-repair-result-v1",
        "root": str(root),
        "source": report["plan"]["identity"]["source"],
        "report_sha256": report["sha256"],
        "report_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": rows,
        "registered_execution_authorized": False,
        "result_claim": None,
        "summary_resets": 0,
        "summary_solver_calls": 0,
    }


if __name__ == "__main__":
    sys.stdout.buffer.write(canonical_bytes(summarize(Path(sys.argv[1]))))
