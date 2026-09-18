"""Causal daily-coverage controller used to establish physical feasibility.

Research baseline, not an economic optimizer. It receives the same detached
observation and actuator authority as MPC; no realized future forcing is read.
"""

from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace

from plant_backend.rl.v2.mpc_execution import public_observation_sha256
from plant_backend.rl.v2.mpc_schema import domain_sha256
from plant_backend.rl.v2.setpoint_command import issue_process_setpoint_command
from plant_backend.rl.v2.setpoint_control import ProcessSetpointTargetsV1


@dataclass(frozen=True)
class CoverageSettings:
    solar_fraction: float = 0.95
    h2_buffer_kg: float = 5.0
    co2_buffer_kg: float = 20.0
    ch4_buffer_kg: float = 10.0
    battery_buffer_kwh: float = 100.0
    h2_energy_kwh_kg: float = 56.0


def distribute(budget, weights, limits):
    """Allocate a current power budget in proportion to remaining work."""
    allocation = [0.0] * len(weights)
    active = {i for i, weight in enumerate(weights) if weight > 1e-8}
    for _ in range(len(weights)):
        if not active or budget <= 0:
            break
        total = math.fsum(weights[i] for i in active)
        proposals = {i: budget * weights[i] / total for i in active}
        capped = {i for i in active if proposals[i] >= limits[i]}
        if not capped:
            for i in active:
                allocation[i] = proposals[i]
            break
        for i in capped:
            allocation[i] = limits[i]
            budget -= limits[i]
        active -= capped
    return allocation


class DailyCoverageController:
    id = "daily-coverage-feasibility-v2"

    def __init__(self, settings=None):
        self.settings = settings or CoverageSettings()
        self.reset()

    def reset(self):
        self.decisions = 0

    def to_dict(self):
        return {
            "id": self.id,
            "settings": asdict(self.settings),
            "information_set": "Current public plant, delivery and cycle observation; static plant and commitment profile; no future truth or scenario identity.",
            "objective": "Deliver current daily obligations and restore opening stocks before nightfall; economic prices unused.",
        }

    def compute_command(self, observation, *, constraints, config, commitment_profile):
        plant, cycle, tuning = observation.plant, observation.cycle, self.settings
        state = plant.state.boundary
        dt_h = plant.decision_interval_s / 3600
        remaining_h = (cycle.cycle_decisions - cycle.decision_in_cycle) * dt_h
        delivery = plant.delivery
        until = delivery.steps_to_next
        remaining_steps = remaining_h * 3600 / config.dt_s
        count = (
            0
            if until is None or until > remaining_steps
            else 1
            + int((remaining_steps - until + 1e-8) // delivery.delivery_period_steps)
        )
        pending = count * delivery.next_requested_kg
        methane = max(0.0, pending - cycle.ch4_margin_kg + tuning.ch4_buffer_kg)
        hydrogen = max(
            0.0,
            methane * config.methanator.h2_kg_per_kg_ch4
            - cycle.h2_margin_kg
            + tuning.h2_buffer_kg,
        )
        carbon = max(
            0.0,
            methane * config.methanator.co2_kg_per_kg_ch4
            - cycle.co2_margin_kg
            + tuning.co2_buffer_kg,
        )
        battery = max(0.0, tuning.battery_buffer_kwh - cycle.battery_margin_kwh)
        # Endogenous stored surplus can bridge a dip; opening battery is not a
        # planned energy source. The common minute dispatcher still owns support.
        solar = max(0.0, state.solar_available_kw)
        budget = tuning.solar_fraction * solar
        weights = [
            hydrogen * tuning.h2_energy_kwh_kg,
            carbon * config.dac.specific_energy_kwh_per_kg,
            methane * config.methanator.specific_energy_kwh_per_kg,
            battery / config.battery.charge_efficiency,
        ]
        limits = [
            constraints.electrolyser_limit_kw(config),
            constraints.dac_limit_kw(config),
            constraints.methanator_limit_kw(config),
            config.battery.max_charge_kw,
        ]
        # Avoid overshooting the small remaining DAC/methanator requirement.
        limits[1] = min(limits[1], weights[1] / dt_h)
        limits[2] = min(limits[2], weights[2] / dt_h)
        if state.health["dac"].offline:
            weights[1] = 0.0
        e, d, m, _ = distribute(budget, weights, limits)
        snap = plant.state.electrolyser_commitment
        mode = snap.mode.value if snap is not None else "legacy"
        start = commitment_profile.start_load_frac * config.electrolyser.rated_kw
        stable = commitment_profile.min_stable_load_frac * config.electrolyser.rated_kw
        if hydrogen < 0.1:
            e = 0.0
        elif mode in {"off", "cooldown", "cold_starting", "hot_starting"}:
            e = start if budget >= start and (state.time_s / 3600) % 24 < 16.5 else 0.0
        elif e < stable:
            # A low-power period can continue on already replenished battery.
            if (
                budget
                - d
                - m
                + max(0.0, cycle.battery_margin_kwh - tuning.battery_buffer_kwh) / dt_h
                >= stable
            ):
                e = stable
            else:
                e = 0.0
        # A started sequence is a commitment, not a fresh start decision on
        # every weather sample. Bridge cloud dips with the normal dispatcher.
        if mode in {"cold_starting", "hot_starting"}:
            e = start
        elif snap is not None and snap.min_up_remaining_s > 0:
            e = max(e, stable)
        targets = ProcessSetpointTargetsV1(e, d, m, config.battery.max_charge_kw)
        evidence = {
            "controller": self.to_dict(),
            "decision": self.decisions,
            "remaining_deliveries_kg": pending,
            "work": {"ch4_kg": methane, "h2_kg": hydrogen, "co2_kg": carbon},
        }
        command = issue_process_setpoint_command(
            targets,
            decision_index=self.decisions,
            observation_sha256=public_observation_sha256(plant),
            hold_duration_s=plant.decision_interval_s,
            issuer_id=self.id,
            source_schema="daily-coverage-feasibility-decision-v1",
            source_sha256=domain_sha256(
                "daily-coverage-feasibility-decision-v1", evidence
            ),
            constraints=constraints,
            config=config,
        )
        self.decisions += 1
        return SimpleNamespace(command=command, invalid=False, evidence=evidence)
