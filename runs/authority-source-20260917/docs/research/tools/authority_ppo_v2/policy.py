"""Two prospectively specified residual authority interfaces for hybrid PPO.

The narrow arm reuses the exact prior experiment's command mapping. The broad
arm expands authority using bounded additive requests and public commitment
state only. Neither arm changes physics or claims to ensure mission success.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
for folder in (ROOT, ROOT / "backend/src", ROOT / "simulator/src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import gymnasium as gym
import numpy as np
from docs.research.tools import coverage_ppo_v1 as legacy
from plant_backend.rl.v2.mpc_execution import public_observation_sha256
from plant_backend.rl.v2.mpc_schema import domain_sha256
from plant_backend.rl.v2.setpoint_command import (
    issue_process_setpoint_command, resolve_process_setpoint_envelope,
)
from plant_backend.rl.v2.setpoint_control import ProcessSetpointTargetsV1
from plant_backend.rl.v2.training.custody_json import canonical_bytes

SCHEMA = "coverage-authority-full-mission-ppo-v2"
ARMS = ("narrow", "broad")
BROAD_FRACTION = 0.30
TARGET_NAMES = legacy.TARGET_NAMES
AuthorityResidualDecision = legacy.CoverageResidualDecision
make_coverage_prior = legacy.make_coverage_prior
augmented_observation = legacy.augmented_observation
actor_parameter_sha256 = legacy.actor_parameter_sha256


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"Unknown authority arm: {arm}")


def interface_identity(base: Any, arm: str) -> dict[str, Any]:
    _check_arm(arm)
    identity = legacy.interface_identity(base)
    identity.update({
        "schema": SCHEMA,
        "arm": arm,
        "residual_fraction": legacy.RESIDUAL_FRACTION if arm == "narrow" else BROAD_FRACTION,
        "authority": "Hybrid controller with bounded learned requests and explicit command projections; no mission-feasibility guarantee.",
        "exploration_comparison": "Identical action-space Gaussian initialization; physical exploration differs because authority mappings differ.",
    })
    if arm == "broad":
        identity["mapping"] = (
            "Request prior target + 0.30*installed rating*clip(action,-1,1), "
            "then clip to current authority. Electrolyser: offline/cooldown/min-down "
            "requests project to zero; ongoing startup requires start load; running "
            "minimum-up requires stable load. Otherwise nearest member of "
            "{0} union [start,limit] when off and {0} union [stable,limit] when active "
            "(midpoint ties to zero). Mandatory/start/stable projection can exceed "
            "the requested additive range and is recorded. If authority is below "
            "required floor, issue zero and report commitment-authority conflict. "
            "All-zero action returns the exact original prior command."
        )
    # SB3 serializes JSON-compatible identities through JSON, converting tuple
    # fields to lists. Canonicalize before binding so reopened models compare
    # exactly to live interfaces as well as persisted result manifests.
    return json.loads(canonical_bytes(identity))


def project_electrolyser(
    requested_kw: float, *, mode: str, start_kw: float, stable_kw: float,
    limit_kw: float, min_up_remaining_s: float = 0,
    min_down_remaining_s: float = 0, offline: bool = False,
) -> tuple[float, str]:
    """Project a request using public current commitment state, without lookahead."""
    clipped = float(np.clip(requested_kw, 0, limit_kw))
    if offline:
        return 0.0, "offline"
    if mode == "cooldown":
        return 0.0, "cooldown"
    if mode == "off" and min_down_remaining_s > 0:
        return 0.0, "minimum_down"
    if mode in {"cold_starting", "hot_starting"}:
        return (start_kw, "mandatory_startup") if limit_kw >= start_kw else (0.0, "commitment_authority_conflict")
    if min_up_remaining_s > 0:
        return (max(stable_kw, clipped), "mandatory_minimum_up") if limit_kw >= stable_kw else (0.0, "commitment_authority_conflict")
    floor = start_kw if mode == "off" else stable_kw
    if clipped == 0:
        return 0.0, "zero"
    if limit_kw < floor:
        return 0.0, "authority_below_operating_floor"
    if clipped >= floor:
        return clipped, "within_feasible_range"
    if clipped <= floor / 2:
        return 0.0, "nearest_feasible_off"
    return floor, "nearest_feasible_floor"


def residual_command(base: Any, prior: Any, action: Any, identity_sha256: str, arm: str) -> AuthorityResidualDecision:
    _check_arm(arm)
    # Reuse original validation of shape, scalar type, finite values, current
    # observation, boundary and operator authority for both arms.
    narrow = legacy.residual_command(base, prior, action, identity_sha256)
    if arm == "narrow":
        return AuthorityResidualDecision(narrow.command, narrow.invalid, {
            **narrow.evidence, "authority_arm": arm,
            "projection_kind": "legacy_multiplicative_authority_cap",
        })
    raw = np.asarray(action, dtype=np.float64)
    bounded = np.clip(raw, -1, 1)
    zero = bool(np.all(raw == 0))
    envelope = resolve_process_setpoint_envelope(constraints=base.constraints, config=base.config)
    nominal = np.asarray([getattr(prior.command.targets, name) for name in TARGET_NAMES], dtype=np.float64)
    ratings = legacy._ratings(base)
    requested = nominal + BROAD_FRACTION * ratings * bounded
    limits = np.asarray([envelope.electrolyser_limit_kw, envelope.dac_limit_kw,
                         envelope.methanator_limit_kw, envelope.battery_charge_limit_kw])
    authority_clipped = np.clip(requested, 0, limits)
    projected = authority_clipped.copy()
    plant = base.cyclic_market_controller_observation.plant
    snap = plant.state.electrolyser_commitment
    if snap is None:
        raise ValueError("Broad authority requires public electrolyser commitment state")
    profile = base.episode_config.electrolyser_commitment
    start = profile.start_load_frac * ratings[0]
    floor = profile.min_stable_load_frac * ratings[0]
    projected[0], reason = project_electrolyser(
        requested[0], mode=snap.mode.value, start_kw=start, stable_kw=floor,
        limit_kw=limits[0], min_up_remaining_s=snap.min_up_remaining_s,
        min_down_remaining_s=snap.min_down_remaining_s,
        offline=plant.state.boundary.health["electrolyser"].offline,
    )
    if zero:
        projected = nominal.copy()
        reason = "exact_zero_prior_preserved"
    targets = prior.command.targets if zero else ProcessSetpointTargetsV1(*map(float, projected))
    body = {
        "schema": SCHEMA, "interface_sha256": identity_sha256, "authority_arm": arm,
        "decision_index": base.decisions_taken,
        "nominal_coverage": prior.evidence, "nominal_command": prior.command.to_dict(),
        "raw_residual": raw.tolist(), "bounded_residual": bounded.tolist(),
        "action_clipped": bool(np.any(raw != bounded)), "exact_zero_command": zero,
        "installed_ratings_kw": ratings.tolist(), "requested_targets_kw": requested.tolist(),
        "authority_clipped_targets_kw": authority_clipped.tolist(),
        "authority_projection_kw": (authority_clipped - requested).tolist(),
        "commitment_projection_kw": (projected - authority_clipped).tolist(),
        "projection_applied": bool(np.any(projected != requested)),
        "electrolyser_mode": snap.mode.value, "electrolyser_projection_reason": reason,
        "electrolyser_start_kw": start, "electrolyser_floor_kw": floor,
        "electrolyser_min_up_remaining_s": snap.min_up_remaining_s,
        "electrolyser_min_down_remaining_s": snap.min_down_remaining_s,
        "commitment_authority_conflict": reason == "commitment_authority_conflict",
        "prior_zero_target_enabled": ((nominal == 0) & (projected > 0)).tolist(),
        "mapped_targets": targets.to_dict(),
    }
    digest = domain_sha256(SCHEMA, body)
    command = prior.command if zero else issue_process_setpoint_command(
        targets, decision_index=base.decisions_taken,
        observation_sha256=public_observation_sha256(base.controller_observation),
        hold_duration_s=base.episode_config.decision_interval_s,
        issuer_id=f"{SCHEMA}-{arm}", source_schema=SCHEMA, source_sha256=digest,
        constraints=base.constraints, config=base.config,
    )
    return AuthorityResidualDecision(command, bool(prior.invalid or np.any(raw != bounded)), {**body, "sha256": digest})


class AuthorityResidualEnv(legacy.CoverageResidualEnv):
    def __init__(self, base: Any, arm: str):
        _check_arm(arm)
        super().__init__(base)
        self.arm = arm
        self.identity = interface_identity(base, arm)
        self.identity_sha256 = hashlib.sha256(canonical_bytes(self.identity)).hexdigest()

    def step(self, action):
        if self.cached is None:
            raise RuntimeError("Reset a live episode before choosing residual actions")
        decision = residual_command(self.base, self.cached, action, self.identity_sha256, self.arm)
        vector, reward, terminated, truncated, info = self.base.step_process_setpoints(
            decision.command, controller_invalid=decision.invalid,
        )
        self.last_decision = decision
        return self._cache(vector, terminal=terminated or truncated), float(reward), terminated, truncated, info


class AuthorityPPOController(legacy.CoveragePPOController):
    id = SCHEMA

    def __init__(self, env: Any, model: Any, arm: str | None = None):
        self.env, self.model = env, model
        self.arm = arm or model.research_identity["arm"]
        self.identity = interface_identity(env, self.arm)
        if model.research_identity != self.identity:
            raise ValueError("Model is not bound to this authority/economic task")
        self.identity_sha256 = hashlib.sha256(canonical_bytes(self.identity)).hexdigest()
        self.prior = make_coverage_prior()
        self.vector = self.decision_index = None

    def compute_command(self, observation, **kwargs):
        if self.vector is None or self.decision_index != self.env.decisions_taken:
            raise ValueError("Supply actual current reset/step vector before inference")
        if observation.sha256 != self.env.cyclic_market_controller_observation.sha256:
            raise ValueError("Public policy observation differs from current boundary")
        prior = self.prior.compute_command(observation, **kwargs)
        vector = augmented_observation(self.vector, self.env, prior)
        import torch
        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(vector.copy())
            raw_mean = self.model.policy.get_distribution(tensor).distribution.mean[0].cpu().numpy().copy()
        action, _ = self.model.predict(vector.copy(), deterministic=True)
        decision = residual_command(self.env, prior, action, self.identity_sha256, self.arm)
        self.vector = None
        return AuthorityResidualDecision(decision.command, decision.invalid, {
            **decision.evidence, "policy_observation": vector.tolist(), "optimizer_seed": self.model.seed,
            "deterministic_raw_action_mean": raw_mean.tolist(),
            "deterministic_action_clipped": bool(np.any(raw_mean != action)),
            "action_delivery": "SB3 bounds deterministic mean to policy action space before command mapping",
        })

    def to_dict(self):
        return {**super().to_dict(), "authority_arm": self.arm}
