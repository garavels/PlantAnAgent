"""Versioned coverage-residual PPO for the frozen feasible off-grid benchmark.

This is a new policy identity. Existing scalar/MPC checkpoints are never imported
or relabelled. A zero residual reproduces the coverage controller exactly; the
learned policy has bounded operating authority, not a feasibility guarantee.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[3]
for folder in (REPO, REPO / "backend/src", REPO / "simulator/src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import gymnasium as gym
import numpy as np

from docs.research.tools.feasible_mission_controller import DailyCoverageController
from docs.research.tools.offgrid_coverage_v1 import OffgridCoverageController
from plant_backend.rl.training.config import PPOConfig
from plant_backend.rl.v2.mpc_execution import public_observation_sha256
from plant_backend.rl.v2.mpc_schema import domain_sha256
from plant_backend.rl.v2.setpoint_command import (
    issue_process_setpoint_command,
    resolve_process_setpoint_envelope,
)
from plant_backend.rl.v2.setpoint_control import ProcessSetpointTargetsV1
from plant_backend.rl.v2.training.config import BETA_PI_LAGRANGIAN_ALGORITHM_ID
from plant_backend.rl.v2.training.custody_io import (
    mkdir_exclusive,
    publish_bytes,
    publish_json,
    read_regular,
)
from plant_backend.rl.v2.training.custody_json import canonical_bytes

SCHEMA = "coverage-residual-full-mission-ppo-v1"
OPTIMIZER_SEEDS = (101, 211, 307)
TRAIN_BASE = 9_910_000
TOTAL_STEPS = 65_536
ENVIRONMENTS = 4
EPISODE_DECISIONS = 672
REWARD_SCALE = 0.001
RESIDUAL_FRACTION = 0.1
INITIAL_STD = 0.2
TARGET_NAMES = (
    "electrolyser_target_kw",
    "dac_target_kw",
    "methanator_target_kw",
    "battery_charge_ceiling_kw",
)
EXCLUDED_UNRELATED_SOURCE = REPO / "backend/src/plant_backend/mission_mpc.py"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_record(path: Path) -> dict[str, Any]:
    raw = read_regular(path)
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def actor_parameter_sha256(model: Any) -> str:
    """Bind actor weights independently of reward/cost critic updates."""
    digest = hashlib.sha256(b"coverage-ppo-actor-parameters-v1\x00")
    count = 0
    for name, tensor in sorted(model.policy.state_dict().items()):
        if name != "log_std" and not name.startswith(
            ("action_net.", "mlp_extractor.policy_net.", "features_extractor.")
        ):
            continue
        count += 1
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if not count:
        raise ValueError("Policy contained no recognized actor parameters")
    return digest.hexdigest()


def _benchmark() -> Any:
    from docs.research.tools import feasible_benchmark_v1

    return feasible_benchmark_v1


def ppo_settings() -> PPOConfig:
    return PPOConfig(gamma=1.0)


def make_coverage_prior() -> DailyCoverageController:
    """Prospectively fixed prior; never substitute a baseline during inference."""
    return OffgridCoverageController()


def source_records() -> list[dict[str, Any]]:
    from plant_backend.rl.v2.training import constrained_ppo, episode_allocator

    paths = {
        Path(__file__),
        Path(_benchmark().__file__),
        Path(sys.modules[DailyCoverageController.__module__].__file__),
        Path(sys.modules[OffgridCoverageController.__module__].__file__),
        Path(constrained_ppo.__file__),
        Path(episode_allocator.__file__),
        REPO / "docs/research/COVERAGE_PPO_V1_PROTOCOL.md",
        REPO / "uv.lock",
    }
    for directory in (
        REPO / "backend/src/plant_backend",
        REPO / "simulator/src/plant_sim",
    ):
        paths.update(directory.rglob("*.py"))
    paths.discard(EXCLUDED_UNRELATED_SOURCE)
    return [file_record(path) for path in sorted(paths)]


def interface_identity(base: Any) -> dict[str, Any]:
    from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPO

    observation_space = gym.spaces.Box(
        np.concatenate((base.observation_space.low, np.zeros(4, dtype=np.float32))),
        np.concatenate((base.observation_space.high, np.ones(4, dtype=np.float32))),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    return {
        "schema": SCHEMA,
        "benchmark_contract_sha256": base.benchmark_contract["contract_sha256"],
        "base_checkpoint_contract": base.checkpoint_contract,
        "base_checkpoint_contract_digest": base.checkpoint_contract_digest,
        "observation_size": 139,
        "action_size": 4,
        "observation_space": ConstrainedPPO._box_space_record(
            "PPO observation space", observation_space
        ),
        "action_space": ConstrainedPPO._box_space_record(
            "PPO action space", action_space
        ),
        "base_observation": "actual returned float32[135]; no extra encoding",
        "appended_features": [f"coverage_{name}_fraction" for name in TARGET_NAMES],
        "normalization": "cached target / installed rating, clipped to [0,1]",
        "residual_fraction": RESIDUAL_FRACTION,
        "mapping": (
            "E at/below stable floor is preserved; otherwise floor+(E-floor)*(1+0.1*r). "
            "Other targets multiply by (1+0.1*r); current authority caps apply. "
            "Zero residual returns the exact original command."
        ),
        "prior": make_coverage_prior().to_dict(),
        "authority": "Bounded residual only; no feasibility or safety guarantee and no policy substitution.",
        "electricity_model": base.electricity_model.to_dict(),
        "registered_evaluation_eligible": False,
    }


def _ratings(base: Any) -> np.ndarray:
    return np.asarray(
        [
            base.config.electrolyser.rated_kw,
            base.config.dac.rated_kw,
            base.config.methanator.rated_kw,
            base.config.battery.max_charge_kw,
        ],
        dtype=np.float64,
    )


def augmented_observation(vector: Any, base: Any, prior: Any | None) -> np.ndarray:
    if (
        not isinstance(vector, np.ndarray)
        or vector.dtype != np.float32
        or vector.shape != (135,)
        or not np.isfinite(vector).all()
    ):
        raise ValueError("Expected actual finite float32[135] environment observation")
    features = np.zeros(4, dtype=np.float32)
    if prior is not None:
        if prior.command.decision_index != base.decisions_taken:
            raise ValueError("Cached coverage command belongs to another boundary")
        values = [getattr(prior.command.targets, name) for name in TARGET_NAMES]
        features = np.clip(np.asarray(values) / _ratings(base), 0, 1).astype(np.float32)
    return np.concatenate((vector, features))


@dataclass(frozen=True)
class CoverageResidualDecision:
    command: Any
    invalid: bool
    evidence: dict[str, Any]


def residual_command(
    base: Any, prior: Any, action: Any, identity_sha256: str
) -> CoverageResidualDecision:
    raw = np.asarray(action)
    if raw.shape != (4,) or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(
            "Coverage residual must contain exactly four finite real values"
        )
    raw = raw.astype(np.float64)
    bounded = np.clip(raw, -1, 1)
    envelope = resolve_process_setpoint_envelope(
        constraints=base.constraints, config=base.config
    )
    if (
        prior.command.decision_index != base.decisions_taken
        or prior.command.observation_sha256
        != public_observation_sha256(base.controller_observation)
        or prior.command.envelope != envelope
    ):
        raise ValueError("Coverage command or operator authority is stale")
    nominal = prior.command.targets
    profile = base.episode_config.electrolyser_commitment
    if profile is None:
        raise ValueError("This policy requires the unchanged commitment profile")
    floor = profile.min_stable_load_frac * base.config.electrolyser.rated_kw
    zero = bool(np.all(raw == 0.0))
    targets = (
        nominal
        if zero
        else ProcessSetpointTargetsV1(
            (
                nominal.electrolyser_target_kw
                if nominal.electrolyser_target_kw <= floor
                else min(
                    envelope.electrolyser_limit_kw,
                    floor
                    + (nominal.electrolyser_target_kw - floor)
                    * (1 + RESIDUAL_FRACTION * bounded[0]),
                )
            ),
            min(
                envelope.dac_limit_kw,
                nominal.dac_target_kw * (1 + RESIDUAL_FRACTION * bounded[1]),
            ),
            min(
                envelope.methanator_limit_kw,
                nominal.methanator_target_kw * (1 + RESIDUAL_FRACTION * bounded[2]),
            ),
            min(
                envelope.battery_charge_limit_kw,
                nominal.battery_charge_ceiling_kw
                * (1 + RESIDUAL_FRACTION * bounded[3]),
            ),
        )
    )
    body = {
        "schema": SCHEMA,
        "interface_sha256": identity_sha256,
        "decision_index": base.decisions_taken,
        "nominal_coverage": prior.evidence,
        "nominal_command": prior.command.to_dict(),
        "raw_residual": raw.tolist(),
        "bounded_residual": bounded.tolist(),
        "action_clipped": bool(np.any(raw != bounded)),
        "exact_zero_command": zero,
        "electrolyser_floor_kw": floor,
        "electrolyser_low_or_commitment_command_preserved": nominal.electrolyser_target_kw
        <= floor,
        "mapped_targets": targets.to_dict(),
    }
    digest = domain_sha256(SCHEMA, body)
    command = (
        prior.command
        if zero
        else issue_process_setpoint_command(
            targets,
            decision_index=base.decisions_taken,
            observation_sha256=public_observation_sha256(base.controller_observation),
            hold_duration_s=base.episode_config.decision_interval_s,
            issuer_id=SCHEMA,
            source_schema=SCHEMA,
            source_sha256=digest,
            constraints=base.constraints,
            config=base.config,
        )
    )
    return CoverageResidualDecision(
        command,
        bool(prior.invalid or np.any(raw != bounded)),
        {**body, "sha256": digest},
    )


class CoverageResidualEnv(gym.Wrapper):
    def __init__(self, base: Any):
        super().__init__(base)
        self.base = base
        self.prior = make_coverage_prior()
        self.identity = interface_identity(base)
        self.identity_sha256 = hashlib.sha256(
            canonical_bytes(self.identity)
        ).hexdigest()
        self.observation_space = gym.spaces.Box(
            np.concatenate((base.observation_space.low, np.zeros(4, dtype=np.float32))),
            np.concatenate((base.observation_space.high, np.ones(4, dtype=np.float32))),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.cached = None
        self.last_decision: CoverageResidualDecision | None = None

    def _cache(self, vector: Any, *, terminal: bool) -> np.ndarray:
        augmented_observation(vector, self.base, None)
        self.cached = (
            None
            if terminal
            else self.prior.compute_command(
                self.base.cyclic_market_controller_observation,
                constraints=self.base.constraints,
                config=self.base.config,
                commitment_profile=self.base.episode_config.electrolyser_commitment,
            )
        )
        return augmented_observation(vector, self.base, self.cached)

    def reset(self, *, seed=None, options=None):
        self.prior.reset()
        self.cached = self.last_decision = None
        vector, info = self.base.reset(seed=seed, options=options)
        return self._cache(vector, terminal=False), info

    def step(self, action):
        if self.cached is None:
            raise RuntimeError("Reset a live episode before choosing residual actions")
        decision = residual_command(
            self.base, self.cached, action, self.identity_sha256
        )
        vector, reward, terminated, truncated, info = self.base.step_process_setpoints(
            decision.command,
            controller_invalid=decision.invalid,
        )
        self.last_decision = decision
        return (
            self._cache(vector, terminal=terminated or truncated),
            float(reward),
            terminated,
            truncated,
            info,
        )


class _ScaleReward(gym.Wrapper):
    def step(self, action):
        vector, reward, terminated, truncated, info = self.env.step(action)
        return vector, float(reward) * REWARD_SCALE, terminated, truncated, info


def make_learner(optimizer_seed: int, *, fixture_steps: int | None = None):
    import torch
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from plant_backend.rl.v2.training.constrained_ppo import (
        ConstrainedPPOSettingsV1,
        build_constrained_model,
    )
    from plant_backend.rl.v2.training.episode_allocator import EpisodeSeedAllocatorV1

    if optimizer_seed not in OPTIMIZER_SEEDS:
        raise ValueError("Optimizer seed is outside the frozen three-run plan")
    allocator = EpisodeSeedAllocatorV1.fixture(
        base_seed=TRAIN_BASE, episode_decisions=EPISODE_DECISIONS
    )

    def factory(index):
        return lambda: _ScaleReward(
            Monitor(
                allocator.wrap(
                    CoverageResidualEnv(_benchmark().BenchmarkEnv(split="train")),
                    environment_index=index,
                )
            )
        )

    vector = DummyVecEnv([factory(index) for index in range(ENVIRONMENTS)])
    try:
        settings = asdict(ppo_settings())
        if fixture_steps is not None:
            settings.update(
                n_steps=fixture_steps,
                batch_size=ENVIRONMENTS * fixture_steps,
                n_epochs=1,
            )
        model = build_constrained_model(
            vector,
            settings=ConstrainedPPOSettingsV1.for_algorithm(
                BETA_PI_LAGRANGIAN_ALGORITHM_ID, optimizer_seed=optimizer_seed
            ),
            ppo_kwargs=settings,
            verbose=1,
        )
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.log_std.fill_(math.log(INITIAL_STD))
        base = _benchmark().BenchmarkEnv(split="train")
        try:
            model.research_identity = interface_identity(base)
        finally:
            base.close()
        return model, allocator, vector
    except BaseException:
        vector.close()
        raise


def plan() -> dict[str, Any]:
    contract = _benchmark().load_benchmark_contract()
    return {
        "schema": SCHEMA,
        "benchmark_contract_sha256": contract["contract_sha256"],
        "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "physical_training_seeds": [TRAIN_BASE, TRAIN_BASE + 99],
        "same_training_scenarios_across_optimizer_seeds": True,
        "total_transitions_per_policy": TOTAL_STEPS,
        "environments": ENVIRONMENTS,
        "resets_per_policy": 100,
        "completed_episodes_per_policy": 96,
        "partial_final_episodes_per_policy": 4,
        "ppo": asdict(ppo_settings()),
        "algorithm": BETA_PI_LAGRANGIAN_ALGORITHM_ID,
        "reward_scale": REWARD_SCALE,
        "initial_gaussian_std": INITIAL_STD,
        "residual_fraction": RESIDUAL_FRACTION,
        "costs": "Unchanged four normalized daily stock deficits, zero limits; gamma=1, GAE lambda=.95.",
        "initialization": "Fresh optimizer-seeded network; exactly zero actor output head reproduces coverage.",
        "prior": make_coverage_prior().to_dict(),
        "selection": "All three fixed final endpoints retained and evaluated; no interim checkpoint selection.",
        "held_out_gate": "All three final checkpoint identities frozen before opening any held-out scenario.",
        "limitation": "Four costs do not mathematically guarantee safety or delivery. Raw EUR penalties, common guard and full-mission evaluation remain required.",
        "threads": {"torch_intraop": 1, "torch_interop": 1},
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "numpy",
                    "gymnasium",
                    "scipy",
                    "torch",
                    "stable-baselines3",
                )
            },
        },
        "excluded_source": {
            "path": str(EXCLUDED_UNRELATED_SOURCE),
            "reason": "Separately developed full-mission MPC; not imported or executed by coverage PPO. Its own evaluation freezes it independently.",
        },
        "source_files": source_records(),
        "registered_evaluation_eligible": False,
    }


def freeze_plan(root: Path) -> dict[str, Any]:
    mkdir_exclusive(root)
    record = plan()
    publish_json(root / "plan.json", record)
    return record


def _require_plan(root: Path) -> dict[str, Any]:
    recorded = json.loads(read_regular(root / "plan.json"))
    if canonical_bytes(recorded) != canonical_bytes(plan()):
        raise ValueError("Training task, settings or source changed after plan freeze")
    return recorded


def train(root: Path, optimizer_seed: int) -> dict[str, Any]:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from plant_backend.rl.v2.training.constrained_ppo import (
        ConstrainedPPO,
        policy_parameter_sha256,
    )
    from plant_backend.rl.v2.training.research_batch import training_callback
    from plant_backend.rl.v2.training.research_effort_batch import preflight_checkpoint

    frozen = _require_plan(root)
    if optimizer_seed not in OPTIMIZER_SEEDS:
        raise ValueError("Unknown frozen optimizer seed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = root / f"seed-{optimizer_seed}"
    mkdir_exclusive(out)
    mkdir_exclusive(out / "diagnostics")
    publish_json(
        out / "attempt.json",
        {
            "started_at": now(),
            "optimizer_seed": optimizer_seed,
            "plan_sha256": hashlib.sha256(canonical_bytes(frozen)).hexdigest(),
        },
    )
    model, allocator, vector = make_learner(optimizer_seed)
    started = time.monotonic()

    class Progress(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 4096 == 0:
                item = {
                    "optimizer_seed": optimizer_seed,
                    "transitions": self.num_timesteps,
                    "elapsed_s": round(time.monotonic() - started, 3),
                }
                with (out / "progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(item) + "\n")
                print(json.dumps(item), flush=True)
            return True

    try:
        buffer = io.BytesIO()
        torch.save(model.policy.state_dict(), buffer)
        publish_bytes(out / "initial-policy.pt", buffer.getvalue())
        initial_tensors = policy_parameter_sha256(model)
        initial_actor = actor_parameter_sha256(model)
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=CallbackList([training_callback(out / "diagnostics"), Progress()]),
        )
        if (
            model.num_timesteps != TOTAL_STEPS
            or allocator.decisions_consumed_by_environment
            != (TOTAL_STEPS // ENVIRONMENTS,) * ENVIRONMENTS
        ):
            raise ValueError("Learner and allocator did not reach the exact endpoint")
        if model.episode_decision_indices != tuple(
            cursor.decision_offset for cursor in allocator.cursors
        ):
            raise ValueError("Learner constraint and episode allocator cursors differ")
        ledgers = model.rollout_ledgers
        if len(ledgers) != 8 or any(
            row.transition_count != 8192 or row.optimizer_steps != 320
            for row in ledgers
        ):
            raise ValueError("Incomplete rollout or minibatch optimization ledger")
        snapshot = allocator.snapshot().to_dict()
        model.pin_external_episode_state(snapshot)
        artifact = out / "final-policy.zip"
        model.save(artifact)
        state = model.embedded_constraint_state
        preflight_checkpoint(artifact, model.research_identity, state)
        final_tensors = policy_parameter_sha256(model)
        final_actor = actor_parameter_sha256(model)
        reopened = ConstrainedPPO.load(artifact, device="cpu")
        attestation = reopened.loaded_constraint_state_attestation()
        if (
            attestation.artifact_sha256 != file_record(artifact)["sha256"]
            or attestation.num_timesteps != TOTAL_STEPS
            or attestation.constraint_state != state
        ):
            raise ValueError(
                "Saved endpoint failed exact checkpoint reopen attestation"
            )
        if (
            policy_parameter_sha256(reopened) != final_tensors
            or initial_tensors == final_tensors
            or actor_parameter_sha256(reopened) != final_actor
            or initial_actor == final_actor
        ):
            raise ValueError(
                "Reopened final tensors differ or no learned tensor changed"
            )
        _require_plan(root)
        result = {
            "schema": SCHEMA,
            "completed_at": now(),
            "optimizer_seed": optimizer_seed,
            "elapsed_s": time.monotonic() - started,
            "num_timesteps": model.num_timesteps,
            "minibatch_updates": sum(row.optimizer_steps for row in ledgers),
            "initial_checkpoint": file_record(out / "initial-policy.pt"),
            "final_checkpoint": file_record(artifact),
            "initial_tensor_sha256": initial_tensors,
            "final_tensor_sha256": final_tensors,
            "initial_actor_sha256": initial_actor,
            "final_actor_sha256": final_actor,
            "actor_parameters_changed": initial_actor != final_actor,
            "policy_identity": model.research_identity,
            "constraint_state": state,
            "allocator": snapshot,
            "checkpoint_reopen_verified": True,
            "source_files": source_records(),
            "benchmark_contract_sha256": frozen["benchmark_contract_sha256"],
            "claim": "Completed actual PPO training; policy performance and full-mission eligibility require evaluation.",
        }
        publish_json(out / "result.json", result)
        return result
    except BaseException as exc:
        publish_json(
            out / "failure.json",
            {
                "at": now(),
                "type": type(exc).__name__,
                "message": str(exc),
                "num_timesteps": model.num_timesteps,
            },
        )
        raise
    finally:
        vector.close()


def load_final(root: Path, optimizer_seed: int):
    from plant_backend.rl.v2.training.constrained_ppo import (
        ConstrainedPPO,
        policy_parameter_sha256,
    )
    from plant_backend.rl.v2.training.research_effort_batch import preflight_checkpoint

    _require_plan(root)
    out = root / f"seed-{optimizer_seed}"
    result = json.loads(read_regular(out / "result.json"))
    artifact = out / "final-policy.zip"
    if (
        result["schema"] != SCHEMA
        or result["optimizer_seed"] != optimizer_seed
        or file_record(artifact) != result["final_checkpoint"]
    ):
        raise ValueError("Final checkpoint provenance differs")
    preflight_checkpoint(
        artifact, result["policy_identity"], result["constraint_state"]
    )
    model = ConstrainedPPO.load(artifact, device="cpu")
    if (
        model.num_timesteps != TOTAL_STEPS
        or model.research_identity != result["policy_identity"]
        or model.embedded_constraint_state != result["constraint_state"]
        or policy_parameter_sha256(model) != result["final_tensor_sha256"]
        or actor_parameter_sha256(model) != result["final_actor_sha256"]
    ):
        raise ValueError("Loaded policy identity, training endpoint or tensors differ")
    return model


def load_initial(root: Path, optimizer_seed: int):
    """Load exact saved initial tensors for the same benchmark, without a reset."""
    import torch
    from plant_backend.rl.v2.training.constrained_ppo import policy_parameter_sha256

    _require_plan(root)
    out = root / f"seed-{optimizer_seed}"
    result = json.loads(read_regular(out / "result.json"))
    artifact = out / "initial-policy.pt"
    if file_record(artifact) != result["initial_checkpoint"]:
        raise ValueError("Initial policy checkpoint bytes changed")
    model, _, vector = make_learner(optimizer_seed)
    try:
        model.policy.load_state_dict(
            torch.load(
                io.BytesIO(read_regular(artifact)),
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        if (
            policy_parameter_sha256(model) != result["initial_tensor_sha256"]
            or actor_parameter_sha256(model) != result["initial_actor_sha256"]
        ):
            raise ValueError("Initial policy tensors differ from training admission")
        if (
            torch.count_nonzero(model.policy.action_net.weight).item()
            or torch.count_nonzero(model.policy.action_net.bias).item()
        ):
            raise ValueError("Initial actor no longer reproduces exact coverage")
        return model
    finally:
        vector.close()


def freeze_endpoints(root: Path) -> dict[str, Any]:
    _require_plan(root)
    records = []
    for seed in OPTIMIZER_SEEDS:
        load_final(root, seed)
        records.append(
            {
                "optimizer_seed": seed,
                "result": file_record(root / f"seed-{seed}/result.json"),
                "checkpoint": file_record(root / f"seed-{seed}/final-policy.zip"),
            }
        )
    body = {
        "schema": SCHEMA,
        "frozen_at": now(),
        "endpoints": records,
        "selection": "All three fixed final endpoints; no exclusions.",
    }
    publish_json(root / "endpoints-frozen.json", body)
    return body


def require_frozen_endpoints(root: Path) -> dict[str, Any]:
    _require_plan(root)
    body = json.loads(read_regular(root / "endpoints-frozen.json"))
    if [row["optimizer_seed"] for row in body["endpoints"]] != list(OPTIMIZER_SEEDS):
        raise ValueError("Held-out gate lacks all three fixed final endpoints")
    for row in body["endpoints"]:
        for field in ("result", "checkpoint"):
            if file_record(Path(row[field]["path"])) != row[field]:
                raise ValueError("Final endpoint changed after held-out freeze")
    return body


class CoveragePPOController:
    """Public controller adapter for the common benchmark evaluator."""

    id = SCHEMA

    def __init__(self, env: Any, model: Any):
        self.env, self.model = env, model
        self.identity = interface_identity(env)
        if model.research_identity != self.identity:
            raise ValueError("Model is not bound to this coverage/economic task")
        self.identity_sha256 = hashlib.sha256(
            canonical_bytes(self.identity)
        ).hexdigest()
        self.prior = make_coverage_prior()
        self.vector = None
        self.decision_index = None

    def reset(self):
        self.prior.reset()
        self.vector = self.decision_index = None

    def set_observation(self, vector):
        augmented_observation(vector, self.env, None)
        self.vector = vector.copy()
        self.decision_index = self.env.decisions_taken

    def compute_command(self, observation, **kwargs):
        if self.vector is None or self.decision_index != self.env.decisions_taken:
            raise ValueError(
                "Supply the actual current reset/step vector before inference"
            )
        if observation.sha256 != self.env.cyclic_market_controller_observation.sha256:
            raise ValueError(
                "Public policy observation differs from the current boundary"
            )
        prior = self.prior.compute_command(observation, **kwargs)
        vector = augmented_observation(self.vector, self.env, prior)
        action, _ = self.model.predict(vector.copy(), deterministic=True)
        decision = residual_command(self.env, prior, action, self.identity_sha256)
        self.vector = None
        return CoverageResidualDecision(
            decision.command,
            decision.invalid,
            {
                **decision.evidence,
                "policy_observation": vector.tolist(),
                "optimizer_seed": self.model.seed,
            },
        )

    def to_dict(self):
        return {
            "id": self.id,
            "optimizer_seed": self.model.seed,
            "training_transitions": self.model.num_timesteps,
            "policy_identity": self.identity,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze-plan", "train", "freeze-endpoints"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--optimizer-seed", type=int, choices=OPTIMIZER_SEEDS)
    args = parser.parse_args()
    if args.command == "freeze-plan":
        result = freeze_plan(args.root.resolve())
    elif args.command == "freeze-endpoints":
        result = freeze_endpoints(args.root.resolve())
    else:
        if args.optimizer_seed is None:
            parser.error("train requires --optimizer-seed")
        result = train(args.root.resolve(), args.optimizer_seed)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
