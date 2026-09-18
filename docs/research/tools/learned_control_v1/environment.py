"""Seed admission and episode mixing around immutable, existing environments.

This module does not implement plant transitions, observations, rewards, fault
schedules, or weather. Ordinary episodes use WholePlantEnv directly and every
other condition uses TailStressEnv directly. Only admitted random seeds and the
distribution from which training episodes are selected are new.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from numbers import Integral
from pathlib import Path

import gymnasium as gym
import numpy as np

from docs.research.tools.wholeplant_ppo_v1 import environment as ordinary
from docs.research.tools.tail_stress_v1 import environment as tail

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "learned-control-fixed-environment-v1"
ENVIRONMENT_ID = "plant-learned-control-fixed-environments-v1"
TRAIN_BASE = 50_000_000
TRAIN_STOP = 51_000_000
DEVELOPMENT_SEEDS = tuple(range(51_000_000, 51_000_016))
FINAL_SEEDS = tuple(range(52_000_000, 52_000_032))
CONDITIONS = ("ordinary", *tail.CONDITIONS)
STRESS_CONDITIONS = tuple(c for c in tail.CONDITIONS if c != "control")
DEFAULT_MIXTURE = {"ordinary": 0.5, **{c: 0.1 for c in STRESS_CONDITIONS}}
canonical = ordinary.canonical


def _digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def condition_spec(condition):
    if condition == "ordinary":
        return {"family": "ordinary", "onset_h": None, "duration_h": 0.0,
                "end_h": None, "random_component_faults": True}
    return tail.condition_spec(condition)


def environment_identity():
    """Scientific identity, including unchanged original environment digests."""
    return {
        "schema": SCHEMA,
        "environment_id": ENVIRONMENT_ID,
        "ordinary_environment_identity_sha256": ordinary.environment_identity_sha256(),
        "tail_environment_identity_sha256": tail.environment_identity_sha256(),
        "arm": "shared-dynamic",
        "conditions": {c: condition_spec(c) for c in CONDITIONS},
        "condition_implementation": {
            "ordinary": "wholeplant_ppo_v1.environment.WholePlantEnv, with original random DAC faults",
            "control_and_stress": "tail_stress_v1.environment.TailStressEnv, with unchanged prescribed schedules",
        },
        "changes": ["seed admission", "training condition sampling probabilities"],
        "unchanged": ["physics", "capacities", "episode duration", "delivery schedule",
                      "scalar economic reward", "observations", "actions", "fault schedules",
                      "weather generation", "initial-state distribution"],
        "observation": ordinary.environment_identity()["observation"],
        "action": ordinary.environment_identity()["action"],
        "episode_decisions": 672,
        "horizon_h": 168.0,
        "condition_visible_to_policy": False,
        "default_training_mixture": dict(DEFAULT_MIXTURE),
        "splits": {
            "train": {"first": TRAIN_BASE, "stop_exclusive": TRAIN_STOP},
            "development": list(DEVELOPMENT_SEEDS),
            "final": list(FINAL_SEEDS),
        },
        "final_gate": "pinned selected recipe and matching source/environment identity required before any final reset",
    }


def environment_identity_sha256():
    return _digest(environment_identity())


def source_records():
    """Include inherited frozen simulator sources and both original adapters."""
    paths = {Path(row["path"]).resolve() for row in ordinary.source_records()}
    paths.update({Path(tail.__file__).resolve(), Path(tail.__file__).with_name("__init__.py").resolve(),
                  ROOT / "docs/research/TAIL_STRESS_V1_PROTOCOL.md",
                  Path(__file__).resolve(), Path(__file__).with_name("__init__.py").resolve()})
    return [{"path": str(p), "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)]


def source_identity_sha256():
    records = [{**row, "path": Path(row["path"]).relative_to(ROOT).as_posix()}
               for row in source_records()]
    return _digest(records)


def _seeds_for_split(split):
    if split == "train":
        return range(TRAIN_BASE, TRAIN_STOP)
    if split == "development":
        return DEVELOPMENT_SEEDS
    if split == "final":
        return FINAL_SEEDS
    raise ValueError(f"Unknown learned-control split: {split}")


def _verify_final_gate(path):
    if path is None:
        raise ValueError("Final seeds require a frozen selected-recipe gate")
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    recipe = payload.get("selected_recipe", {})
    if (payload.get("schema") != "learned-control-v1-evaluation-gate"
            or payload.get("environment_identity_sha256") != environment_identity_sha256()
            or payload.get("source_identity_sha256") != source_identity_sha256()
            or "final" not in payload.get("allowed_splits", [])
            or payload.get("selected_recipe_frozen") is not True
            or not isinstance(recipe, dict)
            or not isinstance(recipe.get("path"), str)
            or not isinstance(recipe.get("sha256"), str)):
        raise ValueError("Final gate does not admit this environment, source and selected recipe")
    recipe_path = Path(recipe["path"])
    if not recipe_path.is_absolute():
        recipe_path = path.parent / recipe_path
    if (not recipe_path.is_file()
            or hashlib.sha256(recipe_path.read_bytes()).hexdigest() != recipe["sha256"]):
        raise ValueError("Selected recipe differs from the final gate")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FixedEnv(gym.Wrapper):
    """An unchanged existing condition with new, disjoint seed admission.

The wrapped environment retains its existing reset, sample and step methods.
Its ``allowed_seeds`` is the sole physical-environment attribute reassigned.
The wrapper's condition, seed split and gate are audit metadata, never inputs
to the policy's observation vector.
    """

    def __init__(self, condition="ordinary", split="train", *, gate_path=None,
                 record_minutes=False):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown learned-control condition: {condition}")
        admitted = _seeds_for_split(split)
        self.condition = condition
        self.split = split
        self.gate_path = None if gate_path is None else Path(gate_path).resolve()
        self._gate_sha256 = _verify_final_gate(self.gate_path) if split == "final" else None
        physical = (ordinary.WholePlantEnv(arm="shared-dynamic", split="train", record_minutes=record_minutes)
                    if condition == "ordinary" else tail.TailStressEnv(condition, record_minutes=record_minutes))
        physical.allowed_seeds = admitted
        super().__init__(physical)
        self.allowed_seeds = admitted
        self._last_seed = None

    def __getattr__(self, name):
        # Gymnasium deliberately stopped generic Wrapper forwarding; the audit
        # and existing baseline APIs require their original read-only attributes.
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, *, seed=None, options=None):
        if self.split == "final":
            if _verify_final_gate(self.gate_path) != self._gate_sha256:
                raise ValueError("Final gate changed after environment admission")
        selected_seed = self._last_seed if seed is None else seed
        if (isinstance(selected_seed, bool) or not isinstance(selected_seed, Integral)
                or selected_seed not in self.allowed_seeds):
            raise ValueError(f"Seed {selected_seed} is outside learned-control split {self.split}")
        self._last_seed = int(selected_seed)
        observation, info = self.env.reset(seed=self._last_seed, options=options)
        info = dict(info)
        info["learned_control_admission"] = {"split": self.split, "seed": self._last_seed,
                                            "condition": self.condition}
        return observation, info

    def audit_manifest(self):
        payload = deepcopy(self.env.audit_manifest())
        payload.pop("manifest_sha256")
        payload["learned_control_admission"] = {
            "environment_identity_sha256": environment_identity_sha256(),
            "condition": self.condition, "split": self.split, "seed": self._last_seed,
            "gate_sha256": self._gate_sha256,
        }
        return {"manifest_sha256": _digest(payload), **payload}


class MixtureEnv(gym.Env):
    """Training-only condition mixture, resampled independently at each reset.

Two deterministic random streams choose the condition and scenario seed. This
keeps the scenario-seed stream independent of the chosen mixture weights. Child
environments are reused, but each episode invokes their full original reset.
    """
    metadata = {}

    def __init__(self, weights=None, *, sampler_seed=0, record_minutes=False):
        values = dict(DEFAULT_MIXTURE if weights is None else weights)
        if not values or any(c not in CONDITIONS for c in values):
            raise ValueError("Mixture contains an unknown condition or is empty")
        if any(isinstance(w, bool) or not np.isfinite(w) or w < 0 for w in values.values()):
            raise ValueError("Mixture weights must be finite nonnegative numbers")
        total = float(sum(values.values()))
        if total <= 0:
            raise ValueError("At least one mixture weight must be positive")
        self.weights = {c: float(values[c]) / total for c in CONDITIONS if values.get(c, 0) > 0}
        self.conditions = tuple(self.weights)
        self.probabilities = np.asarray(list(self.weights.values()), dtype=float)
        self.record_minutes = bool(record_minutes)
        self.split = "train"
        self.allowed_seeds = range(TRAIN_BASE, TRAIN_STOP)
        self._children = {}
        self._active = self._child(self.conditions[0])
        self.action_space = self._active.action_space
        self.observation_space = self._active.observation_space
        self._set_sampler(sampler_seed)
        self.episode_index = 0
        self.training_scenario = None

    def _set_sampler(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("Sampler seed must be a nonnegative integer")
        self.sampler_seed = int(seed)
        streams = np.random.SeedSequence(self.sampler_seed).spawn(2)
        self._condition_rng, self._scenario_rng = (np.random.default_rng(s) for s in streams)

    def _child(self, condition):
        if condition not in self._children:
            self._children[condition] = FixedEnv(condition, split="train", record_minutes=self.record_minutes)
        return self._children[condition]

    def __getattr__(self, name):
        if name in {"_active", "_children"}:
            raise AttributeError(name)
        return getattr(self._active, name)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._set_sampler(seed)
            self.episode_index = 0
        condition = str(self._condition_rng.choice(self.conditions, p=self.probabilities))
        scenario_seed = int(self._scenario_rng.integers(TRAIN_BASE, TRAIN_STOP))
        self._active = self._child(condition)
        observation, info = self._active.reset(seed=scenario_seed, options=options)
        self.training_scenario = {"seed": scenario_seed, "condition": condition,
                                  "episode_index": self.episode_index}
        self.episode_index += 1
        info = dict(info)
        info["training_scenario"] = dict(self.training_scenario)
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self._active.step(action)
        info = dict(info)
        info["training_scenario"] = dict(self.training_scenario)
        return observation, reward, terminated, truncated, info

    def close(self):
        for child in self._children.values():
            child.close()
