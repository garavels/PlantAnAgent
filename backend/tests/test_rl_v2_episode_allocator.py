"""Fixture-only tests for deterministic constrained-training seed custody."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete

from plant_backend.rl.v2.training.episode_allocator import (
    EpisodeSeedAllocatorV1,
    EpisodeSeedCustodyError,
    EpisodeSeedExhaustedError,
    EpisodeSeedSplitV1,
    FIXTURE_SEED_MAX,
    FIXTURE_SEED_MIN,
    RegisteredResumeForbiddenError,
    SEED_DOMAINS,
    VECTOR_ENV_COUNT,
    canonical_sha256,
    classify_episode_seed,
    episode_seed_for,
    find_seed_allocator,
)


FIXTURE_BASE = FIXTURE_SEED_MIN
FIXTURE_HORIZON = 2


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


FIXTURE_PARENT_CHECKPOINT_SHA256 = _sha("fixture-parent-checkpoint")


def _export_fixture_resume(allocator: EpisodeSeedAllocatorV1) -> dict[str, Any]:
    return allocator.export_fixture_resume(
        parent_checkpoint_sha256=FIXTURE_PARENT_CHECKPOINT_SHA256
    )


def _restore_fixture_resume(
    payload: dict[str, Any],
    *,
    expected_resume_sha256: str | None = None,
    expected_parent_checkpoint_sha256: str = FIXTURE_PARENT_CHECKPOINT_SHA256,
) -> EpisodeSeedAllocatorV1:
    return EpisodeSeedAllocatorV1.restore_fixture(
        payload,
        expected_parent_checkpoint_sha256=expected_parent_checkpoint_sha256,
        expected_resume_sha256=(
            payload["sha256"]
            if expected_resume_sha256 is None
            else expected_resume_sha256
        ),
    )


class _AuditableFixtureEnv(gym.Env[np.ndarray, int]):
    metadata: dict[str, Any] = {}
    observation_space = Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = Discrete(1)

    def __init__(
        self,
        *,
        horizon: int = FIXTURE_HORIZON,
        report_wrong_seed: bool = False,
        manifest_salt: str = "",
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.report_wrong_seed = report_wrong_seed
        self.manifest_salt = manifest_salt
        self.received_seeds: list[int] = []
        self._episode_seed: int | None = None
        self._steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is None:
            raise AssertionError("custody wrapper failed to pass an explicit seed")
        super().reset(seed=seed)
        self.received_seeds.append(seed)
        self._episode_seed = seed
        self._steps = 0
        return np.zeros(1, dtype=np.float32), {"public": True}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        self._steps += 1
        return (
            np.asarray([self._steps / self.horizon], dtype=np.float32),
            0.0,
            self._steps == self.horizon,
            False,
            {"public": True},
        )

    def audit_manifest(self) -> dict[str, Any]:
        if self._episode_seed is None:
            raise AssertionError("reset must precede audit_manifest")
        manifest_seed = self._episode_seed + int(self.report_wrong_seed)
        return {
            "manifest_sha256": _sha(
                f"manifest-{self._episode_seed}-{self.manifest_salt}"
            ),
            "alpha_physical_exogenous_episode_sha256": _sha(
                f"alpha-{self._episode_seed}-{self.manifest_salt}"
            ),
            "beta_exogenous_episode_sha256": _sha(
                f"beta-{self._episode_seed}-{self.manifest_salt}"
            ),
            "episode": {"seed": manifest_seed, "private": "not-ledgered"},
            "future_truth": ["not-ledgered"],
        }


class _SynchronousAutoResetHarness:
    """The relevant reset behavior of SB3 DummyVecEnv, without torch/SB3."""

    def __init__(self, environments: list[gym.Env[Any, Any]]) -> None:
        self.envs = environments

    def reset(self) -> list[np.ndarray]:
        return [environment.reset()[0] for environment in self.envs]

    def step(self) -> list[tuple[np.ndarray, float, bool, bool, dict[str, Any]]]:
        results: list[tuple[np.ndarray, float, bool, bool, dict[str, Any]]] = []
        for environment in self.envs:
            observation, reward, terminated, truncated, info = environment.step(0)
            if terminated or truncated:
                terminal_observation = observation.copy()
                observation, reset_info = environment.reset()
                info = {
                    **info,
                    "terminal_observation": terminal_observation,
                    "reset_info": reset_info,
                }
            results.append((observation, reward, terminated, truncated, info))
        return results


def _wrapped_harness(
    *,
    base_seed: int = FIXTURE_BASE,
    horizon: int = FIXTURE_HORIZON,
) -> tuple[
    EpisodeSeedAllocatorV1,
    list[_AuditableFixtureEnv],
    _SynchronousAutoResetHarness,
]:
    allocator = EpisodeSeedAllocatorV1.fixture(
        base_seed=base_seed,
        episode_decisions=horizon,
    )
    raw = [_AuditableFixtureEnv(horizon=horizon) for _ in range(VECTOR_ENV_COUNT)]
    wrapped = [
        allocator.wrap(environment, environment_index=index)
        for index, environment in enumerate(raw)
    ]
    return allocator, raw, _SynchronousAutoResetHarness(wrapped)


def _resign_fixture_resume(payload: dict[str, Any]) -> None:
    snapshot = payload["snapshot"]
    ledger = snapshot["ledger"]
    for entry in ledger["entries"]:
        entry["sha256"] = canonical_sha256(
            {key: value for key, value in entry.items() if key != "sha256"}
        )
    ledger["sha256"] = canonical_sha256(
        {key: value for key, value in ledger.items() if key != "sha256"}
    )
    snapshot["sha256"] = canonical_sha256(
        {key: value for key, value in snapshot.items() if key != "sha256"}
    )
    payload["sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "sha256"}
    )


def test_protocol_seed_domains_are_pairwise_disjoint_and_fixture_is_classified() -> (
    None
):
    for index, left in enumerate(SEED_DOMAINS):
        for right in SEED_DOMAINS[index + 1 :]:
            assert left.last_seed < right.first_seed
    assert classify_episode_seed(FIXTURE_BASE) is EpisodeSeedSplitV1.FIXTURE
    assert classify_episode_seed(FIXTURE_SEED_MAX) is EpisodeSeedSplitV1.FIXTURE


def test_fixed_four_environment_mapping_is_independent_of_run_randomness() -> None:
    first = EpisodeSeedAllocatorV1.fixture()
    second = EpisodeSeedAllocatorV1.fixture()

    expected = tuple(FIXTURE_BASE + index for index in range(VECTOR_ENV_COUNT))
    assert (
        tuple(first.seed_for(index, 0) for index in range(VECTOR_ENV_COUNT)) == expected
    )
    assert (
        tuple(second.seed_for(index, 0) for index in range(VECTOR_ENV_COUNT))
        == expected
    )
    assert (
        episode_seed_for(
            base_seed=FIXTURE_BASE,
            environment_index=3,
            local_episode_ordinal=7,
        )
        == FIXTURE_BASE + 4 * 7 + 3
    )


def test_wrapper_explicitly_seeds_initial_and_automatic_resets() -> None:
    allocator, raw, vector = _wrapped_harness()

    reset_observations = vector.reset()
    assert all(observation.tolist() == [0.0] for observation in reset_observations)
    assert [environment.received_seeds for environment in raw] == [
        [FIXTURE_BASE + index] for index in range(VECTOR_ENV_COUNT)
    ]
    vector.step()
    terminal_results = vector.step()

    assert all(result[2] is True for result in terminal_results)
    assert [environment.received_seeds for environment in raw] == [
        [FIXTURE_BASE + index, FIXTURE_BASE + VECTOR_ENV_COUNT + index]
        for index in range(VECTOR_ENV_COUNT)
    ]
    assert [entry.episode_seed for entry in allocator.ledger] == [
        *(FIXTURE_BASE + index for index in range(VECTOR_ENV_COUNT)),
        *(FIXTURE_BASE + VECTOR_ENV_COUNT + index for index in range(VECTOR_ENV_COUNT)),
    ]
    assert all(cursor.current_episode_ordinal == 1 for cursor in allocator.cursors)
    assert all(cursor.decision_offset == 0 for cursor in allocator.cursors)
    assert all(cursor.next_episode_ordinal == 2 for cursor in allocator.cursors)
    assert find_seed_allocator(vector) is allocator
    assert all(entry.ignored_framework_seed is None for entry in allocator.ledger)


def test_real_sb3_setup_seed_is_audited_but_cannot_select_episode_seed() -> None:
    stable_baselines3 = pytest.importorskip("stable_baselines3")
    dummy_vec_module = pytest.importorskip("stable_baselines3.common.vec_env")
    optimizer_seed = 401
    allocator = EpisodeSeedAllocatorV1.fixture(episode_decisions=FIXTURE_HORIZON)
    raw = [_AuditableFixtureEnv() for _ in range(VECTOR_ENV_COUNT)]
    wrapped = [
        allocator.wrap(environment, environment_index=index)
        for index, environment in enumerate(raw)
    ]
    vector = dummy_vec_module.DummyVecEnv(
        [lambda environment=environment: environment for environment in wrapped]
    )
    try:
        stable_baselines3.PPO(
            "MlpPolicy",
            vector,
            seed=optimizer_seed,
            n_steps=2,
            batch_size=8,
            n_epochs=1,
            device="cpu",
            verbose=0,
        )
        vector.reset()

        assert [environment.received_seeds for environment in raw] == [
            [FIXTURE_BASE + index] for index in range(VECTOR_ENV_COUNT)
        ]
        assert [entry.ignored_framework_seed for entry in allocator.ledger] == [
            optimizer_seed + index for index in range(VECTOR_ENV_COUNT)
        ]
        assert [entry.episode_seed for entry in allocator.ledger] == [
            FIXTURE_BASE + index for index in range(VECTOR_ENV_COUNT)
        ]
    finally:
        vector.close()


def test_manifest_ledger_is_immutable_canonical_and_private_by_construction() -> None:
    allocator, _, vector = _wrapped_harness()
    infos = [environment.reset()[1] for environment in vector.envs]
    entry = allocator.ledger[0]
    first = allocator.snapshot()
    second = allocator.snapshot()

    assert infos == [{"public": True}] * VECTOR_ENV_COUNT
    assert "seed" not in infos[0]
    assert "manifest" not in infos[0]
    assert first.sha256 == second.sha256
    assert first.ledger_sha256 == second.ledger_sha256
    assert "future_truth" not in str(first.to_dict())
    with pytest.raises(FrozenInstanceError):
        entry.episode_seed = FIXTURE_SEED_MAX  # type: ignore[misc]


def test_snapshot_identity_is_recursively_immutable_and_bodies_are_detached() -> None:
    allocator = EpisodeSeedAllocatorV1.fixture()
    snapshot = allocator.snapshot()
    original_sha256 = snapshot.sha256
    body = snapshot.body()
    body["allocator_identity"]["seed_domain"]["first_seed"] = FIXTURE_SEED_MAX

    assert snapshot.sha256 == original_sha256
    assert (
        snapshot.to_dict()["allocator_identity"]["seed_domain"]["first_seed"]
        == FIXTURE_SEED_MIN
    )
    immutable_domain = snapshot.allocator_identity["seed_domain"]
    with pytest.raises(TypeError):
        immutable_domain["first_seed"] = FIXTURE_SEED_MAX


def test_wrapper_rejects_external_or_early_reset_without_consuming_another_seed() -> (
    None
):
    allocator = EpisodeSeedAllocatorV1.fixture(episode_decisions=FIXTURE_HORIZON)
    raw = _AuditableFixtureEnv()
    wrapped = allocator.wrap(raw, environment_index=0)

    with pytest.raises(EpisodeSeedCustodyError, match="contaminated episode domain"):
        wrapped.reset(seed=FIXTURE_BASE)
    wrapped.reset()
    with pytest.raises(EpisodeSeedCustodyError, match="before.*terminated"):
        wrapped.reset()

    assert raw.received_seeds == [FIXTURE_BASE]
    assert [entry.episode_seed for entry in allocator.ledger] == [FIXTURE_BASE]


def test_manifest_seed_mismatch_fails_closed_and_seed_cannot_be_retried() -> None:
    allocator = EpisodeSeedAllocatorV1.fixture(episode_decisions=FIXTURE_HORIZON)
    raw = _AuditableFixtureEnv(report_wrong_seed=True)
    wrapped = allocator.wrap(raw, environment_index=0)

    with pytest.raises(EpisodeSeedCustodyError, match="manifest seed"):
        wrapped.reset()
    with pytest.raises(EpisodeSeedCustodyError, match="failed closed"):
        wrapped.reset()
    assert raw.received_seeds == [FIXTURE_BASE]
    assert allocator.ledger == ()


def test_exact_mid_episode_cursor_round_trips_only_in_fixture_mode() -> None:
    allocator, _, vector = _wrapped_harness(horizon=3)
    vector.reset()
    vector.step()
    exported = _export_fixture_resume(allocator)
    restored = _restore_fixture_resume(exported)

    assert restored.snapshot().to_dict() == allocator.snapshot().to_dict()
    assert all(cursor.current_episode_ordinal == 0 for cursor in restored.cursors)
    assert all(cursor.decision_offset == 1 for cursor in restored.cursors)
    assert all(cursor.next_episode_ordinal == 1 for cursor in restored.cursors)
    assert all(cursor.awaiting_reset is False for cursor in restored.cursors)

    fresh = restored.wrap(_AuditableFixtureEnv(horizon=3), environment_index=0)
    with pytest.raises(EpisodeSeedCustodyError, match="decision offset 1"):
        fresh.reset()
    with pytest.raises(EpisodeSeedCustodyError, match="failed closed"):
        restored.snapshot()


def test_offset_zero_fixture_restore_reattaches_without_consuming_a_seed() -> None:
    allocator, _, vector = _wrapped_harness(horizon=3)
    vector.reset()
    before = allocator.snapshot().to_dict()
    restored = _restore_fixture_resume(_export_fixture_resume(allocator))
    raw = [_AuditableFixtureEnv(horizon=3) for _ in range(VECTOR_ENV_COUNT)]
    wrapped = [
        restored.wrap(environment, environment_index=index)
        for index, environment in enumerate(raw)
    ]

    for environment in wrapped:
        environment.reset()

    assert restored.snapshot().to_dict() == before
    assert len(restored.ledger) == VECTOR_ENV_COUNT
    assert [environment.received_seeds for environment in raw] == [
        [FIXTURE_BASE + index] for index in range(VECTOR_ENV_COUNT)
    ]
    for environment in wrapped:
        environment.step(0)
    assert restored.decisions_consumed_by_environment == (1, 1, 1, 1)


def test_offset_zero_reattachment_rejects_changed_manifest_without_new_ledger() -> None:
    allocator, _, vector = _wrapped_harness(horizon=3)
    vector.reset()
    restored = _restore_fixture_resume(_export_fixture_resume(allocator))
    wrapped = restored.wrap(
        _AuditableFixtureEnv(horizon=3, manifest_salt="changed"),
        environment_index=0,
    )

    with pytest.raises(EpisodeSeedCustodyError, match="different manifest"):
        wrapped.reset()
    assert len(restored.ledger) == VECTOR_ENV_COUNT


def test_restore_rejects_duplicate_seed_even_when_all_digests_are_recomputed() -> None:
    allocator, _, vector = _wrapped_harness()
    vector.reset()
    payload = deepcopy(_export_fixture_resume(allocator))
    first, duplicate = payload["snapshot"]["ledger"]["entries"][:2]
    duplicate.update(
        {
            "environment_index": first["environment_index"],
            "local_episode_ordinal": first["local_episode_ordinal"],
            "global_episode_ordinal": first["global_episode_ordinal"],
            "episode_seed": first["episode_seed"],
        }
    )
    _resign_fixture_resume(payload)

    with pytest.raises(EpisodeSeedCustodyError, match="duplicates episode seed"):
        _restore_fixture_resume(payload)


def test_split_exhaustion_is_rejected_before_an_out_of_domain_reset() -> None:
    base = FIXTURE_SEED_MAX - (VECTOR_ENV_COUNT - 1)
    allocator, raw, vector = _wrapped_harness(base_seed=base)
    vector.reset()
    vector.step()

    with pytest.raises(EpisodeSeedExhaustedError, match="split boundary"):
        vector.step()
    assert [len(environment.received_seeds) for environment in raw] == [
        1
    ] * VECTOR_ENV_COUNT
    assert len(allocator.ledger) == VECTOR_ENV_COUNT


def test_transition_discontinuities_fail_closed() -> None:
    allocator = EpisodeSeedAllocatorV1.fixture(episode_decisions=FIXTURE_HORIZON)
    wrapped = allocator.wrap(_AuditableFixtureEnv(), environment_index=0)
    wrapped.reset()
    wrapped.step(0)

    with pytest.raises(EpisodeSeedCustodyError, match="did not terminate"):
        allocator.record_transition(0, terminated=False, truncated=False)
    with pytest.raises(EpisodeSeedCustodyError, match="failed closed"):
        allocator.snapshot()


def test_registered_mode_rejects_resume_but_still_exposes_nonresumable_evidence() -> (
    None
):
    registered = EpisodeSeedAllocatorV1.registered()
    fixture_resume = _export_fixture_resume(EpisodeSeedAllocatorV1.fixture())

    assert registered.snapshot().to_dict()["allocator_identity"]["mode"] == "registered"
    with pytest.raises(RegisteredResumeForbiddenError, match="resume export"):
        registered.export_fixture_resume(
            parent_checkpoint_sha256=FIXTURE_PARENT_CHECKPOINT_SHA256
        )
    with pytest.raises(RegisteredResumeForbiddenError, match="zero cursors"):
        EpisodeSeedAllocatorV1.registered(resume_from=fixture_resume)


def test_restore_rejects_cursor_tampering_after_valid_digest_rewrite() -> None:
    allocator, _, vector = _wrapped_harness(horizon=3)
    vector.reset()
    vector.step()
    payload = deepcopy(_export_fixture_resume(allocator))
    admitted_resume_sha256 = payload["sha256"]
    payload["snapshot"]["cursors"][0]["decision_offset"] = 2
    _resign_fixture_resume(payload)

    with pytest.raises(EpisodeSeedCustodyError, match="checkpoint-pinned digest"):
        _restore_fixture_resume(
            payload,
            expected_resume_sha256=admitted_resume_sha256,
        )


def test_restore_rejects_plausible_manifest_substitution_against_external_pin() -> None:
    allocator, _, vector = _wrapped_harness(horizon=3)
    vector.reset()
    payload = deepcopy(_export_fixture_resume(allocator))
    admitted_resume_sha256 = payload["sha256"]
    identity = payload["snapshot"]["ledger"]["entries"][0]["manifest_identity"]
    identity.update(
        {
            "manifest_sha256": _sha("substituted-manifest"),
            "alpha_physical_exogenous_episode_sha256": _sha("substituted-alpha"),
            "beta_exogenous_episode_sha256": _sha("substituted-beta"),
        }
    )
    identity["sha256"] = canonical_sha256(
        {key: value for key, value in identity.items() if key != "sha256"}
    )
    _resign_fixture_resume(payload)

    with pytest.raises(EpisodeSeedCustodyError, match="checkpoint-pinned digest"):
        _restore_fixture_resume(
            payload,
            expected_resume_sha256=admitted_resume_sha256,
        )


def test_restore_rejects_parent_checkpoint_substitution() -> None:
    payload = _export_fixture_resume(EpisodeSeedAllocatorV1.fixture())

    with pytest.raises(EpisodeSeedCustodyError, match="parent checkpoint differs"):
        _restore_fixture_resume(
            payload,
            expected_parent_checkpoint_sha256=_sha("different-parent-checkpoint"),
        )
