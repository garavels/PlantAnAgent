"""Focused initial-witness fixtures; fresh disposable models, development only."""

import copy
import gzip
import json

import pytest
import torch

from docs.research.tools.coverage_ppo_v1 import make_learner
from docs.research.tools.feasible_benchmark_v1 import BenchmarkEnv, ROOT
from docs.research.tools.feasible_mission_replay import canonical_sha, run_controller
from docs.research.tools.verify_initial_coverage_v1 import SharedInitialController


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    triples = {seed: make_learner(seed) for seed in (101, 211, 307)}
    env = BenchmarkEnv(split="development")
    reference = json.loads(
        gzip.decompress(
            (
                ROOT
                / "docs/research/results/feasible25_offgrid_20260915/offgrid-coverage-development/9920000.json.gz"
            ).read_bytes()
        )
    )
    env.reset(seed=9920000)
    yield env, {seed: triple[0] for seed, triple in triples.items()}, reference
    env.close()
    for _, _, vector in triples.values():
        vector.close()


def test_three_real_initial_actors_match_reference_trace(setup):
    env, models, reference = setup
    shared = SharedInitialController(env, models, reference)
    replay = run_controller(env, shared, max_decisions=8, retain_minutes=False)
    assert replay["failure"] is None
    assert len(shared.records) == 8
    assert sum(len(row["policies"]) for row in shared.records) == 24
    assert replay["minute_truth_sha256"] == canonical_sha(
        reference["minute_truth_states"][:121]
    )
    assert all(len(row["actual135"]) == 135 for row in shared.records)


def test_nonzero_initial_actor_is_rejected_before_physics(setup):
    env, models, reference = setup
    with torch.no_grad():
        models[211].policy.action_net.bias.fill_(2.0)
    shared = SharedInitialController(env, models, reference)
    replay = run_controller(env, shared, max_decisions=1)
    assert replay["failure"]["message"] == "Initial actor 211 was not exactly zero"
    assert env.decisions_taken == 0 and env.physics_steps_taken == 0


def test_reference_command_mismatch_is_rejected_before_physics(setup):
    env, models, reference = setup
    reference = copy.deepcopy(reference)
    reference["commands"][0]["command"]["hold_duration_s"] = 899.0
    shared = SharedInitialController(env, models, reference)
    replay = run_controller(env, shared, max_decisions=1)
    assert (
        replay["failure"]["message"]
        == "Coverage physical command differs at decision 0"
    )
    assert env.decisions_taken == 0 and env.physics_steps_taken == 0
