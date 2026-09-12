"""Real disjoint physical fixtures for residual evidence and saved-command replay."""

from __future__ import annotations

from copy import deepcopy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from plant_backend.rl.v2.training import research_batch
from plant_backend.rl.v2.training import research_mpc_residual_evaluation as evaluation
from plant_backend.rl.v2.training.custody_json import canonical_bytes


class ConstantPolicy:
    def __init__(self, action: list[float]) -> None:
        self.action = np.asarray(action, dtype=np.float32)
        self.calls = 0

    def predict(self, observation, *, deterministic):
        assert observation.shape == (139,)
        assert deterministic is True
        self.calls += 1
        return self.action.copy(), None


@pytest.fixture(scope="module")
def day(tmp_path_factory):
    config = research_batch.template(tmp_path_factory.mktemp("mpc-residual-evidence"))
    policy = ConstantPolicy([0.25, -0.3, 0.4, 0.5])
    episode = evaluation.evaluate_residual_episode(
        config, policy=policy, seed=9_990_860, fixture=True, max_decisions=3
    )
    assert policy.calls == 3
    assert episode["summary"]["failure"] is None
    return config, episode


def test_serialized_real_mpc_prefix_replays_saved_commands(day, monkeypatch):
    config, episode = day
    # Full compressed JSON round trip exercises tuple/list and float32 publication.
    encoded = gzip.compress(canonical_bytes(episode), mtime=0)
    reopened = json.loads(gzip.decompress(encoded))

    def no_policy_execution(*args, **kwargs):
        raise AssertionError("physical replay must not execute the learned policy")

    monkeypatch.setattr(ConstantPolicy, "predict", no_policy_execution)
    result = evaluation.verify_residual_episode_replay(config, reopened, fixture=True)
    assert result["verified"] is True
    assert result["saved_absolute_commands_executed"] is True
    assert result["learned_policy_reexecuted"] is False
    assert result["physical_steps"] == 45
    assert result["execution_complete"] is False
    assert result["scope"] == "retained_physical_prefix"
    assert len(reopened["summary"]["cycle_settlements"]) == 0
    assert len(reopened["initial_observation"]["normalized"]) == 135
    assert len(reopened["initial_policy_observation"]) == 139


@pytest.mark.parametrize(
    "target",
    [
        "input",
        "raw",
        "command",
        "mapping",
        "physical",
        "reward",
        "summary",
        "interface",
        "identity",
        "deterministic",
        "budget",
        "extra_failure",
    ],
)
def test_tampering_with_mapping_or_physical_evidence_is_rejected(day, target):
    config, episode = day
    changed = deepcopy(episode)
    row = changed["history"][0]
    if target == "input":
        row["policy_observation"][-1] += 0.01
    elif target == "raw":
        row["raw_action"][0] += 0.1
    elif target == "command":
        row["command"]["tampered"] = True
    elif target == "mapping":
        row["residual_evidence"]["sha256"] = "0" * 64
    elif target == "physical":
        changed["minute_truth_states"][1]["battery_soc"] += 0.01
    elif target == "reward":
        row["reward_eur"] += 1
    elif target == "summary":
        changed["summary"]["hard_safety_passed"] = not changed["summary"][
            "hard_safety_passed"
        ]
    elif target == "interface":
        changed["interface_contract_digest"] = "0" * 64
    elif target == "identity":
        changed["controller"]["id"] = "another policy"
    elif target == "deterministic":
        changed["controller"]["deterministic"] = False
    elif target == "budget":
        changed["prefix_limit"] += 1
    elif target == "extra_failure":
        changed["summary"]["failure"] = {
            "phase": "policy_inference",
            "type": "ValueError",
            "message": "invented",
        }
    with pytest.raises(ValueError, match="differ"):
        evaluation.verify_residual_episode_replay(config, changed, fixture=True)


def test_invalid_policy_preserves_replayable_failure_prefix(tmp_path: Path):
    class FailsAfterTwo(ConstantPolicy):
        def predict(self, observation, *, deterministic):
            action, state = super().predict(observation, deterministic=deterministic)
            if self.calls == 3:
                action[0] = np.nan
            return action, state

    config = research_batch.template(tmp_path)
    episode = evaluation.evaluate_residual_episode(
        config,
        policy=FailsAfterTwo([0, 0, 0, 0]),
        seed=9_990_861,
        fixture=True,
        max_decisions=4,
    )
    assert episode["summary"]["failure"]["phase"] == "policy_inference"
    assert episode["summary"]["decisions"] == 2
    assert episode["failed_decision"]["raw_action"][0] == "nan"
    replay = evaluation.verify_residual_episode_replay(config, episode, fixture=True)
    assert replay["physical_steps"] == 30
    assert replay["controller_failure_authenticity_verified"] is False
    for mutation in ("shape", "mapping", "phase", "missing_failure"):
        altered = deepcopy(episode)
        pending = altered["failed_decision"]
        if mutation == "shape":
            pending["raw_action_shape"] = [3]
        elif mutation == "mapping":
            pending["command"] = {"made_up": True}
        elif mutation == "phase":
            altered["summary"]["failure"]["phase"] = "environment_execution"
        else:
            altered["summary"]["failure"] = None
        with pytest.raises(ValueError, match="differ"):
            evaluation.verify_residual_episode_replay(config, altered, fixture=True)


def test_clipped_residual_is_visible_and_replayable(tmp_path: Path):
    config = research_batch.template(tmp_path)
    episode = evaluation.evaluate_residual_episode(
        config,
        policy=ConstantPolicy([2, -3, 0.5, 0]),
        seed=9_990_862,
        fixture=True,
        max_decisions=1,
    )
    assert episode["summary"]["failure"] is None
    row = episode["history"][0]
    assert row["raw_action"] == [2, -3, 0.5, 0]
    assert row["residual_evidence"]["body"]["residual_clipped"] is True
    assert row["controller_invalid"] is True
    assert evaluation.verify_residual_episode_replay(config, episode, fixture=True)[
        "verified"
    ]


def test_post_execution_evidence_failure_is_integrity_error(tmp_path, monkeypatch):
    config = research_batch.template(tmp_path)
    advanced = []

    def broken_capture(base, transition, minutes, *, actor):
        advanced.append(base.decisions_taken)
        raise ValueError("injected archive capture failure")

    monkeypatch.setattr(evaluation.physical, "_interval", broken_capture)
    with pytest.raises(RuntimeError, match="integrity failure in evidence_capture"):
        evaluation.evaluate_residual_episode(
            config,
            policy=ConstantPolicy([0, 0, 0, 0]),
            seed=9_990_863,
            fixture=True,
            max_decisions=1,
        )
    assert advanced == [1]


@pytest.mark.parametrize(
    "seed,fixture,prefix",
    [(42, False, None), (9_990_860, False, None), (9_700_100, False, 1)],
)
def test_forbidden_seed_or_production_prefix_does_not_reset(
    tmp_path, monkeypatch, seed, fixture, prefix
):
    config = research_batch.template(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid admission reached environment construction")

    monkeypatch.setattr(evaluation.physical, "_environment", forbidden)
    with pytest.raises(ValueError):
        evaluation.evaluate_residual_episode(
            config,
            policy=ConstantPolicy([0, 0, 0, 0]),
            seed=seed,
            fixture=fixture,
            max_decisions=prefix,
        )


@pytest.fixture
def clocked_fallback(monkeypatch):
    """Fabricate only numerical unavailability, retaining the native fallback."""
    from types import SimpleNamespace
    from plant_backend import operational_mpc
    from plant_backend.rl.v2.economic_mpc_schema import (
        EconomicMPCSolveError,
        EconomicMPCStageSolveV1,
    )
    from plant_backend.rl.v2.mpc_schema import domain_sha256

    clock = {"elapsed_ns": 100, "status": 1}

    def build(controller, observation, **kwargs):
        return SimpleNamespace(
            sha256=domain_sha256("mpc-evaluator-fixture-model", observation.sha256),
            horizon_intervals=min(
                96, controller.settings.task_interval_count - controller._next_decision
            ),
        )

    def solve(*args, **kwargs):
        stage = EconomicMPCStageSolveV1(
            stage="service_maximization",
            status=clock["status"],
            message="controlled fixture unavailability",
            elapsed_ns=clock["elapsed_ns"],
            objective_value=None,
            mip_gap=None,
            mip_node_count=None,
            solution_sha256=None,
            scipy_version="fixture",
            highs_version="fixture",
            success=False,
        )
        raise EconomicMPCSolveError("controlled fixture", stages=(stage,))

    monkeypatch.setattr(operational_mpc.OperationalMPCController, "build_model", build)
    monkeypatch.setattr(operational_mpc, "solve_economic_mpc_model", solve)
    return clock


@pytest.mark.parametrize("action", [[0, 0, 0, 0], [0.25, -0.5, 0.25, 0.5]])
def test_fallback_timing_changes_keep_saved_command_execution(
    tmp_path, monkeypatch, clocked_fallback, action
):
    from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta

    config = research_batch.template(tmp_path)
    episode = evaluation.evaluate_residual_episode(
        config,
        policy=ConstantPolicy(action),
        seed=9_990_864,
        fixture=True,
        max_decisions=2,
    )
    archived = json.loads(gzip.decompress(gzip.compress(canonical_bytes(episode))))
    assert archived["summary"]["fallback_decisions"] == 2
    before = archived["history"][0]["mpc_decision"]
    assert (
        before["evidence"]["payload"]["operation"]["solver_failure"]["stages"][0][
            "elapsed_ns"
        ]
        == 100
    )
    executed = []
    original = PlantCyclicMarketEnvV2Beta.step_process_setpoints

    def record_command(self, command, **kwargs):
        executed.append(command.to_dict())
        return original(self, command, **kwargs)

    monkeypatch.setattr(
        PlantCyclicMarketEnvV2Beta, "step_process_setpoints", record_command
    )
    clocked_fallback["elapsed_ns"] = 123456
    result = evaluation.verify_residual_episode_replay(config, archived, fixture=True)
    assert result["verified"] is True
    assert result["mpc_reexecuted_for_mapping_check"] is True
    assert executed == [row["command"] for row in archived["history"]]
    assert canonical_bytes(archived) == canonical_bytes(episode)  # No raw rewriting.
    clocked_fallback["status"] = 2
    with pytest.raises(ValueError, match="causal prior semantics differ"):
        evaluation.verify_residual_episode_replay(config, archived, fixture=True)


def test_day_settlement_with_native_fallback_is_replayed(tmp_path, clocked_fallback):
    config = research_batch.template(tmp_path)
    episode = evaluation.evaluate_residual_episode(
        config,
        policy=ConstantPolicy([0.25, -0.25, 0.25, 0]),
        seed=9_990_865,
        fixture=True,
        max_decisions=96,
    )
    assert episode["summary"]["failure"] is None
    assert len(episode["summary"]["cycle_settlements"]) == 1
    clocked_fallback["elapsed_ns"] += 1
    result = evaluation.verify_residual_episode_replay(
        config, json.loads(canonical_bytes(episode)), fixture=True
    )
    assert result["physical_steps"] == 1440
    assert episode["summary"]["fallback_decisions"] == 96


def _rehash_nominal(record):
    """Build a self-consistent tamper, so semantic checks must reject it."""
    from dataclasses import replace
    from plant_backend.rl.v2.mpc_schema import domain_sha256

    payload = record["evidence"]["payload"]
    digest = domain_sha256(evaluation.DECISION_SCHEMA, payload)
    record["evidence"]["sha256"] = digest
    command = evaluation._decode_command(record["command"], name="fixture command")
    record["command"] = replace(command, source_sha256=digest).to_dict()


@pytest.mark.parametrize(
    "field",
    [
        "target",
        "public_binding",
        "model_binding",
        "resource",
        "delivery",
        "mode",
        "source_notes",
        "plan_step",
        "solver_message",
        "solver_solution",
        "invalid",
        "unrecognized_clock",
        "negative_clock",
        "boolean_clock",
        "missing_plan",
        "nominal_digest",
    ],
)
def test_serialized_self_consistent_nominal_semantic_tamper_rejected(day, field):
    from dataclasses import replace
    from plant_backend.rl.v2.mpc_schema import domain_sha256

    config, episode = day
    changed = json.loads(canonical_bytes(episode))
    nominal = changed["initial_mpc_decision"]
    payload = nominal["evidence"]["payload"]
    plan = nominal["evidence"]["plan"]
    assert plan is not None
    if field == "target":
        command = evaluation._decode_command(nominal["command"], name="fixture target")
        targets = replace(
            command.targets,
            electrolyser_target_kw=command.targets.electrolyser_target_kw + 1,
        )
        nominal["command"] = replace(command, targets=targets).to_dict()
        payload["targets"] = targets.to_dict()
    elif field == "public_binding":
        payload["observation_sha256"] = "a" * 64
    elif field == "model_binding":
        payload["model_sha256"] = "b" * 64
    elif field == "resource":
        payload["operation"]["resources"][0]["projected"] += 1
    elif field == "delivery":
        payload["operation"]["deliveries"][0]["predicted_kg"] += 1
    elif field == "mode":
        payload["operation"]["mode"] = "fallback"
    elif field == "source_notes":
        nominal["evidence"]["source_notes"].append("changed")
    elif field == "plan_step":
        plan["steps"][0]["solar_to_dac_kw"] += 1
    elif field == "solver_message":
        plan["stages"][0]["message"] = "a different numerical outcome"
    elif field == "solver_solution":
        plan["stages"][0]["solution_sha256"] = "c" * 64
    elif field == "invalid":
        nominal["invalid"] = not nominal["invalid"]
    elif field == "unrecognized_clock":
        payload["elapsed_ns"] = 12
    elif field == "negative_clock":
        plan["stages"][0]["elapsed_ns"] = -1
    elif field == "boolean_clock":
        plan["stages"][0]["elapsed_ns"] = True
    elif field == "missing_plan":
        nominal["evidence"]["plan"] = None
        payload["plan_sha256"] = None
    elif field == "nominal_digest":
        nominal["command"]["sha256"] = "d" * 64
        with pytest.raises(ValueError):
            evaluation.verify_residual_episode_replay(config, changed, fixture=True)
        return
    if field in {"plan_step", "solver_solution"}:
        body = {key: value for key, value in plan.items() if key != "sha256"}
        body["stages"] = [
            {"stage": s["stage"], "solution_sha256": s["solution_sha256"]}
            for s in plan["stages"]
        ]
        plan["sha256"] = domain_sha256(evaluation.ECONOMIC_MPC_PLAN_DOMAIN, body)
        payload["plan_sha256"] = plan["sha256"]
    _rehash_nominal(nominal)
    with pytest.raises(ValueError):
        evaluation.verify_residual_episode_replay(config, changed, fixture=True)


def test_clock_projection_keeps_original_success_plan_and_rejects_bad_hash(day):
    _, episode = day
    original = episode["history"][0]["residual_evidence"]
    changed = json.loads(canonical_bytes(original))
    changed["body"]["mpc_decision"]["evidence"]["plan"]["stages"][0][
        "elapsed_ns"
    ] += 999
    with pytest.raises(ValueError, match="digest differs"):
        evaluation.residual_evidence_semantic_projection(changed)
    from dataclasses import replace
    from plant_backend.rl.v2.mpc_schema import domain_sha256

    changed["sha256"] = domain_sha256(evaluation.EVIDENCE_SCHEMA, changed["body"])
    command = evaluation._decode_command(changed["command"], name="clock fixture")
    changed["command"] = replace(command, source_sha256=changed["sha256"]).to_dict()
    assert evaluation.residual_evidence_semantic_projection(
        changed
    ) == evaluation.residual_evidence_semantic_projection(original)
    assert (
        changed["body"]["mpc_decision"]["evidence"]["plan"]["sha256"]
        == original["body"]["mpc_decision"]["evidence"]["plan"]["sha256"]
    )


@pytest.mark.parametrize("point", ["reset", "after_step"])
def test_mpc_prior_failure_is_fatal_not_a_controller_prefix(
    tmp_path, monkeypatch, clocked_fallback, point
):
    from plant_backend import operational_mpc

    original = operational_mpc.OperationalMPCController.compute_command
    boundaries = []

    def fail(self, observation, **kwargs):
        boundaries.append(self._next_decision)
        if self._next_decision == (0 if point == "reset" else 1):
            raise ValueError("injected causal MPC failure")
        return original(self, observation, **kwargs)

    monkeypatch.setattr(
        operational_mpc.OperationalMPCController, "compute_command", fail
    )
    with pytest.raises(evaluation.MPCResidualIntegrityError, match="integrity failure"):
        evaluation.evaluate_residual_episode(
            research_batch.template(tmp_path),
            policy=ConstantPolicy([0, 0, 0, 0]),
            seed=9_990_866,
            fixture=True,
            max_decisions=2,
        )
    assert boundaries == ([0] if point == "reset" else [0, 1])


def test_failed_mapping_retains_boundary_without_partial_physical_record(
    tmp_path, monkeypatch, clocked_fallback
):
    original = evaluation.prepare_residual_action

    def fail_on_second(base, decision, action, **kwargs):
        if base.decisions_taken == 1 and np.any(action != 0):
            raise ValueError("injected mapping rejection")
        return original(base, decision, action, **kwargs)

    monkeypatch.setattr(evaluation, "prepare_residual_action", fail_on_second)
    config = research_batch.template(tmp_path)
    episode = evaluation.evaluate_residual_episode(
        config,
        policy=ConstantPolicy([0.1, 0, 0, 0]),
        seed=9_990_867,
        fixture=True,
        max_decisions=3,
    )
    monkeypatch.setattr(evaluation, "prepare_residual_action", original)
    assert episode["summary"]["failure"]["phase"] == "residual_command"
    assert len(episode["history"]) == 1
    assert "residual_evidence" not in episode["failed_decision"]
    assert evaluation.verify_residual_episode_replay(
        config, json.loads(canonical_bytes(episode)), fixture=True
    )["verified"]
    changed = deepcopy(episode)
    changed["prefix_limit"] = 1
    with pytest.raises(ValueError, match="failure descriptor differs"):
        evaluation.verify_residual_episode_replay(config, changed, fixture=True)


@pytest.mark.parametrize(
    "output",
    [[True, False, False, True], ["0", "0", "0", "0"], [1, 2], [[0, 0], [0, 0]]],
)
def test_malformed_inference_retains_unadvanced_prefix(
    tmp_path, clocked_fallback, output
):
    class MalformedPolicy:
        def predict(self, observation, *, deterministic):
            return output, None

    config = research_batch.template(tmp_path)
    episode = evaluation.evaluate_residual_episode(
        config, policy=MalformedPolicy(), seed=9_990_868, fixture=True, max_decisions=2
    )
    assert episode["history"] == []
    assert episode["summary"]["failure"]["phase"] == "policy_inference"
    assert len(episode["minute_truth_states"]) == 1
    assert episode["failed_decision"]["mpc_decision"] == episode["initial_mpc_decision"]
    assert (
        evaluation.verify_residual_episode_replay(
            config, json.loads(canonical_bytes(episode)), fixture=True
        )["physical_steps"]
        == 0
    )


def test_replay_computes_prior_once_at_each_boundary_and_never_reencodes(
    tmp_path, monkeypatch, clocked_fallback
):
    from plant_backend import operational_mpc
    from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta

    config = research_batch.template(tmp_path)
    episode = evaluation.evaluate_residual_episode(
        config,
        policy=ConstantPolicy([0, 0, 0, 0]),
        seed=9_990_869,
        fixture=True,
        max_decisions=2,
    )
    original_compute = operational_mpc.OperationalMPCController.compute_command
    original_observe = PlantCyclicMarketEnvV2Beta._observe
    original_reset = PlantCyclicMarketEnvV2Beta.reset
    computes, observes, resets = [], [], []

    def compute(self, observation, **kwargs):
        computes.append(self._next_decision)
        return original_compute(self, observation, **kwargs)

    def observe(self):
        observes.append(self.decisions_taken)
        return original_observe(self)

    def reset(self, **kwargs):
        resets.append(kwargs)
        return original_reset(self, **kwargs)

    monkeypatch.setattr(
        operational_mpc.OperationalMPCController, "compute_command", compute
    )
    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "_observe", observe)
    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "reset", reset)
    evaluation.verify_residual_episode_replay(config, episode, fixture=True)
    assert computes == [0, 1, 2]
    assert observes == [
        0,
        0,
        1,
        2,
    ]  # Native beta reset encodes base and final beta state.
    assert resets == [{"seed": 9_990_869}]


@pytest.mark.parametrize("loss", ["index", "public", "envelope"])
def test_boundary_loss_after_inference_is_fatal_before_any_execution(
    tmp_path, monkeypatch, clocked_fallback, loss
):
    from dataclasses import replace

    original_environment = evaluation.physical._environment
    holder = []

    def environment(config):
        env = original_environment(config)
        holder.append(env)
        return env

    class MutatingFixturePolicy:
        def predict(self, observation, *, deterministic):
            env = holder[0]
            if loss == "index":
                env.decisions_taken += 1
            elif loss == "public":
                env.control_state = replace(
                    env.control_state,
                    boundary=replace(env.control_state.boundary, battery_soc=0.4),
                )
            else:
                env.constraints = replace(env.constraints, dac_max_kw=100.0)
            return np.zeros(4, dtype=np.float32), None

    monkeypatch.setattr(evaluation.physical, "_environment", environment)
    with pytest.raises(evaluation.MPCResidualIntegrityError, match="prior_integrity"):
        evaluation.evaluate_residual_episode(
            research_batch.template(tmp_path),
            policy=MutatingFixturePolicy(),
            seed=9_990_870,
            fixture=True,
            max_decisions=2,
        )
    assert holder[0].physics_steps_taken == 0
