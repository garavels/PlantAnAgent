"""Single-arm execution and compressed detached evidence on fixture seeds only."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from plant_backend.rl.policies import FixedActionPolicy
from plant_backend.rl.v2 import beta_controller_benchmark as benchmark
from plant_backend.rl.v2.beta_env import PlantCyclicMarketEnvV2Beta
from plant_backend.rl.v2.study import (
    archive,
    arm_evidence as arms,
    arm_execution as execute,
)
from plant_backend.rl.v2.study import compressed_evidence as compressed
from plant_backend.rl.v2.training.custody_json import authenticate
from plant_backend.rl.v2.training.registered_contract import registered_config
from test_rl_v2_beta_controller_benchmark_verifier import _config, _loaded

LIMITS = compressed.EvidenceLimits(
    128 * 1024**2, 512 * 1024**2, 4 * 1024**2, 32 * 1024**2
)
SEED = 9_000_117


@pytest.fixture(scope="module")
def executed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("single-arm-physics").resolve()
    config = _config()
    loaded = _loaded(root, config)
    results, paths, counts = {}, {}, {}
    factory = benchmark._new_beta_environment
    reset = PlantCyclicMarketEnvV2Beta.reset
    close = PlantCyclicMarketEnvV2Beta.close
    act = FixedActionPolicy.act
    for arm in benchmark.BETA_CONTROLLER_ARM_ORDER:
        events: list[tuple[Any, ...]] = []
        calls = 0

        def construct(task: Any) -> Any:
            events.append(("construct",))
            return factory(task)

        def reset_once(self: Any, *args: Any, **kwargs: Any) -> Any:
            events.append(("reset", kwargs.get("seed"), calls))
            return reset(self, *args, **kwargs)

        def close_once(self: Any) -> None:
            events.append(("close",))
            close(self)

        def inference(self: Any, vector: Any) -> Any:
            nonlocal calls
            calls += 1
            return act(self, vector)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(benchmark, "_new_beta_environment", construct)
            patch.setattr(PlantCyclicMarketEnvV2Beta, "reset", reset_once)
            patch.setattr(PlantCyclicMarketEnvV2Beta, "close", close_once)
            patch.setattr(FixedActionPolicy, "act", inference)
            directory = root / arm
            results[arm] = execute.execute_synthetic_arm(
                config,
                arm=arm,
                seed=SEED,
                directory=directory,
                limits=LIMITS,
                checkpoint=loaded.path if arm == "PPO" else None,
            )
        paths[arm], counts[arm] = directory, {"events": events, "policy_calls": calls}
    # A separate permitted fixture run supplies the unchanged four-arm reference.
    legacy = benchmark.execute_beta_controller_benchmark_cell(loaded, config, seed=SEED)
    return {
        "results": results,
        "paths": paths,
        "counts": counts,
        "legacy": legacy,
        "config": config,
        "loaded": loaded,
    }


@pytest.mark.parametrize("arm", benchmark.BETA_CONTROLLER_ARM_ORDER)
def test_single_environment_and_reset_preserve_physics(
    executed: dict[str, Any], arm: str
) -> None:
    result = executed["results"][arm]
    reference = arms._standalone_result(executed["legacy"]["runs"][arm])
    assert (
        result["physical"]["arm_evidence_sha256"]
        == reference["deterministic_evidence_sha256"]
    )
    assert result["physical"]["economic_metrics"] == reference["economic_metrics"]
    assert (
        result["physical"]["classification"]["physical_steps"]
        == reference["physical_steps"]
    )
    events = executed["counts"][arm]["events"]
    assert sum(row[0] == "construct" for row in events) == 1
    assert sum(row[0] == "reset" for row in events) == 1
    assert sum(row[0] == "close" for row in events) == 1
    assert ("reset", SEED, 1000 if arm == "PPO" else 0) in events
    assert result["result_claim"] is None and result["execution_authorized"] is False
    assert result["registered_checkpoint_verified"] is False
    if arm == "PPO":
        assert result["latency"]["warmup_attempted_calls"] == 1000
        assert (
            executed["counts"][arm]["policy_calls"]
            == 1000 + reference["completed_decisions"]
        )
    else:
        assert result["physical"]["actor"]["optimizer_seed"] is None
        assert executed["counts"][arm]["policy_calls"] == 0


@pytest.mark.parametrize("arm", benchmark.BETA_CONTROLLER_ARM_ORDER)
def test_detached_archives_reopen_without_any_execution(
    executed: dict[str, Any], monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail(
            "detached compressed verification attempted plant or policy execution"
        )

    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "__init__", forbidden)
    monkeypatch.setattr(execute, "load_policy", forbidden)
    monkeypatch.setattr(FixedActionPolicy, "act", forbidden)
    monkeypatch.setattr(execute.timing, "capture_latency_host", forbidden)
    result = compressed.verify_synthetic_execution(
        executed["paths"][arm], limits=LIMITS
    )
    assert result == executed["results"][arm]
    descriptor = result["arm_archive"]
    assert descriptor["file"]["logical_bytes"] < descriptor["uncompressed_bytes"]


@pytest.mark.parametrize(
    "seed", [1_000_000, 2_000_000, 3_000_031, 8_999_999, 10_000_000, True, 9_000_000.0]
)
def test_fixture_scope_rejected_before_any_dependency_or_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: Any
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("protected seed reached any execution dependency")

    monkeypatch.setattr(execute, "_execute_arm", forbidden)
    root = tmp_path / "unopened"
    with pytest.raises(ValueError, match="synthetic arm seed"):
        execute.execute_synthetic_arm(
            _config(), arm="H", seed=seed, directory=root, limits=LIMITS
        )
    assert not root.exists()


@pytest.mark.parametrize("where", ["task", "checkpoint"])
def test_registered_task_cannot_enter_fixture_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    from plant_backend.rl.v2.training.custody_io import publish_json

    config = registered_config(
        tmp_path / "unused", "plant-beta-ppo-pi-lagrangian-v1", 101
    )
    checkpoint = None
    if where == "checkpoint":
        checkpoint = tmp_path / "fake-metadata-only"
        checkpoint.mkdir()
        publish_json(
            checkpoint / "metadata.json", {"training_config": config.to_dict()}
        )
        config = _config()

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("registered declaration reached factory or model loading")

    monkeypatch.setattr(execute, "_execute_arm", forbidden)
    root = tmp_path / "unopened"
    with pytest.raises(ValueError, match="registered configuration"):
        execute.execute_synthetic_arm(
            config,
            arm="PPO" if checkpoint else "H",
            seed=SEED,
            checkpoint=checkpoint,
            directory=root,
            limits=LIMITS,
        )
    assert not root.exists()


@pytest.mark.parametrize(
    "arm,has_checkpoint", [("H", True), ("PPO", False), ("unknown", False)]
)
def test_actor_checkpoint_mismatch_is_rejected_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str, has_checkpoint: bool
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid actor/checkpoint declaration reached execution")

    monkeypatch.setattr(execute, "_execute_arm", forbidden)
    root = tmp_path / "unopened"
    with pytest.raises(ValueError):
        execute.execute_synthetic_arm(
            _config(),
            arm=arm,
            seed=SEED,
            directory=root,
            limits=LIMITS,
            checkpoint=tmp_path / "unopened-checkpoint" if has_checkpoint else None,
        )
    assert not root.exists()


def test_consumed_fixture_destination_cannot_run_again(
    executed: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("existing destination reached plant execution")

    monkeypatch.setattr(execute, "_execute_arm", forbidden)
    with pytest.raises(ValueError, match="already exists"):
        execute.execute_synthetic_arm(
            executed["config"],
            arm="H",
            seed=SEED,
            directory=executed["paths"]["H"],
            limits=LIMITS,
        )


def clone(
    executed: dict[str, Any],
    root: Path,
    *,
    arm: str = "PPO",
    mutation: str | None = None,
) -> Path:
    original = executed["paths"][arm]
    envelope, _ = archive.read_archive(
        original / compressed.ARM_FILENAME, **LIMITS.arm()
    )
    document, _ = archive.read_archive(
        original / compressed.TIMING_FILENAME, **LIMITS.timing()
    )
    root.mkdir()
    if mutation == "protected_seed":
        envelope["seed"] = 2_000_000
    elif mutation == "registered_task":
        envelope["training_task"] = registered_config(
            root.parent / "unused", "plant-beta-ppo-pi-lagrangian-v1", 101
        ).to_dict()
    elif mutation == "physical":
        envelope["run"]["economic_metrics"]["net_operating_value_eur"] += 1
    envelope = authenticate(envelope)
    arm_record = archive.publish_archive(
        root / compressed.ARM_FILENAME, envelope, **LIMITS.arm()
    )
    document["arm_archive"] = arm_record
    document["arm_envelope_sha256"] = envelope["sha256"]
    if mutation == "arm_reference":
        document["arm_archive"]["file"]["sha256"] = "0" * 64
    elif mutation == "timing_result":
        document["result"]["channel_samples_ns"]["inference"][0] += 1
    elif mutation == "missing_call":
        document["measurement"]["decisions"].pop()
        document["measurement"] = authenticate(document["measurement"])
    elif mutation == "host":
        document["host_before"]["machine"]["torch_intraop_threads"] += 1
        document["host_before"] = authenticate(document["host_before"])
    elif mutation == "claim":
        document["result_claim"] = "forged registered result"
    archive.publish_archive(
        root / compressed.TIMING_FILENAME, authenticate(document), **LIMITS.timing()
    )
    return root


@pytest.mark.parametrize(
    "mutation",
    [
        "protected_seed",
        "registered_task",
        "physical",
        "arm_reference",
        "timing_result",
        "missing_call",
        "host",
        "claim",
    ],
)
def test_rehashed_compressed_pair_tampering_is_rejected(
    executed: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = clone(executed, tmp_path / "tampered", mutation=mutation)
    if mutation in {"protected_seed", "registered_task"}:

        def forbidden(*args: object, **kwargs: object) -> None:
            pytest.fail("unadmitted compressed scope reached manifest reconstruction")

        monkeypatch.setattr(arms.verifier, "_verify_audit_manifest", forbidden)
    with pytest.raises(ValueError):
        compressed.verify_synthetic_execution(root, limits=LIMITS)


def test_identical_archive_replacement_during_reconstruction_rejected(
    executed: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = clone(executed, tmp_path / "replaced")
    verify = compressed.timing._project_latency

    def replace_file(*args: Any, **kwargs: Any) -> Any:
        result = verify(*args, **kwargs)
        target = root / compressed.ARM_FILENAME
        replacement = root.parent / "replacement.gz"
        replacement.write_bytes(target.read_bytes())
        replacement.replace(target)
        return result

    monkeypatch.setattr(compressed.timing, "_project_latency", replace_file)
    with pytest.raises(ValueError, match="changed during"):
        compressed.verify_synthetic_execution(root, limits=LIMITS)


@pytest.mark.parametrize("failure", ["warmup", "controller"])
def test_failure_retains_timing_and_never_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    config = _config()
    loaded = _loaded(tmp_path, config) if failure == "warmup" else None
    resets = []
    original_reset = PlantCyclicMarketEnvV2Beta.reset

    def reset(self: Any, *args: Any, **kwargs: Any) -> Any:
        resets.append(kwargs.get("seed"))
        return original_reset(self, *args, **kwargs)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("synthetic failure before any physical movement")

    monkeypatch.setattr(PlantCyclicMarketEnvV2Beta, "reset", reset)
    if loaded:
        monkeypatch.setattr(FixedActionPolicy, "act", fail)
    else:
        monkeypatch.setattr(
            execute.LegacyHeuristicSetpointControllerV1, "compute_command", fail
        )
    result = execute.execute_synthetic_arm(
        config,
        arm="PPO" if loaded else "H",
        seed=9_000_118,
        directory=tmp_path / "failed",
        checkpoint=None if loaded is None else loaded.path,
        limits=LIMITS,
    )
    assert result["physical"]["classification"]["physical_steps"] == 0
    assert result["physical"]["classification"]["failure"] is not None
    if loaded:
        assert resets == []
        assert result["latency"]["warmup_attempted_calls"] == 1
    else:
        assert resets == [9_000_118]
        assert result["latency"]["channels"]["controller"]["sample_count"] == 1
