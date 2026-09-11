"""Reset-free custody integration using explicit synthetic host/model dependencies.

These tests do not admit real training: provenance, checkpoint loading and the
endpoint-evidence parser are fixture substitutes. The complete receipt/report
reconstruction, file graph, ledger hashes and scorecards execute unchanged.
"""

from copy import deepcopy
from dataclasses import replace
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from custody_fixtures import FixtureServices, admit, context, sha
from plant_backend.rl.v2.study import training_evidence as bridge
from plant_backend.rl.v2.training import custody_documents as docs
from plant_backend.rl.v2.training import custody_verifier as verifier
from plant_backend.rl.v2.training.constrained_ppo import (
    PILagrangeStateV1,
    RolloutLedgerV1,
)
from plant_backend.rl.v2.training.custody_checkpoint import CheckpointAdmissionFailure
from plant_backend.rl.v2.training.config import BETA_SCALAR_PPO_ALGORITHM_ID
from plant_backend.rl.v2.training.custody_json import (
    authenticate,
    canonical_bytes,
    decode,
)
from plant_backend.rl.v2.training.custody_synthetic import run_synthetic
from plant_backend.rl.v2.training.registered_contract import matrix


@pytest.fixture(autouse=True)
def no_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    from plant_backend.rl.v2.training import custody_checkpoint, ppo
    from plant_backend.rl.v2 import beta_env

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("training-input fixture reached a real model or plant")

    monkeypatch.setattr(ppo, "make_env", forbidden)
    monkeypatch.setattr(custody_checkpoint, "make_env", forbidden)
    monkeypatch.setattr(custody_checkpoint, "load_policy", forbidden)
    monkeypatch.setattr(beta_env.PlantCyclicMarketEnvV2Beta, "reset", forbidden)


def ledgers(algorithm: str) -> list[dict[str, Any]]:
    rows = []
    for index in range(1, 129):
        rows.append(
            RolloutLedgerV1.build(
                algorithm_id=algorithm,
                rollout_index=index,
                transition_count=8192,
                completed_episode_count=0,
                complete_cycle_count=0,
                reward_return=0,
                raw_cost=(0, 0, 0, 0),
                training_cost=(0, 0, 0, 0),
                mean_completed_episode_cost=None,
                cost_value_losses=(0, 0, 0, 0),
                multiplier_state=PILagrangeStateV1(
                    proportional=(0, 0, 0, 0),
                    integral=(0, 0, 0, 0),
                    multipliers=(0, 0, 0, 0),
                    completed_update_count=0,
                ),
                multiplier_min=0,
                multiplier_max=100,
                approximate_kl=0,
                entropy_loss=0,
                clip_fraction=0,
                optimizer_steps=320,
                elapsed_compute_s=0.01,
                previous_sha256=None if not rows else rows[-1]["sha256"],
            ).to_dict()
        )
    return rows


@pytest.fixture
def campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    def build(*, defect: str | None = None) -> SimpleNamespace:
        ctx = context(tmp_path / "campaign")
        fixture = FixtureServices(ctx)
        if defect == "process":
            fixture.process_code = "child_nonzero_exit"
        ledger_map = {
            cell["algorithm"]: ledgers(cell["algorithm"])
            for cell in ctx.plan["cells"][::5]
        }

        def process(cell: dict[str, Any]) -> Any:
            result = fixture.process(cell)
            metadata = {
                "extra": {
                    bridge.BETA_CHECKPOINT_EXTRA_KEY: {
                        "fixture_only": True,
                        "algorithm": cell["algorithm"],
                        "optimizer_seed": cell["optimizer_seed"],
                        "rollouts": ledger_map[cell["algorithm"]],
                    }
                }
            }
            (Path(cell["output_path"]) / "checkpoint/metadata.json").write_bytes(
                canonical_bytes(metadata)
            )
            if defect == "forbidden_artifact" and cell["optimizer_seed"] == 211:
                (Path(cell["output_path"]) / "retained-link").symlink_to(
                    "missing-target"
                )
            return result

        admitted: list[tuple[str, int]] = []

        def checkpoint(config: Any, runtime: Any) -> dict[str, Any]:
            assert config.beta_rl is not None
            admitted.append((config.beta_rl.algorithm, config.seed))
            if defect == "checkpoint" and config.seed == 211:
                raise CheckpointAdmissionFailure("checkpoint_admission_failed")
            if defect == "forbidden_artifact" and config.seed == 211:
                raise CheckpointAdmissionFailure("artifact_inventory_invalid")
            return admit(config, runtime)

        services = replace(fixture.services, process=process, checkpoint=checkpoint)
        if defect == "storage":
            calls = 0

            def budget(root: Path, minimum: int) -> bool:
                nonlocal calls
                calls += 1
                return calls < 3

            services = replace(services, budget=budget)
        report = run_synthetic(ctx, services)
        # Reclassification is solely for testing the unchanged production
        # reconstruction with patched authority/model dependencies below.
        report = authenticate(
            {
                **report,
                "schema": docs.REPORT_SCHEMA,
                "claim_boundary": docs.CLAIM_BOUNDARY,
                "result_claim": (
                    docs.SUCCESS_CLAIM
                    if report["campaign_status"] == "completed"
                    else None
                ),
            }
        )
        path = ctx.root / "report.json"
        path.write_bytes(canonical_bytes(report))
        (ctx.root / "synthetic-report.json").unlink()
        admitted.clear()
        monkeypatch.setattr(
            verifier.capacity, "_host_identity", lambda: deepcopy(ctx.identity.host)
        )
        monkeypatch.setattr(
            verifier.authority,
            "admit_authorization",
            lambda root, host: (ctx.authorization, ctx.identity.source),
        )
        monkeypatch.setattr(
            verifier, "capture_identity", lambda authorization: deepcopy(ctx.identity)
        )
        monkeypatch.setattr(
            bridge, "capture_identity", lambda authorization: deepcopy(ctx.identity)
        )
        monkeypatch.setattr(verifier, "admit_checkpoint", checkpoint)
        parsed: list[tuple[str, int]] = []

        def parse(raw: dict[str, Any]) -> SimpleNamespace:
            assert raw["fixture_only"] is True
            parsed.append((raw["algorithm"], raw["optimizer_seed"]))
            return SimpleNamespace(
                sha256=sha("synthetic checkpoint"),
                runtime_provenance=SimpleNamespace(
                    sha256=ctx.identity.runtime["sha256"]
                ),
                constraint_state={
                    "payload": {"rollout_ledgers": deepcopy(raw["rollouts"])}
                },
            )

        monkeypatch.setattr(bridge, "parse_beta_checkpoint_evidence", parse)
        return SimpleNamespace(
            ctx=ctx,
            path=path,
            report=report,
            admitted=admitted,
            parsed=parsed,
            parse=parse,
        )

    return build


def test_complete_verifier_supplies_every_checkpoint_and_128_rollouts(
    campaign: Any,
) -> None:
    fixture = campaign()
    result = bridge.admit_training(fixture.path)
    expected = [
        (cell["algorithm"], cell["optimizer_seed"]) for cell in matrix(fixture.ctx.root)
    ]
    assert fixture.admitted == expected == fixture.parsed
    assert len(result.slots) == 15
    assert all(
        slot.training_completed and slot.checkpoint_admitted for slot in result.slots
    )
    for slot in result.slots:
        assert slot.rollouts is not None and len(slot.rollouts) == 128
        assert sum(row["transition_count"] for row in slot.rollouts) == 1_048_576
    assert result.evidence["evaluation_authorized"] is False
    assert result.evidence["result_claim"] is None
    assert result.evidence["scorecard"]["checkpoint_admitted_count"] == 15
    assert result.evidence["training_report"]["sha256"] == fixture.report["sha256"]
    result.recheck()
    verdict = verifier.verify_report(fixture.path)
    assert verdict == {
        "verified": True,
        "schema": docs.REPORT_SCHEMA,
        "report_sha256": fixture.report["sha256"],
        "campaign_status": "completed",
        "status_counts": fixture.report["status_counts"],
        "result_claim": docs.SUCCESS_CLAIM,
    }
    assert fixture.admitted == expected * 2  # compact CLI also performs full admission


@pytest.mark.parametrize(
    "defect", ["process", "checkpoint", "storage", "forbidden_artifact"]
)
def test_all_failed_and_unstarted_slots_remain_visible(
    campaign: Any, defect: str
) -> None:
    fixture = campaign(defect=defect)
    result = bridge.admit_training(fixture.path)
    assert len(result.slots) == 15
    admitted = sum(slot.checkpoint_admitted for slot in result.slots)
    assert len(fixture.parsed) == admitted < 15
    for slot, record in zip(result.slots, fixture.report["cells"], strict=True):
        if not slot.checkpoint_admitted:
            assert slot.rollouts is None
            assert slot.failure == (
                "not_started"
                if record["state"] == "not_started"
                else record["failure"]["code"]
            )
            assert slot.training_completed == (record["state"] == "checkpoint_rejected")
    assert result.evidence["training_abort"] == fixture.report["abort"]


@pytest.mark.parametrize("door", ["admit", "report", "compact"])
@pytest.mark.parametrize("kind", ["synthetic", "cached", "wrong_name"])
def test_wrong_entry_points_reject_before_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, door: str, kind: str
) -> None:
    path = tmp_path / (
        "synthetic-report.json" if kind == "wrong_name" else "report.json"
    )
    path.write_bytes(
        canonical_bytes(
            authenticate({"schema": docs.SYNTHETIC_REPORT_SCHEMA})
            if kind != "cached"
            else {"verified": True}
        )
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("unadmitted schema reached source, metadata or model dependencies")

    monkeypatch.setattr(bridge, "_tree_stamps", forbidden)
    monkeypatch.setattr(verifier.capacity, "_host_identity", forbidden)
    monkeypatch.setattr(verifier, "admit_checkpoint", forbidden)
    call = {
        "admit": bridge.admit_training,
        "report": verifier.read_verified_report,
        "compact": verifier.verify_report,
    }[door]
    with pytest.raises(ValueError):
        call(path)


@pytest.mark.parametrize(
    "part", ["policy", "metadata", "config", "log", "report", "extra", "same_bytes"]
)
def test_parent_recheck_rejects_any_changed_graph_node(
    campaign: Any, part: str
) -> None:
    fixture = campaign()
    result = bridge.admit_training(fixture.path)
    cell = fixture.ctx.plan["cells"][0]
    paths = {
        "policy": Path(cell["output_path"]) / "checkpoint/policy.zip",
        "metadata": Path(cell["output_path"]) / "checkpoint/metadata.json",
        "config": Path(cell["config_path"]),
        "log": Path(cell["stdout_path"]),
        "report": fixture.path,
        "extra": fixture.ctx.root / "new-file",
        "same_bytes": fixture.path,
    }
    path = paths[part]
    if part == "same_bytes":
        replacement = fixture.path.parent.parent / "replacement.json"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    else:
        with path.open("ab") as handle:
            handle.write(b"retained mutation")
    with pytest.raises(ValueError, match="graph changed"):
        result.recheck()


@pytest.mark.parametrize("component", ["host", "source", "runtime"])
def test_parent_training_identity_is_not_relaxed_to_evaluation_threads(
    campaign: Any, monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    fixture = campaign()
    result = bridge.admit_training(fixture.path)
    value = deepcopy(getattr(fixture.ctx.identity, component))
    if component == "host":
        value["torch_intraop_threads"] = 1
        value["torch_interop_threads"] = 1
    elif component == "source":
        value["authorization_commit"] = "d" * 40
    else:
        value["sha256"] = "e" * 64
    changed = replace(fixture.ctx.identity, **{component: value})
    monkeypatch.setattr(bridge, "capture_identity", lambda authority: changed)
    with pytest.raises(ValueError, match="source/runtime/host changed"):
        result.recheck()


@pytest.mark.parametrize(
    "mutation",
    ["digest", "runtime", "short", "steps", "chain", "algorithm", "replacement"],
)
def test_bridge_binds_reopened_metadata_and_full_ledger(
    campaign: Any, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    fixture = campaign()

    def parse(raw: dict[str, Any]) -> Any:
        result = fixture.parse(raw)
        rows = result.constraint_state["payload"]["rollout_ledgers"]
        if mutation == "digest":
            result.sha256 = "f" * 64
        elif mutation == "runtime":
            result.runtime_provenance.sha256 = "f" * 64
        elif mutation == "short":
            rows.pop()
        elif mutation == "steps":
            rows[0]["optimizer_steps"] += 1
        elif mutation == "chain":
            rows[0]["reward_return"] += 1
        elif mutation == "algorithm":
            result.constraint_state["payload"]["rollout_ledgers"] = ledgers(
                BETA_SCALAR_PPO_ALGORITHM_ID
            )
        else:
            path = (
                Path(fixture.ctx.plan["cells"][0]["output_path"])
                / "checkpoint/metadata.json"
            )
            replacement = fixture.ctx.root.parent / "replacement"
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        return result

    monkeypatch.setattr(bridge, "parse_beta_checkpoint_evidence", parse)
    with pytest.raises(ValueError):
        bridge.admit_training(fixture.path)


def test_verification_failure_prevents_any_ledger_projection(
    campaign: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = campaign()

    def reject(*args: object, **kwargs: object) -> None:
        raise CheckpointAdmissionFailure("checkpoint_admission_failed")

    monkeypatch.setattr(verifier, "admit_checkpoint", reject)
    with pytest.raises(ValueError, match="terminal status"):
        bridge.admit_training(fixture.path)
    assert fixture.parsed == []


def test_new_public_entry_points_have_no_cached_verdict_or_fixture_hook() -> None:
    for function in (
        bridge.admit_training,
        verifier.read_verified_report,
        verifier.verify_report,
    ):
        assert tuple(inspect.signature(function).parameters) == ("path",)
