"""Authority/record gates with explicit fake training; no policy or plant access."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from study_stage_fixtures import (
    authorize,
    training,
    patch_admission,
    process_record,
    rehash_file,
    SOURCE,
)
from plant_backend.rl.v2.study import registry, stage_authority as authority
from plant_backend.rl.v2.study import (
    stage_documents as documents,
    stage_campaign,
    stage_launch,
)
from plant_backend.rl.v2.training.custody_json import authenticate, canonical_bytes
from test_rl_v2_study_training_evidence import campaign as training_campaign


@pytest.fixture(autouse=True)
def no_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("stage authority attempted a real model or protected reset")

    from plant_backend.rl.v2 import beta_env
    from plant_backend.rl.v2.training import custody_checkpoint

    monkeypatch.setattr(beta_env.PlantCyclicMarketEnvV2Beta, "reset", forbidden)
    monkeypatch.setattr(custody_checkpoint, "load_policy", forbidden)
    monkeypatch.setattr(stage_launch, "_execute_arm", forbidden)


@pytest.fixture
def admitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    trained = training(tmp_path.resolve() / "training")
    authorize(trained)
    patch_admission(monkeypatch, trained)
    return authority.admit_stage(trained.path.parent, "development")


def test_fresh_training_all_15_and_distinct_latency_threads(admitted: Any) -> None:
    assert admitted.training.calls[0] == "fresh_training_admission"
    assert len(admitted.training.slots) == 15
    assert (
        admitted.authorization["latency_host"]["machine"]["torch_intraop_threads"] == 1
    )
    assert (
        admitted.authorization["latency_host"]["machine"]["torch_interop_threads"] == 1
    )
    assert admitted.training.identity.host["torch_intraop_threads"] == 8
    assert admitted.training.identity.host["torch_interop_threads"] == 12
    for cell in admitted.registry["cells"][:18]:
        config = admitted.config_for(cell)
        assert config.beta_rl is not None
        if cell["actor"]["training_order"] is not None:
            assert config.seed == cell["actor"]["optimizer_seed"]
            assert config.beta_rl.algorithm == cell["actor"]["algorithm"]
    assert not admitted.root.exists()


@pytest.mark.parametrize("defect", ["missing", "scope", "permission", "schema"])
def test_authority_scope_rejected_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
) -> None:
    trained = training(tmp_path.resolve() / "training")
    raw = authorize(trained)
    path = Path(
        registry.stage_registry(trained.path.parent, "development")[
            "authorization_input_path"
        ]
    )
    if defect == "missing":
        path.unlink()
    else:
        key, value = {
            "scope": ("stage", "heldout"),
            "permission": ("stage_execution_authorized", False),
            "schema": ("schema", "fixture"),
        }[defect]
        raw[key] = value
        path.write_bytes(canonical_bytes(authenticate(raw)))
    monkeypatch.setattr(
        authority, "admit_training", lambda _: pytest.fail("scope gate was bypassed")
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        authority.admit_stage(trained.path.parent, "development")


@pytest.mark.parametrize(
    "defect",
    [
        "authorization_id",
        "authorized_at_utc",
        "protocol",
        "registry_sha256",
        "training_inputs_sha256",
        "training_report",
        "source_provenance",
        "latency_host",
        "operator_assertions",
        "development",
        "reserve",
        "budget",
        "training_abort",
        "training_unclosed",
    ],
)
def test_rehashed_authority_substitution_rejected(admitted: Any, defect: str) -> None:
    raw = deepcopy(admitted.authorization)
    if defect == "reserve":
        raw["budget"] = replace(
            admitted.budget, filesystem_reserve_bytes=1024**3 - 1
        ).to_dict()
    elif defect == "budget":
        raw["budget"]["capacity_measured"] = True
        raw["budget"] = authenticate(raw["budget"])
    elif defect == "training_abort":
        admitted.training.report["abort"] = {"code": "storage"}
    elif defect == "training_unclosed":
        admitted.training.report["cells"][0]["state"] = "claimed_interrupted"
    else:
        raw[defect] = "rehashed substitution"
    Path(admitted.registry["authorization_input_path"]).write_bytes(
        canonical_bytes(authenticate(raw))
    )
    with pytest.raises((ValueError, TypeError, AttributeError)):
        authority.admit_stage(admitted.training.path.parent, "development")


def test_identical_authority_replacement_and_source_change(
    admitted: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = Path(admitted.registry["authorization_input_path"])
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(path.read_bytes())
    replacement.replace(path)
    with pytest.raises(ValueError, match="replaced"):
        admitted.recheck()
    fresh = authority.admit_stage(admitted.training.path.parent, "development")
    monkeypatch.setattr(
        authority.benchmark,
        "_source_provenance",
        lambda **_: {**SOURCE, "changed": True},
    )
    with pytest.raises(ValueError, match="source"):
        fresh.recheck()


@pytest.mark.parametrize(
    "defect",
    [None, "opening", "abort", "choices", "archive", "budget", "host", "inputs"],
)
def test_heldout_reconstructs_development_and_freezes_every_input(
    admitted: Any,
    monkeypatch: pytest.MonkeyPatch,
    defect: str | None,
) -> None:
    from plant_backend.rl.v2.study import stage_verifier

    choices = {family: 101 for family in registry.BETA_CORE_ALGORITHM_IDS}
    report = {
        "sha256": "d" * 64,
        "abort": None,
        "analysis": {
            "development_opening_requirements": {
                "all_calculated_requirements_met": True
            },
            "frozen_champions": choices,
        },
    }
    calls: list[str] = []
    prior = SimpleNamespace(
        context=admitted,
        report=report,
        archive={"fixture_archive": True},
        recheck=lambda: calls.append("recheck"),
    )

    def verify(root: Path, stage: str) -> Any:
        assert root == admitted.training.path.parent and stage == "development"
        calls.append("fresh_development_reconstruction")
        return prior

    monkeypatch.setattr(stage_verifier, "verify_stage", verify)
    pin = {
        "report_archive": prior.archive,
        "report_sha256": report["sha256"],
        "frozen_champions": choices,
    }
    authorize(admitted.training, "heldout", development=pin)
    path = Path(
        registry.stage_registry(admitted.training.path.parent, "heldout")[
            "authorization_input_path"
        ]
    )
    if defect == "opening":
        report["analysis"]["development_opening_requirements"][
            "all_calculated_requirements_met"
        ] = False
    elif defect == "abort":
        report["abort"] = {"code": "failed"}
    elif defect == "choices":
        rehash_file(
            path,
            lambda raw: raw["development"]["frozen_champions"].update(
                {next(iter(choices)): 211}
            ),
        )
    elif defect == "archive":
        rehash_file(
            path,
            lambda raw: raw["development"].update(report_archive={"different": True}),
        )
    elif defect in {"budget", "host", "inputs"}:
        # The held-out values are independently valid for training; substitution
        # of the development-side pin must still reject them.
        key = {
            "budget": "budget",
            "host": "latency_host",
            "inputs": "training_inputs_sha256",
        }[defect]
        prior.context = replace(
            admitted, authorization={**admitted.authorization, key: "different"}
        )
    if defect is None:
        result = authority.admit_stage(admitted.training.path.parent, "heldout")
        assert result.development is prior
        assert calls == ["fresh_development_reconstruction", "recheck"]
    else:
        with pytest.raises(ValueError):
            authority.admit_stage(admitted.training.path.parent, "heldout")
        assert calls == ["fresh_development_reconstruction"]


def test_opening_claim_chronology_and_exclusive_publication(
    admitted: Any, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="before"):
        documents.opening(
            admitted.authorization,
            admitted.registry,
            parent_pid=1,
            lock={},
            timestamp="2020-01-01T00:00:00.000000Z",
        )
    opened = documents.opening(
        admitted.authorization,
        admitted.registry,
        parent_pid=1,
        lock={},
        timestamp="2026-09-02T00:00:00.000000Z",
    )
    with pytest.raises(ValueError, match="chronology"):
        documents.claim(
            admitted.registry,
            opened,
            admitted.registry["cells"][0],
            None,
            timestamp=admitted.authorization["authorized_at_utc"],
        )
    path = tmp_path / "exclusive.json"
    documents.publish(path, opened)
    with pytest.raises(FileExistsError):
        documents.publish(path, opened)


@pytest.mark.parametrize(
    "defect,expected",
    [
        (None, None),
        ("exit", "child_nonzero_exit"),
        ("signal", "child_signaled"),
        ("timeout", "child_timeout"),
        ("cap", "child_log_limit_exceeded"),
        ("spawn", "child_spawn_failed"),
    ],
)
def test_process_failures_have_fixed_priority(
    admitted: Any, defect: str | None, expected: str | None
) -> None:
    from plant_backend.rl.v2.training.custody_documents import FILE_FIELDS

    cell = admitted.registry["cells"][0]
    process = process_record(admitted.registry, cell)
    log = dict.fromkeys(FILE_FIELDS, "fixture")
    log.update(logical_bytes=0, allocated_bytes=0)
    logs = {name: deepcopy(log) for name in ("stdout", "stderr")}
    if defect == "exit":
        process["exit_code"] = 2
    elif defect in {"signal", "timeout", "cap"}:
        process.update(exit_code=-15, termination_signal=15)
        if defect in {"timeout", "cap"}:
            process["wall_elapsed_ns"] = 60 * 10**9
        if defect == "cap":
            logs["stderr"]["logical_bytes"] = admitted.budget.log_limit_bytes
    elif defect == "spawn":
        process = {
            key: value if key in {"command", "timeout_seconds"} else None
            for key, value in process.items()
        }
        process["started"] = False
    assert (
        documents.process_failure(
            process, logs, documents.command(admitted.registry, cell), admitted.budget
        )
        == expected
    )


def test_parent_and_child_direct_commands_are_closed_before_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        stage_campaign,
        "admit_stage",
        lambda *_: pytest.fail("parent flag guard bypassed"),
    )
    with pytest.raises(ValueError, match="confirmed"):
        stage_campaign.run_stage(tmp_path, "development")
    monkeypatch.delenv(stage_launch.LAUNCH_FD_ENV, raising=False)
    with pytest.raises(ValueError, match="live admitted parent"):
        stage_launch.run_child(tmp_path, "development", 1)


def test_stage_admission_reopens_complete_training_custody_with_fixture_models(
    training_campaign: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full receipt/checkpoint bridge, with only explicit synthetic dependencies."""
    import test_rl_v2_study_training_evidence as fixtures
    from plant_backend.rl.v2.study import training_evidence
    from plant_backend.rl.v2.training import custody_documents as training_documents

    original = fixtures.context

    def with_cpu(root: Path) -> Any:
        ctx = original(root)
        host = {**ctx.identity.host, "cpu_brand": "Explicit custody fixture CPU"}
        permission = authenticate(
            {
                **ctx.authorization,
                "host_contract": {"host": host, "capacity_projection_applies": False},
            }
        )
        identity = replace(ctx.identity, host=host)
        return replace(
            ctx,
            authorization=permission,
            identity=identity,
            plan=training_documents.make_plan(
                root,
                authorization=permission,
                source=identity.source,
                runtime=identity.runtime,
                host=host,
            ),
        )

    monkeypatch.setattr(fixtures, "context", with_cpu)
    fixture = training_campaign()
    trained = training_evidence.admit_training(fixture.path)
    authorize(trained)
    monkeypatch.setattr(
        authority.benchmark, "_source_provenance", lambda **_: deepcopy(SOURCE)
    )
    result = authority.admit_stage(fixture.path.parent, "development")
    assert len(fixture.admitted) == len(fixture.parsed) == 30
    assert all(len(slot.rollouts) == 128 for slot in result.training.slots)
    assert result.training.evidence == trained.evidence
    assert not result.root.exists()


@pytest.mark.parametrize("command", ["run", "child", "verify"])
def test_cli_closed_entry_points_do_not_open_any_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    command: str,
) -> None:
    from plant_backend.rl.v2.study.stage_cli import main

    monkeypatch.delenv(stage_launch.LAUNCH_FD_ENV, raising=False)
    path = tmp_path.resolve() / "unopened"
    args = [command, "--training-root", str(path), "--stage", "development"]
    if command == "child":
        args.extend(["--order", "1"])
    assert main(args) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert "evidence and consumed attempts are retained" in captured.err
    assert not list(tmp_path.iterdir())
