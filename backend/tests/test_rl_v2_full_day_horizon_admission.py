"""Outcome-free adversarial tests for the frozen M24 admission boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from plant_backend.rl.v2 import full_day_horizon_admission as admission
from plant_backend.rl.v2.mpc_schema import canonical_json


def _write_canonical(path: Path, payload: dict[str, Any]) -> str:
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _cell_axes(cell_id: str) -> tuple[float, float, int]:
    horizon, rate, seed = cell_id.split("-")
    return float(horizon[1:]), float(rate[1:]), int(seed[1:])


def _arm(label: str, *, marker: str) -> dict[str, Any]:
    common: dict[str, Any] = {
        "arm": label,
        "status": "complete",
        "manifest_identity": {"manifest_sha256": "1" * 64},
        "completed_decisions": 1,
        "physical_steps": 1,
        "terminated": True,
        "truncated": False,
        "reward_trace_eur": [0.0],
        "reward_trace_sum_eur": 0.0,
        "interval_totals": {},
        "execution_count": 1,
        "application_count": 1,
        "physical_metrics": {"synthetic": 0.0},
        "economic_metrics": {"synthetic": 0.0},
        "failure": None,
        "outcome_secret_test_marker": marker,
    }
    if label == "H":
        common.update(
            {
                "controller": {"id": admission.LEGACY_HEURISTIC_SETPOINT_CONTROLLER_ID},
                "evidence": {"sha256": "2" * 64},
            }
        )
    else:
        settings_sha = "3" * 64
        common.update(
            {
                "controller": {
                    "id": admission.SERVICE_MPC_SETPOINT_CONTROLLER_ID,
                    "settings": {"sha256": settings_sha},
                },
                "evidence": {
                    "controller_id": admission.SERVICE_MPC_SETPOINT_CONTROLLER_ID,
                    "settings_sha256": settings_sha,
                    "evidence_graph_sha256": "4" * 64,
                    "sha256": "5" * 64,
                },
            }
        )
    return common


@dataclass
class SyntheticReference:
    root: Path
    repository: Path
    directory: Path
    report: Path
    output: Path
    identity: admission.ReferenceIdentityV1
    verifier_calls: list[Path]
    source_calls: list[tuple[Path, str]]

    def verifier(self, report: Path) -> dict[str, Any]:
        self.verifier_calls.append(report)
        return {
            "path": str(report),
            "raw_file_sha256": self.identity.raw_report_sha256,
            "report_sha256": self.identity.report_domain_sha256,
            "deterministic_evidence_sha256": (
                self.identity.deterministic_evidence_sha256
            ),
            "status": self.identity.report_status,
            "cell_count": 20,
        }

    def source_provider(
        self, repository: Path, revision: str
    ) -> admission.SourceDiffSnapshotV1:
        self.source_calls.append((repository, revision))
        return _clean_source_snapshot(revision)

    def admit(self) -> admission.ReferenceAdmissionV1:
        return admission.admit_full_day_horizon_probe(
            self.report,
            self.output,
            self.repository,
            self.verifier,
            self.source_provider,
            identity=self.identity,
        )


def _clean_source_snapshot(revision: str) -> admission.SourceDiffSnapshotV1:
    return admission.SourceDiffSnapshotV1(
        base_revision=revision,
        target_revision="d" * 40,
        worktree_clean=True,
        entries=tuple(
            admission.SourceDiffEntryV1(status="A", path=path)
            for path in admission.ALLOWED_NEW_M24_RUNTIME_PATHS
        ),
    )


@pytest.fixture
def reference(tmp_path: Path) -> SyntheticReference:
    repository = tmp_path / "repository"
    repository.mkdir()
    lock_bytes = b"synthetic frozen dependency lock\n"
    (repository / "uv.lock").write_bytes(lock_bytes)
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    directory = tmp_path / "historical-reference"
    directory.mkdir()

    references: list[dict[str, Any]] = []
    filenames = admission.registered_reference_filenames()
    for index, filename in enumerate(filenames, start=1):
        cell_id = filename.split("_", maxsplit=1)[1].removesuffix(".json")
        horizon, rate, seed = _cell_axes(cell_id)
        cell = {
            "id": cell_id,
            "horizon_h": horizon,
            "delivery_rate_kg_h": rate,
            "seed": seed,
        }
        payload = {
            "format": "synthetic-reference-cell-v1",
            "status": "complete",
            "cell": cell,
            "arms": {
                "H": _arm("H", marker=f"historical-H-{index}"),
                "M": _arm("M", marker=f"historical-M6-{index}"),
            },
            "deterministic_evidence_sha256": f"{index + 100:064x}",
            "sha256": f"{index + 200:064x}",
        }
        raw_sha = _write_canonical(directory / filename, payload)
        references.append(
            {
                "index": index,
                "file_name": filename,
                "cell": cell,
                "status": "complete",
                "raw_file_sha256": raw_sha,
                "report_sha256": payload["sha256"],
                "deterministic_evidence_sha256": payload[
                    "deterministic_evidence_sha256"
                ],
                "artifact_verified": True,
                "verification_failure": None,
            }
        )

    source_revision = "a" * 40
    worktree_sha = "b" * 64
    contract_sha = "c" * 64
    report_sha = "d" * 64
    deterministic_sha = "e" * 64
    provenance = {
        "source": {
            "git_dirty": False,
            "identity_complete": True,
            "git_revision": source_revision,
            "worktree_identity_sha256": worktree_sha,
        },
        "dependency_lock": {"sha256": lock_sha},
        "runtime": {"synthetic": True},
    }
    report_payload = {
        "format": "synthetic-reference-report-v1",
        "contract_sha256": contract_sha,
        "status": admission.HISTORICAL_REPORT_STATUS,
        "source_gate": {
            "status": "passed",
            "start": provenance,
            "end": json.loads(canonical_json(provenance)),
        },
        "cells": references,
        "deterministic_evidence_sha256": deterministic_sha,
        "sha256": report_sha,
    }
    report = directory / admission.REFERENCE_REPORT_FILENAME
    raw_report_sha = _write_canonical(report, report_payload)
    identity = admission.ReferenceIdentityV1(
        executed_source_revision=source_revision,
        worktree_identity_sha256=worktree_sha,
        dependency_lock_sha256=lock_sha,
        contract_sha256=contract_sha,
        raw_report_sha256=raw_report_sha,
        report_domain_sha256=report_sha,
        deterministic_evidence_sha256=deterministic_sha,
        cell_filenames=filenames,
    )
    return SyntheticReference(
        root=tmp_path,
        repository=repository,
        directory=directory,
        report=report,
        output=tmp_path / "m24-output",
        identity=identity,
        verifier_calls=[],
        source_calls=[],
    )


def test_valid_admission_returns_all_citations_and_only_full_day_runner_inputs(
    reference: SyntheticReference,
) -> None:
    result = reference.admit()

    assert len(result.cells) == 20
    assert [cell.index for cell in result.cells] == list(range(1, 21))
    assert len(result.selected_cells) == 10
    assert all(cell.citation.horizon_h == 24.0 for cell in result.selected_cells)
    assert list(result.runner_inputs_by_cell_id) == [
        cell.cell_id for cell in result.cells[10:]
    ]
    first_payload, heuristic, m6, citation = next(
        iter(result.runner_inputs_by_cell_id.values())
    )
    assert first_payload["arms"]["H"] is heuristic
    assert first_payload["arms"]["M"] is m6
    assert citation.heuristic.retained_evidence_sha256 == "2" * 64
    assert citation.m6.retained_evidence_sha256 == "5" * 64
    assert citation.m6.evidence_graph_sha256 == "4" * 64
    assert reference.verifier_calls == [reference.report.resolve()]
    assert reference.source_calls == [
        (reference.repository.resolve(), reference.identity.executed_source_revision)
    ]

    serialized = canonical_json(result.to_dict())
    assert "outcome_secret_test_marker" not in serialized
    assert "historical-H-11" not in serialized
    assert "historical-M6-11" not in serialized
    assert len(result.to_dict()["cells"]) == 20


def test_registered_reference_has_exact_twenty_file_order_and_frozen_identities() -> (
    None
):
    identity = admission.REGISTERED_REFERENCE_IDENTITY

    assert len(identity.cell_filenames) == 20
    assert identity.cell_filenames[0] == "01_h6-r35-s20.json"
    assert identity.cell_filenames[10] == "11_h24-r35-s20.json"
    assert identity.cell_filenames[-1] == "20_h24-r42-s24.json"
    assert identity.executed_source_revision == admission.HISTORICAL_EXECUTED_REVISION
    assert identity.worktree_identity_sha256 == (
        admission.HISTORICAL_WORKTREE_IDENTITY_SHA256
    )


@pytest.mark.parametrize(
    "raw, message",
    [
        (b'{"value":1,"value":2}\n', "duplicate"),
        (b'{"value":NaN}\n', "non-finite"),
        (b'{"value":Infinity}\n', "non-finite"),
        (b'{"value":1e999}\n', "non-finite"),
        (b'{ "value":1}\n', "not canonical"),
        (b'{"value":1}', "trailing newline"),
        (b'{"value":1}\n\n', "trailing newline"),
        (b'{"value":1}\r\n', "not canonical"),
    ],
)
def test_canonical_reader_rejects_ambiguous_or_noncanonical_bytes(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    target = tmp_path / "fixture.json"
    target.write_bytes(raw)

    with pytest.raises(ValueError, match=message):
        admission.read_canonical_json(target)


def test_canonical_reader_rejects_nested_duplicate_keys(tmp_path: Path) -> None:
    target = tmp_path / "fixture.json"
    target.write_bytes(b'{"outer":{"same":1,"same":2}}\n')

    with pytest.raises(ValueError, match="duplicate"):
        admission.read_canonical_json(target)


def test_canonical_reader_rejects_symlink_and_nonregular_entry(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    _write_canonical(real, {"value": 1})
    link = tmp_path / "link.json"
    link.symlink_to(real)
    directory = tmp_path / "directory.json"
    directory.mkdir()

    with pytest.raises(ValueError, match="symlink"):
        admission.read_canonical_json(link)
    with pytest.raises(ValueError, match="non-regular"):
        admission.read_canonical_json(directory)


@pytest.mark.parametrize("mutation", ["missing", "extra", "symlink"])
def test_reference_directory_requires_exact_regular_nonsymlink_custody(
    reference: SyntheticReference, mutation: str
) -> None:
    first = reference.directory / reference.identity.cell_filenames[0]
    if mutation == "missing":
        first.unlink()
    elif mutation == "extra":
        _write_canonical(reference.directory / "extra.json", {"extra": True})
    else:
        first.unlink()
        first.symlink_to(reference.report)

    with pytest.raises(ValueError, match="21 entries|regular nonsymlink"):
        reference.admit()
    assert reference.verifier_calls == []
    assert reference.source_calls == []


@pytest.mark.parametrize(
    "field",
    [
        "raw_report_sha256",
        "report_domain_sha256",
        "deterministic_evidence_sha256",
        "contract_sha256",
        "executed_source_revision",
        "worktree_identity_sha256",
        "dependency_lock_sha256",
    ],
)
def test_every_registered_root_identity_is_bound_before_deep_verification(
    reference: SyntheticReference, field: str
) -> None:
    replacement = "f" * (40 if field == "executed_source_revision" else 64)
    reference.identity = replace(reference.identity, **{field: replacement})

    with pytest.raises(ValueError, match="historical|current dependency lock"):
        reference.admit()
    assert reference.verifier_calls == []


def test_reference_report_expected_gate_failure_is_valid_retained_evidence(
    reference: SyntheticReference,
) -> None:
    result = reference.admit()

    assert result.report.status == "complete_gate_failed"


def test_report_reference_must_bind_each_cell_raw_and_domain_identity(
    reference: SyntheticReference,
) -> None:
    payload = admission.read_canonical_json(reference.report)
    payload["cells"][4]["raw_file_sha256"] = "0" * 64
    new_raw = _write_canonical(reference.report, payload)
    reference.identity = replace(reference.identity, raw_report_sha256=new_raw)

    with pytest.raises(ValueError, match="cell raw identity"):
        reference.admit()
    assert reference.verifier_calls == []


@pytest.mark.parametrize(
    "key",
    [
        "raw_file_sha256",
        "report_sha256",
        "deterministic_evidence_sha256",
        "status",
        "cell_count",
    ],
)
def test_injected_deep_verifier_must_reproduce_every_compact_identity(
    reference: SyntheticReference, key: str
) -> None:
    def wrong_verifier(path: Path) -> dict[str, Any]:
        result = reference.verifier(path)
        result[key] = 19 if key == "cell_count" else "wrong"
        return result

    with pytest.raises(ValueError, match=f"wrong {key}"):
        admission.admit_full_day_horizon_probe(
            reference.report,
            reference.output,
            reference.repository,
            wrong_verifier,
            reference.source_provider,
            identity=reference.identity,
        )
    assert reference.source_calls == []


def test_deep_verifier_mutation_is_detected_before_return(
    reference: SyntheticReference,
) -> None:
    target = reference.directory / reference.identity.cell_filenames[-1]

    def mutating_verifier(path: Path) -> dict[str, Any]:
        result = reference.verifier(path)
        payload = admission.read_canonical_json(target)
        payload["post_verifier_mutation"] = True
        _write_canonical(target, payload)
        return result

    with pytest.raises(RuntimeError, match="changed during preflight"):
        admission.admit_full_day_horizon_probe(
            reference.report,
            reference.output,
            reference.repository,
            mutating_verifier,
            reference.source_provider,
            identity=reference.identity,
        )


@pytest.mark.parametrize(
    "entry, message",
    [
        (
            admission.SourceDiffEntryV1(
                status="M", path="backend/src/plant_backend/rl/v2/config.py"
            ),
            "historical protected runtime file",
        ),
        (
            admission.SourceDiffEntryV1(
                status="D", path="simulator/src/plant_sim/plant.py"
            ),
            "historical protected runtime file",
        ),
        (
            admission.SourceDiffEntryV1(
                status="R100",
                old_path="backend/src/plant_backend/controllers/base.py",
                path="backend/src/plant_backend/controllers/core.py",
            ),
            "renaming",
        ),
        (
            admission.SourceDiffEntryV1(
                status="A",
                path="backend/src/plant_backend/rl/v2/undeclared_m24.py",
            ),
            "undeclared",
        ),
        (
            admission.SourceDiffEntryV1(
                status="T", path="backend/src/plant_backend/safety/guard.py"
            ),
            "non-additive",
        ),
    ],
)
def test_source_diff_rejects_every_protected_non_allowlisted_change(
    entry: admission.SourceDiffEntryV1, message: str
) -> None:
    clean = _clean_source_snapshot(admission.HISTORICAL_EXECUTED_REVISION)
    snapshot = replace(clean, entries=(*clean.entries, entry))

    with pytest.raises(ValueError, match=message):
        admission.validate_source_diff(snapshot)


def test_source_diff_allows_docs_tests_and_exact_new_runtime_modules() -> None:
    clean = _clean_source_snapshot(admission.HISTORICAL_EXECUTED_REVISION)
    snapshot = replace(
        clean,
        entries=(
            *clean.entries,
            admission.SourceDiffEntryV1(status="M", path="README.md"),
            admission.SourceDiffEntryV1(status="A", path="backend/tests/test_m24.py"),
            admission.SourceDiffEntryV1(
                status="M", path="docs/research/FULL_DAY_PROTOCOL.md"
            ),
        ),
    )

    result = admission.validate_source_diff(snapshot)

    assert result.admitted_new_runtime_paths == admission.ALLOWED_NEW_M24_RUNTIME_PATHS


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        "uv.lock",
        "backend/pyproject.toml",
        "backend/src/plant_backend/app.py",
        "frontend/src/App.tsx",
        ".github/workflows/ci.yml",
    ],
)
def test_source_diff_rejects_every_undeclared_non_runtime_surface(path: str) -> None:
    clean = _clean_source_snapshot(admission.HISTORICAL_EXECUTED_REVISION)
    snapshot = replace(
        clean,
        entries=(
            *clean.entries,
            admission.SourceDiffEntryV1(status="M", path=path),
        ),
    )

    with pytest.raises(ValueError, match="clarification 002"):
        admission.validate_source_diff(snapshot)


def test_source_diff_requires_historical_base_cleanliness_and_all_modules() -> None:
    clean = _clean_source_snapshot(admission.HISTORICAL_EXECUTED_REVISION)

    with pytest.raises(ValueError, match="historical revision"):
        admission.validate_source_diff(replace(clean, base_revision="0" * 40))
    with pytest.raises(ValueError, match="clean source worktree"):
        admission.validate_source_diff(replace(clean, worktree_clean=False))
    with pytest.raises(ValueError, match="omits declared"):
        admission.validate_source_diff(replace(clean, entries=clean.entries[:-1]))


def test_source_diff_rejects_unresolved_status_even_outside_runtime_surfaces() -> None:
    clean = _clean_source_snapshot(admission.HISTORICAL_EXECUTED_REVISION)
    snapshot = replace(
        clean,
        entries=(
            *clean.entries,
            admission.SourceDiffEntryV1(status="U", path="README.md"),
        ),
    )

    with pytest.raises(ValueError, match="unresolved"):
        admission.validate_source_diff(snapshot)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": "M", "path": "../escape.py"},
        {"status": "M", "path": "/absolute.py"},
        {"status": "M", "path": "backend\\ambiguous.py"},
        {"status": "R100", "path": "new.py"},
        {"status": "A", "path": "new.py", "old_path": "old.py"},
    ],
)
def test_source_diff_entries_reject_ambiguous_paths_and_shapes(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        admission.SourceDiffEntryV1(**kwargs)


def test_source_provider_must_return_authenticated_snapshot(
    reference: SyntheticReference,
) -> None:
    def unauthenticated(_: Path, __: str) -> list[object]:
        return []

    with pytest.raises(TypeError, match="mapping"):
        admission.admit_full_day_horizon_probe(
            reference.report,
            reference.output,
            reference.repository,
            reference.verifier,
            unauthenticated,  # type: ignore[arg-type]
            identity=reference.identity,
        )


def test_current_lock_is_checked_before_source_diff_provider(
    reference: SyntheticReference,
) -> None:
    (reference.repository / "uv.lock").write_bytes(b"changed lock\n")

    with pytest.raises(ValueError, match="current dependency lock"):
        reference.admit()
    assert reference.verifier_calls == [reference.report.resolve()]
    assert reference.source_calls == []


@pytest.mark.parametrize(
    "case",
    ["relative", "inside_repository", "inside_reference", "existing", "missing_parent"],
)
def test_output_must_be_new_absolute_external_path_with_existing_parent(
    reference: SyntheticReference, case: str
) -> None:
    if case == "relative":
        output = Path("relative-output")
        error = ValueError
    elif case == "inside_repository":
        output = reference.repository / "output"
        error = ValueError
    elif case == "inside_reference":
        output = reference.directory / "output"
        error = ValueError
    elif case == "existing":
        output = reference.root / "existing-output"
        output.mkdir()
        error = FileExistsError
    else:
        output = reference.root / "missing-parent" / "output"
        error = ValueError

    with pytest.raises(error):
        admission.validate_output_directory(
            output,
            repository_root=reference.repository,
            reference_directory=reference.directory,
        )


def test_invalid_output_refuses_before_opening_reference_or_source(
    reference: SyntheticReference,
) -> None:
    with pytest.raises(ValueError, match="outside the repository"):
        admission.admit_full_day_horizon_probe(
            reference.report,
            reference.repository / "forbidden-output",
            reference.repository,
            reference.verifier,
            reference.source_provider,
            identity=reference.identity,
        )

    assert reference.verifier_calls == []
    assert reference.source_calls == []


def test_reference_directory_alias_is_rejected(reference: SyntheticReference) -> None:
    alias = reference.root / "reference-alias"
    alias.symlink_to(reference.directory, target_is_directory=True)

    with pytest.raises(ValueError, match="nonsymlink directory"):
        admission.admit_full_day_horizon_probe(
            alias / "report.json",
            reference.output,
            reference.repository,
            reference.verifier,
            reference.source_provider,
            identity=reference.identity,
        )


def test_no_environment_or_controller_execution_module_is_imported() -> None:
    source = Path(admission.__file__).read_text(encoding="utf-8")

    assert "economic_env" not in source
    assert "PlantEnvV2" not in source
    assert "execute_mpc_arm" not in source
    assert "reset(" not in source
    assert "step(" not in source
    assert "compute(" not in source


def test_source_snapshot_mapping_round_trip_is_strict() -> None:
    snapshot = _clean_source_snapshot(admission.HISTORICAL_EXECUTED_REVISION)

    rebuilt = admission.SourceDiffSnapshotV1.from_mapping(snapshot.to_dict())

    assert rebuilt == snapshot
    with pytest.raises(ValueError, match="fields differ"):
        admission.SourceDiffSnapshotV1.from_mapping(
            {**snapshot.to_dict(), "unregistered": True}
        )
    with pytest.raises(ValueError, match="fields differ"):
        admission.SourceDiffEntryV1.from_mapping({"status": "A"})
    with pytest.raises(TypeError, match="must be strings"):
        admission.SourceDiffEntryV1.from_mapping({"status": "A", "path": 1})


def test_broken_output_symlink_is_not_treated_as_a_new_destination(
    reference: SyntheticReference,
) -> None:
    reference.output.symlink_to(reference.root / "missing-target")

    with pytest.raises(FileExistsError, match="already exists"):
        admission.validate_output_directory(
            reference.output,
            repository_root=reference.repository,
            reference_directory=reference.directory,
        )


def test_admission_never_creates_the_output_directory(
    reference: SyntheticReference,
) -> None:
    result = reference.admit()

    assert result.output_directory == reference.output.resolve()
    assert not reference.output.exists()
    assert not os.path.lexists(reference.output)
