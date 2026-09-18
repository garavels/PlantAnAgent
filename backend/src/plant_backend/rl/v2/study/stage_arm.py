"""Registered arm archives bound to admitted stage, actor and checkpoint custody."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from ... import checkpoints
from ..training import custody_io as io
from ..training.custody_json import authenticate, canonical_bytes, require_envelope
from ..training.custody_validation import exact
from . import (
    archive,
    arm_evidence as arms,
    compressed_evidence as compressed,
    latency_evidence as timing,
)
from .arm_execution import ArmExecution
from .contract import PREFIX, analysis_contract
from .latency import latency_contract
from .registry import actor_identity
from .stage_authority import StageAdmission
from . import stage_documents as documents

if TYPE_CHECKING:
    from .stage_launch import ChildBindings

ARM_SCHEMA = PREFIX + "registered-arm-evidence-v1"
TIMING_SCHEMA = PREFIX + "registered-latency-evidence-v1"
ARM_FIELDS = arms.FIELDS | {"custody"}
TIMING_FIELDS = compressed.TIMING_FIELDS | {"custody"}
BOUNDARY = "registered synthetic study evidence; no standalone superiority claim"


def custody(
    authority: dict[str, Any], registry: dict[str, Any], claim: dict[str, Any]
) -> dict[str, Any]:
    return {
        "authorization_sha256": authority["sha256"],
        "registry_sha256": registry["sha256"],
        "claim_sha256": claim["sha256"],
        "training_inputs_sha256": authority["training_inputs_sha256"],
    }


def checkpoint_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    path = Path(descriptor["path"])
    exact(
        "admitted checkpoint metadata",
        io.file_record(path / "metadata.json"),
        descriptor["files"][0],
    )
    metadata = checkpoints.load_metadata(path, check=False)
    return {
        "path": str(path),
        "format": metadata.format,
        "algorithm": metadata.algorithm,
        "trained": metadata.trained,
        "plant_sim_version": metadata.plant_sim_version,
        "total_timesteps": metadata.total_timesteps,
        "metadata_file_sha256": descriptor["metadata_sha256"],
        "policy_artifact_sha256": metadata.policy_artifact_sha256,
        "checkpoint_contract_sha256": metadata.checkpoint_contract_digest,
    }


def _timing_document(
    envelope: dict[str, Any],
    admitted: dict[str, Any],
    arm_record: dict[str, Any],
    measurement: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    authority: dict[str, Any],
) -> dict[str, Any]:
    exact("registered latency host before", before, authority["latency_host"])
    exact("registered latency host after", after, authority["latency_host"])
    before, after, result = timing._project_latency(
        envelope, admitted, measurement, before, after
    )
    return authenticate(
        {
            "schema": TIMING_SCHEMA,
            "latency_contract_sha256": latency_contract()["sha256"],
            "arm_archive": arm_record,
            "arm_envelope_sha256": envelope["sha256"],
            "host_before": before,
            "host_after": after,
            "measurement": measurement,
            "result": result,
            "custody": envelope["custody"],
            "claim_boundary": BOUNDARY,
            "result_claim": None,
        }
    )


def publish_arm(
    context: StageAdmission | ChildBindings,
    cell: dict[str, Any],
    claim: dict[str, Any],
    executed: ArmExecution,
) -> None:
    """Child publication after live-parent launch admission; no permission here."""
    config = context.config_for(cell)
    limits = context.budget.for_arm(cell["actor"]["arm"])
    exact(
        "executed source", executed.source, context.authorization["source_provenance"]
    )
    expected_checkpoint = context.checkpoint_for(cell)
    exact(
        "executed checkpoint",
        executed.checkpoint,
        (
            None
            if expected_checkpoint is None
            else checkpoint_descriptor(expected_checkpoint)
        ),
    )
    envelope = authenticate(
        {
            "schema": ARM_SCHEMA,
            "protocol": documents.protocol(),
            "analysis_contract_sha256": analysis_contract()["sha256"],
            "stage": context.registry["stage"],
            "seed": cell["episode_seed"],
            "actor": actor_identity(cell["actor"]),
            "training_task": config.to_dict(),
            "checkpoint": executed.checkpoint,
            "source_provenance": executed.source,
            "service_mpc_settings": arms._settings(config).to_dict(),
            "manifest_identity": executed.manifest_identity,
            "audit_manifest": executed.audit_manifest,
            "run": executed.run,
            "custody": custody(context.authorization, context.registry, claim),
            "claim_boundary": BOUNDARY,
            "result_claim": None,
        }
    )
    admitted = arms._verify_arm_body(envelope, config=config, seed=cell["episode_seed"])
    record = archive.publish_archive(Path(cell["arm_path"]), envelope, **limits.arm())
    document = _timing_document(
        envelope,
        admitted,
        record,
        executed.measurement,
        executed.host_before,
        executed.host_after,
        context.authorization,
    )
    archive.publish_archive(Path(cell["latency_path"]), document, **limits.timing())


def verify_cell(
    context: StageAdmission, cell: dict[str, Any], claim: dict[str, Any]
) -> dict[str, Any]:
    """Caller admits full stage authority before any protected manifest reconstruction."""
    exact("registered cell", cell, context.registry["cells"][cell["order"] - 1])
    if context.training_failure(cell) is not None:
        raise ValueError("unadmitted training cannot carry executed arm evidence")
    config = context.config_for(cell)
    limits = context.budget.for_arm(cell["actor"]["arm"])
    arm_path, timing_path = Path(cell["arm_path"]), Path(cell["latency_path"])
    stamps = {path: arms.file_stamp(path) for path in (arm_path, timing_path)}
    raw, arm_record = archive.read_archive(arm_path, **limits.arm())
    envelope = require_envelope(
        canonical_bytes(raw), schema=ARM_SCHEMA, fields=ARM_FIELDS
    )
    expected_checkpoint = context.checkpoint_for(cell)
    for key, expected in {
        "protocol": documents.protocol(),
        "analysis_contract_sha256": analysis_contract()["sha256"],
        "stage": context.registry["stage"],
        "seed": cell["episode_seed"],
        "actor": actor_identity(cell["actor"]),
        "training_task": config.to_dict(),
        "source_provenance": context.authorization["source_provenance"],
        "checkpoint": (
            None
            if expected_checkpoint is None
            else checkpoint_descriptor(expected_checkpoint)
        ),
        "custody": custody(context.authorization, context.registry, claim),
        "claim_boundary": BOUNDARY,
        "result_claim": None,
    }.items():
        exact("registered arm " + key, envelope[key], expected)
    admitted = arms._verify_arm_body(envelope, config=config, seed=cell["episode_seed"])
    raw_timing, timing_record = archive.read_archive(timing_path, **limits.timing())
    document = require_envelope(
        canonical_bytes(raw_timing), schema=TIMING_SCHEMA, fields=TIMING_FIELDS
    )
    rebuilt = _timing_document(
        envelope,
        admitted,
        arm_record,
        document["measurement"],
        document["host_before"],
        document["host_after"],
        context.authorization,
    )
    exact("registered timing evidence", document, rebuilt)
    for path, record, limit in (
        (arm_path, arm_record, limits.arm_stored_bytes),
        (timing_path, timing_record, limits.timing_stored_bytes),
    ):
        compressed._recheck(path, record, stamps[path], limit)
    return {
        "arm_archive": arm_record,
        "timing_archive": timing_record,
        "physical": {
            **admitted,
            "verification": "registered_arm_reconstructed",
            "registered_checkpoint_verified": expected_checkpoint is not None,
        },
        "latency": {
            **rebuilt["result"],
            "registered_hardware_verified": True,
            "registered_concurrency_verified": True,
        },
    }
