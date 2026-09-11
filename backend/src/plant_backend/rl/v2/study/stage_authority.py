"""Fresh training/development admission for external one-stage authorization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from .. import beta_controller_benchmark as benchmark
from ..training.custody_authorization import utc_timestamp
from ..training.custody_json import authenticate, canonical_bytes
from ..training.custody_validation import exact
from ..training.registered_contract import matrix, registered_config
from ..training.config import ControllerNeutralTrainingConfigV1
from . import registry as registry_module, stage_documents as documents
from .arm_evidence import file_stamp
from .execution_budget import ExecutionBudget
from .latency_evidence import HOST_SCHEMA, validate_latency_host
from .training_evidence import TrainingAdmission, admit_training


def expected_latency_host(training: TrainingAdmission) -> dict[str, Any]:
    rt = training.identity.runtime
    return validate_latency_host(
        authenticate(
            {
                "schema": HOST_SCHEMA,
                "machine": {
                    **deepcopy(training.identity.host),
                    "torch_intraop_threads": 1,
                    "torch_interop_threads": 1,
                },
                "python": rt["python_version"],
                "dependencies": {
                    "numpy": rt["numpy_version"],
                    "scipy": rt["scipy_version"],
                    "torch": rt["torch_version"],
                    "stable-baselines3": rt["stable_baselines3_version"],
                    "gymnasium": rt["gymnasium_version"],
                    "plant-sim": rt["plant_sim_version"],
                },
            }
        )
    )


@dataclass(frozen=True)
class StageAdmission:
    registry: dict[str, Any]
    authorization: dict[str, Any]
    budget: ExecutionBudget
    training: TrainingAdmission
    development: Any | None
    authority_stamp: tuple[int, ...]

    @property
    def root(self) -> Path:
        return Path(self.registry["stage_root"])

    def config_for(self, cell: dict[str, Any]) -> ControllerNeutralTrainingConfigV1:
        order = cell["actor"]["training_order"] or 1
        item = matrix(self.training.path.parent)[order - 1]
        return registered_config(
            Path(item["output_path"]), item["algorithm"], item["optimizer_seed"]
        )

    def checkpoint_for(self, cell: dict[str, Any]) -> dict[str, Any] | None:
        order = cell["actor"]["training_order"]
        return (
            None
            if order is None
            else self.training.report["cells"][order - 1]["checkpoint"]
        )

    def training_failure(self, cell: dict[str, Any]) -> str | None:
        order = cell["actor"]["training_order"]
        return None if order is None else self.training.slots[order - 1].failure

    def recheck(self) -> None:
        self.training.recheck()
        path = Path(self.registry["authorization_input_path"])
        exact(
            "external stage authority",
            documents.read(path, documents.AUTH_SCHEMA, documents.AUTH_FIELDS),
            self.authorization,
        )
        if file_stamp(path) != self.authority_stamp:
            raise ValueError("external stage authority was replaced")
        exact(
            "stage source",
            benchmark._source_provenance(require_clean=True),
            self.authorization["source_provenance"],
        )
        if self.development is not None:
            self.development.recheck()


def _validate_authority(
    raw: dict[str, Any], registry: dict[str, Any], training: TrainingAdmission
) -> ExecutionBudget:
    try:
        identifier = UUID(raw["authorization_id"])
    except (ValueError, TypeError, AttributeError) as reason:
        raise ValueError("stage authorization identifier is invalid") from reason
    exact(
        "canonical authorization identifier", str(identifier), raw["authorization_id"]
    )
    utc_timestamp(raw["authorized_at_utc"])
    for label, actual, expected in (
        ("protocol", raw["protocol"], documents.protocol()),
        ("stage", raw["stage"], registry["stage"]),
        ("registry", raw["registry_sha256"], registry["sha256"]),
        ("training inputs", raw["training_inputs_sha256"], training.evidence["sha256"]),
        (
            "training report",
            raw["training_report"],
            training.evidence["training_report"],
        ),
        ("operator assertions", raw["operator_assertions"], documents.ASSERTIONS),
        ("latency registration", raw["latency_host"], expected_latency_host(training)),
        (
            "clean source",
            raw["source_provenance"],
            benchmark._source_provenance(require_clean=True),
        ),
    ):
        exact("stage authorization " + label, actual, expected)
    if raw["stage_execution_authorized"] is not True:
        raise ValueError("stage execution has not been authorized")
    if training.report["abort"] is not None or any(
        row["state"]
        in {"not_started", "claimed_interrupted", "campaign_integrity_failed"}
        for row in training.report["cells"]
    ):
        raise ValueError("stage requires a closed, non-aborted training matrix")
    budget = ExecutionBudget.from_dict(raw["budget"])
    if budget.filesystem_reserve_bytes < 1024**3:
        raise ValueError("production study requires at least 1 GiB filesystem reserve")
    return budget


def admit_stage(training_root: Path | str, stage: str) -> StageAdmission:
    """Read-only production gate; no injected verdict, source, host or checkpoint."""
    registry = registry_module.stage_registry(training_root, stage)
    path = Path(registry["authorization_input_path"])
    raw = documents.read(path, documents.AUTH_SCHEMA, documents.AUTH_FIELDS)
    stamp = file_stamp(path)
    # Schema/scope rejection precedes expensive training/checkpoint admission.
    if raw["stage"] != stage or raw["stage_execution_authorized"] is not True:
        raise ValueError("authorization does not open this stage")
    training = admit_training(Path(registry["training_root"]) / "report.json")
    budget = _validate_authority(raw, registry, training)
    development = None
    if stage == "development":
        if raw["development"] is not None:
            raise ValueError("development cannot carry a prior-stage selection")
    else:
        from .stage_verifier import verify_stage

        development = verify_stage(training_root, "development")
        report = development.report
        if (
            report["abort"] is not None
            or report["analysis"]["development_opening_requirements"][
                "all_calculated_requirements_met"
            ]
            is not True
        ):
            raise ValueError("verified development does not permit held-out opening")
        expected = {
            "report_archive": development.archive,
            "report_sha256": report["sha256"],
            "frozen_champions": report["analysis"]["frozen_champions"],
        }
        exact("held-out frozen development", raw["development"], expected)
        exact(
            "held-out budget",
            raw["budget"],
            development.context.authorization["budget"],
        )
        exact(
            "held-out hardware",
            raw["latency_host"],
            development.context.authorization["latency_host"],
        )
        exact(
            "held-out training identities",
            raw["training_inputs_sha256"],
            development.context.authorization["training_inputs_sha256"],
        )
    result = StageAdmission(registry, raw, budget, training, development, stamp)
    result.recheck()
    return result
