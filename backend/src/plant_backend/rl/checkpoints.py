"""Checkpoints: a directory, a policy file, and enough metadata to refuse it.

    checkpoints/ppo-demo/
        metadata.json      what it is, and what it was trained against
        policy.zip         the algorithm's own artefact (Stable-Baselines3)

A checkpoint is a directory rather than a bare file because the artefact alone
cannot say which observation it expects. The metadata records the field *names*
in order, the spec version, the exact observation/action descriptor digest, and
the plant it was trained against. Loading a policy whose inputs or outputs have
changed therefore fails loudly instead of silently feeding it different values.

Loading has exactly four failure modes, and each has its own error: the path is
missing, the metadata is unreadable, the spaces do not match, or the library that
can read the artefact is not installed. All four are `CheckpointError`, all four
carry the path and the reason, and none of them fall back to an untrained policy:
a random policy that looks like a trained one is worse than no policy at all.

No trained checkpoint is committed to this repository. `write_fixed_checkpoint`
exists for tests and for wiring checks; it writes a metadata-only checkpoint that
declares itself `fixed` and is labelled as a mock everywhere it surfaces.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any, Sequence

import plant_sim

from . import spaces

#: A Stable-Baselines3 PPO artefact, saved by `model.save(...)`.
FORMAT_SB3_PPO = "sb3-ppo"
#: A constant action, for tests and integration wiring. Never a trained policy.
FORMAT_FIXED = "fixed"
FORMATS = (FORMAT_SB3_PPO, FORMAT_FIXED)

METADATA_FILE = "metadata.json"
POLICY_FILE = "policy.zip"
#: Environment variable the backend reads at start-up for a checkpoint to load.
CHECKPOINT_ENV = "PLANT_RL_CHECKPOINT"


class CheckpointError(ValueError):
    """A checkpoint could not be used. The message says why, in operator terms."""


class MissingCheckpointError(CheckpointError):
    """Nothing at the given path."""


class CorruptCheckpointError(CheckpointError):
    """The metadata is missing, unreadable, or not describing a checkpoint."""


class IncompatibleCheckpointError(CheckpointError):
    """The checkpoint was trained against a different observation or action space."""


@dataclass(frozen=True)
class PolicyEnvironmentContract:
    """Exact policy-facing interface and optional full task identity.

    V1 callers may omit this value and retain the historical global contract.
    Newer environments pass it explicitly so checkpoint validation is not tied
    to whichever environment happens to be imported by default.
    """

    spec_version: int
    observation_fields: tuple[str, ...]
    action_fields: tuple[str, ...]
    environment_id: str
    environment_contract_digest: str
    checkpoint_contract_digest: str = ""

    def __post_init__(self) -> None:
        if (
            isinstance(self.spec_version, bool)
            or not isinstance(self.spec_version, Integral)
            or self.spec_version <= 0
        ):
            raise ValueError("policy contract spec_version must be positive")
        for label, values in (
            ("observation", self.observation_fields),
            ("action", self.action_fields),
        ):
            if not values or any(
                not isinstance(name, str) or not name for name in values
            ):
                raise ValueError(
                    f"policy contract {label} fields must be non-empty names"
                )
            if len(set(values)) != len(values):
                raise ValueError(f"policy contract {label} fields must be unique")
        if not isinstance(self.environment_id, str) or not self.environment_id:
            raise ValueError("policy contract environment_id must be non-empty")
        _require_digest(
            "policy contract environment_contract_digest",
            self.environment_contract_digest,
        )
        if self.checkpoint_contract_digest:
            _require_digest(
                "policy contract checkpoint_contract_digest",
                self.checkpoint_contract_digest,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": int(self.spec_version),
            "observation_fields": list(self.observation_fields),
            "action_fields": list(self.action_fields),
            "environment_id": self.environment_id,
            "environment_contract_digest": self.environment_contract_digest,
            "checkpoint_contract_digest": self.checkpoint_contract_digest,
        }


def _require_digest(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as reason:
        raise ValueError(f"{name} must be a SHA-256 digest") from reason
    return value


def legacy_policy_contract() -> PolicyEnvironmentContract:
    """Return the frozen V1 contract used by existing runtime callers."""

    return PolicyEnvironmentContract(
        spec_version=spaces.SPEC_VERSION,
        observation_fields=spaces.OBSERVATION_NAMES,
        action_fields=spaces.ACTION_NAMES,
        environment_id=spaces.ENVIRONMENT_ID,
        environment_contract_digest=spaces.ENVIRONMENT_CONTRACT_DIGEST,
    )


@dataclass(frozen=True)
class CheckpointMetadata:
    """Everything needed to decide whether this policy may drive this plant."""

    format: str
    algorithm: str
    spec_version: int = spaces.SPEC_VERSION
    observation_fields: tuple[str, ...] = spaces.OBSERVATION_NAMES
    action_fields: tuple[str, ...] = spaces.ACTION_NAMES
    plant_sim_version: str = plant_sim.__version__
    created_at: str = ""
    #: Training provenance. Empty for a checkpoint that was never trained.
    seed: int | None = None
    total_timesteps: int = 0
    training_config: dict[str, Any] = field(default_factory=dict)
    #: Format-specific payload, e.g. the constant action of a `fixed` checkpoint.
    extra: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    #: Identity and exact descriptor digest of the observation/action contract.
    #: These fields are last to preserve the positional shape of older callers.
    environment_id: str = spaces.ENVIRONMENT_ID
    environment_contract_digest: str = spaces.ENVIRONMENT_CONTRACT_DIGEST
    #: Full task identity. Empty only for historical V1 metadata.
    checkpoint_contract_digest: str = ""
    #: Optional content hash for the algorithm artifact itself.
    policy_artifact_sha256: str = ""

    @property
    def observation_size(self) -> int:
        return len(self.observation_fields)

    @property
    def action_size(self) -> int:
        return len(self.action_fields)

    @property
    def trained(self) -> bool:
        """False for a mock or an untrained artefact: the UI must not claim more."""
        return self.format != FORMAT_FIXED and self.total_timesteps > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "observation_fields": list(self.observation_fields),
            "action_fields": list(self.action_fields),
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "trained": self.trained,
        }

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, path: Path | None = None
    ) -> "CheckpointMetadata":
        where = f" in {path}" if path else ""
        try:
            checkpoint_format = str(raw["format"])
        except (KeyError, TypeError) as reason:
            raise CorruptCheckpointError(
                f"checkpoint metadata has no format{where}"
            ) from reason
        if checkpoint_format not in FORMATS:
            raise CorruptCheckpointError(
                f"unsupported checkpoint format {checkpoint_format!r}{where}: "
                f"expected one of {', '.join(FORMATS)}"
            )
        try:
            metadata = cls(
                format=checkpoint_format,
                algorithm=str(raw.get("algorithm", "unknown")),
                spec_version=int(raw.get("spec_version", 0)),
                observation_fields=tuple(
                    str(name) for name in raw.get("observation_fields", ())
                ),
                action_fields=tuple(str(name) for name in raw.get("action_fields", ())),
                plant_sim_version=str(raw.get("plant_sim_version", "unknown")),
                created_at=str(raw.get("created_at", "")),
                seed=None if raw.get("seed") is None else int(raw["seed"]),
                total_timesteps=int(raw.get("total_timesteps", 0)),
                training_config=dict(raw.get("training_config") or {}),
                extra=dict(raw.get("extra") or {}),
                notes=str(raw.get("notes", "")),
                # Metadata written before contract digests existed is V1. An empty
                # digest is its legacy sentinel and is checked by version/names only.
                environment_id=str(raw.get("environment_id", spaces.ENVIRONMENT_ID)),
                environment_contract_digest=str(
                    raw.get("environment_contract_digest", "")
                ),
                checkpoint_contract_digest=str(
                    raw.get("checkpoint_contract_digest", "")
                ),
                policy_artifact_sha256=str(raw.get("policy_artifact_sha256", "")),
            )
        except (TypeError, ValueError) as reason:
            raise CorruptCheckpointError(
                f"checkpoint metadata has malformed fields{where}: {reason}"
            ) from reason
        if metadata.seed is not None and metadata.seed < 0:
            raise CorruptCheckpointError(f"checkpoint seed must be non-negative{where}")
        if metadata.total_timesteps < 0:
            raise CorruptCheckpointError(
                f"checkpoint total_timesteps must be non-negative{where}"
            )
        return metadata

    def check_compatible(
        self,
        *,
        path: Path | None = None,
        contract: PolicyEnvironmentContract | None = None,
    ) -> None:
        """Raise unless this checkpoint speaks the current observation/action space."""
        where = f" ({path})" if path else ""
        expected = contract or legacy_policy_contract()
        if self.spec_version != expected.spec_version:
            raise IncompatibleCheckpointError(
                f"checkpoint spec version {self.spec_version} does not match "
                f"this environment's {expected.spec_version}{where}: retrain the policy"
            )
        for label, recorded, current in (
            ("observation", self.observation_fields, expected.observation_fields),
            ("action", self.action_fields, expected.action_fields),
        ):
            if recorded == current:
                continue
            if len(recorded) != len(current):
                raise IncompatibleCheckpointError(
                    f"checkpoint expects {len(recorded)} {label} fields, this plant has "
                    f"{len(current)}{where}"
                )
            differing = [
                f"{index}: {was!r} -> {now!r}"
                for index, (was, now) in enumerate(zip(recorded, current))
                if was != now
            ]
            raise IncompatibleCheckpointError(
                f"checkpoint {label} fields differ from this plant's{where}: "
                f"{'; '.join(differing)}"
            )
        if self.environment_id != expected.environment_id:
            raise IncompatibleCheckpointError(
                f"checkpoint environment {self.environment_id!r} does not match "
                f"this task's {expected.environment_id!r}{where}: retrain the policy"
            )
        if contract is not None and not self.environment_contract_digest:
            raise IncompatibleCheckpointError(
                "checkpoint metadata has no policy interface digest"
                f"{where}: retrain the policy"
            )
        if (
            self.environment_contract_digest
            and self.environment_contract_digest != expected.environment_contract_digest
        ):
            raise IncompatibleCheckpointError(
                "checkpoint environment contract digest "
                f"{self.environment_contract_digest!r} does not match this task's "
                f"{expected.environment_contract_digest!r}{where}: the observation/action "
                "semantics differ; retrain the policy"
            )
        if expected.checkpoint_contract_digest:
            if not self.checkpoint_contract_digest:
                raise IncompatibleCheckpointError(
                    "checkpoint metadata has no full task contract digest"
                    f"{where}: retrain the policy"
                )
            if self.checkpoint_contract_digest != expected.checkpoint_contract_digest:
                raise IncompatibleCheckpointError(
                    "checkpoint task contract digest "
                    f"{self.checkpoint_contract_digest!r} does not match this task's "
                    f"{expected.checkpoint_contract_digest!r}{where}: the plant, episode, "
                    "constraints, objective, or assumptions differ; retrain the policy"
                )


def metadata_path(path: Path | str) -> Path:
    return Path(path) / METADATA_FILE


def policy_path(path: Path | str) -> Path:
    return Path(path) / POLICY_FILE


def save_metadata(path: Path | str, metadata: CheckpointMetadata) -> Path:
    """Write `metadata.json` into the checkpoint directory, creating it if needed."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    target = metadata_path(directory)
    payload = metadata.to_dict()
    if not payload.get("created_at"):
        payload["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def load_metadata(
    path: Path | str,
    *,
    check: bool = True,
    contract: PolicyEnvironmentContract | None = None,
) -> CheckpointMetadata:
    """Read and (by default) verify the metadata of a checkpoint directory."""
    directory = Path(path)
    if not directory.exists():
        raise MissingCheckpointError(f"no checkpoint at {directory}")
    target = metadata_path(directory)
    if not target.is_file():
        raise CorruptCheckpointError(
            f"{directory} is not a checkpoint: expected {METADATA_FILE} inside it"
        )
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as reason:
        raise CorruptCheckpointError(f"cannot read {target}: {reason}") from reason
    if not isinstance(raw, dict):
        raise CorruptCheckpointError(f"{target} does not contain a metadata object")
    metadata = CheckpointMetadata.from_dict(raw, path=directory)
    if check:
        metadata.check_compatible(path=directory, contract=contract)
        if metadata.format != FORMAT_FIXED and not policy_path(directory).is_file():
            raise CorruptCheckpointError(
                f"{directory} declares format {metadata.format!r} but has no {POLICY_FILE}"
            )
        if metadata.format != FORMAT_FIXED and metadata.policy_artifact_sha256:
            artifact = policy_path(directory)
            try:
                _require_digest(
                    "policy_artifact_sha256",
                    metadata.policy_artifact_sha256,
                )
                actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
            except (OSError, ValueError) as reason:
                raise CorruptCheckpointError(
                    f"cannot verify {artifact}: {reason}"
                ) from reason
            if actual != metadata.policy_artifact_sha256:
                raise CorruptCheckpointError(
                    f"{artifact} content does not match policy_artifact_sha256"
                )
        elif (
            metadata.format != FORMAT_FIXED
            and contract is not None
            and contract.checkpoint_contract_digest
        ):
            raise CorruptCheckpointError(
                f"{directory} has no policy artifact digest in its metadata"
            )
    return metadata


def write_fixed_checkpoint(
    path: Path | str,
    action: Sequence[float],
    *,
    notes: str = "Mock checkpoint: a constant action. Not a trained policy.",
    contract: PolicyEnvironmentContract | None = None,
) -> Path:
    """Write a `fixed` checkpoint. For tests and wiring checks only."""
    values = [float(value) for value in action]
    expected = contract or legacy_policy_contract()
    if len(values) != len(expected.action_fields):
        raise ValueError(
            f"a fixed action needs {len(expected.action_fields)} entries, got {len(values)}"
        )
    save_metadata(
        path,
        CheckpointMetadata(
            format=FORMAT_FIXED,
            algorithm="fixed-action (mock)",
            spec_version=expected.spec_version,
            observation_fields=expected.observation_fields,
            action_fields=expected.action_fields,
            environment_id=expected.environment_id,
            environment_contract_digest=expected.environment_contract_digest,
            checkpoint_contract_digest=expected.checkpoint_contract_digest,
            extra={"action": values},
            notes=notes,
        ),
    )
    return Path(path)


__all__ = [
    "CHECKPOINT_ENV",
    "FORMATS",
    "FORMAT_FIXED",
    "FORMAT_SB3_PPO",
    "METADATA_FILE",
    "POLICY_FILE",
    "CheckpointError",
    "CheckpointMetadata",
    "CorruptCheckpointError",
    "IncompatibleCheckpointError",
    "MissingCheckpointError",
    "PolicyEnvironmentContract",
    "legacy_policy_contract",
    "load_metadata",
    "metadata_path",
    "policy_path",
    "save_metadata",
    "write_fixed_checkpoint",
]
