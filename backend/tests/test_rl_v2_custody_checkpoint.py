"""Negative checkpoint admission fixtures never construct or reset a plant."""

from pathlib import Path
from typing import cast

import pytest

from plant_backend.rl.v2.training import custody_checkpoint as admission
from plant_backend.rl.v2.training.custody_io import publish_bytes
from plant_backend.rl.v2.training.registered_contract import registered_config
from plant_backend.rl.v2.training.runtime_provenance import (
    BetaTrainingRuntimeProvenanceV1,
)


@pytest.mark.parametrize(
    "defect,code",
    [
        ("absent", "checkpoint_missing"),
        ("empty", "checkpoint_missing"),
        ("incomplete", "artifact_inventory_invalid"),
        ("extra", "artifact_inventory_invalid"),
        ("symlink", "artifact_inventory_invalid"),
        ("malformed", "checkpoint_malformed"),
        ("duplicate_metadata", "checkpoint_malformed"),
        ("nonfinite_config", "checkpoint_malformed"),
    ],
)
def test_checkpoint_rejection_preserves_every_byte_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
    code: str,
) -> None:
    from plant_backend.rl.v2.training.custody_io import inventory

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("rejected checkpoint reached environment or model construction")

    monkeypatch.setattr(admission, "make_env", forbidden)
    monkeypatch.setattr(admission, "load_policy", forbidden)
    root = tmp_path / "output"
    config = registered_config(root, "plant-beta-ppo-pi-lagrangian-v1", 101)
    if defect != "absent":
        root.mkdir()
    if defect not in {"absent", "empty"}:
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        publish_bytes(checkpoint / "policy.zip", b"synthetic invalid artifact")
        if defect != "incomplete":
            publish_bytes(
                checkpoint / "metadata.json",
                b'{"a": 1, "a": 2}' if defect == "duplicate_metadata" else b"{}",
            )
            publish_bytes(
                checkpoint / "training_config.json",
                b'{"a": NaN}' if defect == "nonfinite_config" else b"{}",
            )
        if defect == "extra":
            publish_bytes(root / "unexpected.bin", b"retained extra bytes")
        if defect == "symlink":
            (root / "foreign").symlink_to(tmp_path)
    before = inventory(root)
    # Runtime is deliberately unusable: these checks must reject before trying
    # to admit identity, deserialize a policy, or construct an environment.
    runtime = cast(BetaTrainingRuntimeProvenanceV1, None)
    with pytest.raises(admission.CheckpointAdmissionFailure) as raised:
        admission.admit_checkpoint(config, runtime)
    assert raised.value.code == code
    assert inventory(root) == before
