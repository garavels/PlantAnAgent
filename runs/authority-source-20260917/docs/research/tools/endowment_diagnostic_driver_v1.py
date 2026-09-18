"""One-shot, source-audited endowment diagnostic for the frozen planner matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from plant_sim import PlantConfig

from plant_backend.controllers.base import Constraints
from plant_backend.rl.v2._planner_evidence import (
    capture_runtime_versions,
    capture_source_boundary,
)
from plant_backend.rl.v2.config import EpisodeConfigV2, sample_episode
from plant_backend.rl.v2.contract import describe_checkpoint_contract
from plant_backend.rl.v2.planner import find_repository_root
from plant_backend.rl.v2.planner_schema import (
    canonical_json,
    canonical_payload,
    canonical_payload_preimage,
)
from plant_backend.rl.v2.reward import RewardWeightsV2
from plant_backend.rl.v2.upper_bound import evaluate_delivery_relaxation


CELLS = ((35.0, 0), (35.0, 1), (35.0, 2), (42.0, 0), (42.0, 1), (42.0, 2))
DOMAIN = "planner-endowment-diagnostic-artifact-v1"
DISCLAIMER = (
    "Privileged endowment-preserving continuous-relaxation diagnostic. This is a "
    "restricted LP result, not a task upper bound, feasible controller trajectory, "
    "causal controller, field-performance estimate, or commercial claim."
)


def _driver_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _write_atomic(target: Path, value: object) -> None:
    serialized = canonical_json(value).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    reopened = target.read_bytes()
    if (
        reopened != serialized
        or canonical_json(json.loads(reopened)) != serialized.decode()
    ):
        raise RuntimeError("diagnostic artifact did not reopen byte-identically")


def _run_cell(*, rate: float, seed: int, output_dir: Path, driver_sha256: str) -> None:
    target = output_dir / f"endowment-rate{int(rate)}-seed{seed}.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")

    config = PlantConfig()
    episode_config = EpisodeConfigV2(horizon_h=72.0, delivery_rate_kg_h=rate)
    constraints = Constraints()
    weights = RewardWeightsV2()
    repository_root = find_repository_root()
    start = capture_source_boundary(
        episode=episode_config,
        plant=config,
        constraints=constraints,
        weights=weights,
        root=repository_root,
        require_clean_source=True,
        phase="start_source",
    )
    runtime = capture_runtime_versions()
    checkpoint = describe_checkpoint_contract(
        plant=config,
        episode=episode_config,
        constraints=constraints,
        weights=weights,
    )
    episode = sample_episode(episode_config, config, seed=seed)
    result = evaluate_delivery_relaxation(
        episode=episode,
        episode_config=episode_config,
        config=config,
        constraints=constraints,
        checkpoint_contract=checkpoint,
        preserve_initial_endowment=True,
        require_canonical_episode=True,
        solver_options={},
    )
    end = capture_source_boundary(
        episode=episode_config,
        plant=config,
        constraints=constraints,
        weights=weights,
        root=repository_root,
        require_clean_source=True,
        phase="end_source",
    )
    if start.source.canonical_json != end.source.canonical_json:
        raise RuntimeError("source changed during diagnostic cell")
    if start.lock_sha256 != end.lock_sha256:
        raise RuntimeError("dependency lock changed during diagnostic cell")
    if (
        result.preserve_initial_endowment is not True
        or result.is_task_upper_bound is not False
    ):
        raise RuntimeError("solver result has the wrong diagnostic scope")
    if result.canonical_episode_verified is not True:
        raise RuntimeError("solver result is not bound to a canonical episode")

    result_value = result.to_dict()
    deterministic_result = dict(result_value)
    deterministic_result.pop("elapsed_s")
    deterministic_result_sha256 = canonical_payload(
        "planner-relaxation-result-v1", deterministic_result
    ).sha256
    body = {
        "schema_version": 1,
        "diagnostic": "endowment-preserving-delivery-relaxation",
        "evidence_scope": "privileged_offline_development_diagnostic",
        "disclaimer": DISCLAIMER,
        "delivery_rate_kg_h": rate,
        "seed": seed,
        "driver_sha256": driver_sha256,
        "start_source": start.source.to_dict(),
        "end_source": end.source.to_dict(),
        "start_lock_sha256": start.lock_sha256,
        "end_lock_sha256": end.lock_sha256,
        "runtime_versions": runtime.to_dict(),
        "deterministic_result_sha256": deterministic_result_sha256,
        "result": result_value,
    }
    artifact_sha256 = hashlib.sha256(
        canonical_payload_preimage(DOMAIN, canonical_json(body))
    ).hexdigest()
    artifact = {**body, "artifact_sha256": artifact_sha256}
    _write_atomic(target, artifact)
    parsed = json.loads(target.read_text("utf-8"))
    observed_sha = parsed.pop("artifact_sha256")
    expected_sha = hashlib.sha256(
        canonical_payload_preimage(DOMAIN, canonical_json(parsed))
    ).hexdigest()
    if expected_sha != observed_sha:
        raise RuntimeError("diagnostic artifact digest did not revalidate")
    status = "success" if result.success else f"failure:{result.failure_code}"
    print(
        f"{rate:g}/{seed}: {status}; artifact={target}; sha256={artifact_sha256}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    driver_sha256 = _driver_sha256()
    for rate, seed in CELLS:
        _run_cell(
            rate=rate,
            seed=seed,
            output_dir=output_dir,
            driver_sha256=driver_sha256,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
