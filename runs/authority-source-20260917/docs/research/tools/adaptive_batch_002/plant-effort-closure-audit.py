"""Read-only audit of an already closed adaptive batch; never execute a controller."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import math
import gzip
import hashlib
import io
from pathlib import Path
import subprocess
import sys
from typing import Any, cast

from plant_backend.rl.v2.training import research_effort_batch as batch
from plant_backend.rl.v2.training import research_evaluation as evaluation
from plant_backend.rl.v2.training import (
    research_residual_evaluation as residual_evaluation,
)
from plant_backend.rl.v2.training import runtime_provenance as provenance
from plant_backend.rl.v2.training.config import (
    ControllerNeutralTrainingConfigV1,
    BETA_PI_LAGRANGIAN_ALGORITHM_ID,
)
from plant_backend.rl.v2.beta_config import RESOURCE_NAMES
from plant_backend.rl.v2.training.custody_io import publish_json, read_regular
from plant_backend.rl.v2.training.custody_json import canonical_bytes, decode

PINNED_HEAD = "a442a4a9f5b028568d36bbbddf21a34c3609a7cc"
PINNED_TREE = "7b9376db19948986501bd7cf52f677c8ffc73599"
PINNED_GATE_SHA256: str | None = (
    "60c35ffd2b81dac615aec2d3c77f06c08c1cc14ce50bf84a92f1925c6dc40bcd"
)
PINNED_GATE_PATH = Path(
    "/Volumes/Lexar/PlantAnAgentResearch/effort-readiness-a442a4a/gate.json"
)
ACTORS = batch.ACTORS


class AuditError(ValueError):
    """No report may be published from inconsistent evidence."""


class NotClosed(AuditError):
    """Outcome access is forbidden until normal batch closure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def same(left: object, right: object) -> bool:
    return canonical_bytes(left) == canonical_bytes(right)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class Reader:
    def __init__(self) -> None:
        self.records: dict[Path, dict[str, Any]] = {}

    def read(self, path: Path) -> bytes:
        raw = read_regular(path)
        identity = {"path": str(path), "bytes": len(raw), "sha256": sha(raw)}
        previous = self.records.get(path)
        require(previous is None or previous == identity, f"artifact changed: {path}")
        self.records[path] = identity
        return raw

    def json(self, path: Path) -> dict[str, Any]:
        raw = self.read(path)
        value = decode(raw)
        require(raw == canonical_bytes(value), f"noncanonical JSON: {path}")
        return value

    def check(self, path: Path, expected: object) -> bytes:
        raw = self.read(path)
        require(
            same(self.records[path], expected), f"artifact identity differs: {path}"
        )
        return raw

    def recheck(self) -> None:
        for path in list(self.records):
            self.read(path)


def verify_source(plan: dict[str, Any], *, fixture: bool) -> dict[str, Any]:
    if fixture:
        require(plan["runtime"] == {"fixture": True}, "fixture runtime differs")
        return {"fixture": True, "git_head": PINNED_HEAD, "git_tree": PINNED_TREE}
    saved = provenance.BetaTrainingRuntimeProvenanceV1.from_dict(plan["runtime"])
    require(
        saved.git_head == PINNED_HEAD and saved.worktree_clean,
        "source is not pinned clean batch source",
    )
    provenance.verify_unchanged(saved, provenance.capture(require_clean=True))
    repo = Path(batch.__file__).resolve().parents[6]
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=repo, text=True
    ).strip()
    require(tree == PINNED_TREE, "source tree differs")
    # Thread configuration is recorded in the run; the auditor performs no learning.
    require(
        plan["host"]["torch_intraop_threads"]
        == plan["host"]["torch_interop_threads"]
        == 1,
        "run thread profile differs",
    )
    from plant_backend.rl.v2.training.capacity_benchmark import _host_identity

    live_host = _host_identity()
    # The auditor does no training; its own Torch thread count is irrelevant to
    # the recorded run profile, which is separately required to be exactly one.
    excluded = {"torch_intraop_threads", "torch_interop_threads"}
    require(
        same(
            {k: v for k, v in live_host.items() if k not in excluded},
            {k: v for k, v in plan["host"].items() if k not in excluded},
        ),
        "execution host differs",
    )
    return {
        "git_head": saved.git_head,
        "git_tree": tree,
        "runtime_sha256": saved.sha256,
        "host_sha256": sha(canonical_bytes(plan["host"])),
    }


def receipt_success(
    receipt: dict[str, Any], *, stage: str, name: str, extras: list[str]
) -> bool:
    require(
        receipt["stage"] == stage and receipt["name"] == name,
        f"receipt slot differs: {name}",
    )
    codes = receipt["codes"]
    require(
        isinstance(codes, list)
        and all(isinstance(x, str) for x in codes)
        and codes == sorted(set(codes)),
        "malformed process codes",
    )
    process = receipt["process"]
    require(
        process["command"][1:] == ["-m", batch.MODULE, stage, *extras],
        "child command differs",
    )
    require(
        Path(process["command"][0]).resolve() == Path(sys.executable).resolve(),
        "child interpreter differs",
    )
    expected_timeout = batch.TRAIN_TIMEOUT if stage == "_train" else batch.EVAL_TIMEOUT
    require(process["timeout_seconds"] == expected_timeout, "child timeout differs")
    require(
        process["exit_code"] != batch.INTEGRITY_EXIT
        and "root_custody_lost" not in codes,
        "closed report contains campaign-aborting integrity/interruption failure",
    )
    success = not codes
    if success:
        require(
            process["started"] is True and process["exit_code"] == 0,
            "successful receipt omitted process failure",
        )
    else:
        require(
            process["exit_code"] != 0
            or any(
                x in codes
                for x in (
                    "child_timeout",
                    "child_log_limit_exceeded",
                    "child_spawn_failed",
                )
            ),
            "process failure codes contradict exit",
        )
    require(
        isinstance(receipt["closed_at"], str) and bool(receipt["closed_at"]),
        "receipt is not closed",
    )
    return success


def expected_identity(root: Path, arm: batch.Arm) -> dict[str, Any]:
    # Constructors only; this function never calls reset/observe/step.
    return batch.policy_identity(batch.template(root), arm)


def is_residual(actor: str) -> bool:
    return actor == "initial-residual" or any(
        a.name == actor and a.residual for a in batch.ARMS
    )


def process_failure(
    reader: Reader, root: Path, receipt: dict[str, Any], kind: str
) -> dict[str, Any]:
    path = root / "logs" / f"{receipt['name']}.stderr.log"
    raw = reader.read(path)
    require(len(raw) <= batch.LOG_LIMIT, "stderr exceeded declared process limit")
    return {
        "kind": kind,
        "codes": receipt["codes"],
        "process": receipt["process"],
        "stderr_tail": raw[-4096:].decode("utf-8", errors="replace"),
        "stderr_artifact": reader.records[path],
    }


def verify_loaded_checkpoint(
    reader: Reader, path: Path, result: dict[str, Any], config: Any, arm: batch.Arm
) -> dict[str, Any]:
    from plant_backend.rl.v2.training.constrained_ppo import ConstrainedPPO

    model = ConstrainedPPO.load(path, device="cpu")
    batch.verify_model_spaces(model, config, arm)
    attestation = model.loaded_constraint_state_attestation()
    require(
        attestation.artifact_sha256 == reader.records[path]["sha256"]
        and same(attestation.constraint_state, result["constraint_state"]),
        "loaded checkpoint attestation differs",
    )
    require(
        model.algorithm_id == arm.algorithm
        and model.num_timesteps == batch.TOTAL_STEPS,
        "loaded algorithm or training budget differs",
    )
    require(
        same(getattr(model, "research_identity", None), result["policy_identity"]),
        "loaded model research identity differs",
    )
    reader.check(path, result["final_policy"])
    return {
        "verified": True,
        "policy_inference_or_learning_executed": False,
        "artifact_sha256": attestation.artifact_sha256,
        "policy_tensor_shapes": {
            name: list(value.shape) for name, value in model.policy.state_dict().items()
        },
    }


def verify_allocator(result: dict[str, Any], config: Any) -> dict[str, Any]:
    from plant_backend.rl.v2.training.episode_allocator import (
        EpisodeSeedAllocatorV1,
        FIXTURE_RESUME_SCHEMA,
        canonical_sha256,
    )
    from plant_backend.rl.v2.training.constrained_ppo import ExternalEpisodeStatePinV1

    snapshot = result["allocator"]
    pin = ExternalEpisodeStatePinV1.from_dict(
        result["constraint_state"]["payload"]["external_episode_state_pin"]
    )
    require(
        same(pin.payload, snapshot), "allocator snapshot differs from checkpoint pin"
    )
    parent = result["final_policy"]["sha256"]
    body = {
        "schema": FIXTURE_RESUME_SCHEMA,
        "fixture_only": True,
        "eligible_for_registered_evaluation": False,
        "parent_checkpoint_sha256": parent,
        "snapshot": snapshot,
    }
    envelope = {**body, "sha256": canonical_sha256(body)}
    restored = EpisodeSeedAllocatorV1.restore_fixture(
        envelope,
        expected_parent_checkpoint_sha256=parent,
        expected_resume_sha256=envelope["sha256"],
    )
    require(
        restored.base_seed == batch.TRAIN_SEED_BASE
        and restored.episode_decisions == config.episode.decision_steps(config.plant),
        "allocator seed or horizon differs",
    )
    require(
        restored.decisions_consumed_by_environment == (batch.TOTAL_STEPS // 4,) * 4,
        "allocator transition count differs",
    )
    require(
        len(restored.ledger) == 196 and same(restored.snapshot().to_dict(), snapshot),
        "allocator reset count or canonical snapshot differs",
    )
    return {
        "verified": True,
        "resets_executed": 0,
        "ledger_entries": len(restored.ledger),
        "decisions_consumed_by_environment": list(
            restored.decisions_consumed_by_environment
        ),
    }


def training_record(
    reader: Reader, root: Path, arm: batch.Arm, success: bool, receipt: dict[str, Any]
) -> dict[str, Any]:
    from plant_backend.rl.v2.training.constrained_ppo import (
        RolloutLedgerV1,
        ConstrainedPPOSettingsV1,
    )

    record: dict[str, Any] = {
        "closed_successfully": success,
        "receipt": receipt,
        "diagnostics": [],
        "rollouts": [],
        "result": None,
        "failures": [],
    }
    directory = root / "training" / arm.name
    config = batch.template(root)
    identity = expected_identity(root, arm)
    if (directory / "attempt.json").exists():
        attempt = reader.json(directory / "attempt.json")
        require(
            same(attempt["arm"], asdict(arm))
            and same(attempt["template"], config.to_dict())
            and attempt["seed_base"] == batch.TRAIN_SEED_BASE
            and same(attempt["policy_identity"], identity),
            "training attempt identity differs",
        )
        record["attempt"] = attempt
    else:
        require(not success, "successful training has no attempt")
    for kind, key in (("diagnostic", "diagnostics"), ("rollout", "rollouts")):
        paths = sorted((directory / "diagnostics").glob(f"{kind}-*.json"))
        for index, path in enumerate(paths, 1):
            require(
                path.name == f"{kind}-{index:03d}.json",
                "noncontiguous training evidence prefix",
            )
            record[key].append(reader.json(path))
    previous = None
    count = batch.TOTAL_STEPS // (config.n_envs * config.ppo.n_steps)
    for index, raw in enumerate(record["rollouts"], 1):
        ledger = RolloutLedgerV1.from_dict(raw)
        ledger.verify(expected_previous_sha256=previous)
        require(
            ledger.algorithm_id == arm.algorithm
            and ledger.rollout_index == index
            and ledger.transition_count == 8192
            and ledger.optimizer_steps == 320,
            "rollout budget or algorithm differs",
        )
        previous = ledger.sha256
    require(
        len(record["rollouts"]) <= count and len(record["diagnostics"]) <= count,
        "training evidence exceeds frozen budget",
    )
    for index, diagnostic in enumerate(record["diagnostics"], 1):
        require(
            diagnostic["schema"] == "plant-research-ppo-rollout-diagnostics-v1"
            and diagnostic["num_timesteps"] == index * 8192
            and diagnostic["algorithm_id"] == arm.algorithm,
            "diagnostic rollout or algorithm identity differs",
        )
    if (directory / "initial-policy.pt").exists():
        reader.read(directory / "initial-policy.pt")
        record["initial_policy"] = reader.records[directory / "initial-policy.pt"]
    if not success:
        require(
            not (directory / "result.json").exists()
            and not (directory / "final-policy.zip").exists(),
            "failed training unexpectedly published a final endpoint",
        )
        record["failures"] = [
            process_failure(reader, root, receipt, "training_process")
        ]
        return record
    result = reader.json(directory / "result.json")
    require(
        result["schema"] == batch.SCHEMA
        and same(result["arm"], asdict(arm))
        and same(result["environment_template"], config.to_dict())
        and same(result["policy_identity"], identity),
        "training result identity differs",
    )
    require(
        result["total_timesteps"] == batch.TOTAL_STEPS
        and result["registered_evaluation_eligible"] is False,
        "training endpoint or eligibility differs",
    )
    require(
        result["checkpoint_contract_digest"] == identity["checkpoint_contract_digest"],
        "checkpoint task contract differs",
    )
    reader.check(directory / "initial-policy.pt", result["initial_policy"])
    reader.check(directory / "final-policy.zip", result["final_policy"])
    payload = result["constraint_state"]["payload"]
    require(
        payload["num_timesteps"] == batch.TOTAL_STEPS
        and payload["pending_rollout"] is False
        and payload["aborted_rollout"] is None
        and payload["unledgered_transition_count"] == 0
        and payload["resume_continuation_receipt"] is None
        and payload["ppo_epoch_update_count"] == 160,
        "constraint endpoint differs",
    )
    require(
        same(
            payload["settings"],
            ConstrainedPPOSettingsV1.for_algorithm(
                arm.algorithm, optimizer_seed=101
            ).to_dict(),
        ),
        "checkpoint learner settings differ",
    )
    require(
        len(record["rollouts"]) == len(record["diagnostics"]) == count
        and same(payload["rollout_ledgers"], record["rollouts"])
        and payload["rollout_ledger_sha256"] == previous,
        "published rollout differs from endpoint",
    )
    configuration = payload["ppo_configuration"]
    for name in (
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "learning_rate",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "device",
    ):
        require(
            same(configuration[name], getattr(config.ppo, name)),
            f"PPO configuration differs: {name}",
        )
    require(
        configuration["n_envs"] == config.n_envs
        and configuration["seed"] == config.seed,
        "optimizer seed or vector count differs",
    )
    batch.preflight_checkpoint(
        directory / "final-policy.zip", identity, result["constraint_state"]
    )
    loaded = verify_loaded_checkpoint(
        reader, directory / "final-policy.zip", result, config, arm
    )
    allocator = verify_allocator(result, config)
    record.update(
        initial_policy=result["initial_policy"],
        final_policy=result["final_policy"],
        result=result,
        first_rollout=record["rollouts"][0],
        final_rollout=record["rollouts"][-1],
        loaded_checkpoint=loaded,
        allocator_verification=allocator,
    )
    return record


def compare_initial_tensors(
    reader: Reader, root: Path, training: dict[str, Any]
) -> dict[str, Any]:
    import torch

    weights: dict[str, Any] = {}
    for arm in batch.ARMS:
        entry = training[arm.name]
        if not entry["closed_successfully"]:
            continue
        raw = reader.check(
            root / "training" / arm.name / "initial-policy.pt", entry["initial_policy"]
        )
        data = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        require(
            isinstance(data, dict)
            and bool(data)
            and all(
                isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all())
                for v in data.values()
            ),
            "invalid initial tensor dictionary",
        )
        for key in ("action_net.weight", "action_net.bias"):
            require(
                key in data and torch.equal(data[key], torch.zeros_like(data[key])),
                "initial actor output head is not exactly zero",
            )
        require(
            "log_std" in data
            and torch.equal(
                data["log_std"], torch.full_like(data["log_std"], math.log(0.2))
            ),
            "initial log_std differs from declared intervention",
        )
        expected_shape = (64, 139 if arm.residual else 135)
        for key in (
            "mlp_extractor.policy_net.0.weight",
            "mlp_extractor.value_net.0.weight",
        ):
            require(
                key in data and tuple(data[key].shape) == expected_shape,
                "initial tensor observation interface differs",
            )
        cost_keys = [k for k in data if k.startswith("cost_critic.")]
        require(
            bool(cost_keys) == (arm.algorithm == BETA_PI_LAGRANGIAN_ALGORITHM_ID),
            "initial cost critic family differs",
        )
        shapes = entry["loaded_checkpoint"].get("policy_tensor_shapes")
        if shapes is not None:
            require(
                same({key: list(value.shape) for key, value in data.items()}, shapes),
                "initial policy architecture differs from admitted endpoint",
            )
        weights[arm.name] = data
    pairs = []
    for left, right in (
        (batch.ARMS[0].name, batch.ARMS[1].name),
        (batch.ARMS[2].name, batch.ARMS[3].name),
    ):
        available = left in weights and right in weights
        if available:
            keys = {k for k in weights[left] if not k.startswith("cost_critic.")}
            require(
                keys == {k for k in weights[right] if not k.startswith("cost_critic.")}
                and all(torch.equal(weights[left][k], weights[right][k]) for k in keys),
                f"matched within-interface actor/reward initialization differs: {left}/{right}",
            )
        pairs.append(
            {
                "left": left,
                "right": right,
                "available": available,
                "equal": True if available else None,
            }
        )
    return {
        "performed": bool(weights),
        "passed": True if len(weights) == 4 else None,
        "available_initializers": list(weights),
        "pairs": pairs,
        "cross_interface_equality_claimed": False,
        "failed_initial_comparator_substitution": False,
    }


def verify_episode(
    reader: Reader,
    root: Path,
    actor: str,
    seed: int,
    replay_success: bool,
    equivalence_evidence: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    directory = root / "evaluations" / f"{seed}-{actor}"
    summary = reader.json(directory / "summary.json")
    require(
        summary["actor"] == actor and summary["seed"] == seed,
        "summary slot identity differs",
    )
    encoded = reader.check(directory / "episode.json.gz", summary["archive"])
    require(
        len(encoded) <= batch.ARCHIVE_LIMIT, "archive exceeds declared stored limit"
    )
    raw = gzip.decompress(encoded)
    episode = decode(raw)
    require(raw == canonical_bytes(episode), "decoded episode is not canonical")
    require(
        episode["schema"]
        == (residual_evaluation.SCHEMA if is_residual(actor) else evaluation.SCHEMA)
        and episode["seed"] == seed
        and episode["fixture"] is False
        and episode["prefix_limit"] is None,
        "episode namespace or scope differs",
    )
    require(
        same(episode["config"], batch.template(root).to_dict()),
        "episode template differs",
    )
    require(
        episode["evaluator_source_sha256"]
        == sha(
            read_regular(
                Path(
                    str(
                        (
                            residual_evaluation if is_residual(actor) else evaluation
                        ).__file__
                    )
                )
            )
        ),
        "episode evaluator source differs",
    )
    value = episode["summary"]
    expected_actor = (
        residual_evaluation.ACTOR
        if is_residual(actor)
        else (actor if actor in batch.BASELINES else "PPO")
    )
    if is_residual(actor):
        identity = expected_identity(root, batch.ARMS[2])
        require(
            same(episode["interface_contract"], identity["interface_contract"])
            and episode["interface_contract_digest"]
            == identity["interface_contract_digest"],
            "residual episode interface differs",
        )
        require(
            same(
                episode["controller"],
                {"id": actor, "deterministic": True, "interface": "residual-H-139"},
            ),
            "residual controller identity differs",
        )
    require(
        same(value, summary["summary"])
        and value["actor"] == expected_actor
        and value["policy_id"] == actor,
        "episode/summary/controller identity differs",
    )
    require(
        value["registered_evaluation_eligible"] is False
        and value["expected_decisions"] == 672
        and value["expected_physical_steps"] == 10080,
        "episode target differs",
    )
    require(
        len(episode["history"]) == value["decisions"]
        and len(episode["minute_truth_states"]) == value["physical_steps"] + 1,
        "episode trace length differs",
    )
    require(
        value["physical_steps"] == value["decisions"] * 15,
        "incomplete physical interval",
    )
    complete = (
        value["failure"] is None
        and value["terminated"] is True
        and value["truncated"] is False
        and value["decisions"] == 672
    )
    require(value["execution_complete"] is complete, "episode completeness differs")
    if value["hard_safety_classifier_failure"] is not None:
        require(
            value["hard_safety_passed"] is False,
            "classifier failure claimed safety pass",
        )
    cycles = value["cycle_settlements"]
    require(len(cycles) == value["decisions"] // 96, "daily settlements missing")
    require(
        value["resource_order"] == list(RESOURCE_NAMES),
        "resource order differs",
    )
    require(
        value["cycle_deficit_vectors"]
        == [[row["deficits"][name] for name in RESOURCE_NAMES] for row in cycles],
        "cycle deficit vector differs",
    )
    require(
        value["all_cycles_feasible"]
        is (complete and len(cycles) == 7 and all(row["feasible"] for row in cycles)),
        "cycle feasibility aggregate differs",
    )
    if replay_success:
        replay = reader.json(directory / "replay.json")
        expected = {
            "schema": "plant-adaptive-research-replay-v1",
            "verified": True,
            "seed": seed,
            "actor": expected_actor,
            "decisions": value["decisions"],
            "physical_steps": value["physical_steps"],
            "execution_complete": complete,
            "scope": "complete_episode" if complete else "retained_physical_prefix",
            "controller_reexecuted": False,
            "controller_failure_authenticity_verified": False,
            "registered_evaluation_eligible": False,
            "evidence_sha256": sha(raw),
        }
        if is_residual(actor):
            expected["schema"] = "plant-adaptive-residual-replay-v1"
            expected.pop("controller_reexecuted")
            expected.update(
                saved_absolute_commands_executed=True,
                heuristic_reexecuted_for_mapping_check=True,
                learned_policy_reexecuted=False,
            )
        require(same(replay, expected), "replay/episode cross-binding differs")
    if actor in ("H", "initial-residual") and complete and replay_success:
        equivalence_evidence[(actor, seed)] = physical_projection(
            episode, residual=is_residual(actor)
        )
    return cast(dict[str, Any], value)


def physical_projection(episode: dict[str, Any], *, residual: bool) -> dict[str, Any]:
    """Hash every declared physical field; deliberately exclude only identity/method extras."""
    history_keys = (
        "decision_index",
        "observation",
        "after_observation",
        "command",
        "controller_invalid",
        "reward_eur",
        "terminated",
        "truncated",
        "info",
        "execution",
    )
    summary = {
        k: v for k, v in episode["summary"].items() if k not in ("actor", "policy_id")
    }
    fields = {
        k: episode[k]
        for k in (
            "manifest",
            "initial_observation",
            "final_observation",
            "minute_truth_states",
        )
    }
    fields["history"] = [
        {k: row[k] for k in history_keys} for row in episode["history"]
    ]
    fields["summary_physical_economic_safety_settlements"] = summary
    if residual:
        require(
            all(
                row["raw_action_shape"] == [4]
                and same(row["raw_action"], [0.0] * 4)
                and row["residual_evidence"]["body"]["exact_zero_heuristic_command"]
                is True
                for row in episode["history"]
            ),
            "initial residual policy did not issue zero adjustments",
        )
    return {
        "digests": {k: sha(canonical_bytes(v)) for k, v in fields.items()},
        "decisions": len(episode["history"]),
        "minute_states": len(episode["minute_truth_states"]),
    }


def compare_initial_residual_with_heuristic(
    reader: Reader,
    root: Path,
    actors: dict[str, Any],
    evidence: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    slots = []
    for seed in batch.DEV_SEEDS:
        names = ("H", "initial-residual")
        available = all((actor, seed) in evidence for actor in names)
        comparisons = {}
        if available:
            comparisons = {
                key: evidence[(names[0], seed)]["digests"][key]
                == evidence[(names[1], seed)]["digests"][key]
                for key in evidence[(names[0], seed)]["digests"]
            }
        equal = all(comparisons.values()) if available else None
        slots.append(
            {
                "seed": seed,
                "available": available,
                "equal": equal,
                "reason": (
                    None
                    if available
                    else "one or both fixed slots lack a complete replayed episode; no substituted pair"
                ),
                "comparisons": comparisons,
                "physical_projection_digests": {
                    name: evidence[(name, seed)]
                    for name in names
                    if (name, seed) in evidence
                },
                "archive_records": {
                    name: reader.records[
                        root / "evaluations" / f"{seed}-{name}" / "episode.json.gz"
                    ]
                    for name in names
                    if (root / "evaluations" / f"{seed}-{name}" / "episode.json.gz")
                    in reader.records
                },
            }
        )
    present = [s for s in slots if s["available"]]
    all_available = len(present) == 8
    return {
        "all_eight_pairs_available": all_available,
        "all_available_pairs_equal": (
            all(s["equal"] for s in present) if present else None
        ),
        "all_eight_pairs_equal": (
            all(s["equal"] for s in present) if all_available else None
        ),
        "slots": slots,
        "policy_inference_reexecuted": False,
        "scope": "exact saved command, minute state, raw observation, economics, safety and daily-settlement equality; no partial-prefix match claim",
    }


def audit_closed_batch(
    root: Path = batch.ROOT, *, fixture: bool = False
) -> dict[str, Any]:
    root = Path(root)
    if fixture:
        require(
            root != batch.ROOT
            and str(root.resolve()).startswith(
                ("/private/tmp/", "/private/var/folders/")
            ),
            "fixtures must stay in temporary storage",
        )
    else:
        require(
            root == batch.ROOT and root == root.resolve(), "production root differs"
        )
    reader = Reader()
    # This is the only artifact opened before the closure gate.
    try:
        report = reader.json(root / "report.json")
    except FileNotFoundError as exc:
        raise NotClosed("batch report is absent; live outcomes remain unread") from exc
    if (
        report.get("status") != "closed"
        or report.get("failure") is not None
        or not report.get("closed_at")
    ):
        raise NotClosed("batch has no normal closed report; outcomes remain unread")
    require(
        report["schema"] == batch.SCHEMA
        and report["registered_evaluation_eligible"] is False,
        "closed report schema differs",
    )
    auditor_path = Path(__file__).resolve()
    reader.read(auditor_path)
    plan_raw = reader.check(root / "plan.json", report["plan"])
    plan = decode(plan_raw)
    require(
        plan_raw == canonical_bytes(plan)
        and same(plan["specification"], batch.specification(root)),
        "fixed batch plan differs",
    )
    source = verify_source(plan, fixture=fixture)
    if not fixture:
        require(
            PINNED_GATE_SHA256 is not None
            and plan["gate"]["sha256"] == PINNED_GATE_SHA256
            and Path(plan["gate"]["path"]) == PINNED_GATE_PATH,
            "launch gate identity differs or has not been pinned",
        )
    gate_path = Path(plan["gate"]["path"])
    gate = decode(reader.check(gate_path, plan["gate"]))
    require(
        gate["passed"] is True
        and gate["source_commit"] == PINNED_HEAD
        and gate.get("source_tree") == PINNED_TREE
        and gate.get("source_clean_and_unchanged") is True,
        "source gate differs",
    )
    pending = list(report["receipts"])
    observed_receipts: list[dict[str, Any]] = []

    def take(stage: str, name: str, extras: list[str]) -> tuple[dict[str, Any], bool]:
        require(bool(pending), f"expected receipt missing: {name}")
        saved = reader.json(root / "receipts" / f"{name}.json")
        require(
            same(pending.pop(0), saved),
            f"report receipt ordering/content differs: {name}",
        )
        success = receipt_success(saved, stage=stage, name=name, extras=extras)
        observed_receipts.append(saved)
        return saved, success

    training = {}
    for arm in batch.ARMS:
        receipt, success = take("_train", f"train-{arm.name}", ["--cell", arm.name])
        training[arm.name] = training_record(reader, root, arm, success, receipt)
    tensor_check = compare_initial_tensors(reader, root, training)
    equivalence_evidence: dict[tuple[str, int], dict[str, Any]] = {}
    actors: dict[str, Any] = {actor: {"slots": []} for actor in ACTORS}
    for seed in batch.DEV_SEEDS:
        for actor in ACTORS:
            receipt, success = take(
                "_evaluate",
                f"evaluate-{seed}-{actor}",
                ["--actor", actor, "--seed", str(seed)],
            )
            replay_ok = False
            failures = []
            summary = None
            trained = (
                True
                if actor in batch.BASELINES
                else training[
                    (
                        batch.INITIAL_SOURCES[actor].name
                        if actor in batch.INITIAL_SOURCES
                        else actor
                    )
                ]["closed_successfully"]
            )
            if not trained:
                failures.append(
                    {
                        "kind": "training_unavailable",
                        "arm": (
                            batch.INITIAL_SOURCES[actor].name
                            if actor in batch.INITIAL_SOURCES
                            else actor
                        ),
                    }
                )
                require(
                    not success,
                    "evaluation used an unsuccessfully closed training child",
                )
            if success:
                attempt = reader.json(
                    root / "evaluations" / f"{seed}-{actor}" / "attempt.json"
                )
                require(
                    attempt["actor"] == actor and attempt["seed"] == seed,
                    "evaluation attempt slot differs",
                )
                replay_receipt, replay_ok = take(
                    "_replay",
                    f"replay-{seed}-{actor}",
                    ["--actor", actor, "--seed", str(seed)],
                )
                if not replay_ok:
                    failures.append(
                        process_failure(reader, root, replay_receipt, "replay_process")
                    )
                summary = verify_episode(
                    reader, root, actor, seed, replay_ok, equivalence_evidence
                )
                if summary["failure"] is not None:
                    failures.append(
                        {
                            "kind": "controller_or_execution",
                            "detail": summary["failure"],
                        }
                    )
                if summary["hard_safety_classifier_failure"] is not None:
                    failures.append(
                        {
                            "kind": "safety_classifier",
                            "detail": summary["hard_safety_classifier_failure"],
                        }
                    )
            else:
                failures.append(
                    process_failure(reader, root, receipt, "evaluation_process")
                )
            actors[actor]["slots"].append(
                {
                    "seed": seed,
                    "summary": summary,
                    "eligible_for_descriptive_comparison": bool(
                        trained
                        and success
                        and replay_ok
                        and summary is not None
                        and summary["execution_complete"]
                    ),
                    "failures": failures,
                }
            )
    require(not pending, "unexpected extra report receipts")
    expected_names = {f"{r['name']}.json" for r in observed_receipts}
    require(
        {
            p.name
            for p in (root / "receipts").glob("*.json")
            if not p.name.startswith("._")
        }
        == expected_names,
        "unexpected receipt artifacts",
    )
    equivalence = compare_initial_residual_with_heuristic(
        reader, root, actors, equivalence_evidence
    )
    auditor_path = Path(__file__).resolve()
    reader.read(auditor_path)
    reader.recheck()
    if not fixture:
        require(
            same(source, verify_source(plan, fixture=False)),
            "source changed during audit",
        )
    return {
        "schema": "plant-effort-closed-batch-audit-v1",
        "integrity_verified": True,
        "batch_closed": True,
        "auditor_source": reader.records[auditor_path],
        "root": str(root),
        "source": source,
        "report": reader.records[root / "report.json"],
        "plan": reader.records[root / "plan.json"],
        "training": training,
        "actors": actors,
        "initial_tensor_comparison": tensor_check,
        "initial_residual_heuristic_equivalence": equivalence,
        "integrity": {
            "passed": True,
            "training_slots": len(training),
            "evaluation_slots": sum(len(a["slots"]) for a in actors.values()),
            "replay_slots": sum(r["stage"] == "_replay" for r in observed_receipts),
            "hashed_artifacts": len(reader.records),
            "simulator_or_controller_executed": False,
            "physical_replay": "cross-bound previously completed replay receipts; not rerun by this auditor",
        },
        "artifact_records": list(reader.records.values()),
        "performance_means_computed": False,
        "registered_evaluation_eligible": False,
        "limits": [
            "Development custody audit only; no policy selection, performance means or material-benefit threshold.",
            "Successful physical replays are cross-bound; no environment reset, policy inference or controller execution is performed by this auditor.",
            "Diagnostics have file hashes and rollout/algorithm joins; they are not embedded cryptographic commitments in the checkpoint.",
            "Native comparator exception authenticity is not independently established by physical-prefix replay.",
            "Initial-residual/H equivalence compares all complete replayed pairs, retaining unequal or unavailable slots without a replacement.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packet = audit_closed_batch()
    if args.output:
        publish_json(args.output, packet)
    else:
        sys.stdout.buffer.write(canonical_bytes(packet))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
