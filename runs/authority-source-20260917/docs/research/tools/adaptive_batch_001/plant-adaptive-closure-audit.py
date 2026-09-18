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

from plant_backend.rl.v2.training import research_batch as batch
from plant_backend.rl.v2.training import research_evaluation as evaluation
from plant_backend.rl.v2.training import runtime_provenance as provenance
from plant_backend.rl.v2.training.config import (
    ControllerNeutralTrainingConfigV1,
    BETA_PI_LAGRANGIAN_ALGORITHM_ID,
)
from plant_backend.rl.v2.beta_config import RESOURCE_NAMES
from plant_backend.rl.v2.training.custody_io import publish_json, read_regular
from plant_backend.rl.v2.training.custody_json import canonical_bytes, decode

PINNED_HEAD = "238fe7ca763254b70af63a4ef0690a9d4523aa60"
PINNED_TREE = "becdfaa68e7924a7af75ca2794b32f09dbf8d4e2"
PINNED_GATE_SHA256 = "88b4fd2d8fa6c8f38c8ecb7f30b8d09bca89539e9f751de75bd6163e59c1cf98"
ACTORS = (*batch.BASELINES, "initial", *(a.name for a in batch.ARMS))


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
    return {
        "git_head": saved.git_head,
        "git_tree": tree,
        "runtime_sha256": saved.sha256,
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


def expected_contract(config: ControllerNeutralTrainingConfigV1) -> str:
    from plant_backend.rl.v2.beta_env import describe_beta_checkpoint_contract

    assert config.beta_task is not None
    return str(
        describe_beta_checkpoint_contract(
            plant=config.plant,
            episode=config.episode,
            constraints=config.constraints,
            weights=config.diagnostic_reward,
            assumptions=config.economic_assumptions,
            beta_task=config.beta_task,
        )["checkpoint_contract_digest"]
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


def training_record(
    reader: Reader, root: Path, arm: batch.Arm, success: bool, receipt: dict[str, Any]
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "closed_successfully": success,
        "receipt": receipt,
        "diagnostics": [],
        "result": None,
        "failures": [],
    }
    directory = root / "training" / arm.name
    if not success:
        record["failures"] = [
            process_failure(reader, root, receipt, "training_process")
        ]
        return record
    config = batch.template(root)
    attempt = reader.json(directory / "attempt.json")
    require(
        same(attempt["arm"], asdict(arm))
        and same(attempt["template"], config.to_dict())
        and attempt["seed_base"] == batch.TRAIN_SEED_BASE,
        "training attempt identity differs",
    )
    result = reader.json(directory / "result.json")
    require(
        result["schema"] == batch.SCHEMA
        and same(result["arm"], asdict(arm))
        and same(result["environment_template"], config.to_dict()),
        "training result identity differs",
    )
    require(
        result["total_timesteps"] == batch.TOTAL_STEPS
        and result["registered_evaluation_eligible"] is False,
        "training endpoint or eligibility differs",
    )
    require(
        result["checkpoint_contract_digest"] == expected_contract(config),
        "checkpoint task contract differs",
    )
    reader.check(directory / "initial-policy.pt", result["initial_policy"])
    reader.check(directory / "final-policy.zip", result["final_policy"])
    payload = result["constraint_state"]["payload"]
    require(
        payload["num_timesteps"] == batch.TOTAL_STEPS
        and payload["pending_rollout"] is False,
        "constraint endpoint differs",
    )
    ledgers = payload["rollout_ledgers"]
    count = batch.TOTAL_STEPS // (config.n_envs * config.ppo.n_steps)
    require(len(ledgers) == count, "rollout ledger count differs")
    diagnostics = []
    for index, ledger in enumerate(ledgers, start=1):
        ledger_path = directory / "diagnostics" / f"rollout-{index:03d}.json"
        require(
            same(reader.json(ledger_path), ledger),
            "published rollout differs from endpoint",
        )
        diagnostic_path = directory / "diagnostics" / f"diagnostic-{index:03d}.json"
        diagnostic = reader.json(diagnostic_path)
        require(
            diagnostic["schema"] == "plant-research-ppo-rollout-diagnostics-v1"
            and diagnostic["num_timesteps"]
            == index * config.n_envs * config.ppo.n_steps
            and diagnostic["algorithm_id"] == arm.algorithm,
            "diagnostic rollout or algorithm identity differs",
        )
        diagnostics.append(diagnostic)
    record.update(
        {
            "initial_policy": result["initial_policy"],
            "final_policy": result["final_policy"],
            "result": result,
            "diagnostics": diagnostics,
            "first_rollout": ledgers[0],
            "final_rollout": ledgers[-1],
        }
    )
    return record


def compare_initial_tensors(
    reader: Reader, root: Path, training: dict[str, Any]
) -> dict[str, Any]:
    if not all(value["closed_successfully"] for value in training.values()):
        return {
            "performed": False,
            "reason": "one or more training children failed; no substitute initializer",
        }
    import torch

    weights = {}
    for arm in batch.ARMS:
        raw = reader.check(
            root / "training" / arm.name / "initial-policy.pt",
            training[arm.name]["initial_policy"],
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
        weights[arm.name] = data
    reference = weights[batch.ARMS[0].name]
    shared_keys = {key for key in reference if not key.startswith("cost_critic.")}
    require("log_std" in shared_keys, "initial policy omits log_std")
    for arm in batch.ARMS:
        current = weights[arm.name]
        require(
            {key for key in current if not key.startswith("cost_critic.")}
            == shared_keys,
            "actor/reward tensor key set differs",
        )
        for key in shared_keys:
            if key == "log_std":
                expected = torch.full_like(current[key], math.log(arm.initial_std))
                require(
                    torch.equal(current[key], expected),
                    "initial log_std differs from declared intervention",
                )
            else:
                require(
                    torch.equal(current[key], reference[key]),
                    f"matched actor/reward initialization differs: {arm.name}/{key}",
                )
    cost_keys = {key for key in reference if key.startswith("cost_critic.")}
    require(bool(cost_keys), "PI initialization omits cost critic")
    for arm in (
        a for a in batch.ARMS if a.algorithm == BETA_PI_LAGRANGIAN_ALGORITHM_ID
    ):
        current = weights[arm.name]
        require(
            {key for key in current if key.startswith("cost_critic.")} == cost_keys
            and all(torch.equal(current[key], reference[key]) for key in cost_keys),
            "matched PI cost critic initialization differs",
        )
    return {
        "performed": True,
        "passed": True,
        "actor_reward_tensor_keys": len(shared_keys),
        "pi_cost_tensor_keys": len(cost_keys),
        "exception": "E-pi-scaled-lowstd log_std equals log(0.2)",
    }


def verify_episode(
    reader: Reader, root: Path, actor: str, seed: int, replay_success: bool
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
        episode["schema"] == evaluation.SCHEMA
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
        == sha(read_regular(Path(evaluation.__file__))),
        "episode evaluator source differs",
    )
    value = episode["summary"]
    expected_actor = actor if actor in batch.BASELINES else "PPO"
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
        require(same(replay, expected), "replay/episode cross-binding differs")
    return cast(dict[str, Any], value)


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
            plan["gate"]["sha256"] == PINNED_GATE_SHA256, "launch gate identity differs"
        )
    gate_path = Path(plan["gate"]["path"])
    gate = decode(reader.check(gate_path, plan["gate"]))
    require(
        gate["passed"] is True and gate["source_commit"] == PINNED_HEAD,
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
                else training[batch.ARMS[0].name if actor == "initial" else actor][
                    "closed_successfully"
                ]
            )
            if not trained:
                failures.append(
                    {
                        "kind": "training_unavailable",
                        "arm": batch.ARMS[0].name if actor == "initial" else actor,
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
                summary = verify_episode(reader, root, actor, seed, replay_ok)
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
    auditor_path = Path(__file__).resolve()
    reader.read(auditor_path)
    reader.recheck()
    if not fixture:
        require(
            same(source, verify_source(plan, fixture=False)),
            "source changed during audit",
        )
    return {
        "schema": "plant-adaptive-closed-batch-audit-v1",
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
