"""Synthetic custody data and artifact writers; no trainer or environment access."""

from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

from plant_backend.rl.v2.training import capacity_benchmark as capacity
from plant_backend.rl.v2.training import custody_authorization as auth
from plant_backend.rl.v2.training import custody_documents as docs
from plant_backend.rl.v2.training import custody_io as io
from plant_backend.rl.v2.training.config import ControllerNeutralTrainingConfigV1
from plant_backend.rl.v2.training.custody_json import (
    authenticate,
    canonical_bytes,
    decode,
)
from plant_backend.rl.v2.training.custody_state import (
    CampaignContext,
    CollectionServices,
    IdentitySnapshot,
    command_for,
)
from plant_backend.rl.v2.training import runtime_provenance as runtime


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def fixture_runtime() -> runtime.BetaTrainingRuntimeProvenanceV1:
    return runtime.BetaTrainingRuntimeProvenanceV1(
        git_head="a" * 40,
        worktree_clean=True,
        worktree_status_sha256=hashlib.sha256(b"").hexdigest(),
        worktree_content_sha256=runtime._empty_worktree_content_sha256(),
        uv_lock_sha256=sha("lock"),
        dependency_entrypoints_sha256=sha("entrypoints"),
        python_version="3.12.11",
        python_implementation="CPython",
        platform="fixture",
        numpy_version="2.2.6",
        scipy_version="1.15.3",
        torch_version="2.7.1",
        stable_baselines3_version="2.9.0",
        gymnasium_version="1.2.0",
        plant_sim_version="0.5.2",
        plant_sim_module_path="simulator/src/plant_sim/__init__.py",
        plant_sim_module_sha256=sha("simulator"),
    )


def context(root: Path) -> CampaignContext:
    rt = fixture_runtime()
    host = {
        "platform": "fixture",
        "system": "fixture",
        "machine": "fixture",
        "processor": None,
        "cpu_brand": None,
        "physical_cpu_count": 4,
        "logical_cpu_count": 8,
        "physical_memory_bytes": 16 * 1024**3,
        "torch_intraop_threads": 4,
        "torch_interop_threads": 2,
        "thread_environment": {
            name: None for name in capacity.CAPACITY_THREAD_ENVIRONMENT
        },
        "clock": {
            "name": "time.perf_counter_ns",
            "monotonic": True,
            "adjustable": False,
            "resolution_s": 1e-9,
        },
    }
    source = {
        "authorization_commit": rt.git_head,
        "approved_implementation_commit": "b" * 40,
        "approved_implementation_tree": "c" * 40,
        "uv_lock_sha256": rt.uv_lock_sha256,
        "authorization_parent": "b" * 40,
        "changed_paths": [auth.AUTHORIZATION_PATH],
    }
    authorization = authenticate(
        {
            "schema": auth.AUTHORIZATION_SCHEMA,
            "authorization_id": "f28e6e04-c99e-4705-adb5-01412b2f81e1",
            "authorized_at": "2026-09-11T12:00:00.000000Z",
            "protocols": {"synthetic": True},
            "capacity_report": {
                "path": "/private/tmp/unopened-synthetic-capacity.json",
                "raw_file_sha256": auth.CAPACITY_RAW_SHA256,
                "sha256": auth.CAPACITY_DOMAIN_SHA256,
            },
            "approved_implementation": {
                "commit": "b" * 40,
                "tree": "c" * 40,
                "uv_lock_sha256": rt.uv_lock_sha256,
            },
            "campaign_root": str(root),
            "host_contract": {"host": host, "capacity_projection_applies": False},
            "campaign_contract": auth.EXECUTION,
            "registered_training_authorized": True,
            "development_evaluation_authorized": False,
            "held_out_evaluation_authorized": False,
            "non_retry_statement": auth.NON_RETRY_STATEMENT,
        }
    )
    plan = docs.make_plan(
        root,
        authorization=authorization,
        source=source,
        runtime=rt.to_dict(),
        host=host,
    )
    return CampaignContext(
        root, authorization, plan, IdentitySnapshot(source, rt.to_dict(), host)
    )


def admit(
    config: ControllerNeutralTrainingConfigV1,
    rt: runtime.BetaTrainingRuntimeProvenanceV1,
) -> dict[str, Any]:
    path = Path(config.output_dir) / "checkpoint"
    files = [
        io.file_record(path / name)
        for name in ("metadata.json", "policy.zip", "training_config.json")
    ]
    return {
        "path": str(path),
        "metadata_sha256": files[0]["sha256"],
        "checkpoint_evidence_sha256": sha("synthetic checkpoint"),
        "runtime_provenance_sha256": rt.sha256,
        "environment_contract_sha256": sha("environment"),
        "checkpoint_contract_sha256": sha("checkpoint"),
        "files": files,
        "logical_bytes": sum(item["logical_bytes"] for item in files),
        "allocated_bytes": sum(item["allocated_bytes"] for item in files),
        "rollout_entry_count": 128,
        "allocator_entry_count": 1564,
        "optimizer_steps": 40960,
        "registered_evaluation_eligible": True,
    }


class FixtureServices:
    def __init__(self, ctx: CampaignContext) -> None:
        self.context = ctx
        self.current = deepcopy(ctx.identity)
        self.started: list[int] = []
        self.process_code: str | None = None
        self.mutation: str | None = None
        self.services = CollectionServices(
            self.process,
            admit,
            lambda: deepcopy(self.current),
            lambda root, minimum: True,
        )

    def process(self, cell: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
        assert len(list((self.context.root / "configs").iterdir())) == 15
        assert io.read_regular(Path(cell["claim_path"]))
        config = ControllerNeutralTrainingConfigV1.from_file(cell["config_path"])
        assert canonical_bytes(config.to_dict()) == io.read_regular(
            Path(cell["config_path"])
        )
        self.started.append(cell["order"])
        for name in ("stdout", "stderr"):
            io.publish_bytes(Path(cell[f"{name}_path"]), b"")
        output = Path(cell["output_path"])
        io.mkdir_exclusive(output)
        io.mkdir_exclusive(output / "checkpoint")
        for name, raw in (
            ("policy.zip", b"synthetic policy; not a loadable trained artifact"),
            ("metadata.json", b"{}\n"),
            ("training_config.json", canonical_bytes(config.to_dict())),
        ):
            io.publish_bytes(output / "checkpoint" / name, raw)
        process: dict[str, Any] = {
            "command": command_for(cell),
            "timeout_seconds": 36000,
            "started": True,
            "exit_code": 0,
            "termination_signal": None,
            "wall_elapsed_ns": 1_000_000_000,
            "child_user_cpu_ns": 500_000_000,
            "child_system_cpu_ns": 0,
        }
        codes: set[str] = set()
        if cell["order"] == 2 and self.process_code:
            codes.add(self.process_code)
            if self.process_code in {"child_timeout", "child_signaled"}:
                process["exit_code"] = -15
                process["termination_signal"] = 15
                codes.add("child_signaled")
            if self.process_code == "child_timeout":
                process["wall_elapsed_ns"] = 36000 * 1_000_000_000
            elif self.process_code == "child_nonzero_exit":
                process["exit_code"] = 3
            elif self.process_code == "child_spawn_failed":
                process = {
                    key: (value if key in {"command", "timeout_seconds"} else None)
                    for key, value in process.items()
                }
                process["started"] = False
            elif self.process_code == "child_log_limit_exceeded":
                with Path(cell["stdout_path"]).open("r+b") as handle:
                    handle.truncate(67_108_864)
        if cell["order"] == 2 and self.mutation:
            if self.mutation == "source_mutation":
                rt = replace(
                    runtime.BetaTrainingRuntimeProvenanceV1.from_dict(
                        self.current.runtime
                    ),
                    git_head="d" * 40,
                )
                self.current = replace(
                    self.current,
                    source={**self.current.source, "authorization_commit": rt.git_head},
                    runtime=rt.to_dict(),
                )
            elif self.mutation == "runtime_mutation":
                rt = replace(
                    runtime.BetaTrainingRuntimeProvenanceV1.from_dict(
                        self.current.runtime
                    ),
                    torch_version="changed-fixture",
                )
                self.current = replace(self.current, runtime=rt.to_dict())
            elif self.mutation == "host_mutation":
                self.current = replace(
                    self.current,
                    host={**self.current.host, "cpu_brand": "changed-fixture"},
                )
            elif self.mutation in {"authorization_mutation", "plan_mutation"}:
                path = self.context.root / (
                    "authorization.json"
                    if self.mutation == "authorization_mutation"
                    else "plan.json"
                )
                value = decode(path.read_bytes())
                value["sha256"] = "0" * 64
                path.write_bytes(canonical_bytes(value))
            elif self.mutation == "root_custody_lost":
                codes.add("root_custody_lost")
        return process, codes
