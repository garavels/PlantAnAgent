"""Read-only PPO launch proposal. Opens no episode, checkpoint or campaign root."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from plant_backend.rl.v2.study.execution_budget import ExecutionBudget
from plant_backend.rl.v2.training import capacity_benchmark as capacity
from plant_backend.rl.v2.training import custody_authorization as authority
from plant_backend.rl.v2.training import custody_io as io
from plant_backend.rl.v2.training.registered_contract import matrix, TEMPLATE_SHA256
from plant_backend.rl.v2.training.runtime_provenance import capture

CAMPAIGN_ROOT = Path("/Volumes/Lexar/PlantAnAgentResearch/registered-beta-v1")
CAPACITY_REPORT = Path(
    "/Users/filippogombac/Developer/PlantAnAgentResearchArtifacts/capacity-4f505b8/report.json"
)


def build() -> dict[str, Any]:
    import torch

    torch.set_num_threads(8)
    torch.set_num_interop_threads(12)
    runtime = capture(require_clean=True)
    repository = authority.repository_root()
    if CAMPAIGN_ROOT.exists() or CAMPAIGN_ROOT.is_symlink():
        raise ValueError("proposed registered campaign root must remain uncreated")
    if (repository / authority.AUTHORIZATION_PATH).exists():
        raise ValueError("readiness packet is for the pre-authorization implementation")
    io.require_directory(CAMPAIGN_ROOT.parent)
    raw = io.read_regular(CAPACITY_REPORT)
    if hashlib.sha256(raw).hexdigest() != authority.CAPACITY_RAW_SHA256:
        raise ValueError("recovered capacity report bytes differ")
    report = json.loads(raw)
    if report["sha256"] != authority.CAPACITY_DOMAIN_SHA256:
        raise ValueError("capacity report identity differs")
    host = capacity._verify_host(capacity._host_identity())
    if host != report["host"]:
        raise ValueError("host differs from the retained capacity measurement")
    historical = json.loads(
        (
            repository / "docs/research/results/v2_training_readiness_12936c7.json"
        ).read_text()
    )
    # Keep every earlier conservative cap; increase only the filesystem margin
    # for Lexar's 256 KiB allocation units. This is not a stage authorization.
    budget = replace(
        ExecutionBudget.from_dict(historical["evaluation_budget_review"]["budget"]),
        filesystem_reserve_bytes=4 * 1024**3,
    )
    disk = os.statvfs(CAMPAIGN_ROOT.parent)
    free = disk.f_bavail * disk.f_frsize
    result = {
        "schema": "plant-ppo-launch-readiness-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "commit": runtime.git_head,
            "tree": subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"], cwd=repository, text=True
            ).strip(),
            "worktree_clean": True,
        },
        "protocols": authority.protocol_identity(repository),
        "runtime_provenance": runtime.to_dict(),
        "host": host,
        "capacity_report": {
            "path": str(CAPACITY_REPORT),
            "raw_sha256": authority.CAPACITY_RAW_SHA256,
            "sha256": authority.CAPACITY_DOMAIN_SHA256,
            "report_only_recovered": True,
            "historical_checkpoint_reverified": False,
        },
        "training_projection": historical["training_projection"],
        "campaign_root": str(CAMPAIGN_ROOT),
        "campaign_root_created": False,
        "authorization_created": False,
        "execution_authorized": False,
        "registered_task_template_sha256": TEMPLATE_SHA256,
        "training_contract": authority.EXECUTION,
        "cells": matrix(CAMPAIGN_ROOT),
        "evaluation_budget_proposal": budget.to_dict(),
        "filesystem": {
            "path": str(CAMPAIGN_ROOT.parent),
            "free_bytes": free,
            "fragment_bytes": disk.f_frsize,
            "training_floor_met": free >= authority.EXECUTION["minimum_free_bytes"],
            "training_plus_two_stage_planning_reserve_met": free
            >= budget.opening_free_bytes("development")
            + authority.EXECUTION["minimum_free_bytes"],
        },
        "scientific_limitations": [
            "Research E-beta failed opening planning on three retained fixtures; it can make all 32-pair policy scores ineligible.",
            "Product MPC continuity is not research feasibility; it is not substituted in registered comparisons.",
            "PPO improvement and held-out eligibility are unknown; negative development leaves held-out sealed.",
            "Storage is sufficient for proposed caps; complete trained-policy/MPC evaluation capacity is unmeasured.",
            "Training duration is a historical host projection; USB artifact I/O is not a new capacity measurement.",
        ],
    }
    if capture(require_clean=True) != runtime:
        raise ValueError("runtime changed during readiness observation")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="New compact proposal file outside tracked source"
    )
    args = parser.parse_args()
    result = build()
    io.publish_json(args.output, result)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "source": result["source"],
                "filesystem": result["filesystem"],
                "execution_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
