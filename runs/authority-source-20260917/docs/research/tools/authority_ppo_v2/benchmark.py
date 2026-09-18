"""New split/identity over the unchanged feasible25 physical environment."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from docs.research.tools import feasible_benchmark_v1 as legacy
from docs.research.tools.feasible_mission_replay import canonical_sha, file_pin
from plant_backend.rl.v2.contract import contract_digest

ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = ROOT / "docs/research/benchmarks/authority_ppo_v2.json"
TRAIN_BASE = 9_940_000
ENVIRONMENT_ID = "plant-env-feasible25-offgrid-authority-v2"
STAGE = "authority-ppo-research-v2"
INTERFACE_CONTRACT = {
    **legacy.INTERFACE_CONTRACT,
    "schema": "feasible-mission-authority-interface-v2",
    "environment_id": ENVIRONMENT_ID,
}
INTERFACE_DIGEST = contract_digest(INTERFACE_CONTRACT)


def load_benchmark_contract(path: Path = CONTRACT_PATH):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != "plant-authority-ppo-benchmark-v2":
        raise ValueError("Unknown authority benchmark schema")
    if payload.get("contract_sha256") != canonical_sha(
        {k: v for k, v in payload.items() if k != "contract_sha256"}
    ):
        raise ValueError("Authority benchmark checksum differs")
    parent = legacy.load_benchmark_contract()
    if payload["parent_benchmark_sha256"] != parent["contract_sha256"]:
        raise ValueError("Parent physical benchmark identity differs")
    for key in ("source_template", "delivery_rate_kg_h", "electricity", "acceptance", "inventory_sampling"):
        if payload[key] != parent[key]:
            raise ValueError(f"Physical task changed: {key}")
    splits = list(payload["splits"].values())
    if any(len(values) != len(set(values)) for values in splits):
        raise ValueError("Duplicate split seed")
    if any(set(a) & set(b) for i, a in enumerate(splits) for b in splits[i + 1:]):
        raise ValueError("Authority benchmark split overlap")
    historical = set().union(*map(set, parent["splits"].values()))
    for name in ("train", "development", "held_out"):
        if historical & set(payload["splits"][name]):
            raise ValueError("Fresh split overlaps historical scenarios")
    return payload


class BenchmarkEnv(legacy.BenchmarkEnv):
    """Reuse all historical physical/accounting methods without editing them."""

    def __init__(self, split="train", *, electricity_model=None):
        contract = load_benchmark_contract()
        if split not in contract["splits"]:
            raise ValueError("Unknown authority benchmark split")
        # Legacy initialization constructs no episode and performs no reset.
        super().__init__(split="train", electricity_model=electricity_model)
        self.benchmark_contract = contract
        self.split = split
        self.allowed_seeds = frozenset(contract["splits"][split])
        payload = deepcopy(self.checkpoint_contract)
        payload.pop("checkpoint_contract_digest")
        payload.update(
            environment_id=ENVIRONMENT_ID,
            interface_contract=INTERFACE_CONTRACT,
            interface_contract_digest=INTERFACE_DIGEST,
            benchmark_contract_sha256=contract["contract_sha256"],
        )
        self._checkpoint_contract = {
            "checkpoint_contract_digest": contract_digest(payload), **payload
        }

    def reset(self, *, seed=None, options=None):
        result = super().reset(seed=seed, options=options)
        self.frozen_source.source_pin = file_pin(CONTRACT_PATH)
        return result

    def _identify(self, info):
        info = super()._identify(info)
        info.update(environment_id=ENVIRONMENT_ID,
                    environment_contract_digest=INTERFACE_DIGEST, stage=STAGE)
        return info

    def audit_manifest(self):
        payload = super().audit_manifest()
        payload.pop("manifest_sha256")
        payload["environment"] = {
            "id": ENVIRONMENT_ID, "stage": STAGE, "interface": INTERFACE_CONTRACT
        }
        accounting = {k: payload.pop(k) for k in
                      ("delivery_accounting", "economic_accounting", "cycle_accounting")}
        return {"manifest_sha256": canonical_sha(payload), **payload, **accounting}
