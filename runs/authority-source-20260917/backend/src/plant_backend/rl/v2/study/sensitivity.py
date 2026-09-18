"""Generic policy-minus-E-beta additive rescoring, with no controller execution."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from ..training.custody_json import authenticate, canonical_bytes
from .contract import COMPONENT_NAMES, MULTIPLIERS, PREFIX, analysis_contract
from .statistics import finite_real

SCHEMA = PREFIX + "pairwise-sensitivity-v1"


def _components(raw: Mapping[str, object]) -> dict[str, float]:
    if set(raw) != set(COMPONENT_NAMES):
        raise ValueError("sensitivity requires every exact additive ledger component")
    return {name: finite_real(raw[name]) for name in COMPONENT_NAMES}


def _sign(value: float) -> str:
    return "policy_higher" if value > 0 else ("E-beta_higher" if value < 0 else "tie")


def _weights() -> dict[str, dict[str, float]]:
    default = {name: 1.0 for name in COMPONENT_NAMES}
    settings = {"default": default}
    groups = {
        "methane_endowment": ("methane_endowment_liability_change",),
        "declared_wear": ("battery_throughput_reserve", "electrolyser_life_reserve"),
        "solar_opportunity": ("solar_opportunity_cost",),
    }
    for group, components in groups.items():
        for multiplier in MULTIPLIERS:
            settings[f"{group}.{multiplier:g}"] = {
                **default,
                **{name: multiplier for name in components},
            }
    settings["no_methane_endowment"] = {
        **default,
        "methane_endowment_liability_change": 0.0,
    }
    settings["no_declared_wear"] = {
        **default,
        "battery_throughput_reserve": 0.0,
        "electrolyser_life_reserve": 0.0,
    }
    service_solar = {
        "delivered_service_value",
        "unmet_delivery_penalty",
        "solar_opportunity_cost",
    }
    settings["service_plus_solar_only"] = {
        name: float(name in service_solar) for name in COMPONENT_NAMES
    }
    return settings


def rescore_pair(
    policy: Mapping[str, object], e_beta: Mapping[str, object]
) -> dict[str, Any]:
    """Compute every preregistered coefficient setting from retained components."""
    left, right = _components(policy), _components(e_beta)
    deltas = {name: finite_real(left[name] - right[name]) for name in COMPONENT_NAMES}
    settings: list[dict[str, Any]] = []
    for name, weights in _weights().items():
        try:
            delta = finite_real(
                math.fsum(
                    deltas[component] * weights[component]
                    for component in COMPONENT_NAMES
                )
            )
        except OverflowError as reason:
            raise ValueError("sensitivity arithmetic is non-finite") from reason
        settings.append(
            {
                "name": name,
                "weights": weights,
                "delta_eur": delta,
                "higher_arm": _sign(delta),
            }
        )
    default_sign = settings[0]["higher_arm"]
    return authenticate(
        {
            "schema": SCHEMA,
            "analysis_contract_sha256": analysis_contract()["sha256"],
            "orientation": "policy minus E-beta",
            "unit": "EUR",
            "inputs": {"policy_components_eur": left, "e_beta_components_eur": right},
            "component_deltas_eur": deltas,
            "settings": settings,
            "ranking_reversals_from_default": [
                item["name"]
                for item in settings[1:]
                if item["higher_arm"] != default_sign
            ],
            "claim_boundary": "synthetic coefficient sensitivity; no rerun, fitted coefficient or alternative selection objective",
        }
    )


def verify_sensitivity_result(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every setting; this does not independently admit input ledgers."""
    inputs = raw.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "policy_components_eur",
        "e_beta_components_eur",
    }:
        raise ValueError("sensitivity input ledger fields differ")
    if any(not isinstance(value, dict) for value in inputs.values()):
        raise ValueError("sensitivity input ledgers must be mappings")
    expected = rescore_pair(
        inputs["policy_components_eur"], inputs["e_beta_components_eur"]
    )
    if canonical_bytes(raw) != canonical_bytes(expected):
        raise ValueError("sensitivity calculation differs from its additive inputs")
    return {
        "verification": "sensitivity_reconstructed",
        "source_evidence_verified": False,
        "result_claim": None,
    }
