"""Committed numerical representations for the frozen constrained-RL study."""

from __future__ import annotations

from typing import Any

from ..training.config import (
    BETA_CORE_ALGORITHM_IDS as BETA_CORE_ALGORITHM_IDS,
    BETA_OPTIMIZER_SEEDS as BETA_OPTIMIZER_SEEDS,
)
from ..training.custody_json import authenticate

PREFIX = "plant-beta-constrained-rl-study-"
CONTRACT_SCHEMA = PREFIX + "analysis-contract-v1"
PROTOCOL_ID = "plant-beta-constrained-rl-protocol-v1"
PROTOCOL_SHA256 = "4d0269f37c5cf1fb398be655505ec3d63b081f55196eb81eec67db2a797f5a22"
EPISODE_COUNT = 32
BOOTSTRAP_SEED = 20260905
BOOTSTRAP_REPLICATES = 10_000
PROFILE_THRESHOLDS_EUR = (
    -10000,
    -5000,
    -2000,
    -1000,
    -500,
    -250,
    -100,
    -50,
    -25,
    -10,
    0,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2000,
    5000,
    10000,
)
COMPONENT_NAMES = (
    "delivered_service_value",
    "unmet_delivery_penalty",
    "solar_opportunity_cost",
    "battery_throughput_reserve",
    "electrolyser_life_reserve",
    "methane_endowment_liability_change",
    "battery_endowment_liability_change",
)
MULTIPLIERS = (0.0, 0.5, 1.0, 2.0)


def analysis_contract() -> dict[str, Any]:
    """Return the exact source-bound contract, without inspecting study data."""
    return authenticate(
        {
            "schema": CONTRACT_SCHEMA,
            "protocol": {"id": PROTOCOL_ID, "raw_sha256": PROTOCOL_SHA256},
            "family_order": list(BETA_CORE_ALGORITHM_IDS),
            "optimizer_seeds": list(BETA_OPTIMIZER_SEEDS),
            "episode_count": EPISODE_COUNT,
            "missing_policy": "no paired value aggregate; retained in denominator of five",
            "numeric_type": "float64; finite real values; booleans and strings rejected",
            "sample_std_ddof": 1,
            "iqm_sorted_slice": [8, 24],
            "cvar20_smallest_count": 7,
            "selection_key": [
                "iqm",
                "cvar20",
                "median",
                "positive_fraction",
                "negative_optimizer_seed",
            ],
            "improvement": "strictly greater than zero; exact zero is a tie",
            "bootstrap": {
                "generator": "numpy.random.Generator(PCG64)",
                "seed": BOOTSTRAP_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "restart": "fresh generator for each calculation; shared draws across its statistics",
                "episode_resampling": "32 paired episode indices with replacement",
                "family_resampling": "optimizer indices first, then a common paired episode block across rows",
                "family_statistic": "equal-weight mean of complete policies' 32-episode statistics",
                "missingness": "conditional on complete policies; retain all five optimizer slots",
                "matched_family": "identical optimizer IDs only; both complete vectors required",
                "quantiles": [0.025, 0.975],
                "quantile_method": "linear",
            },
            "binomial": {
                "interval": "two-sided 95% Wilson",
                "z": "statistics.NormalDist().inv_cdf(0.975)",
                "zero_denominator": "null rate and interval; counts retained",
            },
            "profile": {
                "unit": "paired delta EUR",
                "thresholds": list(PROFILE_THRESHOLDS_EUR),
                "comparison": "strictly greater than threshold",
                "family_interval": "same optimizer/episode bootstrap; never 160-trial binomial",
            },
            "sensitivity": {
                "orientation": "policy minus E-beta",
                "components": list(COMPONENT_NAMES),
                "multipliers": list(MULTIPLIERS),
                "groups": {
                    "methane_endowment": ["methane_endowment_liability_change"],
                    "declared_wear": [
                        "battery_throughput_reserve",
                        "electrolyser_life_reserve",
                    ],
                    "solar_opportunity": ["solar_opportunity_cost"],
                },
                "counterfactuals": [
                    "default",
                    "no_methane_endowment",
                    "no_declared_wear",
                    "service_plus_solar_only",
                ],
                "reversal": "every sign change, including to or from exact zero",
                "selection_role": "none",
            },
            "claim_boundary": "numerical analysis only; no verified eligibility or execution authorization",
        }
    )
