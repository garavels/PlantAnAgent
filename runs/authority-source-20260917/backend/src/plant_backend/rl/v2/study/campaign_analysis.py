"""All-slot stage analysis from admitted inputs; never execution authority.

The detached stage layer must reconstruct the supplied custody/host/publication
facts and all physical, timing and training inputs. Arithmetic alone cannot
publish the restricted controller claim or open held-out seeds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from ..training.custody_json import authenticate, canonical_bytes
from . import pairwise, registry
from .contract import (
    BETA_CORE_ALGORITHM_IDS,
    BETA_OPTIMIZER_SEEDS,
    EPISODE_COUNT,
    PREFIX,
    analysis_contract,
)
from .latency import INFERENCE_LIMIT_NS, latency_summary
from .operational import policy_scorecard
from .selection import DevelopmentPolicy, select_family
from .sensitivity import _weights, verify_sensitivity_result
from .statistics import family_summary, finite_real, policy_summary
from .training_scorecards import TrainingSlot, training_scorecard

SCHEMA = PREFIX + "campaign-analysis-v1"
CELL_STATES = (
    "not_started",
    "training_not_admitted",
    "process_failed",
    "evidence_rejected",
    "verified",
)
CHANNELS = ("inference", "controller", "end_to_end", "dispatcher", "solver")
VALUE_STATISTICS = ("mean", "median", "iqm", "cvar20", "minimum", "maximum")


@dataclass(frozen=True)
class CellInput:
    identity: Mapping[str, Any]
    state: str
    failure: str | None
    arm: Mapping[str, Any] | None
    latency: Mapping[str, Any] | None


@dataclass(frozen=True)
class CustodyFacts:
    training_verified: bool = False
    stage_verified: bool = False
    registered_hardware_verified: bool = False
    identities_frozen: bool = False
    published_fields_verified: bool = False

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in asdict(self).values()):
            raise ValueError("custody facts must be exact booleans")


def _vector(
    pairs: Sequence[Mapping[str, Any]], *, admitted: bool
) -> list[float] | None:
    if len(pairs) != EPISODE_COUNT:
        raise ValueError("a policy must retain all 32 pairs")
    if not admitted or any(
        pair["value_comparison_eligible"] is not True for pair in pairs
    ):
        return None
    return [finite_real(pair["delta_eur"]) for pair in pairs]


def pooled_latency(
    seeds: Sequence[int],
    records: Sequence[Mapping[str, Any] | None],
    *,
    arm: str,
    expected_host_sha256: str | None,
) -> dict[str, Any]:
    if len(seeds) != EPISODE_COUNT or len(records) != EPISODE_COUNT:
        raise ValueError("latency pooling requires all 32 slots")
    if (
        any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != EPISODE_COUNT
    ):
        raise ValueError("latency pooling requires unique exact episode identities")
    if expected_host_sha256 is not None and (
        not isinstance(expected_host_sha256, str)
        or len(expected_host_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_host_sha256)
    ):
        raise ValueError("registered latency host digest is malformed")
    samples: dict[str, list[int]] = {name: [] for name in CHANNELS}
    present = []
    for record in records:
        if record is None:
            continue
        if (
            record["arm"] != arm
            or set(record["channel_samples_ns"]) != set(CHANNELS)
            or set(record["channels"]) != set(CHANNELS)
        ):
            raise ValueError("pooled latency actor or channel fields differ")
        for name in CHANNELS:
            raw = record["channel_samples_ns"][name]
            if canonical_bytes(latency_summary(raw)) != canonical_bytes(
                record["channels"][name]
            ):
                raise ValueError("pooled latency summary differs from all raw samples")
            samples[name].extend(raw)
        present.append(record)
    channels = {name: latency_summary(raw) for name, raw in samples.items()}
    hosts_match = (
        expected_host_sha256 is not None
        and bool(present)
        and all(
            record.get("host_sha256") == expected_host_sha256
            and record.get("torch_single_thread") is True
            for record in present
        )
    )
    warmups_complete = arm != "PPO" or all(
        record["warmup_status"] == "complete" for record in present
    )
    return {
        "episode_seeds": list(seeds),
        "measurement_count": len(present),
        "missing_episode_seeds": [
            seed for seed, record in zip(seeds, records, strict=True) if record is None
        ],
        "raw_measurement_sha256": [
            None if record is None else record["raw_sha256"] for record in records
        ],
        "channel_samples_ns": samples,
        "channels": channels,
        "host_matches_registered_identity": hosts_match,
        "warmups_complete": warmups_complete,
        "all_32_measured": len(present) == EPISODE_COUNT,
        "inference_failures": sum(record["inference_failures"] for record in present),
        "controller_failures": sum(record["controller_failures"] for record in present),
        "interval_failures": sum(record["interval_failures"] for record in present),
        "source_evidence_verified": False,
        "result_claim": None,
    }


def policy_sensitivity(
    pairs: Sequence[Mapping[str, Any]], *, admitted: bool
) -> dict[str, Any]:
    complete = _vector(pairs, admitted=admitted) is not None
    settings: list[dict[str, Any]] = []
    for pair in pairs:
        if pair["sensitivity"] is not None:
            verify_sensitivity_result(pair["sensitivity"])
    for name in _weights():
        values = []
        for pair in pairs:
            sensitivity = pair["sensitivity"]
            values.append(
                None
                if sensitivity is None
                else next(
                    row["delta_eur"]
                    for row in sensitivity["settings"]
                    if row["name"] == name
                )
            )
        settings.append(
            {
                "name": name,
                "values_eur": values,
                "summary": policy_summary(values) if complete else None,
            }
        )
    reference = settings[0]["summary"]
    reversals = []
    if reference is not None:
        for setting in settings[1:]:
            for metric in VALUE_STATISTICS:
                initial = reference["statistics"][metric]
                current = setting["summary"]["statistics"][metric]
                if (initial > 0) - (initial < 0) != (current > 0) - (current < 0):
                    reversals.append(
                        {
                            "setting": setting["name"],
                            "statistic": metric,
                            "default": initial,
                            "changed": current,
                        }
                    )
    return {
        "complete_policy_vector": complete,
        "settings": settings,
        "per_episode_reversals": [
            (
                None
                if pair["sensitivity"] is None
                else pair["sensitivity"]["ranking_reversals_from_default"]
            )
            for pair in pairs
        ],
        "aggregate_sign_reversals": reversals,
        "selection_role": "none",
    }


def heldout_clauses(
    *,
    champion: Mapping[str, Any] | None,
    pi_policies: Sequence[Mapping[str, Any]],
    facts: CustodyFacts,
) -> dict[str, Any]:
    """Ten numerical/evidence conditions; their truth is not execution authority."""
    facts.__post_init__()
    if len(pi_policies) != 5:
        raise ValueError("held-out family must retain all five policies")
    expected_actors = registry.actors()[3:8]
    if canonical_bytes([policy["actor"] for policy in pi_policies]) != canonical_bytes(
        expected_actors
    ):
        raise ValueError(
            "held-out rule requires the five exact independent PI identities"
        )
    if champion is not None:
        matches = [
            policy for policy in pi_policies if policy["actor"] == champion["actor"]
        ]
        if len(matches) != 1 or canonical_bytes(champion) != canonical_bytes(
            matches[0]
        ):
            raise ValueError(
                "held-out champion must be the unchanged frozen family member"
            )
    pairs = [] if champion is None else champion["primary_pairs"]
    stats = None if champion is None else champion["primary_summary"]
    policy_arms = [] if champion is None else champion["physical_arms"]
    all_arms = len(policy_arms) == EPISODE_COUNT and all(
        arm is not None for arm in policy_arms
    )
    timing = None if champion is None else champion["latency"]
    checks = {
        "1_all_32_primary_pairs": len(pairs) == EPISODE_COUNT
        and all(pair["value_comparison_eligible"] is True for pair in pairs)
        and facts.training_verified
        and facts.stage_verified
        and facts.identities_frozen,
        "2_zero_unexpected_safety_forced_stop_invalid_command": all_arms
        and all(
            arm["classification"]["hard_safety_passed"] is True for arm in policy_arms
        ),
        "3_every_cycle_return_feasible": all_arms
        and all(arm["classification"]["cycle_feasible"] is True for arm in policy_arms),
        "4_all_32_service_comparable": len(pairs) == EPISODE_COUNT
        and all(pair["checks"]["service_comparable"] is True for pair in pairs),
        "5_iqm_bootstrap_lower_positive": stats is not None
        and stats["bootstrap_95"]["iqm"][0] > 0,
        "6_at_least_24_positive_pairs": stats is not None
        and stats["sign_counts"]["positive"] >= 24,
        "7_cvar20_nonnegative": stats is not None
        and stats["statistics"]["cvar20"] >= 0,
        "8_registered_inference_p99_at_most_100_ms": timing is not None
        and facts.registered_hardware_verified
        and timing["all_32_measured"]
        and timing["host_matches_registered_identity"]
        and timing["warmups_complete"]
        and timing["channels"]["inference"]["p99"] is not None
        and timing["channels"]["inference"]["p99"] <= INFERENCE_LIMIT_NS,
        "9_four_of_five_pi_positive_medians": sum(
            policy["primary_summary"] is not None
            and policy["primary_summary"]["statistics"]["median"] > 0
            for policy in pi_policies
        )
        >= 4,
        "10_all_scorecards_sensitivities_published": facts.published_fields_verified,
    }
    return {
        "checks": checks,
        "all_calculated_requirements_met": all(checks.values()),
        "failed_checks": [key for key, value in checks.items() if not value],
        "source_evidence_verified": False,
        "result_claim": None,
    }


def analyze_campaign(
    *,
    stage: str,
    episode_seeds: Sequence[int],
    cells: Sequence[CellInput],
    training_slots: Sequence[TrainingSlot],
    facts: CustodyFacts,
    frozen_champions: Mapping[str, int | None] | None = None,
    expected_host_sha256: str | None = None,
    cycle_count: int = 7,
    expected_training_rollouts: int = 128,
) -> dict[str, Any]:
    """Compose all preregistered outputs from supplied independently admitted sources.

    Fixture callers may use only synthetic inputs. Production callers must first
    admit the exact registry, training artifacts, every arm and timing witness.
    This function never opens a file, checkpoint, plant or episode manifest.
    """
    if stage not in registry.STAGES:
        raise ValueError("unknown study analysis stage")
    facts.__post_init__()
    identities = registry.cells_for_seeds(list(episode_seeds))
    if len(cells) != len(identities):
        raise ValueError("campaign analysis requires all 576 exact slots")
    if stage == "development" and frozen_champions is not None:
        raise ValueError("development choices must be computed from the full registry")
    if stage == "heldout" and (
        frozen_champions is None
        or set(frozen_champions) != set(BETA_CORE_ALGORITHM_IDS)
        or any(
            value is not None
            and (type(value) is not int or value not in BETA_OPTIMIZER_SEEDS)
            for value in frozen_champions.values()
        )
    ):
        raise ValueError("held-out analysis requires all frozen development choices")
    training = training_scorecard(
        training_slots, expected_rollouts=expected_training_rollouts
    )
    statuses = {(row.algorithm, row.optimizer_seed): row for row in training_slots}
    grouped: dict[str, list[CellInput]] = {
        actor["id"]: [] for actor in registry.actors()
    }
    source = None
    for cell, identity in zip(cells, identities, strict=True):
        if type(cell) is not CellInput or canonical_bytes(
            cell.identity
        ) != canonical_bytes(identity):
            raise ValueError("campaign cell identity/order differs")
        actor = identity["actor"]
        if (
            cell.state not in CELL_STATES
            or (cell.state == "verified") != (cell.arm is not None)
            or (cell.state == "verified") != (cell.failure is None)
            or (
                cell.failure is not None
                and (not isinstance(cell.failure, str) or not cell.failure)
            )
        ):
            raise ValueError("campaign cell evidence/failure declaration differs")
        if cell.arm is None and cell.latency is not None:
            raise ValueError("timing cannot be admitted without physical evidence")
        if actor["arm"] == "PPO":
            slot = statuses[(actor["algorithm"], actor["optimizer_seed"])]
            if not slot.checkpoint_admitted and cell.state != "training_not_admitted":
                raise ValueError(
                    "unadmitted training must retain every skipped policy slot"
                )
            if not slot.checkpoint_admitted and cell.failure != slot.failure:
                raise ValueError(
                    "skipped policy slot must retain its training custody reason"
                )
            if slot.checkpoint_admitted and cell.state == "training_not_admitted":
                raise ValueError("admitted policy cannot be silently skipped")
        elif cell.state == "training_not_admitted":
            raise ValueError("baseline cannot be excluded by policy training failure")
        if cell.arm is not None:
            if (
                cell.arm["seed"] != identity["episode_seed"]
                or cell.arm["actor"] != registry.actor_identity(actor)
                or cell.arm["operational"]["actor"] != cell.arm["actor"]
            ):
                raise ValueError(
                    "admitted arm differs from its exact registry actor/seed"
                )
            if source is not None and canonical_bytes(source) != canonical_bytes(
                cell.arm["source_provenance"]
            ):
                raise ValueError("campaign source identity differs between arms")
            source = cell.arm["source_provenance"]
            if cell.arm["stage"] not in {stage, "fixture"}:
                raise ValueError("physical arm belongs to a different study stage")
            if (
                cell.latency is not None
                and cell.latency["arm_envelope_sha256"] != cell.arm["envelope_sha256"]
            ):
                raise ValueError("latency references a different physical arm")
        grouped[actor["id"]].append(cell)
    actor_reports: dict[str, dict[str, Any]] = {}
    for actor in registry.actors():
        rows = grouped[actor["id"]]
        arms = [None if row.arm is None else deepcopy(dict(row.arm)) for row in rows]
        actor_reports[actor["id"]] = {
            "actor": actor,
            "cell_states": [row.state for row in rows],
            "cell_failures": [row.failure for row in rows],
            "physical_arms": arms,
            "operational": policy_scorecard(
                episode_seeds,
                [None if arm is None else arm["operational"] for arm in arms],
                cycle_count=cycle_count,
            ),
            "latency": pooled_latency(
                episode_seeds,
                [row.latency for row in rows],
                arm=actor["arm"],
                expected_host_sha256=expected_host_sha256,
            ),
        }
    baseline = actor_reports["e-beta"]["physical_arms"]
    families: dict[str, Any] = {}
    selections: dict[str, Any] = {}
    champions: dict[str, int | None] = {}
    for family in BETA_CORE_ALGORITHM_IDS:
        members = [actor for actor in registry.actors() if actor["algorithm"] == family]
        inputs: dict[int, DevelopmentPolicy] = {}
        vectors: dict[int, list[float] | None] = {}
        policies: list[dict[str, Any]] = []
        for actor in members:
            seed = actor["optimizer_seed"]
            report = actor_reports[actor["id"]]
            pairs = [
                pairwise._comparison(policy, comparator)
                for policy, comparator in zip(
                    report["physical_arms"], baseline, strict=True
                )
            ]
            trained = (
                statuses[(family, seed)].checkpoint_admitted and facts.training_verified
            )
            closed = facts.stage_verified and all(
                state != "not_started" for state in report["cell_states"]
            )
            vector = _vector(pairs, admitted=trained and closed)
            report.update(
                primary_pairs=pairs,
                primary_vector_eur=vector,
                primary_summary=None if vector is None else policy_summary(vector),
                sensitivity=policy_sensitivity(pairs, admitted=trained and closed),
            )
            policies.append(report)
            vectors[seed] = vector
            inputs[seed] = DevelopmentPolicy(
                trained,
                closed,
                tuple(
                    pair["delta_eur"] if pair["value_comparison_eligible"] else None
                    for pair in pairs
                ),
            )
        families[family] = {
            "policies": policies,
            "primary_family_summary": family_summary(vectors),
        }
        sensitivity_settings: list[dict[str, Any]] = []
        for index, name in enumerate(_weights()):
            rescored = {
                policy["actor"]["optimizer_seed"]: (
                    policy["sensitivity"]["settings"][index]["values_eur"]
                    if policy["primary_vector_eur"] is not None
                    else None
                )
                for policy in policies
            }
            sensitivity_settings.append(
                {"name": name, "summary": family_summary(rescored)}
            )
        reference = sensitivity_settings[0]["summary"]["statistics"]
        reversals = []
        if reference is not None:
            for setting in sensitivity_settings[1:]:
                for metric in VALUE_STATISTICS:
                    initial = reference[metric]
                    current = setting["summary"]["statistics"][metric]
                    if (initial > 0) - (initial < 0) != (current > 0) - (current < 0):
                        reversals.append(
                            {
                                "setting": setting["name"],
                                "statistic": metric,
                                "default": initial,
                                "changed": current,
                            }
                        )
        families[family]["sensitivity"] = {
            "settings": sensitivity_settings,
            "aggregate_sign_reversals": reversals,
            "selection_role": "none",
        }
        if stage == "development":
            selections[family] = select_family(family, inputs)
            champions[family] = selections[family]["champion_optimizer_seed"]
        else:
            assert frozen_champions is not None
            champions[family] = frozen_champions[family]
    scalar_family = BETA_CORE_ALGORITHM_IDS[2]
    for family in BETA_CORE_ALGORITHM_IDS[:2]:
        secondary = []
        vectors = {}
        for policy, scalar in zip(
            families[family]["policies"],
            families[scalar_family]["policies"],
            strict=True,
        ):
            seed = policy["actor"]["optimizer_seed"]
            if seed != scalar["actor"]["optimizer_seed"]:
                raise ValueError("secondary contrast must match optimizer identifiers")
            pairs = [
                pairwise._comparison(left, right, comparator="scalar_ppo")
                for left, right in zip(
                    policy["physical_arms"], scalar["physical_arms"], strict=True
                )
            ]
            admitted = (
                facts.training_verified
                and facts.stage_verified
                and statuses[(family, seed)].checkpoint_admitted
                and statuses[(scalar_family, seed)].checkpoint_admitted
            )
            vector = _vector(pairs, admitted=admitted)
            vectors[seed] = vector
            secondary.append(
                {
                    "optimizer_seed": seed,
                    "pairs": pairs,
                    "vector_eur": vector,
                    "summary": None if vector is None else policy_summary(vector),
                }
            )
        families[family]["secondary_vs_scalar"] = {
            "orientation": "policy minus scalar PPO; exact optimizer and episode matches",
            "policies": secondary,
            "family_summary": family_summary(vectors),
        }
    pi = BETA_CORE_ALGORITHM_IDS[0]
    primary = next(
        (
            row
            for row in families[pi]["policies"]
            if row["actor"]["optimizer_seed"] == champions[pi]
        ),
        None,
    )
    opening = None
    success = None
    if stage == "development":
        pi_selection = selections[pi]
        checks = {
            "all_five_pi_training_and_development_closed": facts.training_verified
            and facts.stage_verified
            and all(
                statuses[(pi, seed)].training_completed
                and statuses[(pi, seed)].checkpoint_admitted
                for seed in BETA_OPTIMIZER_SEEDS
            )
            and all(
                all(state != "not_started" for state in row["cell_states"])
                for row in families[pi]["policies"]
            ),
            "at_least_four_pi_admissible": pi_selection["development_admission"][
                "successes"
            ]
            >= 4,
            "champion_iqm_positive_and_cvar_nonnegative": primary is not None
            and primary["primary_summary"] is not None
            and primary["primary_summary"]["statistics"]["iqm"] > 0
            and primary["primary_summary"]["statistics"]["cvar20"] >= 0,
            "all_identities_choices_destinations_frozen": facts.identities_frozen
            and facts.registered_hardware_verified,
            "sensitivities_verified_and_published": facts.published_fields_verified,
        }
        opening = {
            "checks": checks,
            "all_calculated_requirements_met": all(checks.values()),
            "execution_authorized": False,
        }
    else:
        success = heldout_clauses(
            champion=primary, pi_policies=families[pi]["policies"], facts=facts
        )
    return authenticate(
        {
            "schema": SCHEMA,
            "analysis_contract_sha256": analysis_contract()["sha256"],
            "stage": stage,
            "episode_seeds": list(episode_seeds),
            "cell_count": len(cells),
            "state_counts": {
                state: sum(cell.state == state for cell in cells)
                for state in CELL_STATES
            },
            "declared_custody_facts": asdict(facts),
            "training": training,
            "actors": actor_reports,
            "families": families,
            "development_selections": selections if stage == "development" else None,
            "frozen_champions": champions,
            "development_opening_requirements": opening,
            "heldout_success_requirements": success,
            "source_evidence_verified": False,
            "result_claim": None,
        }
    )
