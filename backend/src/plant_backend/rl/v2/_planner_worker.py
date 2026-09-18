"""Fresh-process worker for one registered feasible-planner cell."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time

from plant_sim import PlantConfig

from ...controllers.base import Constraints
from ._planner_actions import bind_relaxation_guidance
from ._planner_core import (
    build_unadmitted_failure_result,
    execute_feasible_planner,
)
from ._planner_port import PlannerPlantInputsV1
from ._planner_runtime import admit_planner_runtime
from .config import EpisodeConfigV2, sample_episode
from .contract import describe_checkpoint_contract
from .planner_schema import (
    FeasiblePlannerResultV1,
    build_planner_settings,
    canonical_json,
    planner_result_from_dict,
    raw_canonical_sha256,
)
from .reward import RewardWeightsV2
from .upper_bound import (
    DEFAULT_FEASIBILITY_TOLERANCE,
    DEFAULT_MARGINAL_SIGN_TOLERANCE,
    DEFAULT_OPTIMALITY_TOLERANCE_KG,
    DEFAULT_PHYSICAL_TOLERANCE,
    DeliveryRelaxationError,
    build_delivery_relaxation,
    solve_delivery_relaxation,
)


_WALL_LIMIT_NS = 5_400_000_000_000
_RSS_LIMIT_BYTES = 4_294_967_296


def _peak_rss_bytes() -> int:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError("peak RSS reading must be numeric")
    integer = int(raw)
    if integer < 0 or float(integer) != float(raw):
        raise ValueError("peak RSS reading must be a non-negative integer")
    return integer if platform.system() == "Darwin" else integer * 1024


def _failure_result(
    error: Exception,
    *,
    start_ns: int,
    phase: str,
) -> FeasiblePlannerResultV1:
    now = time.perf_counter_ns()
    elapsed = max(0, now - start_ns)
    try:
        rss = _peak_rss_bytes()
    except Exception:
        rss = 0
    observed: int | None = None
    if elapsed > _WALL_LIMIT_NS:
        code = "wall_time_limit"
        observed = elapsed
    elif rss > _RSS_LIMIT_BYTES:
        code = "rss_limit"
        observed = rss
    elif isinstance(error, MemoryError):
        code = "memory_error"
    elif phase == "input_validation":
        code = "invalid_input"
    elif isinstance(error, DeliveryRelaxationError):
        code = "relaxation_unavailable"
    else:
        code = "relaxation_binding_mismatch"
    return build_unadmitted_failure_result(
        evidence_scope="privileged_offline_development",
        phase=(
            "input_validation" if phase == "input_validation" else "relaxation_binding"
        ),
        code=code,
        message=error,
        elapsed_ns=elapsed,
        peak_rss_bytes=rss,
        observed_resource_value=observed,
    )


def execute_registered_cell(
    *,
    delivery_rate_kg_h: float,
    seed: int,
    start_ns: int,
) -> FeasiblePlannerResultV1:
    """Construct, solve, bind, and search one exact registered cell."""

    phase = "input_validation"
    try:
        if isinstance(delivery_rate_kg_h, bool) or delivery_rate_kg_h not in {
            35.0,
            42.0,
        }:
            raise ValueError("delivery rate is not registered")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in {0, 1, 2}:
            raise ValueError("seed is not registered")
        if isinstance(start_ns, bool) or not isinstance(start_ns, int) or start_ns < 0:
            raise ValueError("start_ns must be a non-negative integer")
        config = PlantConfig()
        episode_config = EpisodeConfigV2(
            horizon_h=72.0,
            delivery_rate_kg_h=float(delivery_rate_kg_h),
        )
        constraints = Constraints()
        weights = RewardWeightsV2()
        checkpoint = describe_checkpoint_contract(
            plant=config,
            episode=episode_config,
            constraints=constraints,
            weights=weights,
        )
        episode = sample_episode(episode_config, config, seed=seed)

        phase = "relaxation_binding"
        model = build_delivery_relaxation(
            episode=episode,
            episode_config=episode_config,
            config=config,
            constraints=constraints,
            checkpoint_contract=checkpoint,
            preserve_initial_endowment=False,
            require_canonical_episode=True,
        )
        relaxation = solve_delivery_relaxation(
            model,
            feasibility_tolerance=DEFAULT_FEASIBILITY_TOLERANCE,
            physical_tolerance=DEFAULT_PHYSICAL_TOLERANCE,
            optimality_tolerance_kg=DEFAULT_OPTIMALITY_TOLERANCE_KG,
            marginal_sign_tolerance=DEFAULT_MARGINAL_SIGN_TOLERANCE,
            solver_options={},
        )
        if relaxation.success is not True:
            raise DeliveryRelaxationError(
                "the registered delivery relaxation did not solve successfully"
            )
        binding = bind_relaxation_guidance(model, relaxation)
        settings = build_planner_settings(
            expected_decisions=episode_config.decision_steps(config),
            expected_physics_steps=episode_config.physics_steps(config),
            plant_config_sha256=raw_canonical_sha256(config.to_dict()),
            constraints_sha256=raw_canonical_sha256(constraints.to_dict()),
            reward_weights_sha256=raw_canonical_sha256(weights.to_dict()),
            episode_config_sha256=raw_canonical_sha256(episode_config.to_dict()),
            checkpoint_contract_digest=binding.checkpoint_contract_digest,
            relaxation_model_sha256=binding.model_sha256,
            relaxation_result_sha256=binding.result_sha256,
            relaxation_witness_sha256=binding.witness_sha256,
            relaxation_guidance_sha256=binding.guidance_sha256,
        )
        admission = admit_planner_runtime(
            config=config,
            episode_config=episode_config,
            constraints=constraints,
            weights=weights,
            settings=settings,
            binding=binding,
            seed=seed,
            evidence_scope="privileged_offline_development",
            canonical_episode_verified=True,
        )
        inputs = PlannerPlantInputsV1(
            config=config,
            episode_config=episode_config,
            constraints=constraints,
            weights=weights,
            admission=admission,
        )
    except Exception as exc:
        return _failure_result(exc, start_ns=start_ns, phase=phase)
    return execute_feasible_planner(inputs, start_ns=start_ns)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m plant_backend.rl.v2._planner_worker",
        description="Execute one private registered feasible-planner worker.",
    )
    parser.add_argument(
        "--delivery-rate",
        type=float,
        required=True,
        choices=(35.0, 42.0),
    )
    parser.add_argument("--seed", type=int, required=True, choices=(0, 1, 2))
    return parser


def main(argv: list[str] | None = None) -> int:
    start_ns = time.perf_counter_ns()
    arguments = build_parser().parse_args(argv)
    try:
        result = execute_registered_cell(
            delivery_rate_kg_h=arguments.delivery_rate,
            seed=arguments.seed,
            start_ns=start_ns,
        )
        serialized = canonical_json(result.to_dict())
        rebuilt = planner_result_from_dict(json.loads(serialized))
        if rebuilt.deterministic_result_sha256 != result.deterministic_result_sha256:
            raise ValueError("worker result identity changed during reconstruction")
        sys.stdout.buffer.write(serialized.encode("utf-8"))
        sys.stdout.buffer.flush()
        return 0
    except Exception as exc:
        message = " ".join(str(exc).split()) or type(exc).__name__
        print(f"planner worker protocol failure: {message[:240]}", file=sys.stderr)
        return 70


if __name__ == "__main__":  # pragma: no cover - subprocess entry point
    raise SystemExit(main())


__all__ = ["build_parser", "execute_registered_cell", "main"]
