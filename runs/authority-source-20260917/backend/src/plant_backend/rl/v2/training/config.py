"""Strict, serializable training contract for the four-target V2 tasks."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, TypeVar, cast

from plant_sim import (
    BatteryConfig,
    DACConfig,
    ElectrolyserCommitmentProfile,
    ElectrolyserConfig,
    MethanatorConfig,
    PlantConfig,
    SolarConfig,
    StorageConfig,
)

from ....controllers.base import Constraints
from ...training.config import PPOConfig
from .. import beta_spaces, economic_spaces, setpoint_spaces
from ..beta_config import (
    CyclicMarketTaskConfigV1,
    CyclicReturnConfigV1,
)
from ..beta_disturbances import DisturbanceConfigV1
from ..config import EpisodeConfigV2
from ..economics import DayAheadTariffV1, EconomicAssumptionsV1
from ..market import MarketConfigV1, MarketRegimeV1
from ..reward import RewardWeightsV2
from ..weather import WeatherConfigV2
from .custody_json import canonical_bytes, decode


SUPPORTED_CONTROLLER_NEUTRAL_ENVIRONMENTS = (
    economic_spaces.ENVIRONMENT_ID,
    setpoint_spaces.ENVIRONMENT_ID,
    beta_spaces.ENVIRONMENT_ID,
)

BETA_CONSTRAINED_RL_PROTOCOL_ID = "plant-beta-constrained-rl-protocol-v1"
BETA_PI_LAGRANGIAN_ALGORITHM_ID = "plant-beta-ppo-pi-lagrangian-v1"
BETA_I_LAGRANGIAN_ALGORITHM_ID = "plant-beta-ppo-i-lagrangian-v1"
BETA_SCALAR_PPO_ALGORITHM_ID = "plant-beta-sb3-ppo-scalar-v1"
BETA_CORE_ALGORITHM_IDS = (
    BETA_PI_LAGRANGIAN_ALGORITHM_ID,
    BETA_I_LAGRANGIAN_ALGORITHM_ID,
    BETA_SCALAR_PPO_ALGORITHM_ID,
)
BETA_FIXTURE_RUN_KIND = "fixture"
BETA_REGISTERED_RUN_KIND = "registered"
BETA_RUN_KINDS = (BETA_FIXTURE_RUN_KIND, BETA_REGISTERED_RUN_KIND)
BETA_OPTIMIZER_SEEDS = (101, 211, 307, 401, 503)
BETA_REGISTERED_EPISODE_SEED_BASE = 1_000_000
BETA_FIXTURE_EPISODE_SEED_MIN = 9_000_000
BETA_FIXTURE_EPISODE_SEED_MAX = 9_999_999
BETA_REGISTERED_TOTAL_TIMESTEPS = 1_048_576
BETA_REGISTERED_ENV_COUNT = 4
BETA_FIXTURE_MAX_COLLECTED_TIMESTEPS = 8_192


def _object(name: str, value: object | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object")
    return dict(value)


def _reject_unknown(name: str, raw: Mapping[str, Any], known: set[str]) -> None:
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown {name}: {', '.join(sorted(unknown))}")


def _typed_scalar(name: str, value: object, template: object) -> object:
    """Normalize JSON scalars so equal settings have one canonical task identity."""

    if isinstance(template, bool):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")
        return value
    if isinstance(template, int):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        return int(value)
    if isinstance(template, float):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        return number
    if isinstance(template, str):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        return value
    return value


def _tuple2(name: str, value: object) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{name} must be a two-number array")
    return (
        cast(float, _typed_scalar(f"{name}[0]", value[0], 0.0)),
        cast(float, _typed_scalar(f"{name}[1]", value[1], 0.0)),
    )


_Dataclass = TypeVar("_Dataclass")


def _flat_dataclass(base: _Dataclass, raw: object | None, *, name: str) -> _Dataclass:
    values = _object(name, raw)
    known = {item.name for item in fields(cast(Any, base))}
    _reject_unknown(name, values, known)
    normalized = {
        key: _typed_scalar(f"{name}.{key}", value, getattr(base, key))
        for key, value in values.items()
    }
    return cast(_Dataclass, replace(cast(Any, base), **normalized))


def _plant_from_dict(raw: object | None) -> PlantConfig:
    values = _object("plant settings", raw)
    known = {item.name for item in fields(PlantConfig)}
    _reject_unknown("plant settings", values, known)
    base = PlantConfig()
    nested = {
        "solar": SolarConfig,
        "battery": BatteryConfig,
        "electrolyser": ElectrolyserConfig,
        "h2_tank": StorageConfig,
        "dac": DACConfig,
        "co2_tank": StorageConfig,
        "methanator": MethanatorConfig,
        "ch4_tank": StorageConfig,
    }
    updates: dict[str, Any] = {}
    for key, value in values.items():
        if key == "dt_s":
            updates[key] = _typed_scalar("plant.dt_s", value, base.dt_s)
            continue
        component_type = nested[key]
        updates[key] = _flat_dataclass(
            getattr(base, key),
            value,
            name=f"plant.{key} settings",
        )
        if not isinstance(updates[key], component_type):  # pragma: no cover - defensive
            raise TypeError(f"plant.{key} settings produced the wrong component type")
    return replace(base, **updates)


def _weather_from_dict(raw: object | None) -> WeatherConfigV2:
    values = _object("weather settings", raw)
    known = {item.name for item in fields(WeatherConfigV2)}
    _reject_unknown("weather settings", values, known)
    if "forecast_windows_h" in values:
        windows = values["forecast_windows_h"]
        if not isinstance(windows, (list, tuple)):
            raise TypeError("forecast_windows_h must be an array")
        values["forecast_windows_h"] = tuple(
            _tuple2(f"forecast_windows_h[{index}]", window)
            for index, window in enumerate(windows)
        )
    return _flat_dataclass(
        WeatherConfigV2(), values, name="weather settings"
    ).validate()


def _commitment_from_dict(raw: object) -> ElectrolyserCommitmentProfile | None:
    if raw is None:
        return None
    values = _object("electrolyser commitment settings", raw)
    known = {item.name for item in fields(ElectrolyserCommitmentProfile)}
    _reject_unknown("electrolyser commitment settings", values, known)
    if "specific_energy_multiplier_anchors" in values:
        anchors = values["specific_energy_multiplier_anchors"]
        if not isinstance(anchors, (list, tuple)):
            raise TypeError("specific_energy_multiplier_anchors must be an array")
        values["specific_energy_multiplier_anchors"] = tuple(
            _tuple2(f"specific_energy_multiplier_anchors[{index}]", anchor)
            for index, anchor in enumerate(anchors)
        )
    return _flat_dataclass(
        ElectrolyserCommitmentProfile(),
        values,
        name="electrolyser commitment settings",
    )


def _episode_from_dict(raw: object | None) -> EpisodeConfigV2:
    values = _object("episode settings", raw)
    known = {item.name for item in fields(EpisodeConfigV2)}
    _reject_unknown("episode settings", values, known)
    for key in ("battery_soc", "h2_fill", "co2_fill", "ch4_coverage_h"):
        if key in values:
            values[key] = _tuple2(f"episode.{key}", values[key])
    if "weather" in values:
        values["weather"] = _weather_from_dict(values["weather"])
    if "electrolyser_commitment" in values:
        values["electrolyser_commitment"] = _commitment_from_dict(
            values["electrolyser_commitment"]
        )
    return _flat_dataclass(EpisodeConfigV2(), values, name="episode settings")


def _constraints_from_dict(raw: object | None) -> Constraints:
    values = _object("constraint settings", raw)
    _reject_unknown(
        "constraint settings",
        values,
        {item.name for item in fields(Constraints)},
    )
    normalized: dict[str, float | None] = {}
    for key, value in values.items():
        optional = key != "battery_min_soc"
        if value is None:
            if not optional:
                raise TypeError(
                    "constraint settings.battery_min_soc must be a real number"
                )
            normalized[key] = None
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            suffix = " or null" if optional else ""
            raise TypeError(f"constraint settings.{key} must be a real number{suffix}")
        try:
            number = float(value)
        except OverflowError as reason:
            raise ValueError(f"constraint settings.{key} must be finite") from reason
        if not math.isfinite(number):
            raise ValueError(f"constraint settings.{key} must be finite")
        normalized[key] = number
    # Keep the runtime envelope's existing range checks and merge semantics. The
    # training boundary is stricter only about which JSON values may reach it.
    return Constraints().merged(normalized)


def _reward_from_dict(raw: object | None) -> RewardWeightsV2:
    result = _flat_dataclass(
        RewardWeightsV2(),
        raw,
        name="V2 reward settings",
    )
    if not isinstance(result, RewardWeightsV2):  # pragma: no cover - defensive
        raise TypeError("V2 reward settings produced the wrong type")
    return result.validate()


def _tariff_from_dict(raw: object | None) -> DayAheadTariffV1:
    values = _object("tariff settings", raw)
    _reject_unknown("tariff settings", values, {"id", "hourly_eur_per_mwh"})
    if "hourly_eur_per_mwh" in values:
        prices = values["hourly_eur_per_mwh"]
        if not isinstance(prices, (list, tuple)):
            raise TypeError("hourly_eur_per_mwh must be an array")
        values["hourly_eur_per_mwh"] = tuple(
            cast(
                float,
                _typed_scalar(f"hourly_eur_per_mwh[{index}]", value, 0.0),
            )
            for index, value in enumerate(prices)
        )
    return _flat_dataclass(
        DayAheadTariffV1(), values, name="tariff settings"
    ).validate()


def _assumptions_from_dict(raw: object | None) -> EconomicAssumptionsV1:
    values = _object("economic assumption settings", raw)
    known = {item.name for item in fields(EconomicAssumptionsV1)}
    _reject_unknown("economic assumption settings", values, known)
    if "tariff" in values:
        values["tariff"] = _tariff_from_dict(values["tariff"])
    return _flat_dataclass(
        EconomicAssumptionsV1(),
        values,
        name="economic assumption settings",
    ).validate()


def _ppo_from_dict(raw: object | None) -> PPOConfig:
    result = _flat_dataclass(PPOConfig(), raw, name="PPO settings")
    if not isinstance(result, PPOConfig):  # pragma: no cover - defensive
        raise TypeError("PPO settings produced the wrong type")
    return result


def _cycle_from_dict(raw: object | None) -> CyclicReturnConfigV1:
    result = _flat_dataclass(
        CyclicReturnConfigV1(),
        raw,
        name="cyclic return settings",
    )
    if not isinstance(result, CyclicReturnConfigV1):  # pragma: no cover
        raise TypeError("cyclic return settings produced the wrong type")
    return result


def _market_from_dict(raw: object | None) -> MarketConfigV1:
    values = _object("market settings", raw)
    known = {item.name for item in fields(MarketConfigV1)}
    _reject_unknown("market settings", values, known)
    updates: dict[str, Any] = {}
    for key, value in values.items():
        if key == "regime":
            if not isinstance(value, str):
                raise TypeError("market settings.regime must be a string")
            try:
                updates[key] = MarketRegimeV1(value)
            except ValueError as reason:
                choices = ", ".join(item.value for item in MarketRegimeV1)
                raise ValueError(
                    f"market settings.regime must be one of: {choices}"
                ) from reason
        else:
            updates[key] = _typed_scalar(
                f"market settings.{key}",
                value,
                getattr(MarketConfigV1(), key),
            )
    return replace(MarketConfigV1(), **updates).validate()


def _disturbances_from_dict(raw: object | None) -> DisturbanceConfigV1:
    values = _object("disturbance settings", raw)
    known = {item.name for item in fields(DisturbanceConfigV1)}
    _reject_unknown("disturbance settings", values, known)
    if "duration_h" in values:
        values["duration_h"] = _tuple2(
            "disturbance settings.duration_h", values["duration_h"]
        )
    result = _flat_dataclass(
        DisturbanceConfigV1(),
        values,
        name="disturbance settings",
    )
    if not isinstance(result, DisturbanceConfigV1):  # pragma: no cover
        raise TypeError("disturbance settings produced the wrong type")
    return result.validate()


def _beta_task_from_dict(raw: object | None) -> CyclicMarketTaskConfigV1:
    values = _object("cyclic-market beta settings", raw)
    _reject_unknown(
        "cyclic-market beta settings",
        values,
        {"cycle", "market", "disturbances"},
    )
    return CyclicMarketTaskConfigV1(
        cycle=_cycle_from_dict(values.get("cycle")),
        market=_market_from_dict(values.get("market")),
        disturbances=_disturbances_from_dict(values.get("disturbances")),
    )


def _validate_ppo(config: PPOConfig, *, n_envs: int) -> None:
    for name in ("policy", "device"):
        value = getattr(config, name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"PPO {name} must be a non-empty string")
    for name in ("n_steps", "batch_size", "n_epochs"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"PPO {name} must be an integer")
        if value <= 0:
            raise ValueError(f"PPO {name} must be positive")
    numeric_ranges = {
        "learning_rate": (0.0, None, False),
        "gamma": (0.0, 1.0, True),
        "gae_lambda": (0.0, 1.0, True),
        "clip_range": (0.0, None, False),
        "ent_coef": (0.0, None, True),
        "vf_coef": (0.0, None, True),
        "max_grad_norm": (0.0, None, False),
    }
    for name, (minimum, maximum, inclusive_minimum) in numeric_ranges.items():
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"PPO {name} must be a real number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"PPO {name} must be finite")
        below = number < minimum if inclusive_minimum else number <= minimum
        if below or (maximum is not None and number > maximum):
            if maximum is not None:
                left = "[" if inclusive_minimum else "("
                interval = f"{left}{minimum:g}, {maximum:g}]"
            else:
                operator = ">=" if inclusive_minimum else ">"
                interval = f"{operator} {minimum:g}"
            raise ValueError(f"PPO {name} must be in {interval}")
    rollout_size = int(config.n_steps) * n_envs
    if rollout_size <= 1:
        raise ValueError("PPO rollout must contain at least two transitions")
    if config.batch_size > rollout_size:
        raise ValueError("PPO batch_size cannot exceed n_steps * n_envs")
    if rollout_size % config.batch_size:
        raise ValueError("PPO batch_size must divide n_steps * n_envs exactly")


def _assumptions_to_dict(assumptions: EconomicAssumptionsV1) -> dict[str, Any]:
    return {
        "tariff": {
            "id": assumptions.tariff.id,
            "hourly_eur_per_mwh": list(assumptions.tariff.hourly_eur_per_mwh),
        },
        **{
            item.name: getattr(assumptions, item.name)
            for item in fields(EconomicAssumptionsV1)
            if item.name != "tariff"
        },
    }


def _beta_task_to_training_dict(
    task: CyclicMarketTaskConfigV1,
) -> dict[str, Any]:
    return {
        "cycle": asdict(task.cycle),
        "market": {
            "regime": task.market.regime.value,
            "forecast_ensemble_members": task.market.forecast_ensemble_members,
            "forecast_innovation_scale": task.market.forecast_innovation_scale,
        },
        "disturbances": task.disturbances.to_dict(),
    }


@dataclass(frozen=True, slots=True)
class BetaRLTrainingRunV1:
    """Outcome-custody selector for one beta constrained-RL run.

    Historical controller-neutral PPO configurations omit this record and keep
    their existing seed behavior.  Its presence opts a beta run into either the
    bounded fixture gate or the exact registered campaign.
    """

    protocol_id: str = BETA_CONSTRAINED_RL_PROTOCOL_ID
    algorithm: str = BETA_PI_LAGRANGIAN_ALGORITHM_ID
    run_kind: str = BETA_FIXTURE_RUN_KIND
    episode_seed_base: int = BETA_FIXTURE_EPISODE_SEED_MIN

    def validate(self) -> "BetaRLTrainingRunV1":
        if self.protocol_id != BETA_CONSTRAINED_RL_PROTOCOL_ID:
            raise ValueError(
                "beta RL protocol identifier differs from the frozen protocol"
            )
        if self.algorithm not in BETA_CORE_ALGORITHM_IDS:
            raise ValueError(
                "beta RL algorithm is not one of the three frozen core families"
            )
        if self.run_kind not in BETA_RUN_KINDS:
            raise ValueError("beta RL run_kind must be fixture or registered")
        if isinstance(self.episode_seed_base, bool) or not isinstance(
            self.episode_seed_base, int
        ):
            raise TypeError("beta RL episode_seed_base must be an integer")
        if self.run_kind == BETA_REGISTERED_RUN_KIND:
            if self.episode_seed_base != BETA_REGISTERED_EPISODE_SEED_BASE:
                raise ValueError(
                    "registered beta RL must start at episode seed 1,000,000"
                )
        elif not (
            BETA_FIXTURE_EPISODE_SEED_MIN
            <= self.episode_seed_base
            <= BETA_FIXTURE_EPISODE_SEED_MAX
        ):
            raise ValueError(
                "fixture beta RL episode seeds must remain in 9,000,000..9,999,999"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "algorithm": self.algorithm,
            "run_kind": self.run_kind,
            "episode_seed_base": self.episode_seed_base,
        }

    @classmethod
    def from_dict(cls, raw: object | None) -> "BetaRLTrainingRunV1":
        values = _object("beta RL training run", raw)
        _reject_unknown(
            "beta RL training run",
            values,
            {item.name for item in fields(cls)},
        )
        return replace(cls(), **values).validate()


@dataclass(frozen=True)
class ControllerNeutralTrainingConfigV1:
    """One reproducible PPO run against a four-target Environment V2 task."""

    environment_id: str = economic_spaces.ENVIRONMENT_ID
    seed: int = 0
    total_timesteps: int = 1_000_000
    n_envs: int = 4
    checkpoint_every: int = 100_000
    output_dir: str = "runs/v2-economic-ppo"
    resume_from: str | None = None
    plant: PlantConfig = field(default_factory=PlantConfig)
    episode: EpisodeConfigV2 = field(default_factory=EpisodeConfigV2)
    constraints: Constraints = field(default_factory=Constraints)
    diagnostic_reward: RewardWeightsV2 = field(default_factory=RewardWeightsV2)
    economic_assumptions: EconomicAssumptionsV1 = field(
        default_factory=EconomicAssumptionsV1
    )
    beta_task: CyclicMarketTaskConfigV1 | None = None
    beta_rl: BetaRLTrainingRunV1 | None = None
    ppo: PPOConfig = field(default_factory=PPOConfig)
    notes: str = ""

    def validate(self) -> "ControllerNeutralTrainingConfigV1":
        if not isinstance(self.environment_id, str):
            raise TypeError("environment_id must be a string")
        if self.environment_id not in SUPPORTED_CONTROLLER_NEUTRAL_ENVIRONMENTS:
            raise ValueError(
                "environment_id must identify a supported controller-neutral V2 task"
            )
        for name in ("seed", "total_timesteps", "n_envs", "checkpoint_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.total_timesteps <= 0 or self.n_envs <= 0:
            raise ValueError("total_timesteps and n_envs must be positive")
        if self.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be non-negative")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty")
        if self.resume_from is not None and (
            not isinstance(self.resume_from, str) or not self.resume_from.strip()
        ):
            raise ValueError("resume_from must be null or a non-empty path")
        if not isinstance(self.notes, str):
            raise TypeError("notes must be a string")
        for name, value, expected in (
            ("plant", self.plant, PlantConfig),
            ("episode", self.episode, EpisodeConfigV2),
            ("constraints", self.constraints, Constraints),
            ("diagnostic_reward", self.diagnostic_reward, RewardWeightsV2),
            (
                "economic_assumptions",
                self.economic_assumptions,
                EconomicAssumptionsV1,
            ),
            ("ppo", self.ppo, PPOConfig),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        self.episode.validate(self.plant)
        _constraints_from_dict(self.constraints.to_dict())
        self.diagnostic_reward.validate()
        self.economic_assumptions.validate()
        if self.environment_id == beta_spaces.ENVIRONMENT_ID:
            if not isinstance(self.beta_task, CyclicMarketTaskConfigV1):
                raise TypeError(
                    "beta_task must be CyclicMarketTaskConfigV1 for the beta environment"
                )
            self.beta_task.validate(
                episode=self.episode,
                plant=self.plant,
                assumptions=self.economic_assumptions,
            )
        elif self.beta_task is not None:
            raise ValueError("beta_task is only valid for the cyclic-market beta")
        if self.beta_rl is not None:
            if not isinstance(self.beta_rl, BetaRLTrainingRunV1):
                raise TypeError("beta_rl must be BetaRLTrainingRunV1 or None")
            self.beta_rl.validate()
            if self.environment_id != beta_spaces.ENVIRONMENT_ID:
                raise ValueError("beta_rl is only valid for the cyclic-market beta")
        _validate_ppo(self.ppo, n_envs=self.n_envs)
        if self.beta_rl is not None:
            self._validate_beta_rl_run()
        return self

    def _validate_beta_rl_run(self) -> None:
        if self.beta_rl is None:  # pragma: no cover - private caller guards this
            return
        if self.n_envs != BETA_REGISTERED_ENV_COUNT:
            raise ValueError(
                "beta RL custody requires exactly four vector environments"
            )
        if self.seed not in BETA_OPTIMIZER_SEEDS:
            raise ValueError(
                "beta RL optimizer seed is not in the frozen five-seed set"
            )
        if self.checkpoint_every != 0:
            raise ValueError("beta RL permits only a final eligible checkpoint")

        rollout_size = self.ppo.n_steps * self.n_envs
        collected_timesteps = (
            math.ceil(self.total_timesteps / rollout_size) * rollout_size
        )
        decisions_per_environment = collected_timesteps // self.n_envs
        episode_decisions = self.episode.decision_steps(self.plant)
        reset_count_per_environment = decisions_per_environment // episode_decisions + 1
        final_episode_seed = (
            self.beta_rl.episode_seed_base
            + self.n_envs * (reset_count_per_environment - 1)
            + self.n_envs
            - 1
        )
        output = Path(self.output_dir).expanduser()
        resolved_output = output.resolve()
        repository = Path(__file__).resolve().parents[6]
        repository_run_root = (repository / "runs").resolve()

        if self.beta_rl.run_kind == BETA_REGISTERED_RUN_KIND:
            if self.total_timesteps != BETA_REGISTERED_TOTAL_TIMESTEPS:
                raise ValueError(
                    "registered beta RL timestep budget must be exactly 1,048,576"
                )
            if self.resume_from is not None:
                raise ValueError("registered beta RL forbids resume_from")
            if self.ppo != PPOConfig():
                raise ValueError(
                    "registered beta RL PPO settings differ from the frozen defaults"
                )
            if episode_decisions != 672:
                raise ValueError(
                    "registered beta RL requires the frozen 672-decision episode"
                )
            if final_episode_seed != 1_001_563:
                raise ValueError(
                    "registered beta RL allocator endpoint differs from the protocol"
                )
            if not output.is_absolute():
                raise ValueError("registered beta RL output_dir must be absolute")
            if resolved_output.is_relative_to(repository):
                raise ValueError(
                    "registered beta RL output_dir must be external to the repository"
                )
            from .registered_contract import require_task_template

            require_task_template(self)
        else:
            if collected_timesteps > BETA_FIXTURE_MAX_COLLECTED_TIMESTEPS:
                raise ValueError(
                    "fixture beta RL is bounded to at most 8,192 collected decisions"
                )
            if final_episode_seed > BETA_FIXTURE_EPISODE_SEED_MAX:
                raise ValueError(
                    "fixture beta RL allocator would leave the fixture seed domain"
                )
            if resolved_output.is_relative_to(
                repository
            ) and not resolved_output.is_relative_to(repository_run_root):
                raise ValueError(
                    "fixture beta RL output_dir inside the repository must be under "
                    "the ignored runs/ artifact root"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "seed": self.seed,
            "total_timesteps": self.total_timesteps,
            "n_envs": self.n_envs,
            "checkpoint_every": self.checkpoint_every,
            "output_dir": self.output_dir,
            "resume_from": self.resume_from,
            "plant": self.plant.to_dict(),
            "episode": self.episode.to_dict(),
            "constraints": self.constraints.to_dict(),
            "diagnostic_reward": self.diagnostic_reward.to_dict(),
            "economic_assumptions": _assumptions_to_dict(self.economic_assumptions),
            "beta_task": (
                None
                if self.beta_task is None
                else _beta_task_to_training_dict(self.beta_task)
            ),
            "beta_rl": None if self.beta_rl is None else self.beta_rl.to_dict(),
            "ppo": asdict(self.ppo),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any] | None,
    ) -> "ControllerNeutralTrainingConfigV1":
        values = _object("controller-neutral training settings", raw)
        _reject_unknown(
            "controller-neutral training settings",
            values,
            {item.name for item in fields(cls)},
        )
        parsers = {
            "plant": _plant_from_dict,
            "episode": _episode_from_dict,
            "constraints": _constraints_from_dict,
            "diagnostic_reward": _reward_from_dict,
            "economic_assumptions": _assumptions_from_dict,
            "beta_task": (
                lambda value: None if value is None else _beta_task_from_dict(value)
            ),
            "beta_rl": (
                lambda value: (
                    None if value is None else BetaRLTrainingRunV1.from_dict(value)
                )
            ),
            "ppo": _ppo_from_dict,
        }
        updates: dict[str, Any] = {}
        for key, value in values.items():
            parser = parsers.get(key)
            updates[key] = parser(value) if parser is not None else value
        return replace(cls(), **updates).validate()

    @classmethod
    def from_file(cls, path: Path | str) -> "ControllerNeutralTrainingConfigV1":
        target = Path(path)
        if not target.is_file():
            raise ValueError(f"no controller-neutral training config at {target}")
        from .registered_contract import is_registered_config_path

        campaign_path = is_registered_config_path(target)
        try:
            encoded = target.read_bytes()
            raw = decode(encoded)
        except (OSError, json.JSONDecodeError) as reason:
            raise ValueError(f"cannot read {target}: {reason}") from reason
        config = cls.from_dict(raw)
        # A campaign config cannot downgrade itself by deleting beta_rl. This
        # classification uses reserved paths, independently of config data.
        if campaign_path or (
            config.beta_rl is not None
            and config.beta_rl.run_kind == BETA_REGISTERED_RUN_KIND
        ):
            from .custody_io import read_regular
            from .registered_contract import require_canonical_config

            if read_regular(target.absolute()) != encoded:
                raise ValueError("registered configuration changed during admission")
            require_canonical_config(encoded, config)
        return config

    def write(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_bytes(self.to_dict()))
        return target


__all__ = [
    "BETA_CONSTRAINED_RL_PROTOCOL_ID",
    "BETA_CORE_ALGORITHM_IDS",
    "BETA_FIXTURE_RUN_KIND",
    "BETA_I_LAGRANGIAN_ALGORITHM_ID",
    "BETA_OPTIMIZER_SEEDS",
    "BETA_PI_LAGRANGIAN_ALGORITHM_ID",
    "BETA_REGISTERED_RUN_KIND",
    "BETA_SCALAR_PPO_ALGORITHM_ID",
    "BetaRLTrainingRunV1",
    "SUPPORTED_CONTROLLER_NEUTRAL_ENVIRONMENTS",
    "ControllerNeutralTrainingConfigV1",
]
