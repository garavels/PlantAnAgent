"""V7: the plant as a reinforcement-learning environment, and a policy that can drive it.

    from plant_backend.rl import PlantEnv

    env = PlantEnv()
    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

The layering is the point, and it is the same layering the rest of the backend
already has:

    spaces.py        the contract: what a policy sees, what it may say
    observation.py   telemetry -> normalised vector
    actions.py       vector -> Action (range conversion only, no safety logic)
    reward.py        what "running the plant well" means, term by term
    episodes.py      what one episode is: a sampled Scenario
    metrics.py       what a person compares two controllers with
    env.py           Gymnasium env: decide -> validate -> simulator.step
    checkpoints.py   metadata, compatibility, and four honest failures
    policies.py      loading a checkpoint into something callable
    training/        offline, no application server involved
    evaluation/      heuristic vs checkpoint, same seeds, same days

The simulator is wrapped, never duplicated, and every action still passes through
`safety.validate()` on its way to the plant. No policy has been trained here yet:
the heuristic controller remains the default and the only controller that can
drive the plant without a checkpoint being supplied on purpose.
"""

from .env import PlantEnv
from .episodes import Episode, EpisodeConfig
from .metrics import EpisodeMetrics
from .reward import RewardBreakdown, RewardWeights
from .spaces import ACTION_NAMES, ACTION_SIZE, OBSERVATION_NAMES, OBSERVATION_SIZE, describe

__all__ = [
    "ACTION_NAMES",
    "ACTION_SIZE",
    "OBSERVATION_NAMES",
    "OBSERVATION_SIZE",
    "Episode",
    "EpisodeConfig",
    "EpisodeMetrics",
    "PlantEnv",
    "RewardBreakdown",
    "RewardWeights",
    "describe",
]
