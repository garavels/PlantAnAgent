"""Matched persistence input for the unchanged competent daylight baseline."""
from __future__ import annotations

import numpy as np

from docs.research.tools.comparison_ppo_v2.baselines import BaselineConfig, BaselinePolicy, INDEX, OBSERVATION_NAMES
from docs.research.tools.free_ppo_v1.analyze import canonical_sha

FIELDS = ("solar_forecast_0_1h_mean", "solar_forecast_0_1h_p10", "solar_forecast_1_6h_mean")
INDICES = tuple(INDEX[name] for name in FIELDS)
SCHEMA = "forecast-persistence-policy-v1"


class PersistencePolicy:
    def __init__(self, config: BaselineConfig | None = None):
        self.baseline = BaselinePolicy("daylight", config=config)

    @staticmethod
    def transform(observation):
        raw = np.asarray(observation)
        if raw.dtype.kind not in "iuf" or not np.isfinite(raw).all() or raw.ndim not in (1, 2) or raw.shape[-1] != len(OBSERVATION_NAMES):
            raise ValueError("Persistence requires finite raw135 observations")
        transformed = raw.copy()
        transformed[..., list(INDICES)] = raw[..., INDEX["solar_available"], None]
        return transformed

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        return self.baseline.predict(self.transform(observation), deterministic=deterministic)

    def inference(self, observation):
        raw = np.asarray(observation)
        if raw.ndim != 1:
            raise ValueError("An inference witness records one actual observation")
        transformed = self.transform(raw)
        action, _ = self.baseline.predict(transformed, deterministic=True)
        return action, {
            "observation": raw.tolist(), "observation_sha256": canonical_sha(raw.tolist()),
            "transformed_observation": transformed.tolist(),
            "transformed_observation_sha256": canonical_sha(transformed.tolist()),
            "replacements": [{"field": name, "index": index, "original": float(raw[index]),
                              "replacement": float(transformed[index])} for name, index in zip(FIELDS, INDICES)],
            "action": action.tolist(),
        }

    def metadata(self):
        return {"schema": SCHEMA, "label": "daylight-persistence", "baseline": self.baseline.metadata(),
            "input": "Actual raw public135; transformed only within this policy",
            "replaced_fields": list(FIELDS), "replaced_indices": list(INDICES),
            "replacement_source": "solar_available", "replacement_source_index": INDEX["solar_available"],
            "forecastless_interpretation": "Current-solar persistence replaces consumed forecast means and p10 jointly; inventory and commitment rules remain",
            "fitted_or_selected_using_outcomes": False, "additional_privileged_information": False}


def verify_witness(evidence, policy):
    """Reconstruct exact transformed input and action; reject altered provenance."""
    raw = np.asarray(evidence["observation"], dtype=np.float64)
    _, expected = policy.inference(raw)
    if evidence != expected:
        raise ValueError("Persistence inference witness differs from actual transformation or action")
    return True
