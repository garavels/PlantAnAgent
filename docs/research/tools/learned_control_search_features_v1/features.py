"""Training-only PCA basis and its exact algebraic fold into the existing head."""
from __future__ import annotations

import numpy as np
import torch


OUTPUTS = 14
COMPONENTS = 4
DIMENSION = OUTPUTS * (COMPONENTS + 1)


def whitening_from_covariance(mean, covariance, samples):
    mean = np.asarray(mean, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if (mean.ndim != 1 or covariance.shape != (len(mean), len(mean))
            or len(mean) < COMPONENTS or samples < 2
            or not np.isfinite(mean).all() or not np.isfinite(covariance).all()):
        raise ValueError("Expected finite hidden-activation population moments")
    covariance = (covariance + covariance.T) / 2
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:COMPONENTS]
    selected, directions = eigenvalues[order], vectors[:, order]
    if np.any(selected <= 1e-12):
        raise ValueError("Four nondegenerate principal activation directions are required")
    # Eigenvectors have arbitrary signs; pin an unambiguous coordinate convention.
    for column in range(COMPONENTS):
        dominant = int(np.argmax(np.abs(directions[:, column])))
        if directions[dominant, column] < 0:
            directions[:, column] *= -1
    projection = directions / np.sqrt(selected)[None, :]
    whitened_covariance = projection.T @ covariance @ projection
    if (not np.allclose(directions.T @ directions, np.eye(COMPONENTS), rtol=0, atol=1e-10)
            or not np.allclose(whitened_covariance, np.eye(COMPONENTS), rtol=0, atol=1e-8)):
        raise ValueError("PCA orthonormality or population whitening failed")
    return {"schema": "parent-actor-hidden-pca-v1", "samples": int(samples),
        "hidden_dimension": len(mean), "components": COMPONENTS,
        "mean": mean.tolist(), "directions": directions.tolist(),
        "eigenvalues": selected.tolist(), "projection": projection.tolist(),
        "population_whitened_covariance": whitened_covariance.tolist(),
        "explained_variance_fraction": float(selected.sum() / np.trace(covariance)),
        "definition": "z=(h-mean)@projection; projection=top4_orthonormal_eigenvectors/sqrt(population_eigenvalues)",
        "sign_convention": "Each eigenvector's largest absolute component is positive; ties use first index",
        "numerical_convention": "Float64 batch-centered population covariance, denominatorN; no randomized PCA or held-out observations"}


def derive_basis(policy, observations, batch_size=2048):
    observations = np.asarray(observations)
    if observations.ndim != 2 or observations.shape[1] != 149 or len(observations) < 2:
        raise ValueError("PCA requires the original training-only149-coordinate observations")
    if not np.isfinite(observations).all() or batch_size < 1:
        raise ValueError("Observations and activation batches must be finite and nonempty")
    network = policy.model.policy
    count, mean, m2 = 0, None, None
    for first in range(0, len(observations), batch_size):
        normalized = policy.normalize_observation(observations[first:first+batch_size])
        tensor, _ = network.obs_to_tensor(normalized)
        with torch.no_grad():
            features = network.extract_features(tensor)
            actor_features = features if network.share_features_extractor else features[0]
            hidden = network.mlp_extractor.forward_actor(actor_features).detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(hidden).all():
            raise ValueError("Parent actor produced nonfinite training activations")
        batch_mean = hidden.mean(axis=0)
        centered = hidden - batch_mean
        batch_m2 = centered.T @ centered
        size = len(hidden)
        if count == 0:
            mean, m2 = batch_mean, batch_m2
        else:
            delta = batch_mean - mean
            m2 += batch_m2 + np.outer(delta, delta) * (count * size / (count + size))
            mean += delta * (size / (count + size))
        count += size
    return whitening_from_covariance(mean, m2 / count, count)


def apply_coefficients(model, parent_weight, parent_bias, parameters, scales, basis):
    coefficients = np.asarray(parameters, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    if (coefficients.shape != (DIMENSION,) or scales.shape != (OUTPUTS,)
            or not np.isfinite(coefficients).all() or not np.isfinite(scales).all()
            or np.any(scales <= 0)):
        raise ValueError("Expected finite70coefficients and14positive output scales")
    head = model.policy.action_net
    weight = torch.as_tensor(parent_weight, dtype=head.weight.dtype, device=head.weight.device)
    bias = torch.as_tensor(parent_bias, dtype=head.bias.dtype, device=head.bias.device)
    if weight.shape != head.weight.shape or bias.shape != head.bias.shape or head.weight.shape[0] != 21:
        raise ValueError("Parent head differs from the unchanged21-row hybrid action head")
    projection = np.asarray(basis["projection"], dtype=np.float64)
    mean = np.asarray(basis["mean"], dtype=np.float64)
    if projection.shape != (head.weight.shape[1], COMPONENTS) or mean.shape != (head.weight.shape[1],):
        raise ValueError("Frozen PCA basis differs from the parent hidden dimension")
    with torch.no_grad():
        # Every candidate starts at the complete parent head, never its predecessor.
        head.weight.copy_(weight)
        head.bias.copy_(bias)
        if not np.any(coefficients):
            return
        theta = coefficients.reshape(OUTPUTS, COMPONENTS + 1)
        delta_weight = scales[:, None] * (theta[:, 1:] @ projection.T)
        delta_bias = scales * theta[:, 0] - delta_weight @ mean
        head.weight[:OUTPUTS].add_(torch.as_tensor(delta_weight, dtype=head.weight.dtype, device=head.weight.device))
        head.bias[:OUTPUTS].add_(torch.as_tensor(delta_bias, dtype=head.bias.dtype, device=head.bias.device))


def validate_fold(policy, observations, parent_weight, parent_bias, scales, basis, batch_size):
    """Check a fixed nonzero coefficient pattern on every training activation.

    This engineering probe restores the exact parent and is never an evaluated
    search candidate. It records actual float32-fold error against float64 math.
    """
    parameters = np.sin(np.arange(1, DIMENSION+1, dtype=np.float64)) * .03
    theta = parameters.reshape(OUTPUTS, COMPONENTS+1)
    projection, mean = np.asarray(basis["projection"]), np.asarray(basis["mean"])
    weight, bias, scales = np.asarray(parent_weight), np.asarray(parent_bias), np.asarray(scales)
    network = policy.model.policy
    maximum_absolute, maximum_scaled, count = 0., 0., 0
    try:
        apply_coefficients(policy.model, parent_weight, parent_bias, parameters, scales, basis)
        for first in range(0, len(observations), batch_size):
            tensor, _ = network.obs_to_tensor(policy.normalize_observation(observations[first:first+batch_size]))
            with torch.no_grad():
                features = network.extract_features(tensor)
                actor_features = features if network.share_features_extractor else features[0]
                hidden_tensor = network.mlp_extractor.forward_actor(actor_features)
                actual = network.action_net(hidden_tensor).detach().cpu().numpy().astype(np.float64)
            hidden = hidden_tensor.detach().cpu().numpy().astype(np.float64)
            expected = hidden @ weight.T + bias
            z = (hidden - mean) @ projection
            expected[:, :OUTPUTS] += scales * (theta[:, 0] + z @ theta[:, 1:].T)
            error = np.abs(actual - expected)
            maximum_absolute = max(maximum_absolute, float(error.max()))
            maximum_scaled = max(maximum_scaled, float((error / np.maximum(1., np.abs(expected))).max()))
            count += len(hidden)
        if maximum_scaled > 1e-5:
            raise ValueError("Folded float32 head differs from the declared PCA affine map")
        return {"training_rows_checked": count, "max_absolute_output_error": maximum_absolute,
            "max_error_divided_by_max_one_abs_expected": maximum_scaled, "scaled_error_tolerance": 1e-5,
            "probe_coefficients": "0.03*sin(1..70), fixed engineering probe only",
            "zero_coefficients_restore_parent_exactly": True}
    finally:
        apply_coefficients(policy.model, parent_weight, parent_bias, np.zeros(DIMENSION), scales, basis)


__all__ = ["OUTPUTS", "COMPONENTS", "DIMENSION", "derive_basis", "whitening_from_covariance", "apply_coefficients", "validate_fold"]
