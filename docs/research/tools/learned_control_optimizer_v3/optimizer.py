"""Diagonal preconditioning of actual Adam parameter changes, not gradients."""
from __future__ import annotations

import torch


class UpdateScaledAdam(torch.optim.Adam):
    """Run ordinary Adam moments, then scale each proposed parameter increment.

    Scales are reconstructed by the policy constructor and are not optimizer
    moments. Parameters without a gradient are untouched, including their Adam
    counters. Unit scales follow the exact ordinary-Adam parameter write path.
    """

    def __init__(self, params, *, update_scales, update_labels=None, **kwargs):
        if kwargs.get("differentiable", False) or kwargs.get("capturable", False):
            raise ValueError("This CPU optimizer does not support differentiable or capturable steps")
        super().__init__(params, **kwargs)
        parameters = [p for group in self.param_groups for p in group["params"]]
        if len(parameters) != len(update_scales):
            raise ValueError("Exactly one update scale per optimizer parameter is required")
        self._update_scales = {}
        labels = ["scaled_parameter"] * len(parameters) if update_labels is None else update_labels
        if len(labels) != len(parameters):
            raise ValueError("Exactly one diagnostic label per parameter is required")
        self._update_labels = {}
        self.last_update_diagnostics = {}
        for parameter, value, label in zip(parameters, update_scales, labels):
            scale = torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device)
            if not torch.isfinite(scale).all() or not torch.all(scale > 0):
                raise ValueError("Update preconditioners must be finite and strictly positive")
            if torch.broadcast_shapes(parameter.shape, scale.shape) != parameter.shape:
                raise ValueError("Update scale must broadcast exactly to its parameter shape")
            if not torch.all(scale == 1):
                self._update_scales[parameter] = scale.detach().clone()
                self._update_labels[parameter] = label

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        previous = {parameter: parameter.detach().clone()
                    for parameter in self._update_scales if parameter.grad is not None}
        super().step()
        self.last_update_diagnostics = {}
        for parameter, before in previous.items():
            scale = self._update_scales[parameter].to(parameter)
            # Scale the already computed Adam proposal, including epsilon and
            # bias correction. The moment state itself remains ordinary Adam.
            proposed = parameter - before
            mask = torch.broadcast_to(scale != 1, parameter.shape)
            parameter.copy_(torch.where(mask, before + scale * proposed, parameter))
            actual = parameter - before
            proposed, actual = proposed[mask], actual[mask]
            record = self.last_update_diagnostics.setdefault(self._update_labels[parameter], {
                "scaled_elements": 0, "proposed_nonzero_elements": 0,
                "rounded_to_zero_elements": 0, "actual_changed_elements": 0,
                "proposed_squared_displacement": 0., "actual_squared_displacement": 0.})
            record["scaled_elements"] += proposed.numel()
            record["proposed_nonzero_elements"] += int(torch.count_nonzero(proposed))
            record["rounded_to_zero_elements"] += int(torch.count_nonzero((proposed != 0) & (actual == 0)))
            record["actual_changed_elements"] += int(torch.count_nonzero(actual))
            record["proposed_squared_displacement"] += float(proposed.double().square().sum())
            record["actual_squared_displacement"] += float(actual.double().square().sum())
        return loss


__all__ = ["UpdateScaledAdam"]
