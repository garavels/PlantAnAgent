"""V9: the HEPA network itself -- shapes, causality, horizon conditioning, hazards.

Everything here runs on the CPU on a model two orders of magnitude smaller than the
paper's: what is under test is the architecture's contracts, not its accuracy. The
whole file is skipped when the optional model dependencies are absent, which is why
the *meaning* of the inputs and outputs is tested in `test_hepa_contracts.py`
instead.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="HEPA models need `uv sync --group hepa`")

from plant_backend.hepa import telemetry  # noqa: E402
from plant_backend.hepa.models import Hepa, HepaConfig  # noqa: E402
from plant_backend.hepa.models.encoder import instance_normalise  # noqa: E402
from plant_backend.hepa.models.heads import anchor_coverage, anchor_steps  # noqa: E402
from plant_backend.hepa.pretraining import objective as stage1  # noqa: E402
from plant_backend.hepa.finetuning import objective as stage2  # noqa: E402

#: Small enough to train a few steps in a test, shaped like the real thing:
#: patchable context, horizon range a whole number of patches.
TINY = HepaConfig(
    d_model=32,
    num_layers=1,
    num_heads=2,
    patch_size=4,
    context_len=16,
    max_horizon_steps=32,
)


@pytest.fixture
def model() -> Hepa:
    torch.manual_seed(0)
    return Hepa(TINY)


def _batch(size: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    context = torch.randn(
        size, TINY.context_len, telemetry.FEATURE_COUNT, generator=generator
    )
    future = torch.randn(
        size, TINY.max_horizon_steps, telemetry.FEATURE_COUNT, generator=generator
    )
    return context, future


# --- configuration --------------------------------------------------------
def test_the_config_round_trips_and_defaults_to_the_papers_shape() -> None:
    default = HepaConfig()
    assert default.d_model == 256
    assert default.num_layers == 2
    assert default.num_heads == 4
    assert default.patch_size == 16
    assert default.input_dim == telemetry.FEATURE_COUNT
    assert HepaConfig.from_dict(TINY.to_dict()) == TINY
    with pytest.raises(ValueError):
        HepaConfig.from_dict({"unknown_setting": 3})


def test_a_context_that_cannot_be_patched_is_refused() -> None:
    with pytest.raises(ValueError):
        Hepa(HepaConfig(context_len=17, patch_size=4, d_model=32, num_heads=2, num_layers=1))


# --- encoder --------------------------------------------------------------
def test_the_encoder_produces_one_latent_per_window(model: Hepa) -> None:
    context, _ = _batch()
    latent = model.encode(context)
    assert latent.shape == (4, TINY.d_model)
    assert torch.all(torch.isfinite(latent))


def test_instance_normalisation_is_per_window_and_per_channel() -> None:
    values = torch.randn(3, 16, 5) * 4.0 + 7.0
    normalised = instance_normalise(values)
    assert torch.allclose(normalised.mean(dim=1), torch.zeros(3, 5), atol=1e-5)
    assert torch.allclose(normalised.std(dim=1, unbiased=False), torch.ones(3, 5), atol=1e-3)


def test_the_encoder_attends_only_backwards(model: Hepa) -> None:
    """A later token must not change the representation of an earlier one.

    Tested on the transformer stack rather than on raw samples, because the paper's
    per-context instance normalisation is deliberately window-wide: it rescales the
    whole context with the context's own statistics, so *every* token moves when a
    sample changes. That is not leakage -- the window ends at t -- but it means
    causality lives in the attention mask, which is what this exercises.
    """
    model.eval()
    tokens = torch.randn(1, TINY.context_len // TINY.patch_size, TINY.d_model)
    disturbed = tokens.clone()
    disturbed[:, -1] += 5.0
    mask = torch.nn.Transformer.generate_square_subsequent_mask(tokens.shape[1])
    with torch.no_grad():
        out = model.encoder.transformer(tokens, mask=mask, is_causal=True)
        out_after = model.encoder.transformer(disturbed, mask=mask, is_causal=True)
    assert torch.allclose(out[:, :-1], out_after[:, :-1], atol=1e-5)
    assert not torch.allclose(out[:, -1], out_after[:, -1], atol=1e-3)


def test_the_encoded_context_shape_follows_the_patch_grid(model: Hepa) -> None:
    context, _ = _batch(1)
    with torch.no_grad():
        assert model.encoder(context).shape == (
            1,
            TINY.context_len // TINY.patch_size,
            TINY.d_model,
        )


def test_a_window_that_is_not_a_whole_number_of_patches_is_refused(model: Hepa) -> None:
    with pytest.raises(ValueError):
        model.encode(torch.zeros(1, TINY.patch_size + 1, telemetry.FEATURE_COUNT))
    with pytest.raises(ValueError):
        model.encode(torch.zeros(1, TINY.context_len, telemetry.FEATURE_COUNT + 1))


def test_a_target_representation_comes_from_the_same_weights(model: Hepa) -> None:
    context, future = _batch()
    horizons = torch.full((4,), float(TINY.max_horizon_steps))
    target = model.target_latent(future, horizons)
    assert target.shape == (4, TINY.d_model)
    # the encoder is weight-shared: no separate target parameters exist
    names = {name for name, _ in model.named_parameters()}
    assert not any(name.startswith("target_encoder") for name in names)


# --- predictor ------------------------------------------------------------
def test_the_predictor_answers_one_or_many_horizons(model: Hepa) -> None:
    context, _ = _batch()
    latent = model.encode(context)
    single = model.predict_latent(latent, torch.full((4,), 8.0))
    assert single.shape == (4, TINY.d_model)
    many = model.predict_latent(latent, torch.tensor([[4.0, 8.0, 16.0]]).repeat(4, 1))
    assert many.shape == (4, 3, TINY.d_model)


def test_the_horizon_actually_conditions_the_prediction(model: Hepa) -> None:
    context, _ = _batch(2)
    latent = model.encode(context)
    near = model.predict_latent(latent, torch.full((2,), 4.0))
    far = model.predict_latent(latent, torch.full((2,), 32.0))
    assert not torch.allclose(near, far, atol=1e-3)


# --- survival head --------------------------------------------------------
def test_hazard_anchors_cover_each_horizon_exactly() -> None:
    steps = anchor_steps(TINY)
    assert steps.numel() == TINY.horizon_anchors
    assert float(steps[0]) == TINY.patch_size
    assert float(steps[-1]) == TINY.max_horizon_steps
    horizons = torch.tensor([[4.0, 12.0, 32.0]])
    coverage = anchor_coverage(TINY, horizons)
    assert coverage.shape == (1, 3, steps.numel())
    # each horizon is covered exactly once, block by block, with no overlap
    assert torch.allclose(coverage.sum(dim=-1), horizons)
    assert torch.all(coverage >= 0.0) and torch.all(coverage <= TINY.patch_size)


def test_event_probabilities_are_probabilities_and_monotone(model: Hepa) -> None:
    context, _ = _batch()
    probabilities = model.event_probabilities(context, [4, 8, 16, 32])
    assert probabilities.shape == (4, TINY.event_count, 4)
    assert torch.all(probabilities >= 0.0) and torch.all(probabilities <= 1.0)
    # p(t, Δt) = 1 - prod (1 - hazard) can only grow with Δt
    assert torch.all(torch.diff(probabilities, dim=-1) >= -1e-6)


def test_logits_and_probabilities_describe_the_same_answer(model: Hepa) -> None:
    context, _ = _batch()
    steps = [4, 16]
    probabilities = model.event_probabilities(context, steps)
    logits = model.event_logits(context, steps)
    assert torch.allclose(torch.sigmoid(logits), probabilities, atol=1e-5)


# --- stage-1 objective ----------------------------------------------------
def test_sampled_horizons_stay_inside_the_trained_range() -> None:
    generator = torch.Generator().manual_seed(3)
    sampled = stage1.sample_horizons(
        512, max_steps=TINY.max_horizon_steps, patch_size=TINY.patch_size, generator=generator
    )
    assert sampled.shape == (512,)
    assert float(sampled.min()) >= 1.0
    assert float(sampled.max()) <= TINY.max_horizon_steps
    # log-uniform: more mass on the short horizons than a uniform draw would give
    assert float((sampled <= TINY.max_horizon_steps / 4).float().mean()) > 0.4


def test_the_pretraining_loss_is_l1_plus_sigreg(model: Hepa) -> None:
    context, future = _batch()
    horizons = torch.full((4,), 16.0)
    predicted = model.predict_latent(model.encode(context), horizons)
    target = model.target_latent(future, horizons)
    loss = stage1.pretrain_loss(predicted, target, sigreg_weight=0.1, projections=8)
    parts = loss.to_dict()
    assert parts["prediction"] == pytest.approx(
        float(torch.nn.functional.l1_loss(predicted, target).detach()), abs=1e-5
    )
    assert parts["loss"] == pytest.approx(
        0.9 * parts["prediction"] + 0.1 * parts["sigreg"], abs=1e-5
    )
    assert parts["sigreg"] >= 0.0


def test_sigreg_prefers_a_gaussian_embedding() -> None:
    generator = torch.Generator().manual_seed(5)
    gaussian = torch.randn(256, 32, generator=generator)
    collapsed = torch.zeros(256, 32)
    assert float(stage1.sigreg(gaussian, projections=16, generator=generator)) < float(
        stage1.sigreg(collapsed, projections=16, generator=generator)
    )


def test_one_optimisation_step_reduces_the_pretraining_loss(model: Hepa) -> None:
    context, future = _batch(8)
    horizons = torch.full((8,), 16.0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(5):
        predicted = model.predict_latent(model.encode(context), horizons)
        target = model.target_latent(future, horizons)
        loss = stage1.pretrain_loss(predicted, target, sigreg_weight=0.1, projections=8)
        optimiser.zero_grad(set_to_none=True)
        loss.total.backward()
        optimiser.step()
        losses.append(float(loss.total.detach()))
    assert losses[-1] < losses[0]


# --- stage-2 objective ----------------------------------------------------
def test_positive_weights_balance_each_event_and_horizon() -> None:
    labels = torch.zeros(10, 2, 1)
    labels[:2, 0, 0] = 1.0  # 2 positives, 8 negatives
    valid = torch.ones(10, 1, dtype=torch.bool)
    weights = stage2.positive_weights(labels, valid)
    assert weights.shape == (2, 1)
    assert float(weights[0, 0]) == pytest.approx(4.0)
    # an event with no positives is left alone rather than given an infinite weight
    assert float(weights[1, 0]) == pytest.approx(1.0)
    assert float(weights.max()) <= stage2.MAX_POSITIVE_WEIGHT

    # a single positive among many negatives is capped, not thousands-fold
    rare = torch.zeros(10_000, 1, 1)
    rare[0, 0, 0] = 1.0
    capped = stage2.positive_weights(rare, torch.ones(10_000, 1, dtype=torch.bool))
    assert float(capped[0, 0]) == pytest.approx(stage2.MAX_POSITIVE_WEIGHT)


def test_the_event_loss_ignores_windows_whose_horizon_is_unknown() -> None:
    logits = torch.zeros(4, 1, 2)
    labels = torch.zeros(4, 1, 2)
    labels[:, :, 1] = 1.0
    valid = torch.ones(4, 2, dtype=torch.bool)
    everything = stage2.event_loss(logits, labels, valid)
    valid[:, 1] = False
    masked = stage2.event_loss(logits, labels, valid)
    assert float(everything) > 0.0
    assert float(masked) == pytest.approx(float(everything), abs=1e-6)  # same per-entry loss
    nothing = stage2.event_loss(logits, labels, torch.zeros(4, 2, dtype=torch.bool))
    assert float(nothing) == pytest.approx(0.0)


def test_freezing_the_encoder_leaves_only_the_head_trainable(model: Hepa) -> None:
    context, _ = _batch()
    before = [parameter.detach().clone() for parameter in model.encoder.parameters()]
    model.freeze_encoder()
    assert model.encoder_frozen
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.event_head.parameters())

    model.train()
    assert not model.encoder.training  # a frozen encoder stays in eval mode: no dropout
    optimiser = torch.optim.AdamW(model.finetune_parameters(), lr=1e-2)
    logits = model.event_logits(context, [4, 16])
    loss = stage2.event_loss(
        logits, torch.ones(4, TINY.event_count, 2), torch.ones(4, 2, dtype=torch.bool)
    )
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    optimiser.step()
    for was, now in zip(before, model.encoder.parameters()):
        assert torch.equal(was, now)
    assert model.parameter_counts()["encoder"] > 0

    model.unfreeze_encoder()
    assert not model.encoder_frozen
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_the_model_never_sees_the_future_at_prediction_time(model: Hepa) -> None:
    """`event_probabilities` takes a context and nothing else."""
    context, future = _batch()
    first = model.event_probabilities(context, [8])
    assert torch.allclose(first, model.event_probabilities(context, [8]))
    del future  # there is no argument it could have been passed as
