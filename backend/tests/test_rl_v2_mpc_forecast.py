from __future__ import annotations

import math
from dataclasses import replace

import pytest
from plant_sim import PlantConfig

from plant_backend.rl.v2.mpc_forecast import (
    FORECAST_PROJECTION_VERSION,
    clear_sky_equivalent_hours,
    mean_peak_fraction,
    project_issued_mean_solar,
    public_forecast_issue_sha256,
)
from plant_backend.rl.v2.weather import (
    FORECAST_WINDOWS_H,
    ForecastWindowV2,
    PublicForecastIssueV2,
)


def _compatible_issue(
    *,
    boundary_time_s: float,
    config: PlantConfig,
    resource_scale: float,
) -> PublicForecastIssueV2:
    windows = []
    for start_h, end_h in FORECAST_WINDOWS_H:
        equivalent_h = clear_sky_equivalent_hours(
            boundary_time_s=boundary_time_s,
            start_offset_s=start_h * 3_600.0,
            end_offset_s=end_h * 3_600.0,
            sunrise_h=config.solar.sunrise_h,
            sunset_h=config.solar.sunset_h,
        )
        mean = equivalent_h / (end_h - start_h) * resource_scale
        windows.append(
            ForecastWindowV2(
                name=f"{start_h:g}-{end_h:g}h",
                start_offset_s_from_issue=start_h * 3_600.0,
                end_offset_s_from_issue=end_h * 3_600.0,
                mean_peak_power_fraction=mean,
                p10_peak_power_fraction=0.5 * mean,
            )
        )
    return PublicForecastIssueV2(
        issued_at_physics_boundary_step=20,
        issued_at_offset_s=1_200.0,
        windows=tuple(windows),
    )


def test_projection_preserves_every_band_and_public_clear_sky_shape() -> None:
    config = PlantConfig()
    boundary_time_s = 5.5 * 3_600.0
    issue = _compatible_issue(
        boundary_time_s=boundary_time_s,
        config=config,
        resource_scale=0.63,
    )

    trajectory = project_issued_mean_solar(
        issue,
        boundary_time_s=boundary_time_s,
        config=config,
        interval_s=900.0,
    )

    assert trajectory.projection_version == FORECAST_PROJECTION_VERSION
    assert trajectory.source_issue_sha256 == public_forecast_issue_sha256(issue)
    assert trajectory.interval_count == 96
    assert (
        trajectory.sha256
        == project_issued_mean_solar(
            issue,
            boundary_time_s=boundary_time_s,
            config=config,
            interval_s=900.0,
        ).sha256
    )
    for audit, window in zip(trajectory.bands, issue.windows, strict=True):
        values = trajectory.available_kw[
            audit.first_interval : audit.first_interval + audit.interval_count
        ]
        assert mean_peak_fraction(
            values, peak_kw=config.solar.peak_kw
        ) == pytest.approx(
            window.mean_peak_power_fraction,
            rel=0.0,
            abs=1e-12,
        )
        assert audit.resource_scale == pytest.approx(0.63, abs=1e-12)
        assert audit.absolute_mean_error <= 1e-12
    for clear, available in zip(
        trajectory.clear_sky_mean_peak_fractions,
        trajectory.available_peak_fractions,
        strict=True,
    ):
        assert available == pytest.approx(clear * 0.63, abs=1e-15)
        if clear == 0.0:
            assert available == 0.0


def test_projection_handles_sunrise_sunset_and_midnight_without_point_sampling() -> (
    None
):
    config = PlantConfig()
    boundary_time_s = 18.75 * 3_600.0
    issue = _compatible_issue(
        boundary_time_s=boundary_time_s,
        config=config,
        resource_scale=0.4,
    )
    trajectory = project_issued_mean_solar(
        issue,
        boundary_time_s=boundary_time_s,
        config=config,
        interval_s=900.0,
    )

    # The first three intervals approach the configured 19:30 sunset; the
    # following night is exactly dark, and the next-day sunrise reappears.
    assert (
        trajectory.available_kw[0]
        > trajectory.available_kw[1]
        > trajectory.available_kw[2]
        > 0.0
    )
    assert all(value == 0.0 for value in trajectory.available_kw[3:45])
    assert trajectory.available_kw[45] > 0.0
    assert trajectory.available_kw[-1] > 0.0


def test_shortened_horizon_is_a_byte_identical_prefix_without_renormalization() -> None:
    config = PlantConfig()
    issue = _compatible_issue(
        boundary_time_s=6.0 * 3_600.0,
        config=config,
        resource_scale=0.7,
    )
    trajectory = project_issued_mean_solar(
        issue,
        boundary_time_s=6.0 * 3_600.0,
        config=config,
        interval_s=900.0,
    )

    assert trajectory.prefix(1) == trajectory.available_kw[:1]
    assert trajectory.prefix(37) == trajectory.available_kw[:37]
    assert trajectory.prefix(96) == trajectory.available_kw
    with pytest.raises(ValueError, match="outside the forecast horizon"):
        trajectory.prefix(0)
    with pytest.raises(ValueError, match="outside the forecast horizon"):
        trajectory.prefix(97)


def test_projection_rejects_public_energy_above_clear_sky_support() -> None:
    config = PlantConfig()
    boundary_time_s = 6.0 * 3_600.0
    issue = _compatible_issue(
        boundary_time_s=boundary_time_s,
        config=config,
        resource_scale=1.0,
    )
    first = issue.windows[0]
    impossible = replace(
        issue,
        windows=(
            replace(
                first,
                mean_peak_power_fraction=min(
                    1.0,
                    first.mean_peak_power_fraction + 0.02,
                ),
                p10_peak_power_fraction=first.p10_peak_power_fraction,
            ),
            *issue.windows[1:],
        ),
    )

    with pytest.raises(ValueError, match="exceeds clear-sky energy"):
        project_issued_mean_solar(
            impossible,
            boundary_time_s=boundary_time_s,
            config=config,
            interval_s=900.0,
        )


def test_projection_identity_depends_only_on_declared_public_inputs() -> None:
    config = PlantConfig()
    issue = _compatible_issue(
        boundary_time_s=8.0 * 3_600.0,
        config=config,
        resource_scale=0.55,
    )

    first = project_issued_mean_solar(
        issue,
        boundary_time_s=8.0 * 3_600.0,
        config=config,
        interval_s=900.0,
    )
    unrelated_private_truth_a = (0.0,) * 100
    unrelated_private_truth_b = (1.0,) * 100
    assert unrelated_private_truth_a != unrelated_private_truth_b
    second = project_issued_mean_solar(
        issue,
        boundary_time_s=8.0 * 3_600.0,
        config=config,
        interval_s=900.0,
    )
    assert first.to_dict() == second.to_dict()

    changed_p10 = replace(
        issue,
        windows=(
            replace(
                issue.windows[0],
                p10_peak_power_fraction=issue.windows[0].p10_peak_power_fraction * 0.9,
            ),
            *issue.windows[1:],
        ),
    )
    same_mean_path = project_issued_mean_solar(
        changed_p10,
        boundary_time_s=8.0 * 3_600.0,
        config=config,
        interval_s=900.0,
    )
    assert same_mean_path.available_kw == first.available_kw
    assert same_mean_path.source_issue_sha256 != first.source_issue_sha256
    assert same_mean_path.sha256 != first.sha256


def test_trajectory_schema_rejects_a_band_not_bound_to_its_intervals() -> None:
    config = PlantConfig()
    issue = _compatible_issue(
        boundary_time_s=8.0 * 3_600.0,
        config=config,
        resource_scale=0.55,
    )
    trajectory = project_issued_mean_solar(
        issue,
        boundary_time_s=8.0 * 3_600.0,
        config=config,
        interval_s=900.0,
    )

    with pytest.raises(ValueError, match="interval ranges must be contiguous"):
        replace(
            trajectory,
            bands=(
                replace(trajectory.bands[0], first_interval=1),
                *trajectory.bands[1:],
            ),
        )
    with pytest.raises(ValueError, match="scale is not bound"):
        replace(
            trajectory,
            bands=(
                replace(
                    trajectory.bands[0],
                    resource_scale=trajectory.bands[0].resource_scale * 0.9,
                ),
                *trajectory.bands[1:],
            ),
        )


@pytest.mark.parametrize("interval_s", [0.0, 700.0, math.inf])
def test_projection_rejects_invalid_or_misaligned_resolution(interval_s: float) -> None:
    config = PlantConfig()
    issue = _compatible_issue(
        boundary_time_s=6.0 * 3_600.0,
        config=config,
        resource_scale=0.5,
    )
    with pytest.raises((TypeError, ValueError)):
        project_issued_mean_solar(
            issue,
            boundary_time_s=6.0 * 3_600.0,
            config=config,
            interval_s=interval_s,
        )


def test_clear_sky_integral_is_additive_across_midnight() -> None:
    arguments = {
        "boundary_time_s": 17.0 * 3_600.0,
        "sunrise_h": 6.0,
        "sunset_h": 19.5,
    }
    complete = clear_sky_equivalent_hours(
        **arguments,
        start_offset_s=0.0,
        end_offset_s=30.0 * 3_600.0,
    )
    left = clear_sky_equivalent_hours(
        **arguments,
        start_offset_s=0.0,
        end_offset_s=7.0 * 3_600.0,
    )
    right = clear_sky_equivalent_hours(
        **arguments,
        start_offset_s=7.0 * 3_600.0,
        end_offset_s=30.0 * 3_600.0,
    )
    assert complete == pytest.approx(left + right, abs=1e-14)
