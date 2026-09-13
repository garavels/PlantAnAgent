"""Public-only sampled fixtures; no seeds, resets, inference or solver calls."""

from dataclasses import FrozenInstanceError, asdict, replace
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
from plant_sim import PlantConfig, PlantSimulator

from plant_backend import operational_mpc
from plant_backend.rl.v2.economic_mpc_model import EconomicMPCModelV1
from plant_backend.rl.v2.economic_mpc_schema import EconomicMPCSettingsV1
from plant_backend.rl.v2.mpc_forecast import (
    FORECAST_PROJECTION_VERSION,
    MPCForecastTrajectoryV1,
    clear_sky_equivalent_hours,
    project_issued_mean_solar,
    project_issued_p10_solar,
    public_forecast_issue_sha256,
)
from plant_backend.rl.v2.operating_mpc_forecast import (
    OPERATING_FORECAST_PROJECTION_VERSION,
    OperatingMPCForecastTrajectoryV1,
    operating_constant_power_rows,
    project_operating_issued_mean_solar,
)
from plant_backend.rl.v2.weather import (
    FORECAST_WINDOWS_H,
    ForecastWindowV2,
    PublicForecastIssueV2,
)


@pytest.fixture(autouse=True)
def forbid_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("projection fixtures cannot reset or solve")

    monkeypatch.setattr(PlantSimulator, "reset", forbidden)
    monkeypatch.setattr(operational_mpc, "solve_economic_mpc_model", forbidden)


def _independent_samples(config: PlantConfig, boundary_s: float) -> tuple[float, ...]:
    """Independent scalar formula, deliberately not the production projector."""
    samples = []
    for i in range(round(86400.0 / config.dt_s)):
        hour = ((boundary_s + i * config.dt_s) / 3600.0) % 24.0
        sunrise, sunset = config.solar.sunrise_h, config.solar.sunset_h
        samples.append(
            math.sin(math.pi * (hour - sunrise) / (sunset - sunrise))
            if sunrise < hour < sunset
            else 0.0
        )
    return tuple(samples)


def _sampled_issue(
    config: PlantConfig, boundary_s: float, scale: float = 0.63
) -> PublicForecastIssueV2:
    samples = _independent_samples(config, boundary_s)
    windows = []
    for start, end in FORECAST_WINDOWS_H:
        band = samples[
            round(start * 3600 / config.dt_s) : round(end * 3600 / config.dt_s)
        ]
        mean = scale * math.fsum(band) / len(band)
        windows.append(
            ForecastWindowV2(
                f"{start:g}-{end:g}h", start * 3600.0, end * 3600.0, mean, 0.5 * mean
            )
        )
    return PublicForecastIssueV2(0, 0.0, tuple(windows))


def _saved_sunset_issue() -> PublicForecastIssueV2:
    # Detached public issue from the closed batch004 diagnosis at
    # adaptive-rl-batch-004-analysis/forecast-diagnosis/
    # plant-mpc-forecast-sampling-diagnosis.json, SHA256
    # 61567cd02c3cd4ef2310871340ff2a04ee64e5e7530342354dfb2821891d5d1e.
    # Consumed context 9630007/decision341; no seed execution or private tape.
    means = (0.007359442500906482, 0.0, 0.019033101920693046, 0.4518178463113053)
    lowers = (0.007278474985295449, 0.0, 0.01631637773061769, 0.37926893648992754)
    return PublicForecastIssueV2(
        5115,
        306900.0,
        tuple(
            ForecastWindowV2(
                f"{start:g}-{end:g}h", start * 3600.0, end * 3600.0, mean, lower
            )
            for (start, end), mean, lower in zip(
                FORECAST_WINDOWS_H, means, lowers, strict=True
            )
        ),
    )


def _row_model(horizon: int) -> EconomicMPCModelV1:
    # Firm rows only need group indices, never an optimizer or physical episode.
    groups = {
        name: tuple(range(i * horizon, (i + 1) * horizon))
        for i, name in enumerate(
            (
                "solar_to_battery_kw",
                "solar_to_electrolyser_kw",
                "solar_to_dac_kw",
                "solar_to_methanator_kw",
            )
        )
    }
    return cast(
        EconomicMPCModelV1,
        SimpleNamespace(horizon_intervals=horizon, group=groups.__getitem__),
    )


@pytest.mark.parametrize(
    "dt,interval,hour",
    [
        (60.0, 900.0, 5.75),
        (60.0, 900.0, 6.0),
        (60.0, 900.0, 19.25),
        (60.0, 900.0, 19.5),
        (30.0, 300.0, 23.0),
        (120.0, 600.0, 0.0),
    ],
)
@pytest.mark.parametrize("scale", [0.0, 0.63, 1.0])
def test_sampled_geometry_preserves_every_band_and_exact_darkness(
    dt: float, interval: float, hour: float, scale: float
) -> None:
    config = replace(PlantConfig(), dt_s=dt)
    boundary = hour * 3600.0
    issue = _sampled_issue(config, boundary, scale)
    projection = project_operating_issued_mean_solar(
        issue, boundary_time_s=boundary, config=config, interval_s=interval
    )
    assert isinstance(projection, MPCForecastTrajectoryV1)
    assert projection.projection_version == OPERATING_FORECAST_PROJECTION_VERSION
    assert projection.source_issue_sha256 == public_forecast_issue_sha256(issue)
    assert projection.interval_count == round(86400 / interval)
    samples = _independent_samples(config, boundary)
    stride = round(interval / dt)
    for i, clear in enumerate(projection.clear_sky_mean_peak_fractions):
        group = samples[i * stride : (i + 1) * stride]
        assert clear == pytest.approx(math.fsum(group) / stride, abs=1e-15)
        assert projection.clear_sky_min_peak_fractions[i] == pytest.approx(
            min(group), abs=1e-15
        )
        if not any(group):
            assert clear == projection.available_kw[i] == 0.0
        assert projection.available_peak_fractions[i] <= clear + 1e-12
    for band, window in zip(projection.bands, issue.windows, strict=True):
        values = projection.available_peak_fractions[
            band.first_interval : band.first_interval + band.interval_count
        ]
        assert math.fsum(values) / len(values) == pytest.approx(
            window.mean_peak_power_fraction, rel=0.0, abs=1e-12
        )
        assert band.absolute_mean_error <= 1e-12


def test_exact_consumed_sunset_issue_is_legal_under_physical_sampling() -> None:
    issue = _saved_sunset_issue()
    assert (
        public_forecast_issue_sha256(issue)
        == "be907c0bcfb1c20c8187cd993b2a6593d93504b5e18d9c43b02dfda729033bd0"
    )
    config = PlantConfig()
    with pytest.raises(ValueError, match="exceeds clear-sky energy"):
        project_issued_mean_solar(
            issue, boundary_time_s=328500.0, config=config, interval_s=900.0
        )
    projection = project_operating_issued_mean_solar(
        issue, boundary_time_s=328500.0, config=config, interval_s=900.0
    )
    assert projection.bands[0].clear_sky_mean_peak_fraction == pytest.approx(
        0.007754685426807396, abs=1e-15
    )
    assert projection.bands[0].resource_scale == pytest.approx(
        0.007359442500906482 / 0.007754685426807396, abs=1e-12
    )
    assert projection.bands[0].reconstructed_mean_peak_fraction == pytest.approx(
        issue.windows[0].mean_peak_power_fraction, abs=1e-15
    )
    assert all(value == 0.0 for value in projection.available_kw[1:43])
    assert projection.available_kw[43] > 0.0


@pytest.mark.parametrize("horizon", [1, 3, 7, 37, 96])
def test_prefix_and_firm_power_share_full_sampled_band_scale(horizon: int) -> None:
    config = PlantConfig()
    boundary = 19.25 * 3600
    issue = _sampled_issue(config, boundary, scale=0.95)
    projection = project_operating_issued_mean_solar(
        issue, boundary_time_s=boundary, config=config, interval_s=900.0
    )
    assert projection.prefix(horizon) == projection.available_kw[:horizon]
    full_rows, full_power = operating_constant_power_rows(
        _row_model(96), projection, config
    )
    rows, power = operating_constant_power_rows(_row_model(horizon), projection, config)
    assert power == full_power[:horizon]
    samples = _independent_samples(config, boundary)
    for i, (row, firm) in enumerate(zip(rows, power, strict=True)):
        expected = min(samples[15 * i : 15 * (i + 1)]) * 0.95 * config.solar.peak_kw
        assert firm == pytest.approx(expected, rel=0.0, abs=1e-10)
        assert row.upper == firm == full_rows[i].upper
        assert row.name == f"beta_constant_solar_power[{i}]"
        assert row.lower is None
        assert row.coefficients == tuple(
            (i + group * horizon, 1.0) for group in range(4)
        )


def test_p10_only_change_binds_complete_issue_without_changing_mean_path() -> None:
    config = PlantConfig()
    issue = _sampled_issue(config, 6.0 * 3600)
    altered = replace(
        issue,
        windows=(
            replace(issue.windows[0], p10_peak_power_fraction=0.0),
            *issue.windows[1:],
        ),
    )
    projections = [
        project_operating_issued_mean_solar(
            x, boundary_time_s=6 * 3600, config=config, interval_s=900.0
        )
        for x in [issue, altered]
    ]
    assert projections[0].available_kw == projections[1].available_kw
    assert projections[0].source_issue_sha256 != projections[1].source_issue_sha256
    assert projections[0].sha256 != projections[1].sha256


@pytest.mark.parametrize("hour", [6.0, 19.25, 20.0])
def test_impossible_energy_or_power_in_darkness_still_rejected(hour: float) -> None:
    config = PlantConfig()
    issue = _sampled_issue(config, hour * 3600, scale=1.0)
    invalid = replace(
        issue,
        windows=(
            replace(
                issue.windows[0],
                mean_peak_power_fraction=issue.windows[0].mean_peak_power_fraction
                + 1e-6,
            ),
            *issue.windows[1:],
        ),
    )
    with pytest.raises(ValueError, match="clear-sky energy|public darkness"):
        project_operating_issued_mean_solar(
            invalid, boundary_time_s=hour * 3600, config=config, interval_s=900.0
        )


@pytest.mark.parametrize(
    "dt,interval", [(0.0, 900.0), (61.0, 900.0), (60.0, 1000.0), (60.0, 2400.0)]
)
def test_misaligned_physics_controller_or_band_grid_rejected(
    dt: float, interval: float
) -> None:
    config = PlantConfig()
    issue = _sampled_issue(config, 6 * 3600)
    with pytest.raises(ValueError, match="positive|align"):
        project_operating_issued_mean_solar(
            issue,
            boundary_time_s=6 * 3600,
            config=replace(config, dt_s=dt),
            interval_s=interval,
        )


@pytest.mark.parametrize(
    "alteration",
    [
        {"physics_dt_s": 30.0},
        {"sunrise_h": 6.1},
        {"sunset_h": 19.4},
        {"projection_version": FORECAST_PROJECTION_VERSION},
        {"clear_sky_min_peak_fractions": (0.0,) * 96},
        {"clear_sky_min_peak_fractions": (False,) * 96},
        {"clear_sky_mean_peak_fractions": (0.0,) * 96},
    ],
)
def test_projection_tampering_rejected(alteration: dict[str, Any]) -> None:
    config = PlantConfig()
    projection = project_operating_issued_mean_solar(
        _sampled_issue(config, 6 * 3600),
        boundary_time_s=6 * 3600,
        config=config,
        interval_s=900.0,
    )
    with pytest.raises((ValueError, TypeError)):
        replace(projection, **alteration)


@pytest.mark.parametrize(
    "field,value",
    [("dt_s", 30.0), ("peak_kw", 6000.0), ("sunrise_h", 5.0), ("sunset_h", 20.0)],
)
def test_firm_rows_reject_other_geometry_or_timestep(field: str, value: float) -> None:
    config = PlantConfig()
    projection = project_operating_issued_mean_solar(
        _sampled_issue(config, 6 * 3600),
        boundary_time_s=6 * 3600,
        config=config,
        interval_s=900.0,
    )
    other = (
        replace(config, dt_s=value)
        if field == "dt_s"
        else replace(config, solar=replace(config.solar, **{field: value}))
    )
    with pytest.raises(ValueError, match="bound sampled geometry"):
        operating_constant_power_rows(_row_model(96), projection, other)


def test_sample_identity_serialization_is_immutable_and_consistent() -> None:
    config = PlantConfig()
    issue = _sampled_issue(config, 6 * 3600)
    projection = project_operating_issued_mean_solar(
        issue, boundary_time_s=6 * 3600, config=config, interval_s=900.0
    )
    clone = OperatingMPCForecastTrajectoryV1(
        **{**asdict(projection), "bands": projection.bands}
    )
    assert clone.to_dict() == projection.to_dict()
    assert projection.to_dict()["sampling"] == {
        "rule": "SolarArray.clear_sky_kw at each physical interval start",
        "physics_dt_s": 60.0,
        "sunrise_h": 6.0,
        "sunset_h": 19.5,
    }
    body = projection.to_dict()
    body["sampling"]["physics_dt_s"] = 99.0
    assert projection.to_dict()["sampling"]["physics_dt_s"] == 60.0
    with pytest.raises(FrozenInstanceError):
        setattr(projection, "physics_dt_s", 99.0)
    with pytest.raises(ValueError, match="outside the forecast horizon"):
        projection.prefix(97)


def test_legacy_mean_and_p10_golden_digests_remain_unchanged() -> None:
    config = PlantConfig()
    boundary = 18.75 * 3600.0
    windows = []
    for start, end in FORECAST_WINDOWS_H:
        mean = (
            0.63
            * clear_sky_equivalent_hours(
                boundary_time_s=boundary,
                start_offset_s=start * 3600.0,
                end_offset_s=end * 3600.0,
                sunrise_h=config.solar.sunrise_h,
                sunset_h=config.solar.sunset_h,
            )
            / (end - start)
        )
        windows.append(
            ForecastWindowV2(
                f"{start:g}-{end:g}h", start * 3600.0, end * 3600.0, mean, 0.5 * mean
            )
        )
    issue = PublicForecastIssueV2(20, 1200.0, tuple(windows))
    mean_projection = project_issued_mean_solar(
        issue, boundary_time_s=boundary, config=config, interval_s=900.0
    )
    assert (
        mean_projection.sha256
        == "c950e4d5a0825e343117b3c9d38ed4ab321b82d4da830c2ee60af6d6e611210d"
    )
    assert (
        project_issued_p10_solar(
            issue, boundary_time_s=boundary, config=config, interval_s=900.0
        ).sha256
        == "10090364d08cca3d33d15c606c6db0f11cc6911337418bc8219f13bd3eda899f"
    )
    with pytest.raises(ValueError, match="unsupported"):
        replace(
            mean_projection, projection_version=OPERATING_FORECAST_PROJECTION_VERSION
        )
    with pytest.raises(TypeError, match="sampled operating trajectory"):
        operating_constant_power_rows(
            _row_model(96),
            cast(OperatingMPCForecastTrajectoryV1, mean_projection),
            config,
        )
    assert (
        EconomicMPCSettingsV1().physical_settings.forecast_projection_version
        == FORECAST_PROJECTION_VERSION
    )
    settings = operational_mpc.OperatingMPCSettingsV1()
    assert (
        settings.physical_settings.forecast_projection_version
        == OPERATING_FORECAST_PROJECTION_VERSION
    )
    assert (
        settings.to_dict()["forecast_projection_version"]
        == OPERATING_FORECAST_PROJECTION_VERSION
    )
    assert settings.sha256 != EconomicMPCSettingsV1().sha256
    assert (
        settings.physical_settings.sha256
        != EconomicMPCSettingsV1().physical_settings.sha256
    )
