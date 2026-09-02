"""Electrolyser: turns electricity into hydrogen, but only within its load window."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

from ..commitment import (
    ElectrolyserCommitmentProfile,
    ElectrolyserCommitmentSnapshot,
    ElectrolyserCommitmentState,
    ElectrolyserCommitmentStep,
    ElectrolyserMode,
)
from ..config import ElectrolyserConfig
from .base import Component


class Electrolyser(Component):
    """Consumes power between `min_load` and `rated_kw`, produces H2 linearly.

    The minimum load is what makes power allocation interesting: a trickle of
    solar is worth nothing to the stack, so an agent has to decide between
    buffering it in the battery or discharging the battery to lift the stack
    above its floor.

    Degradation is the V4 seam: a degraded stack keeps drawing the same power
    but converts less of it, so `specific_energy_kwh_per_kg` rises with the
    fault severity and the stack runs hotter -- the diagnostic an agent can
    correlate with the missing hydrogen.

    When ``commitment`` is supplied, a technology-labelled state machine adds
    start-up, standby, ramp and part-load dynamics.  Omitting it retains the
    established memoryless conversion path exactly.
    """

    name = "electrolyser"

    def __init__(
        self,
        config: ElectrolyserConfig | None = None,
        *,
        commitment: ElectrolyserCommitmentProfile | None = None,
    ) -> None:
        if commitment is not None and not isinstance(
            commitment, ElectrolyserCommitmentProfile
        ):
            raise TypeError(
                "commitment must be an ElectrolyserCommitmentProfile or None"
            )
        self.config = config or ElectrolyserConfig()
        self.commitment_profile = commitment
        self._commitment_state = ElectrolyserCommitmentState()
        self.power_kw = 0.0
        self.online = False

    def reset(self) -> None:
        self.power_kw = 0.0
        self.online = False
        self._commitment_state = ElectrolyserCommitmentState()

    @property
    def commitment_enabled(self) -> bool:
        return self.commitment_profile is not None

    @property
    def commitment_snapshot(self) -> ElectrolyserCommitmentSnapshot | None:
        """Return an immutable state/provenance view, or ``None`` in legacy mode."""
        profile = self.commitment_profile
        if profile is None:
            return None
        state = self._commitment_state
        if state.mode is ElectrolyserMode.RUNNING:
            one_minute_up = profile.ramp_up_frac_per_min * self.config.rated_kw
            one_minute_down = profile.ramp_down_frac_per_min * self.config.rated_kw
            ramp_up_headroom_kw = max(
                0.0,
                min(self.config.rated_kw, state.previous_power_kw + one_minute_up)
                - state.previous_power_kw,
            )
            ramp_down_headroom_kw = max(
                0.0,
                state.previous_power_kw
                - max(
                    profile.min_stable_load_frac * self.config.rated_kw,
                    state.previous_power_kw - one_minute_down,
                ),
            )
        else:
            ramp_up_headroom_kw = 0.0
            ramp_down_headroom_kw = 0.0
        return ElectrolyserCommitmentSnapshot(
            profile_id=profile.profile_id,
            technology=profile.technology,
            calibration_basis=profile.calibration_basis,
            dynamics_version=profile.dynamics_version,
            mode=state.mode,
            mode_elapsed_s=state.mode_elapsed_s,
            startup_remaining_s=state.startup_remaining_s,
            cooldown_remaining_s=state.cooldown_remaining_s,
            min_up_remaining_s=state.min_up_remaining_s,
            min_down_remaining_s=state.min_down_remaining_s,
            shortfall_elapsed_s=state.shortfall_elapsed_s,
            previous_power_kw=state.previous_power_kw,
            ramp_up_headroom_kw=ramp_up_headroom_kw,
            ramp_down_headroom_kw=ramp_down_headroom_kw,
            cold_starts=state.cold_starts,
            hot_starts=state.hot_starts,
            aborted_starts=state.aborted_starts,
            forced_stops=state.forced_stops,
            controlled_stops=state.controlled_stops,
            startup_energy_kwh=state.startup_energy_kwh,
            standby_energy_kwh=state.standby_energy_kwh,
            operating_time_s=state.operating_time_s,
            low_load_time_s=state.low_load_time_s,
            ramp_limited_intervals=state.ramp_limited_intervals,
            absolute_load_movement_kw=state.absolute_load_movement_kw,
            mode_time_s=state.mode_time_s,
        )

    @property
    def min_load_kw(self) -> float:
        return self.config.min_load_frac * self.config.rated_kw

    @property
    def load_frac(self) -> float:
        return self.power_kw / self.config.rated_kw

    @property
    def specific_energy_kwh_per_kg(self) -> float:
        """Electricity per kg of H2 *now*: nominal, worsened by degradation."""
        if self.commitment_profile is not None:
            nominal = max(
                self.config.specific_energy_kwh_per_kg,
                self.commitment_profile.nominal_specific_energy_kwh_per_kg,
            )
            multiplier = self.commitment_profile.specific_energy_multiplier(
                max(self.load_frac, self.commitment_profile.min_stable_load_frac)
            )
            return nominal * multiplier / max(self.health, 1e-6)
        return self.config.specific_energy_kwh_per_kg / max(self.health, 1e-6)

    @property
    def specific_energy_multiplier(self) -> float:
        """Current part-load multiplier (one on the legacy conversion path)."""
        if self.commitment_profile is None:
            return 1.0
        return self.commitment_profile.specific_energy_multiplier(
            max(self.load_frac, self.commitment_profile.min_stable_load_frac)
        )

    @property
    def efficiency(self) -> float:
        """Yield relative to the nameplate stack, 0..1."""
        if self.commitment_profile is not None:
            if not self.online:
                return 0.0
            return self.health / self.specific_energy_multiplier
        return self.health

    def validate_commitment_timestep(self, dt_h: float) -> None:
        """Validate commitment boundaries against the simulator physics grid."""
        if self.commitment_profile is not None:
            self.commitment_profile.validate_timestep(dt_h * 3_600.0)

    @property
    def temperature_c(self) -> float:
        """Stack temperature: ambient when idle, rising with load and degradation."""
        if not self.online:
            return self.config.ambient_temperature_c
        return (
            self.config.nominal_temperature_c
            + self.config.fault_temperature_rise_c * self.effect.degradation
        )

    def intake_limit_kw(self, dt_h: float, h2_headroom_kg: float) -> float:
        """Max power the stack can usefully take: rated power, capped by tank room."""
        if self.offline:
            return 0.0
        if self.commitment_profile is not None:
            # Starts and standby consume energy without producing H2, so storage
            # headroom is applied only inside the running branch.
            return self.config.rated_kw
        headroom_kw = h2_headroom_kg * self.specific_energy_kwh_per_kg / dt_h
        return min(self.config.rated_kw, max(0.0, headroom_kw))

    def run(self, power_kw: float, dt_h: float) -> float:
        """Run for `dt_h` at `power_kw`; returns hydrogen produced (kg).

        Below the minimum load the stack simply stays off and produces nothing --
        the caller is responsible for re-routing the power it did not use.
        """
        if self.commitment_profile is not None:
            return self.advance_commitment(
                power_kw,
                dt_h,
                h2_headroom_kg=math.inf,
                commitment_intent=power_kw > 0.0,
            ).h2_produced_kg
        if self.offline or power_kw <= 0.0 or power_kw < self.min_load_kw:
            self.power_kw = 0.0
            self.online = False
            return 0.0
        self.power_kw = min(power_kw, self.config.rated_kw)
        self.online = True
        return self.power_kw * dt_h / self.specific_energy_kwh_per_kg

    def advance_commitment(
        self,
        offered_power_kw: float,
        dt_h: float,
        *,
        h2_headroom_kg: float,
        commitment_intent: bool,
    ) -> ElectrolyserCommitmentStep:
        """Advance exactly one committed interval and return accepted power/H2.

        Modes are sampled at interval boundaries.  A transition completed at
        the end of this call is visible immediately in ``commitment_snapshot``,
        while ``power_kw``, ``online`` and the returned mass still describe the
        interval that just ended.  In particular, the last start-up interval
        consumes start power and makes no H2; production can begin only on the
        following call.
        """
        profile = self.commitment_profile
        if profile is None:
            raise RuntimeError("commitment dynamics are not enabled")
        if isinstance(offered_power_kw, bool) or not isinstance(
            offered_power_kw, (int, float)
        ):
            raise TypeError("offered_power_kw must be a real number")
        if not math.isfinite(float(offered_power_kw)) or offered_power_kw < 0.0:
            raise ValueError("offered_power_kw must be finite and >= 0")
        if isinstance(dt_h, bool) or not isinstance(dt_h, (int, float)):
            raise TypeError("dt_h must be a real number")
        if not math.isfinite(float(dt_h)) or dt_h <= 0.0:
            raise ValueError("dt_h must be finite and > 0")
        if isinstance(h2_headroom_kg, bool) or not isinstance(
            h2_headroom_kg, (int, float)
        ):
            raise TypeError("h2_headroom_kg must be a real number")
        if math.isnan(float(h2_headroom_kg)) or h2_headroom_kg < 0.0:
            raise ValueError("h2_headroom_kg must be >= 0 and not NaN")
        if not isinstance(commitment_intent, bool):
            raise TypeError("commitment_intent must be a bool")

        offered_kw = min(float(offered_power_kw), self.config.rated_kw)
        dt_s = float(dt_h) * 3_600.0
        profile.validate_timestep(dt_s)
        state = self._commitment_state

        if self.offline:
            if state.mode in {
                ElectrolyserMode.COLD_STARTING,
                ElectrolyserMode.RUNNING,
                ElectrolyserMode.HOT_STANDBY,
                ElectrolyserMode.HOT_STARTING,
            }:
                return self._stop_at_boundary(
                    state,
                    dt_s,
                    forced=True,
                    aborted=state.mode in {
                        ElectrolyserMode.COLD_STARTING,
                        ElectrolyserMode.HOT_STARTING,
                    },
                )
            return self._inactive_interval(state, dt_s)

        if state.mode is ElectrolyserMode.COOLDOWN:
            return self._cooldown_interval(state, dt_s)

        if state.mode is ElectrolyserMode.OFF:
            if state.min_down_remaining_s > 0.0:
                return self._off_interval(state, dt_s)
            required_kw = profile.start_load_frac * self.config.rated_kw
            if not commitment_intent or offered_kw + 1e-9 < required_kw:
                return self._off_interval(state, dt_s)
            starting = replace(
                state,
                mode=ElectrolyserMode.COLD_STARTING,
                mode_elapsed_s=0.0,
                startup_remaining_s=profile.cold_start_s,
                min_up_remaining_s=0.0,
                shortfall_elapsed_s=0.0,
                cold_starts=state.cold_starts + 1,
            )
            return self._starting_interval(starting, dt_s, required_kw)

        if not commitment_intent:
            return self._stop_at_boundary(
                state,
                dt_s,
                forced=(
                    state.min_up_remaining_s > 0.0
                    or state.mode
                    in {
                        ElectrolyserMode.COLD_STARTING,
                        ElectrolyserMode.HOT_STARTING,
                    }
                ),
                aborted=state.mode
                in {
                    ElectrolyserMode.COLD_STARTING,
                    ElectrolyserMode.HOT_STARTING,
                },
            )

        if state.mode in {
            ElectrolyserMode.COLD_STARTING,
            ElectrolyserMode.HOT_STARTING,
        }:
            required_kw = profile.start_load_frac * self.config.rated_kw
            if offered_kw + 1e-9 < required_kw:
                return self._shortfall_interval(state, dt_s)
            return self._starting_interval(state, dt_s, required_kw)

        conservative_headroom_kw = (
            h2_headroom_kg
            * max(
                self.config.specific_energy_kwh_per_kg,
                profile.nominal_specific_energy_kwh_per_kg,
            )
            / dt_h
        )
        min_running_kw = profile.min_stable_load_frac * self.config.rated_kw
        standby_kw = profile.hot_standby_load_frac * self.config.rated_kw

        if state.mode is ElectrolyserMode.HOT_STANDBY:
            if (
                offered_kw + 1e-9 >= min_running_kw
                and conservative_headroom_kw + 1e-9 >= min_running_kw
            ):
                starting = replace(
                    state,
                    mode=ElectrolyserMode.HOT_STARTING,
                    mode_elapsed_s=0.0,
                    startup_remaining_s=profile.hot_start_s,
                    shortfall_elapsed_s=0.0,
                    hot_starts=state.hot_starts + 1,
                )
                return self._starting_interval(
                    starting,
                    dt_s,
                    profile.start_load_frac * self.config.rated_kw,
                )
            if offered_kw + 1e-9 >= standby_kw:
                staying = replace(
                    state,
                    mode_elapsed_s=state.mode_elapsed_s + dt_s,
                    min_up_remaining_s=max(0.0, state.min_up_remaining_s - dt_s),
                    shortfall_elapsed_s=0.0,
                )
                return self._record_interval(
                    staying, ElectrolyserMode.HOT_STANDBY, dt_s, standby_kw, 0.0
                )
            return self._shortfall_interval(state, dt_s)

        # The only remaining active mode is running.
        if (
            offered_kw + 1e-9 < min_running_kw
            or conservative_headroom_kw + 1e-9 < min_running_kw
        ):
            if offered_kw + 1e-9 >= standby_kw:
                standby = replace(
                    state,
                    mode=ElectrolyserMode.HOT_STANDBY,
                    mode_elapsed_s=dt_s,
                    min_up_remaining_s=max(0.0, state.min_up_remaining_s - dt_s),
                    shortfall_elapsed_s=0.0,
                )
                return self._record_interval(
                    standby, ElectrolyserMode.HOT_STANDBY, dt_s, standby_kw, 0.0
                )
            return self._shortfall_interval(state, dt_s)

        raw_target_kw = min(
            offered_kw,
            conservative_headroom_kw,
            self.config.rated_kw,
        )
        ramp_up_kw = profile.ramp_up_frac_per_min * self.config.rated_kw * dt_s / 60.0
        ramp_down_kw = (
            profile.ramp_down_frac_per_min * self.config.rated_kw * dt_s / 60.0
        )
        upper_kw = min(self.config.rated_kw, state.previous_power_kw + ramp_up_kw)
        lower_kw = max(min_running_kw, state.previous_power_kw - ramp_down_kw)

        # Offered power is a hard physical ceiling.  If it falls faster than
        # the declared running ramp can follow, production settles to standby
        # instead of consuming power that was never routed to the unit.  The
        # fallback remains a ramp-limit exposure even though no power can be
        # invented to make the running trajectory feasible.
        if raw_target_kw + 1e-9 < lower_kw:
            standby = replace(
                state,
                mode=ElectrolyserMode.HOT_STANDBY,
                mode_elapsed_s=dt_s,
                min_up_remaining_s=max(0.0, state.min_up_remaining_s - dt_s),
                shortfall_elapsed_s=0.0,
            )
            return self._record_interval(
                standby,
                ElectrolyserMode.HOT_STANDBY,
                dt_s,
                standby_kw,
                0.0,
                ramp_limited=True,
            )

        accepted_kw = min(raw_target_kw, max(min_running_kw, upper_kw))
        ramp_limited = accepted_kw + 1e-9 < raw_target_kw
        load_frac = accepted_kw / self.config.rated_kw
        nominal = max(
            self.config.specific_energy_kwh_per_kg,
            profile.nominal_specific_energy_kwh_per_kg,
        )
        specific_energy = (
            nominal
            * profile.specific_energy_multiplier(load_frac)
            / max(self.health, 1e-6)
        )
        h2_kg = min(h2_headroom_kg, accepted_kw * dt_h / specific_energy)
        running = replace(
            state,
            mode_elapsed_s=state.mode_elapsed_s + dt_s,
            min_up_remaining_s=max(0.0, state.min_up_remaining_s - dt_s),
            shortfall_elapsed_s=0.0,
        )
        return self._record_interval(
            running,
            ElectrolyserMode.RUNNING,
            dt_s,
            accepted_kw,
            h2_kg,
            ramp_limited=ramp_limited,
        )

    def _starting_interval(
        self,
        state: ElectrolyserCommitmentState,
        dt_s: float,
        required_kw: float,
    ) -> ElectrolyserCommitmentStep:
        """Consume one complete start interval; transition only at its end."""
        profile = self.commitment_profile
        assert profile is not None
        occupied_mode = state.mode
        remaining_s = max(0.0, state.startup_remaining_s - dt_s)
        min_up_s = (
            max(0.0, state.min_up_remaining_s - dt_s)
            if occupied_mode is ElectrolyserMode.HOT_STARTING
            else state.min_up_remaining_s
        )
        if remaining_s <= 1e-9:
            advanced = replace(
                state,
                mode=ElectrolyserMode.RUNNING,
                mode_elapsed_s=0.0,
                startup_remaining_s=0.0,
                min_up_remaining_s=(
                    profile.min_up_s
                    if occupied_mode is ElectrolyserMode.COLD_STARTING
                    else min_up_s
                ),
                shortfall_elapsed_s=0.0,
            )
        else:
            advanced = replace(
                state,
                mode_elapsed_s=state.mode_elapsed_s + dt_s,
                startup_remaining_s=remaining_s,
                min_up_remaining_s=min_up_s,
                shortfall_elapsed_s=0.0,
            )
        return self._record_interval(
            advanced, occupied_mode, dt_s, required_kw, 0.0
        )

    def _shortfall_interval(
        self,
        state: ElectrolyserCommitmentState,
        dt_s: float,
    ) -> ElectrolyserCommitmentStep:
        """Pause active clocks, then trip when the grace is exhausted."""
        profile = self.commitment_profile
        assert profile is not None
        elapsed_s = state.shortfall_elapsed_s + dt_s
        if elapsed_s + 1e-9 >= profile.shortfall_grace_s:
            tripped = replace(
                state,
                mode=ElectrolyserMode.COOLDOWN,
                mode_elapsed_s=0.0,
                startup_remaining_s=0.0,
                cooldown_remaining_s=profile.cooldown_s,
                min_up_remaining_s=0.0,
                min_down_remaining_s=profile.min_down_s,
                shortfall_elapsed_s=0.0,
                aborted_starts=state.aborted_starts
                + int(
                    state.mode
                    in {
                        ElectrolyserMode.COLD_STARTING,
                        ElectrolyserMode.HOT_STARTING,
                    }
                ),
                forced_stops=state.forced_stops + 1,
            )
            return self._record_interval(tripped, state.mode, dt_s, 0.0, 0.0)
        paused = replace(
            state,
            mode_elapsed_s=state.mode_elapsed_s + dt_s,
            shortfall_elapsed_s=elapsed_s,
        )
        return self._record_interval(paused, state.mode, dt_s, 0.0, 0.0)

    def _stop_at_boundary(
        self,
        state: ElectrolyserCommitmentState,
        dt_s: float,
        *,
        forced: bool,
        aborted: bool,
    ) -> ElectrolyserCommitmentStep:
        """Enter unpowered cooldown at the start of this interval."""
        profile = self.commitment_profile
        assert profile is not None
        cooldown_s = max(0.0, profile.cooldown_s - dt_s)
        min_down_s = max(0.0, profile.min_down_s - dt_s)
        mode = ElectrolyserMode.COOLDOWN if cooldown_s > 1e-9 else ElectrolyserMode.OFF
        stopped = replace(
            state,
            mode=mode,
            mode_elapsed_s=dt_s if mode is ElectrolyserMode.COOLDOWN else 0.0,
            startup_remaining_s=0.0,
            cooldown_remaining_s=cooldown_s,
            min_up_remaining_s=0.0,
            min_down_remaining_s=min_down_s,
            shortfall_elapsed_s=0.0,
            aborted_starts=state.aborted_starts + int(aborted),
            forced_stops=state.forced_stops + int(forced),
            controlled_stops=state.controlled_stops + int(not forced),
        )
        return self._record_interval(
            stopped, ElectrolyserMode.COOLDOWN, dt_s, 0.0, 0.0
        )

    def _cooldown_interval(
        self,
        state: ElectrolyserCommitmentState,
        dt_s: float,
    ) -> ElectrolyserCommitmentStep:
        cooldown_s = max(0.0, state.cooldown_remaining_s - dt_s)
        min_down_s = max(0.0, state.min_down_remaining_s - dt_s)
        if cooldown_s <= 1e-9:
            advanced = replace(
                state,
                mode=ElectrolyserMode.OFF,
                mode_elapsed_s=0.0,
                cooldown_remaining_s=0.0,
                min_down_remaining_s=min_down_s,
            )
        else:
            advanced = replace(
                state,
                mode_elapsed_s=state.mode_elapsed_s + dt_s,
                cooldown_remaining_s=cooldown_s,
                min_down_remaining_s=min_down_s,
            )
        return self._record_interval(
            advanced, ElectrolyserMode.COOLDOWN, dt_s, 0.0, 0.0
        )

    def _off_interval(
        self,
        state: ElectrolyserCommitmentState,
        dt_s: float,
    ) -> ElectrolyserCommitmentStep:
        advanced = replace(
            state,
            mode_elapsed_s=state.mode_elapsed_s + dt_s,
            min_down_remaining_s=max(0.0, state.min_down_remaining_s - dt_s),
            shortfall_elapsed_s=0.0,
        )
        return self._record_interval(
            advanced, ElectrolyserMode.OFF, dt_s, 0.0, 0.0
        )

    def _inactive_interval(
        self,
        state: ElectrolyserCommitmentState,
        dt_s: float,
    ) -> ElectrolyserCommitmentStep:
        if state.mode is ElectrolyserMode.COOLDOWN:
            return self._cooldown_interval(state, dt_s)
        return self._off_interval(state, dt_s)

    def _record_interval(
        self,
        state: ElectrolyserCommitmentState,
        occupied_mode: ElectrolyserMode,
        dt_s: float,
        accepted_kw: float,
        h2_kg: float,
        *,
        ramp_limited: bool = False,
    ) -> ElectrolyserCommitmentStep:
        """Publish interval outputs and update exposure-only audit counters."""
        profile = self.commitment_profile
        assert profile is not None
        previous_kw = self._commitment_state.previous_power_kw
        load_frac = accepted_kw / self.config.rated_kw
        state = state.with_mode_time(occupied_mode, dt_s)
        state = replace(
            state,
            previous_power_kw=accepted_kw,
            startup_energy_kwh=state.startup_energy_kwh
            + (
                accepted_kw * dt_s / 3_600.0
                if occupied_mode
                in {ElectrolyserMode.COLD_STARTING, ElectrolyserMode.HOT_STARTING}
                else 0.0
            ),
            standby_energy_kwh=state.standby_energy_kwh
            + (
                accepted_kw * dt_s / 3_600.0
                if occupied_mode is ElectrolyserMode.HOT_STANDBY
                else 0.0
            ),
            operating_time_s=state.operating_time_s
            + (dt_s if occupied_mode is ElectrolyserMode.RUNNING and h2_kg > 0.0 else 0.0),
            low_load_time_s=state.low_load_time_s
            + (
                dt_s
                if occupied_mode is ElectrolyserMode.RUNNING
                and h2_kg > 0.0
                and load_frac <= profile.low_load_threshold_frac + 1e-12
                else 0.0
            ),
            ramp_limited_intervals=state.ramp_limited_intervals + int(ramp_limited),
            absolute_load_movement_kw=state.absolute_load_movement_kw
            + abs(accepted_kw - previous_kw),
        )
        self._commitment_state = state
        self.power_kw = accepted_kw
        # Existing ``online`` means saleable-H2 production, not merely powered.
        self.online = occupied_mode is ElectrolyserMode.RUNNING and h2_kg > 0.0
        return ElectrolyserCommitmentStep(
            accepted_power_kw=accepted_kw,
            h2_produced_kg=h2_kg,
        )

    def snapshot(self) -> dict[str, Any]:
        result = {
            "power_kw": self.power_kw,
            "load_frac": self.load_frac,
            "online": self.online,
            "rated_kw": self.config.rated_kw,
            "efficiency": self.efficiency,
            "temperature_c": self.temperature_c,
            **self.health_snapshot(),
        }
        if self.commitment_profile is not None:
            commitment = self.commitment_snapshot
            assert commitment is not None
            result["specific_energy_multiplier"] = self.specific_energy_multiplier
            result["specific_energy_kwh_per_kg"] = self.specific_energy_kwh_per_kg
            result["commitment"] = commitment.to_dict()
        return result
