"""Wall-clock pacing for the runtime.

The clock is the only asynchronous thing in the backend: every `TICK_MS` it asks
the runtime to advance `speed` timesteps and publishes the result. While paused
it advances nothing -- simulated time stops, exactly as it did when the browser
owned the clock.

Simulated time therefore depends only on how many ticks were taken, never on how
long the work took, so a slow machine produces the same trace as a fast one.

The supervisory agent is offered each batch of frames after they are applied, and
decides for itself whether to investigate (usually not). It thinks in a background
task, so a slow or failing model delays nothing here: the tick that woke it has
already been published by the time it has an opinion.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from .broadcast import Broadcaster
from .runtime import TICK_MS, PlantRuntime

log = logging.getLogger(__name__)

#: How often a paused clock re-checks whether it should be running.
IDLE_POLL_S = 0.05


class SimulationClock:
    """Drives `PlantRuntime.advance` on a fixed wall-clock cadence."""

    def __init__(
        self,
        runtime: PlantRuntime,
        broadcast: Broadcaster,
        tick_s: float = TICK_MS / 1000.0,
    ) -> None:
        self.runtime = runtime
        self.broadcast = broadcast
        self.tick_s = tick_s
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if not self.running:
            self._task = asyncio.create_task(self._run(), name="plant-clock")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        while True:
            if not self.runtime.playing:
                await asyncio.sleep(IDLE_POLL_S)
                continue
            await asyncio.sleep(self.tick_s)
            if not self.runtime.playing:  # paused while we were sleeping
                continue
            frames = self.runtime.advance(self.runtime.speed)
            self.broadcast.publish(self.runtime.payload(frames))
            self._supervise(frames)

    def _supervise(self, frames: list[dict[str, Any]]) -> None:
        """Offer the batch to the supervisor. The plant outranks it: never raises."""
        try:
            self.runtime.supervisor.observe(frames)
        except Exception:  # noqa: BLE001 - a broken supervisor must not stop the clock
            log.warning("supervisor could not evaluate this tick", exc_info=True)
