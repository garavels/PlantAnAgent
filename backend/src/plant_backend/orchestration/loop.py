"""Wall-clock pacing for the runtime.

The clock is the only asynchronous thing in the backend: every `TICK_MS` it asks
the runtime to advance `speed` timesteps and publishes the result. While paused
it advances nothing -- simulated time stops, exactly as it did when the browser
owned the clock.

Simulated time therefore depends only on how many ticks were taken, never on how
long the work took, so a slow machine produces the same trace as a fast one.
"""

from __future__ import annotations

import asyncio
import contextlib

from .broadcast import Broadcaster
from .runtime import TICK_MS, PlantRuntime

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
