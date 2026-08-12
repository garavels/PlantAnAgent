"""Fan-out of plant updates to every connected control room.

One queue per subscriber, bounded and lossy on purpose: a browser tab that
cannot keep up should skip frames, not slow the plant down or grow memory
without limit. Each payload is self-contained, so a dropped one costs the client
nothing but resolution.
"""

from __future__ import annotations

import asyncio
from typing import Any

#: Payloads buffered per subscriber before the oldest are dropped.
QUEUE_LIMIT = 32


class Broadcaster:
    """In-memory pub/sub. No broker, no persistence -- one process, one plant."""

    def __init__(self, limit: int = QUEUE_LIMIT) -> None:
        self.limit = limit
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    @property
    def subscribers(self) -> int:
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=self.limit)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def publish(self, payload: dict[str, Any]) -> None:
        """Hand `payload` to every subscriber, dropping the stalest if needed."""
        for queue in self._subscribers:
            while queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - race with a reader
                    break
            queue.put_nowait(payload)
