from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from .backend import ModelBackend
from .models import GenerationResponse


@dataclass
class _Pending:
    prompt: str
    max_tokens: int
    enqueued_at: float
    future: asyncio.Future[GenerationResponse]


class DynamicBatcher:
    def __init__(
        self,
        backend: ModelBackend,
        max_batch_size: int = 8,
        max_wait_ms: float = 8,
    ) -> None:
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def submit(self, prompt: str, max_tokens: int) -> GenerationResponse:
        await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[GenerationResponse] = loop.create_future()
        await self.queue.put(_Pending(prompt, max_tokens, time.perf_counter(), future))
        return await future

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            deadline = time.perf_counter() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                except asyncio.TimeoutError:  # an alias of TimeoutError from 3.11
                    break
            started = time.perf_counter()
            try:
                outputs = await self.backend.generate_batch(
                    [item.prompt for item in batch],
                    max(item.max_tokens for item in batch),
                )
                finished = time.perf_counter()
                for item, output in zip(batch, outputs, strict=True):
                    if not item.future.cancelled():
                        item.future.set_result(
                            GenerationResponse(
                                text=output,
                                cached=False,
                                queue_ms=(started - item.enqueued_at) * 1000,
                                inference_ms=(finished - started) * 1000,
                                batch_size=len(batch),
                            )
                        )
            except Exception as error:
                for item in batch:
                    if not item.future.cancelled():
                        item.future.set_exception(error)
            finally:
                for _ in batch:
                    self.queue.task_done()
