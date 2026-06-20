from __future__ import annotations

import asyncio
import logging
import random

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    coro_factory,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    desc: str = "operation",
) -> any:
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.1)
                total_delay = delay + jitter
                logger.warning(
                    "%s attempt %d/%d failed: %s. Retrying in %.2fs",
                    desc, attempt + 1, max_attempts, exc, total_delay,
                )
                await asyncio.sleep(total_delay)
    logger.error("%s failed after %d attempts: %s", desc, max_attempts, last_exc)
    raise last_exc
