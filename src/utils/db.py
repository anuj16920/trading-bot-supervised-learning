"""Database connection management for AQRF.

Async PostgreSQL pool with retry logic and connection health checks.
"""
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import asyncpg
import structlog

logger = structlog.get_logger(__name__)

# Connection pool singleton
_pool: Optional[asyncpg.Pool] = None


async def get_pool(
    dsn: Optional[str] = None,
    min_size: int = 2,
    max_size: int = 10,
) -> asyncpg.Pool:
    """Get or create asyncpg connection pool.

    Args:
        dsn: PostgreSQL connection string. If None, builds from env vars.
        min_size: Minimum connections in pool
        max_size: Maximum connections in pool

    Returns:
        asyncpg.Pool instance
    """
    global _pool

    if _pool is not None and not _pool._closing:
        return _pool

    if dsn is None:
        dsn = (
            f"postgresql://{os.getenv('DB_USER', 'aqrf')}:"
            f"{os.getenv('DB_PASSWORD', 'aqrf_secure_2024')}@"
            f"{os.getenv('DB_HOST', 'localhost')}:"
            f"{os.getenv('DB_PORT', '5432')}/"
            f"{os.getenv('DB_NAME', 'forex_data')}"
        )

    _pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=min_size,
        max_size=max_size,
        command_timeout=60,
        server_settings={
            "jit": "off",  # JIT can slow short queries
            "application_name": "aqrf",
        },
    )

    logger.info(
        "db_pool_created",
        min_size=min_size,
        max_size=max_size,
        host=os.getenv('DB_HOST', 'localhost'),
    )
    return _pool


@asynccontextmanager
async def get_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Get a connection from the pool with automatic cleanup."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            yield conn
        except asyncpg.PostgresError as e:
            logger.error("db_query_error", error=str(e))
            raise


async def close_pool() -> None:
    """Close the connection pool gracefully."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("db_pool_closed")


async def health_check() -> bool:
    """Check database connectivity.

    Returns:
        True if database is reachable
    """
    try:
        async with get_connection() as conn:
            result = await conn.fetchval("SELECT 1")
            return result == 1
    except Exception as e:
        logger.error("db_health_check_failed", error=str(e))
        return False
