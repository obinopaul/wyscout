import logging
import os
from contextlib import AbstractAsyncContextManager
from typing import Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool

from backend.core.settings import settings

logger = logging.getLogger(__name__)

# Module-level variables for Telogical Postgres connections
_telogical_async_pool: Optional[AsyncConnectionPool] = None
_telogical_async_saver: Optional[AsyncPostgresSaver] = None


def validate_postgres_config() -> None:
    """
    Validate that all required PostgreSQL configuration is present.
    Raises ValueError if any required configuration is missing.
    """
    required_vars = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]

    missing = [var for var in required_vars if not getattr(settings, var, None)]
    if missing:
        raise ValueError(
            f"Missing required PostgreSQL configuration: {', '.join(missing)}. "
            "These environment variables must be set to use PostgreSQL persistence."
        )


def get_postgres_connection_string() -> str:
    """Build and return the PostgreSQL connection string from settings."""
    if settings.POSTGRES_PASSWORD is None:
        raise ValueError("POSTGRES_PASSWORD is not set")
    return (
        f"postgresql://{settings.POSTGRES_USER}:"
        f"{settings.POSTGRES_PASSWORD.get_secret_value()}@"
        f"{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/"
        f"{settings.POSTGRES_DB}"
    )


def get_postgres_saver() -> AbstractAsyncContextManager[AsyncPostgresSaver]:
    """Initialize and return a PostgreSQL saver instance."""
    validate_postgres_config()
    return AsyncPostgresSaver.from_conn_string(get_postgres_connection_string())


def get_postgres_store():
    """
    Get a PostgreSQL store instance.

    Returns an AsyncPostgresStore instance that needs to be used with async context manager
    pattern according to the documentation:

    async with AsyncPostgresStore.from_conn_string(conn_string) as store:
        await store.setup()  # Run migrations
        # Use store...
    """
    validate_postgres_config()
    connection_string = get_postgres_connection_string()
    return AsyncPostgresStore.from_conn_string(connection_string)


# Telogical-specific Postgres functions
async def get_telogical_postgres_saver() -> AsyncPostgresSaver:
    """
    Get a Postgres saver for Telogical agents.
    This uses environment variables specific to the Telogical setup.
    """
    global _telogical_async_pool, _telogical_async_saver
    
    if _telogical_async_saver is None:
        # Get connection info from environment variables
        db_host = os.getenv("DB_HOST", "c-telogical-postgresql-eus.7n7rqafletob6o.postgres.cosmos.azure.com")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "citus")
        db_user = os.getenv("DB_USER", "citus")
        db_password = os.getenv("DB_PASSWORD")
        
        if not db_password:
            raise ValueError("DB_PASSWORD environment variable is required for Telogical Postgres connection")
        
        db_uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=require"
        
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
        }
        
        _telogical_async_pool = AsyncConnectionPool(
            conninfo=db_uri,
            max_size=20,  # Adjust as needed
            kwargs=connection_kwargs,
        )
        
        _telogical_async_saver = AsyncPostgresSaver(_telogical_async_pool)
        await _telogical_async_saver.setup()  # Ensure tables are created
    
    return _telogical_async_saver
