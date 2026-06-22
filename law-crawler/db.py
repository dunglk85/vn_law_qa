"""Database configuration for law-crawler.

Uses MySQL with environment variable configuration.
"""
import logging
import os

from peewee import MySQLDatabase


def setup_logging(name: str | None = None) -> logging.Logger:
    """Configure and return a logger with consistent formatting."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return logging.getLogger(name or __name__)


_DEFAULT_PORT = 3306


def _parse_port(value: str) -> int:
    """Parse port string, falling back to default on invalid input."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return _DEFAULT_PORT


def mysql_config() -> dict:
    """Read MySQL connection parameters from environment variables."""
    return {
        "user": os.getenv("LAW_DB_USER", "root"),
        "password": os.getenv("LAW_DB_PASSWORD", ""),
        "host": os.getenv("LAW_DB_HOST", "localhost"),
        "port": _parse_port(os.getenv("LAW_DB_PORT", "3306")),
        "database": os.getenv("LAW_DB_NAME", "law"),
    }


def get_connection_url() -> str:
    """Get MySQL connection URL from environment (lazy)."""
    cfg = mysql_config()
    return f"mysql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"


def get_peewee_db() -> MySQLDatabase:
    """Get a Peewee MySQLDatabase instance from current environment (lazy)."""
    cfg = mysql_config()
    return MySQLDatabase(
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=cfg["port"],
    )


# Module-level Peewee instance (created from env vars at import time).
# Uses parameterized constructor — password is never stored as a plaintext URL.
_cfg = mysql_config()
db = MySQLDatabase(
    database=_cfg["database"],
    user=_cfg["user"],
    password=_cfg["password"],
    host=_cfg["host"],
    port=_cfg["port"],
)


def connect_db() -> None:
    """Connect to database with logging."""
    logger = logging.getLogger(__name__)
    try:
        db.connect()
        logger.info(
            "Connected to MySQL at %s:%s/%s",
            _cfg["host"],
            _cfg["port"],
            _cfg["database"],
        )
    except Exception as exc:
        logger.error("Failed to connect to MySQL: %s", exc)
        raise


def close_db() -> None:
    """Close database connection with logging."""
    logger = logging.getLogger(__name__)
    if not db.is_closed():
        db.close()
        logger.info("Database connection closed")


get_db = get_peewee_db  # backward-compatible alias


def get_sqlalchemy_engine():
    """Get a SQLAlchemy engine configured from environment variables.

    Returns the engine unconnected; connection happens lazily on first use.
    """
    from sqlalchemy import create_engine

    cfg = mysql_config()
    url = (
        f"mysql+mysqlconnector://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url)
