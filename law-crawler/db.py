"""Database configuration for law-crawler.

Uses MySQL with environment variable configuration.
"""
import logging
import os

from peewee import MySQLDatabase

logger = logging.getLogger(__name__)


def mysql_config() -> dict:
    """Read MySQL connection parameters from environment variables.

    Returns:
        Dict with keys: user, password, host, port, database.
    """
    return {
        "user": os.getenv("LAW_DB_USER", "root"),
        "password": os.getenv("LAW_DB_PASSWORD", ""),
        "host": os.getenv("LAW_DB_HOST", "localhost"),
        "port": int(os.getenv("LAW_DB_PORT", "3306")),
        "database": os.getenv("LAW_DB_NAME", "law"),
    }


def get_connection_url() -> str:
    """Get MySQL connection URL from environment (lazy)."""
    cfg = mysql_config()
    return f"mysql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"


def get_db() -> MySQLDatabase:
    """Get a MySQL database instance from current environment (lazy)."""
    cfg = mysql_config()
    return MySQLDatabase(
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
        host=cfg["host"],
        port=cfg["port"],
    )


# Module-level instance for models (created at import time from env vars).
_db_config = mysql_config()
DATABASE = f"mysql://{_db_config['user']}:{_db_config['password']}@{_db_config['host']}:{_db_config['port']}/{_db_config['database']}"
db = MySQLDatabase(
    database=_db_config["database"],
    user=_db_config["user"],
    password=_db_config["password"],
    host=_db_config["host"],
    port=_db_config["port"],
)


def connect_db() -> None:
    """Connect to database with logging."""
    try:
        db.connect()
        logger.info(
            "Connected to MySQL at %s:%s/%s",
            _db_config["host"],
            _db_config["port"],
            _db_config["database"],
        )
    except Exception as exc:
        logger.error("Failed to connect to MySQL: %s", exc)
        raise


def close_db() -> None:
    """Close database connection with logging."""
    if not db.is_closed():
        db.close()
        logger.info("Database connection closed")
