"""Database configuration for law-crawler.

Uses MySQL with environment variable configuration.
"""
import logging
import os

from peewee import MySQLDatabase

logger = logging.getLogger(__name__)

_DB_USER = os.getenv("LAW_DB_USER", "root")
_DB_PASS = os.getenv("LAW_DB_PASSWORD", "")
_DB_HOST = os.getenv("LAW_DB_HOST", "localhost")
_DB_PORT = int(os.getenv("LAW_DB_PORT", "3306"))
_DB_NAME = os.getenv("LAW_DB_NAME", "law")

DATABASE = f"mysql://{_DB_USER}:{_DB_PASS}@{_DB_HOST}:{_DB_PORT}/{_DB_NAME}"

db = MySQLDatabase(
    database=_DB_NAME,
    user=_DB_USER,
    password=_DB_PASS,
    host=_DB_HOST,
    port=_DB_PORT,
)


def connect_db() -> None:
    """Connect to database with logging."""
    try:
        db.connect()
        logger.info("Connected to MySQL at %s:%s/%s", _DB_HOST, _DB_PORT, _DB_NAME)
    except Exception as exc:
        logger.error("Failed to connect to MySQL: %s", exc)
        raise


def close_db() -> None:
    """Close database connection with logging."""
    if not db.is_closed():
        db.close()
        logger.info("Database connection closed")
