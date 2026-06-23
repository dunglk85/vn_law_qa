"""Shared settings and paths for the law-crawler medallion pipeline."""
import logging
import os
from pathlib import Path

import yaml


def setup_logging(name: str | None = None) -> logging.Logger:
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return logging.getLogger(name or __name__)


def _env_int(env_key: str, default: int) -> int:
    raw = os.getenv(env_key)
    if raw is None:
        return default
    try:
        val = int(raw)
    except (ValueError, TypeError):
        raise ValueError(
            f"Environment variable {env_key} must be an integer, got {raw!r}"
        ) from None
    return val


def _env_positive_int(env_key: str, default: int) -> int:
    val = _env_int(env_key, default)
    if val < 1:
        raise ValueError(
            f"Environment variable {env_key} must be >= 1, got {val}"
        )
    return val


def _env_float(env_key: str, default: float) -> float:
    raw = os.getenv(env_key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        raise ValueError(
            f"Environment variable {env_key} must be a number, got {raw!r}"
        ) from None

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
METRICS = ROOT / "metrics"
PHAP_DIEN_DIR = ROOT / "phap-dien"

_params_path = ROOT / "params.yaml"
_params = (yaml.safe_load(_params_path.read_text()) if _params_path.exists() else {}) or {}

# Bronze layer — raw ingested data
BRONZE = DATA / "bronze"
BRONZE_PHAP_DIEN = BRONZE / "phap_dien"
BRONZE_VBQPPL = BRONZE / "vbqppl"

# Silver layer — cleaned & validated data
SILVER = DATA / "silver"
SILVER_PHAP_DIEN = SILVER / "phap_dien"
SILVER_VBQPPL = SILVER / "vbqppl"

# Gold layer — business-ready data for RAG consumption
GOLD = DATA / "gold"

# VBQPPL web crawl settings
VBPL_BASE_URL = "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx"
REQUEST_TIMEOUT = _env_int("LAW_REQUEST_TIMEOUT", _params.get("request_timeout", 10))
SAVE_EVERY = _env_positive_int("LAW_SAVE_EVERY", _params.get("save_every", 10))
MAX_RETRIES = _env_positive_int("LAW_MAX_RETRIES", _params.get("max_retries", 3))

# Checkpoint for Pháp Điển crawl
CHECKPOINT = os.getenv("LAW_CHECKPOINT", _params.get("checkpoint", ""))

# Chunk settings for RAG
CHUNK_SIZE = _env_int("LAW_CHUNK_SIZE", _params.get("chunk_size", 1000))
CHUNK_OVERLAP = _env_int("LAW_CHUNK_OVERLAP", _params.get("chunk_overlap", 200))

# Polite crawl delay between requests (seconds)
# Configurable to respect server rate limits without banning the crawler.
CRAWL_DELAY = _env_float("LAW_CRAWL_DELAY", _params.get("crawl_delay", 0.5))

# User-Agent for HTTP requests
USER_AGENT = os.getenv("LAW_CRAWLER_USER_AGENT", _params.get("user_agent", "law-crawler/1.0 (research project; contact@example.com)"))
