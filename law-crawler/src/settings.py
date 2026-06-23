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

# Config resolution: params.yaml (DVC) → env var → hardcoded default
def _get_int(key: str, default: int) -> int:
    yaml_val = _params.get(key, default)
    return _env_int(f"LAW_{key.upper()}", yaml_val)

def _get_positive_int(key: str, default: int) -> int:
    yaml_val = _params.get(key, default)
    return _env_positive_int(f"LAW_{key.upper()}", yaml_val)

def _get_float(key: str, default: float) -> float:
    yaml_val = _params.get(key, default)
    return _env_float(f"LAW_{key.upper()}", yaml_val)

def _get_str(key: str, default: str) -> str:
    yaml_val = _params.get(key, default)
    return os.getenv(f"LAW_{key.upper()}", yaml_val)

REQUEST_TIMEOUT = _get_int("request_timeout", 10)
SAVE_EVERY = _get_positive_int("save_every", 10)
MAX_RETRIES = _get_positive_int("max_retries", 3)
CHECKPOINT = _get_str("checkpoint", "")
CHUNK_SIZE = _get_int("chunk_size", 1000)
CHUNK_OVERLAP = _get_int("chunk_overlap", 200)
CRAWL_DELAY = _get_float("crawl_delay", 0.5)
USER_AGENT = _get_str("user_agent", "law-crawler/1.0 (research project; contact@example.com)")
