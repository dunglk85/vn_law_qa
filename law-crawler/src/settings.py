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
REQUEST_TIMEOUT = int(os.getenv("LAW_REQUEST_TIMEOUT", _params.get("request_timeout", 10)))
SAVE_EVERY = int(os.getenv("LAW_SAVE_EVERY", _params.get("save_every", 10)))
MAX_RETRIES = int(os.getenv("LAW_MAX_RETRIES", _params.get("max_retries", 3)))

# Checkpoint for Pháp Điển crawl
CHECKPOINT = os.getenv("LAW_CHECKPOINT", _params.get("checkpoint", ""))

# Chunk settings for RAG
CHUNK_SIZE = int(os.getenv("LAW_CHUNK_SIZE", _params.get("chunk_size", 1000)))
CHUNK_OVERLAP = int(os.getenv("LAW_CHUNK_OVERLAP", _params.get("chunk_overlap", 200)))

# Polite crawl delay between requests (seconds)
# Configurable to respect server rate limits without banning the crawler.
CRAWL_DELAY = float(os.getenv("LAW_CRAWL_DELAY", _params.get("crawl_delay", 0.5)))

# User-Agent for HTTP requests
USER_AGENT = os.getenv("LAW_CRAWLER_USER_AGENT", _params.get("user_agent", "law-crawler/1.0 (research project; contact@example.com)"))
