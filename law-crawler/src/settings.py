"""Shared settings and paths for the law-crawler medallion pipeline."""
import logging
import os
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def setup_logging(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or __name__)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
METRICS = ROOT / "metrics"
PHAP_DIEN_DIR = ROOT / "phap-dien"

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
REQUEST_TIMEOUT = 10
SAVE_EVERY = 10
MAX_RETRIES = 3

# Checkpoint for Pháp Điển crawl
CHECKPOINT = os.getenv("LAW_CHECKPOINT", "")

# Chunk settings for RAG
CHUNK_SIZE = int(os.getenv("LAW_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("LAW_CHUNK_OVERLAP", "200"))
