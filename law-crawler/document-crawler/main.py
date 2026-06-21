"""Crawler for VBQPPL documents (văn bản quy phạm pháp luật).

Fetches legal document full text from vbpl.vn and stores in MySQL.
"""
import logging
import os
import re
import time
import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).parent.parent))
from db import mysql_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_cfg = mysql_config()
engine = create_engine(
    f"mysql+mysqlconnector://{_cfg['user']}:{_cfg['password']}@{_cfg['host']}:{_cfg['port']}/{_cfg['database']}"
)

VBPL_BASE_URL = "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx"
REQUEST_TIMEOUT = 10
SAVE_EVERY = 10


def get_item_id(url: str | None) -> str | None:
    """Extract ItemID from a vbpl.vn URL.

    Args:
        url: Full vbpl.vn URL.

    Returns:
        ItemID string, or None if not found.
    """
    if url is None:
        return None
    match = re.search(r"ItemID=(\d+).*#(.*)", url)
    return match.group(1) if match else None


def fetch_document(item_id: str) -> str | None:
    """Fetch full text of a legal document by ItemID.

    Args:
        item_id: The vbpl.vn ItemID.

    Returns:
        HTML content of the document, or None on failure.
    """
    url = f"{VBPL_BASE_URL}?ItemID={item_id}"
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        div_text = soup.find_all("div", class_="fulltext")
        if not div_text:
            logger.warning("No fulltext div for ItemID %s", item_id)
            return None
        noidung = div_text[0].find_all("div")[1]
        return str(noidung)
    except requests.RequestException as exc:
        logger.error("Failed to fetch ItemID %s: %s", item_id, exc)
        return None


def save_data(list_id: list[str], list_noidung: list[str]) -> None:
    """Save document IDs and content to MySQL.

    Args:
        list_id: List of ItemIDs.
        list_noidung: List of HTML content strings.
    """
    if not list_id:
        return
    df_to_write = pd.DataFrame({"id": list_id, "noidung": list_noidung})
    df_to_write.to_sql("vbpl", con=engine, if_exists="append", index=False)
    logger.info("Saved %d documents to database", len(list_id))


def main() -> None:
    """Run the VBQPPL document crawler."""
    logger.info("Starting VBQPPL document crawler")

    # Get distinct ItemIDs from existing articles
    df = pd.read_sql(
        "SELECT vbqppl_link FROM pddieu GROUP BY vbqppl_link;", con=engine
    )
    logger.info("Found %d distinct links", len(df))

    # Extract ItemIDs
    all_ids = [get_item_id(df.iloc[i]["vbqppl_link"]) for i in range(len(df))]
    df_ids = pd.DataFrame(all_ids).dropna().drop_duplicates()
    logger.info("Found %d unique ItemIDs to crawl", len(df_ids))

    batch_ids: list[str] = []
    batch_content: list[str] = []

    for i in range(len(df_ids)):
        item_id = df_ids.iloc[i][0]
        logger.info("[%d/%d] Fetching ItemID %s", i + 1, len(df_ids), item_id)

        content = fetch_document(item_id)
        if content:
            batch_ids.append(item_id)
            batch_content.append(content)

        if (i + 1) % SAVE_EVERY == 0:
            save_data(batch_ids, batch_content)
            batch_ids.clear()
            batch_content.clear()

        # Rate limiting
        time.sleep(0.5)

    save_data(batch_ids, batch_content)
    logger.info("VBQPPL crawler finished")


if __name__ == "__main__":
    main()
