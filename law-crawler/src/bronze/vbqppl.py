"""Bronze layer: VBQPPL document crawler — web → raw Parquet.

Fetches full-text legal documents from vbpl.vn using ItemIDs
extracted from the Pháp Điển bronze layer. Stores raw HTML content
in Parquet.
"""
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.settings import (
    BRONZE_PHAP_DIEN,
    BRONZE_VBQPPL,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    SAVE_EVERY,
    VBPL_BASE_URL,
    setup_logging,
)

logger = setup_logging(__name__)


def get_item_id(url: str | None) -> str | None:
    if url is None:
        return None
    match = re.search(r"ItemID=(\d+)", url)
    return match.group(1) if match else None


_USER_AGENT = "law-crawler/1.0 (research project; contact@example.com)"


def fetch_document(item_id: str) -> str | None:
    url = f"{VBPL_BASE_URL}?ItemID={item_id}"
    headers = {"User-Agent": _USER_AGENT}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            div_text = soup.find_all("div", class_="fulltext")
            if not div_text:
                logger.warning("No fulltext div for ItemID %s", item_id)
                return None
            inner_divs = div_text[0].find_all("div")
            if len(inner_divs) < 2:
                logger.warning(
                    "Unexpected HTML structure for ItemID %s: %d inner divs",
                    item_id, len(inner_divs),
                )
                return None
            return str(inner_divs[1])
        except requests.RequestException as exc:
            logger.warning("Attempt %d/%d failed for ItemID %s: %s", attempt, MAX_RETRIES, item_id, exc)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                logger.error("All retries failed for ItemID %s", item_id)
                return None


def main() -> None:
    logger.info("=== Bronze: VBQPPL document crawl ===")

    BRONZE_VBQPPL.mkdir(parents=True, exist_ok=True)

    dieu_path = BRONZE_PHAP_DIEN / "dieu.parquet"
    if not dieu_path.exists():
        logger.warning("Pháp Điển bronze not found at %s — run bronze phap_dien first", dieu_path)
        return

    df_dieu = pd.read_parquet(dieu_path)
    links = df_dieu["vbqppl_link"].dropna().drop_duplicates()
    logger.info("Found %d distinct vbqppl links", len(links))

    item_ids = links.apply(get_item_id).dropna().drop_duplicates()
    logger.info("Found %d unique ItemIDs to crawl", len(item_ids))

    collected: list[dict] = []
    for i, item_id in enumerate(item_ids, 1):
        logger.info("[%d/%d] Fetching ItemID %s", i, len(item_ids), item_id)
        content = fetch_document(item_id)
        if content:
            collected.append({"id": item_id, "noidung": content})

        if i % SAVE_EVERY == 0 and collected:
            pd.DataFrame(collected).to_parquet(
                BRONZE_VBQPPL / "vbpl.parquet", index=False,
            )
            logger.info("Saved checkpoint at %d documents", len(collected))

        time.sleep(0.5)

    if collected:
        pd.DataFrame(collected).to_parquet(
            BRONZE_VBQPPL / "vbpl.parquet", index=False,
        )

    logger.info("Bronze VBQPPL done: %d documents crawled", len(collected))


if __name__ == "__main__":
    main()
