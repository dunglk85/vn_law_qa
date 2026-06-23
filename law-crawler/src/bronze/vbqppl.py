"""Bronze layer: VBQPPL document crawler — web → raw Parquet.

Fetches full-text legal documents from vbpl.vn using ItemIDs
extracted from the Pháp Điển bronze layer. Stores raw HTML content
in Parquet.
"""
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.settings import (
    BRONZE_PHAP_DIEN,
    BRONZE_VBQPPL,
    CRAWL_DELAY,
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    SAVE_EVERY,
    USER_AGENT,
    VBPL_BASE_URL,
    setup_logging,
)

logger = setup_logging(__name__)


def get_item_id(url: str | None) -> str | None:
    if url is None:
        return None
    match = re.search(r"ItemID=(\d+)", url)
    return match.group(1) if match else None


def fetch_document(item_id: str) -> str | None:
    url = f"{VBPL_BASE_URL}?ItemID={item_id}"
    headers = {"User-Agent": USER_AGENT}
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

    # ── C1 fix: resume from existing parquet ─────────────────────────────────
    # Load any previously crawled documents so a crashed run can continue
    # without re-fetching IDs that were already successfully retrieved.
    existing_path = BRONZE_VBQPPL / "vbpl.parquet"
    if existing_path.exists():
        existing_df = pd.read_parquet(existing_path)
        collected: list[dict] = existing_df.to_dict("records")
        already_fetched: set[str] = set(existing_df["id"].astype(str))
        logger.info("Resuming — %d documents already in parquet, skipping those IDs", len(collected))
    else:
        collected = []
        already_fetched = set()
    # ─────────────────────────────────────────────────────────────────────────

    new_count = 0
    for i, item_id in enumerate(item_ids, 1):
        if str(item_id) in already_fetched:
            logger.debug("[%d/%d] Skipping already-fetched ItemID %s", i, len(item_ids), item_id)
            continue

        logger.info("[%d/%d] Fetching ItemID %s", i, len(item_ids), item_id)
        content = fetch_document(item_id)
        if content:
            collected.append({"id": item_id, "noidung": content})
            already_fetched.add(str(item_id))
            new_count += 1

        if new_count > 0 and new_count % SAVE_EVERY == 0:
            df_save = pd.DataFrame(collected)
            tmp_path = existing_path.with_suffix(".tmp")
            df_save.to_parquet(tmp_path, index=False)
            try:
                tmp_path.replace(existing_path)
            except OSError:
                import shutil
                shutil.move(str(tmp_path), str(existing_path))
            logger.info("Saved checkpoint at %d total documents (%d new)", len(collected), new_count)

        time.sleep(CRAWL_DELAY)  # I4 fix: configurable via LAW_CRAWL_DELAY env / params.yaml crawl_delay

    if new_count > 0 and collected:
        df_save = pd.DataFrame(collected)
        tmp_path = existing_path.with_suffix(".tmp")
        df_save.to_parquet(tmp_path, index=False)
        try:
            tmp_path.replace(existing_path)
        except OSError:
            import shutil
            shutil.move(str(tmp_path), str(existing_path))

    logger.info("Bronze VBQPPL done: %d new documents crawled (%d total)", new_count, len(collected))


if __name__ == "__main__":
    main()
