"""Bronze layer: VBQPPL document crawler — web → raw Parquet.

Fetches full-text legal documents from vbpl.vn using ItemIDs
extracted from the Pháp Điển bronze layer. Stores raw HTML content
in Parquet.
"""
import re
import time

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import InvalidSessionIdException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

from src.settings import (
    BRONZE_PHAP_DIEN,
    BRONZE_VBQPPL,
    CRAWL_DELAY,
    MAX_RETRIES,
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


def _create_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"""
    })
    return driver


def fetch_document(driver: webdriver.Chrome, item_id: str) -> tuple[str | None, webdriver.Chrome]:
    url = f"{VBPL_BASE_URL}/{item_id}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "main"))
            )
            time.sleep(1)
            soup = BeautifulSoup(driver.page_source, "lxml")

            main_tag = soup.select_one("main")
            if main_tag:
                main_text = main_tag.get_text(strip=True)
                if "Văn bản không tồn tại" in main_text or "404" in main_text[:100]:
                    logger.info("ItemID %s returned soft 404 (document no longer available), skipping", item_id)
                    return None, driver

            # Try to find document content by common selectors
            content = None
            for sel in ["#fulltext", ".fulltext", ".prov-content", '[class*="content"]', "main"]:
                el = soup.select_one(sel)
                if el and len(el.get_text(strip=True)) > 200:
                    content = str(el)
                    break
            if content:
                return content, driver
            logger.warning("No content found for ItemID %s", item_id)
            return None, driver

        except InvalidSessionIdException:
            logger.warning("Session invalid on attempt %d/%d for ItemID %s — recreating driver", attempt, MAX_RETRIES, item_id)
            driver.quit()
            driver = _create_driver()
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                logger.error("All retries failed for ItemID %s", item_id)
                return None, driver

        except Exception as exc:
            is_404 = "404" in str(exc)
            logger.warning("Attempt %d/%d failed for ItemID %s: %s", attempt, MAX_RETRIES, item_id, exc)
            if is_404:
                logger.info("ItemID %s returned 404 (document no longer available), skipping", item_id)
                return None, driver
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                logger.error("All retries failed for ItemID %s", item_id)
                return None, driver


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

    existing_path = BRONZE_VBQPPL / "vbpl.parquet"
    if existing_path.exists():
        existing_df = pd.read_parquet(existing_path)
        collected: list[dict] = existing_df.to_dict("records")
        already_fetched: set[str] = set(existing_df["id"].astype(str))
        logger.info("Resuming — %d documents already in parquet, skipping those IDs", len(collected))
    else:
        collected = []
        already_fetched = set()

    driver = _create_driver()
    try:
        new_count = 0
        for i, item_id in enumerate(item_ids, 1):
            if str(item_id) in already_fetched:
                continue

            logger.info("[%d/%d] Fetching ItemID %s", i, len(item_ids), item_id)
            content, driver = fetch_document(driver, item_id)
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

            time.sleep(CRAWL_DELAY)

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
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
