"""Silver layer: VBQPPL — split full-text documents into chapters/articles.

Reads raw VBQPPL HTML from bronze, parses into structured
chapters (Chương) and articles (Điều), writes to silver.
"""
import re
import sys
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.settings import BRONZE_VBQPPL, SILVER_VBQPPL, setup_logging

logger = setup_logging(__name__)

_CHAPTER_RE = re.compile(r"^Chương\b", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"^Điều\b", re.IGNORECASE)


def split_document(id_vb: str, contents: str, start_id: int) -> tuple[list[dict], int]:
    try:
        soup = BeautifulSoup(contents, "html.parser").find("div", id="toanvancontent")
        if soup is None:
            logger.warning("No #toanvancontent div in document %s", id_vb)
            return [], start_id
        texts = [p.get_text().replace("\n", "").lstrip() for p in soup.find_all("p")]
    except Exception as exc:
        logger.warning("Failed to parse document %s: %s", id_vb, exc)
        return [], start_id

    chi_muc: list[dict] = []
    current_id = start_id
    id_chuong = None
    control = 0
    text = ""

    def flush(current_text: str, old_control: int) -> None:
        nonlocal current_id
        if not current_text.strip():
            return
        chi_muc.append({
            "id_vb": id_vb,
            "id": current_id,
            "noi_dung": current_text,
            "chi_muc_cha": None if old_control == 1 else id_chuong,
        })
        current_id += 1

    for line in texts:
        if _CHAPTER_RE.match(line):
            if text:
                flush(text, control)
                text = ""
            id_chuong = current_id
            control = 1
        elif _ARTICLE_RE.match(line):
            if text:
                flush(text, control)
                text = ""
            control = 2

        if control in (1, 2):
            text += line + "\n"

    flush(text, control)
    return chi_muc, current_id


def main() -> None:
    logger.info("=== Silver: VBQPPL document splitting ===")

    vbpl_path = BRONZE_VBQPPL / "vbpl.parquet"
    if not vbpl_path.exists():
        logger.warning("VBQPPL bronze not found at %s", vbpl_path)
        return

    df = pd.read_parquet(vbpl_path)
    logger.info("Loaded %d documents from bronze", len(df))

    existing_path = SILVER_VBQPPL / "vb_chimuc.parquet"
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        current_id = int(existing["id"].max()) + 1
    else:
        current_id = 1
    logger.info("Starting split at id=%d", current_id)

    all_chi_muc: list[dict] = []
    for j in range(len(df)):
        id_vb = str(df.iloc[j]["id"])
        contents = str(df.iloc[j]["noidung"])
        logger.info("Splitting document %s", id_vb)
        records, current_id = split_document(id_vb, contents, current_id)
        all_chi_muc.extend(records)

    if all_chi_muc:
        df_out = pd.DataFrame(all_chi_muc)
        out_path = SILVER_VBQPPL / "vb_chimuc.parquet"
        tmp_path = out_path.with_suffix(".tmp")
        df_out.to_parquet(tmp_path, index=False)
        tmp_path.replace(out_path)
        logger.info("Saved %d split records", len(df_out))

    logger.info("Silver VBQPPL done")


if __name__ == "__main__":
    main()
