"""Split VBQPPL documents into structured chapters and articles.

Reads full-text HTML from MySQL, parses into chapters (Chương)
and articles (Điều), stores in vb_chimuc table.
"""
import logging
import sys
from pathlib import Path

import pandas as pd
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


def split_document(id_vb: str, contents: str, start_id: int) -> tuple[list[dict], int]:
    """Split a single document HTML into chapters and articles.

    Args:
        id_vb: Document identifier.
        contents: HTML content string.
        start_id: Starting ID for generated records.

    Returns:
        Tuple of (list of split records, next available ID).
    """
    try:
        soup = BeautifulSoup(contents, "html.parser").find("div", id="toanvancontent")
        texts = [p.get_text().replace("\n", "").lstrip() for p in soup.find_all("p")]
    except Exception as exc:
        logger.warning("Failed to parse document %s: %s", id_vb, exc)
        return [], start_id

    chi_muc: list[dict] = []
    current_id = start_id
    id_chuong = None
    control = 0
    text = ""

    def flush(current_text: str, old_control: int, new_control: int) -> None:
        """Flush accumulated text as a record."""
        nonlocal current_id
        if not current_text.strip():
            return
        record = {
            "id_vb": id_vb,
            "id": current_id,
            "noi_dung": current_text,
            "chi_muc_cha": None if old_control == 1 else id_chuong,
        }
        chi_muc.append(record)
        current_id += 1

    i = 0
    while i < len(texts):
        line = texts[i]

        if line.startswith("Chương") or line.startswith("CHƯƠNG"):
            if text:
                flush(text, control, 1)
                text = ""
            id_chuong = current_id
            control = 1
        elif line.startswith("Đi"):
            if text:
                flush(text, control, 2)
                text = ""
            control = 2

        if control in (1, 2):
            text += line + "\n"

        i += 1

    flush(text, control, 2)
    return chi_muc, current_id


def main() -> None:
    """Run the document splitter."""
    logger.info("Starting document splitter")

    df = pd.read_sql("SELECT id, noidung FROM vbpl;", con=engine)
    logger.info("Found %d documents to split", len(df))

    all_chi_muc: list[dict] = []
    current_id = 3012

    for j in range(len(df)):
        id_vb = df.iloc[j]["id"]
        contents = df.iloc[j]["noidung"]
        logger.info("Splitting document %s", id_vb)

        records, current_id = split_document(id_vb, contents, current_id)
        all_chi_muc.extend(records)

    if all_chi_muc:
        df_to_write = pd.DataFrame(all_chi_muc)
        df_to_write.to_sql("vb_chimuc", con=engine, if_exists="append", index=False)
        logger.info("Saved %d split records", len(all_chi_muc))

    logger.info("Document splitter finished")


if __name__ == "__main__":
    main()
