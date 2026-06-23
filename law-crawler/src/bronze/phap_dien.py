"""Bronze layer: Pháp Điển crawler — raw HTML/JSON → Parquet.

Parses HTML law documents from the phap-dien/ directory and writes
raw structured data to Parquet files. No cleaning or validation —
that happens in the Silver layer.
"""
import json
import sys
import uuid
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from helper import convert_roman_to_num, extract_input
from src.settings import (
    BRONZE_PHAP_DIEN,
    CHECKPOINT,
    PHAP_DIEN_DIR,
    setup_logging,
)

logger = setup_logging(__name__)


def load_json(filename: str) -> list[dict]:
    filepath = PHAP_DIEN_DIR / filename
    logger.info("Loading %s", filepath)
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def ingest_chude() -> pd.DataFrame:
    chudes = load_json("chude.json")
    logger.info("Ingesting %d chủ đề", len(chudes))
    df = pd.DataFrame(chudes)
    df = df.rename(columns={"Text": "ten", "STT": "stt", "Value": "id"})
    return df[["id", "ten", "stt"]]


def ingest_demuc() -> pd.DataFrame:
    demucs = load_json("demuc.json")
    logger.info("Ingesting %d đề mục", len(demucs))
    df = pd.DataFrame(demucs)
    df = df.rename(columns={"Text": "ten", "STT": "stt", "Value": "id", "ChuDe": "chude_id"})
    return df[["id", "ten", "stt", "chude_id"]]


def ingest_nodes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ingest all tree nodes (chapters, articles, tables, files, cross-refs)."""
    tree_nodes = load_json("treeNode.json")
    logger.info("Ingesting tree nodes")

    demuc_dir = PHAP_DIEN_DIR / "demuc"
    all_chuong: list[dict] = []
    all_dieu: list[dict] = []
    all_tables: list[dict] = []
    all_files: list[dict] = []
    all_lienquan: list[dict] = []

    demuc_files = sorted(f for f in demuc_dir.iterdir() if f.suffix == ".html")
    total = len(demuc_files)
    is_skipping = bool(CHECKPOINT)

    if is_skipping:
        checkpoint_file = demuc_dir / CHECKPOINT
        if not checkpoint_file.exists():
            logger.warning("Checkpoint '%s' not found, starting from beginning", CHECKPOINT)
            is_skipping = False

    for idx, filepath in enumerate(demuc_files, 1):
        file_name = filepath.name
        if file_name == CHECKPOINT:
            is_skipping = False
        if is_skipping:
            continue

        logger.info("Processing [%d/%d] %s", idx, total, file_name)
        demuc_id = file_name.split(".")[0]
        demuc_nodes = [n for n in tree_nodes if n["DeMucID"] == demuc_id]
        if not demuc_nodes:
            continue

        with open(filepath, encoding="utf-8") as f:
            demuc_html = BeautifulSoup(f.read(), "html.parser")

        demuc_chuong = [n for n in demuc_nodes if n["TEN"].startswith("Chương ")]
        chuong_list: list[dict] = []

        for chuong in demuc_chuong:
            row = {
                "mapc": chuong["MAPC"],
                "ten": chuong["TEN"],
                "chimuc": chuong["ChiMuc"],
                "stt": convert_roman_to_num(chuong["ChiMuc"]),
                "demuc_id": chuong["DeMucID"],
            }
            all_chuong.append(row)
            chuong_list.append(row)

        if not chuong_list:
            chuong_list.append({
                "mapc": f"synthetic_{demuc_id}_{uuid.uuid4().hex[:8]}",
                "ten": "", "chimuc": "0",
                "stt": 0, "demuc_id": demuc_id,
            })

        demuc_dieus = [n for n in demuc_nodes if n not in demuc_chuong]
        stt = 0

        for dieu in demuc_dieus:
            if len(chuong_list) == 1:
                chuong_id = chuong_list[0]["mapc"]
            else:
                chuong_id = ""
                for c in chuong_list:
                    if dieu["MAPC"].startswith(c["mapc"]):
                        chuong_id = c["mapc"]
                        break

            mapc = dieu["MAPC"]
            dieu_el = demuc_html.select(f'a[name="{mapc}"]')
            if not dieu_el:
                continue
            dieu_el = dieu_el[0]

            ten = str(dieu_el.nextSibling).strip() if dieu_el.nextSibling else ""
            ghi_chu_html = dieu_el.parent.nextSibling
            vbqppl = ghi_chu_html.text.strip() if ghi_chu_html else None
            vbqppl_link = (
                ghi_chu_html.select("a")[0].get("href")  # I3 fix: .get() avoids KeyError when href is absent
                if ghi_chu_html and ghi_chu_html.select("a")
                else None
            )

            noidung_html = dieu_el.parent.find_next("p", {"class": "pNoiDung"})
            if not noidung_html:
                continue

            noidung_parts: list[str] = []
            for content in noidung_html.contents:
                if content.name == "table":
                    all_tables.append({"dieu_id": mapc, "html": str(content)})
                    continue
                text = content.text.strip() if hasattr(content, "text") else str(content).strip()
                if text:
                    noidung_parts.append(text)

            all_dieu.append({
                "mapc": mapc,
                "ten": ten,
                "chimuc": dieu.get("ChiMuc", ""),
                "stt": stt,
                "noidung": "\n".join(noidung_parts),
                "vbqppl": vbqppl,
                "vbqppl_link": vbqppl_link,
                "demuc_id": dieu.get("DeMucID", ""),
                "chuong_id": chuong_id,
            })

            element = noidung_html.nextSibling
            while element and element.name == "a":
                link = element.get("href")
                if link:
                    all_files.append({"dieu_id": dieu["MAPC"], "link": link, "path": ""})
                element = element.nextSibling

            if (
                element
                and element.name == "p"
                and element.get("class")
                and element["class"][0] == "pChiDan"
            ):
                for lq in element.select("a"):
                    if "onclick" not in lq.attrs or not lq["onclick"]:
                        continue
                    raw = extract_input(lq["onclick"])
                    if raw is None:
                        continue
                    all_lienquan.append({
                        "dieu_id1": dieu["MAPC"],
                        "dieu_id2": raw.replace("'", ""),
                    })

            stt += 1

    return (
        pd.DataFrame(all_chuong),
        pd.DataFrame(all_dieu),
        pd.DataFrame(all_tables),
        pd.DataFrame(all_files),
        pd.DataFrame(all_lienquan),
    )


def main() -> None:
    logger.info("=== Bronze: Pháp Điển ingestion ===")

    BRONZE_PHAP_DIEN.mkdir(parents=True, exist_ok=True)

    df_chude = ingest_chude()
    df_chude.to_parquet(BRONZE_PHAP_DIEN / "chude.parquet", index=False)
    logger.info("Wrote %d chủ đề to bronze", len(df_chude))

    df_demuc = ingest_demuc()
    df_demuc.to_parquet(BRONZE_PHAP_DIEN / "demuc.parquet", index=False)
    logger.info("Wrote %d đề mục to bronze", len(df_demuc))

    df_chuong, df_dieu, df_tables, df_files, df_lienquan = ingest_nodes()
    df_chuong.to_parquet(BRONZE_PHAP_DIEN / "chuong.parquet", index=False)
    df_dieu.to_parquet(BRONZE_PHAP_DIEN / "dieu.parquet", index=False)
    df_tables.to_parquet(BRONZE_PHAP_DIEN / "table.parquet", index=False)
    df_files.to_parquet(BRONZE_PHAP_DIEN / "file.parquet", index=False)
    df_lienquan.to_parquet(BRONZE_PHAP_DIEN / "muclienquan.parquet", index=False)

    logger.info(
        "Bronze Pháp Điển done: %d chương, %d điều, %d tables, %d files, %d liên quan",
        len(df_chuong), len(df_dieu), len(df_tables), len(df_files), len(df_lienquan),
    )


if __name__ == "__main__":
    main()
