"""Crawler for Pháp Điển Việt Nam (Vietnamese Law System).

Parses HTML law documents from phap-dien/ directory and stores
structured data in MySQL via Peewee ORM.
"""
import json
import logging
import os
import uuid
from pathlib import Path

from bs4 import BeautifulSoup
from peewee import IntegrityError

from db import db
from helper import convert_roman_to_num, extract_input
from models.models import (
    PDChuDe,
    PDChuong,
    PDDeMuc,
    PDDieu,
    PDFile,
    PDMucLienQuan,
    PDTable,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PHAP_DIEN_DIR = Path(__file__).parent / "phap-dien"
CHECKPOINT = os.getenv("LAW_CHECKPOINT", "")


def load_json(filename: str) -> list[dict]:
    """Load a JSON file from phap-dien directory."""
    filepath = PHAP_DIEN_DIR / filename
    logger.info("Loading %s", filepath)
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def insert_chude() -> None:
    """Insert all subjects (Chủ đề) from chude.json."""
    chudes = load_json("chude.json")
    logger.info("Inserting %d chủ đề", len(chudes))
    try:
        with db.atomic():
            PDChuDe.bulk_create(
                [PDChuDe(ten=c["Text"], stt=c["STT"], id=c["Value"]) for c in chudes]
            )
        logger.info("Inserted tất cả chủ đề")
    except IntegrityError as exc:
        logger.warning("Chủ đề already loaded or conflict: %s", exc)


def insert_demuc() -> None:
    """Insert all table of contents (Đề mục) from demuc.json."""
    demucs = load_json("demuc.json")
    logger.info("Inserting %d đề mục", len(demucs))
    try:
        with db.atomic():
            PDDeMuc.bulk_create(
                [
                    PDDeMuc(ten=d["Text"], stt=d["STT"], id=d["Value"], chude_id=d["ChuDe"])
                    for d in demucs
                ]
            )
    except IntegrityError as exc:
        logger.warning("Đề mục already loaded or conflict: %s", exc)


def process_demuc_file(
    file_name: str,
    tree_nodes: list[dict],
) -> list[dict]:
    """Process a single đề mục HTML file.

    Parses chapters, articles, tables, and cross-references.

    Args:
        file_name: Name of the HTML file to process.
        tree_nodes: List of tree node dictionaries.

    Returns:
        List of cross-reference dicts to insert later.
    """
    demuc_path = PHAP_DIEN_DIR / "demuc" / file_name
    demuc_id = file_name.split(".")[0]
    dieus_lienquan: list[dict] = []

    with open(demuc_path, encoding="utf-8") as f:
        demuc_html = BeautifulSoup(f.read(), "html.parser")

    demuc_nodes = [n for n in tree_nodes if n["DeMucID"] == demuc_id]
    if not demuc_nodes:
        logger.warning("Không tìm thấy node cho đề mục: %s", file_name)
        return dieus_lienquan

    # Process chapters
    demuc_chuong = [n for n in demuc_nodes if n["TEN"].startswith("Chương ")]
    chuongs_data: list[PDChuong] = []
    for chuong in demuc_chuong:
        mapc = chuong["MAPC"]
        stt = convert_roman_to_num(chuong["ChiMuc"])
        try:
            PDChuong.create(
                ten=chuong["TEN"],
                mapc=mapc,
                chimuc=chuong["ChiMuc"],
                stt=stt,
                demuc_id=chuong["DeMucID"],
            )
        except IntegrityError:
            continue
        chuongs_data.append(PDChuong(
            ten=chuong["TEN"],
            mapc=mapc,
            chimuc=chuong["ChiMuc"],
            stt=stt,
            demuc_id=chuong["DeMucID"],
        ))

    logger.info("Insert %d chương của đề mục %s", len(demuc_chuong), file_name)

    # Create placeholder chapter if none exist
    if not chuongs_data:
        chuong_data = PDChuong(
            ten="",
            mapc=str(uuid.uuid4()),
            chimuc="0",
            stt=0,
            demuc_id=demuc_id,
        )
        chuongs_data.append(chuong_data)

    # Process articles
    demuc_dieus = [n for n in demuc_nodes if n not in demuc_chuong]
    logger.info(
        "Đề mục %s có %d chương và %d điều",
        file_name,
        len(demuc_chuong),
        len(demuc_dieus),
    )

    stt = 0
    for dieu in demuc_dieus:
        # Assign chapter
        if len(chuongs_data) == 1:
            dieu["ChuongID"] = chuongs_data[0].mapc
        else:
            for chuong in chuongs_data:
                if dieu["MAPC"].startswith(chuong.mapc):
                    dieu["ChuongID"] = chuong.mapc
                    break

        mapc = dieu["MAPC"]
        dieu_html = demuc_html.select(f'a[name="{mapc}"]')
        if not dieu_html:
            continue
        dieu_html = dieu_html[0]

        ten = dieu_html.nextSibling
        ghi_chu_html = dieu_html.parent.nextSibling
        vbqppl = ghi_chu_html.text if ghi_chu_html else None
        vbqppl_link = (
            ghi_chu_html.select("a")[0]["href"]
            if ghi_chu_html and ghi_chu_html.select("a")
            else None
        )

        noidung_html = dieu_html.parent.find_next("p", {"class": "pNoiDung"})
        noidung = ""
        tables: list[str] = []
        for content in noidung_html.contents:
            if content.name == "table":
                tables.append(str(content))
                continue
            noidung += str(content.text.strip()) + "\n"

        try:
            PDDieu.create(
                ten=ten,
                mapc=mapc,
                chimuc=dieu["ChiMuc"],
                stt=stt,
                noidung=noidung,
                vbqppl=vbqppl,
                vbqppl_link=vbqppl_link,
                demuc_id=dieu["DeMucID"],
                chuong_id=dieu["ChuongID"],
            )
        except IntegrityError:
            continue

        for table in tables:
            try:
                PDTable.create(dieu_id=mapc, html=table)
            except IntegrityError:
                pass

        # Extract attached files
        element = noidung_html.nextSibling
        while element and element.name == "a":
            link = element.get("href")
            if link:
                try:
                    PDFile.create(dieu_id=dieu["MAPC"], link=link, path="")
                except IntegrityError:
                    pass
            element = element.nextSibling

        # Extract cross-references
        if (
            element
            and element.name == "p"
            and element.get("class")
            and element["class"][0] == "pChiDan"
        ):
            lienquans_html = element.select("a")
            for lienquan_html in lienquans_html:
                if "onclick" not in lienquan_html.attrs or not lienquan_html["onclick"]:
                    continue
                raw = extract_input(lienquan_html["onclick"])
                if raw is None:
                    continue
                mapc_lienquan = raw.replace("'", "")
                dieus_lienquan.append(
                    {"dieu_id1": dieu["MAPC"], "dieu_id2": mapc_lienquan}
                )

        stt += 1

    return dieus_lienquan


def insert_nodes() -> None:
    """Insert all tree nodes (chapters, articles, files) from HTML files."""
    tree_nodes = load_json("treeNode.json")
    logger.info("Inserting tree nodes")

    demuc_dir = PHAP_DIEN_DIR / "demuc"
    all_dieus_lienquan: list[dict] = []

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
        with db.atomic():
            dieus_lienquan = process_demuc_file(file_name, tree_nodes)
        all_dieus_lienquan.extend(dieus_lienquan)

    # Insert cross-references
    for ref in all_dieus_lienquan:
        try:
            with db.atomic():
                PDMucLienQuan.create(
                    dieu_id1=ref["dieu_id1"], dieu_id2=ref["dieu_id2"]
                )
            logger.info(
                "Inserted liên quan %s - %s", ref["dieu_id1"], ref["dieu_id2"]
            )
        except IntegrityError:
            logger.warning(
                "Không thể insert liên quan %s - %s",
                ref["dieu_id1"],
                ref["dieu_id2"],
            )

    logger.info("Inserted tất cả nodes")


def main() -> None:
    """Run the Pháp Điển crawler."""
    logger.info("Starting Pháp Điển crawler")

    db.connect()
    try:
        insert_chude()
        insert_demuc()
        insert_nodes()
    finally:
        db.close()

    logger.info("Crawler finished")


if __name__ == "__main__":
    main()
