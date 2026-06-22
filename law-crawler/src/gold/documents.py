"""Gold layer: Flattened law documents — ready for RAG embedding.

Joins silver Pháp Điển tables into a single denormalized view.
Each row is one article (điều) with full hierarchical context:
  Chủ đề → Đề mục → Chương → Điều

Also produces enriched VBQPPL document views.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.settings import GOLD, SILVER_PHAP_DIEN, SILVER_VBQPPL, setup_logging

logger = setup_logging(__name__)


def _read_silver(name: str) -> pd.DataFrame:
    path = SILVER_PHAP_DIEN / f"{name}.parquet"
    if not path.exists():
        logger.warning("Silver file not found: %s", path)
        return pd.DataFrame()
    return pd.read_parquet(path)


def build_phap_dien_documents() -> pd.DataFrame:
    df_chude = _read_silver("chude")
    df_demuc = _read_silver("demuc")
    df_chuong = _read_silver("chuong")
    df_dieu = _read_silver("dieu")

    if df_dieu.empty:
        logger.warning("No điều data — skipping gold Pháp Điển")
        return pd.DataFrame()

    # Resolve Peewee FK naming: silver may store FK values with _id suffix
    demuc_fk = "demuc_id" if "demuc_id" in df_demuc.columns else "demuc_id_id"
    chuong_fk = "chuong_id" if "chuong_id" in df_chuong.columns else "chuong_id_id"

    # Join: dieu → chuong → demuc → chude
    if not df_chuong.empty:
        df_dieu = df_dieu.merge(
            df_chuong[["mapc", "ten", "demuc_id"]].rename(columns={"ten": "chuong_ten", "demuc_id": "_chuong_demuc"}),
            left_on="chuong_id", right_on="mapc", how="left", suffixes=("", "_chuong"),
        )
        df_dieu["chuong_ten"] = df_dieu["chuong_ten"].fillna("")
        demuc_join_col = "_chuong_demuc"
        has_demuc_link = not df_dieu["_chuong_demuc"].isna().all()
    else:
        demuc_join_col = "demuc_id"
        has_demuc_link = True
        df_dieu["chuong_ten"] = ""

    if not df_demuc.empty and has_demuc_link:
        df_dieu = df_dieu.merge(
            df_demuc[["id", "ten", "chude_id"]].rename(columns={"ten": "demuc_ten"}),
            left_on=demuc_join_col, right_on="id", how="left", suffixes=("", "_demuc"),
        )
        df_dieu["demuc_ten"] = df_dieu["demuc_ten"].fillna("")
    else:
        df_dieu["demuc_ten"] = ""

    if not df_chude.empty and "chude_id" in df_dieu.columns:
        df_dieu = df_dieu.merge(
            df_chude[["id", "ten"]].rename(columns={"ten": "chude_ten"}),
            left_on="chude_id", right_on="id", how="left", suffixes=("", "_chude"),
        )
        df_dieu["chude_ten"] = df_dieu["chude_ten"].fillna("")
    else:
        df_dieu["chude_ten"] = ""

    # Build a rich text field with full context for embedding
    df_dieu["full_context"] = df_dieu.apply(
        lambda row: (
            f"Chủ đề: {row.get('chude_ten', '')}\n"
            f"Đề mục: {row.get('demuc_ten', '')}\n"
            f"Chương: {row.get('chuong_ten', '')}\n"
            f"Điều: {row.get('ten', '')}\n"
            + (f"VBQPPL: {row.get('vbqppl', '')}\n" if row.get("vbqppl") else "")
            + f"{row.get('noidung', '')}"
        ),
        axis=1,
    )

    output_cols = [
        "mapc", "ten", "noidung", "vbqppl", "vbqppl_link",
        "chuong_ten", "demuc_ten", "chude_ten", "full_context",
    ]
    result = df_dieu[[c for c in output_cols if c in df_dieu.columns]]
    return result.reset_index(drop=True)


def build_vbqppl_chimuc() -> pd.DataFrame:
    path = SILVER_VBQPPL / "vb_chimuc.parquet"
    if not path.exists():
        logger.warning("Silver vb_chimuc not found")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    logger.info("Gold VBQPPL: %d rows from silver", len(df))
    return df


def main() -> None:
    logger.info("=== Gold: Flattened documents ===")

    df_pd = build_phap_dien_documents()
    if not df_pd.empty:
        path = GOLD / "law_documents.parquet"
        df_pd.to_parquet(path, index=False)
        logger.info("Gold law_documents: %d articles written to %s", len(df_pd), path)
    else:
        logger.warning("No Pháp Điển gold documents produced")

    df_vb = build_vbqppl_chimuc()
    if not df_vb.empty:
        path = GOLD / "vbqppl_documents.parquet"
        df_vb.to_parquet(path, index=False)
        logger.info("Gold vbqppl_documents: %d rows written to %s", len(df_vb), path)

    logger.info("Gold layer done")


if __name__ == "__main__":
    main()
