"""Silver layer: Pháp Điển — clean, validate, deduplicate bronze → silver.

Reads raw Parquet from bronze, applies cleaning rules and schema
enforcement, then writes validated data to silver.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.settings import BRONZE_PHAP_DIEN, SILVER_PHAP_DIEN, setup_logging

logger = setup_logging(__name__)

# Canonical column names produced by the bronze layer.
# If the bronze schema ever changes, these assertions will fail fast
# at the top of the relevant clean_* function rather than silently
# using a wrong column name (M1 fix — removes the _id suffix heuristic).
_CHUONG_DEMUC_COL = "demuc_id"
_DIEU_FK_COLS = ["demuc_id", "chuong_id"]
_TABLE_DIEU_COL = "dieu_id"
_FILE_DIEU_COL = "dieu_id"


def _read_bronze(name: str) -> pd.DataFrame:
    path = BRONZE_PHAP_DIEN / f"{name}.parquet"
    if not path.exists():
        logger.warning("Bronze file not found: %s", path)
        return pd.DataFrame()
    return pd.read_parquet(path)


def clean_chude(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["id"])
    df["ten"] = df["ten"].str.strip()
    df["stt"] = pd.to_numeric(df["stt"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def clean_demuc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["id"])
    df["ten"] = df["ten"].str.strip()
    df["stt"] = pd.to_numeric(df["stt"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def clean_chuong(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["mapc"])
    df["ten"] = df["ten"].str.strip()
    df["chimuc"] = df["chimuc"].str.strip()
    df["stt"] = pd.to_numeric(df["stt"], errors="coerce").fillna(0).astype(int)
    # M1 fix: assert canonical column name rather than silently falling back
    if _CHUONG_DEMUC_COL not in df.columns:
        raise KeyError(
            f"Expected column {_CHUONG_DEMUC_COL!r} in chuong bronze data; "
            f"found: {list(df.columns)}"
        )
    df[_CHUONG_DEMUC_COL] = df[_CHUONG_DEMUC_COL].str.strip()
    return df.reset_index(drop=True)


def clean_dieu(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["mapc"])
    df["ten"] = df["ten"].str.strip()
    df["noidung"] = df["noidung"].str.strip()
    df["chimuc"] = pd.to_numeric(df["chimuc"], errors="coerce").fillna(0).astype(int)
    df["stt"] = pd.to_numeric(df["stt"], errors="coerce").fillna(0).astype(int)
    df["vbqppl"] = df["vbqppl"].fillna("").str.strip()
    # M1 fix: assert canonical FK column names rather than silently falling back
    for col in _DIEU_FK_COLS:
        if col not in df.columns:
            raise KeyError(
                f"Expected column {col!r} in dieu bronze data; "
                f"found: {list(df.columns)}"
            )
        df[col] = df[col].fillna("").str.strip()
    return df.reset_index(drop=True)


def clean_tables(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop_duplicates()
    df["html"] = df["html"].str.strip()
    # M1 fix: assert canonical column name
    if _TABLE_DIEU_COL not in df.columns:
        raise KeyError(
            f"Expected column {_TABLE_DIEU_COL!r} in tables bronze data; "
            f"found: {list(df.columns)}"
        )
    df[_TABLE_DIEU_COL] = df[_TABLE_DIEU_COL].str.strip()
    return df.reset_index(drop=True)


def clean_files(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop_duplicates()
    df["link"] = df["link"].str.strip()
    df["path"] = df["path"].fillna("")
    # M1 fix: assert canonical column name
    if _FILE_DIEU_COL not in df.columns:
        raise KeyError(
            f"Expected column {_FILE_DIEU_COL!r} in files bronze data; "
            f"found: {list(df.columns)}"
        )
    df[_FILE_DIEU_COL] = df[_FILE_DIEU_COL].str.strip()
    return df.reset_index(drop=True)


def clean_lienquan(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["dieu_id1", "dieu_id2"])
    df["dieu_id1"] = df["dieu_id1"].str.strip()
    df["dieu_id2"] = df["dieu_id2"].str.strip()
    return df.reset_index(drop=True)


def validate_cross_references(df_lienquan: pd.DataFrame, df_dieu: pd.DataFrame) -> dict:
    if df_lienquan.empty or df_dieu.empty:
        return {"total_refs": 0, "valid_refs": 0, "orphan_refs": 0}

    valid_mapc = set(df_dieu["mapc"].dropna())
    valid1 = df_lienquan["dieu_id1"].isin(valid_mapc)
    valid2 = df_lienquan["dieu_id2"].isin(valid_mapc)

    total = len(df_lienquan)
    valid = int((valid1 & valid2).sum())
    return {"total_refs": total, "valid_refs": valid, "orphan_refs": total - valid}


def main() -> None:
    logger.info("=== Silver: Pháp Điển cleaning ===")

    SILVER_PHAP_DIEN.mkdir(parents=True, exist_ok=True)

    datasets = {
        "chude": (clean_chude, _read_bronze("chude")),
        "demuc": (clean_demuc, _read_bronze("demuc")),
        "chuong": (clean_chuong, _read_bronze("chuong")),
        "dieu": (clean_dieu, _read_bronze("dieu")),
        "table": (clean_tables, _read_bronze("table")),
        "file": (clean_files, _read_bronze("file")),
        "muclienquan": (clean_lienquan, _read_bronze("muclienquan")),
    }

    results: dict[str, pd.DataFrame] = {}
    for name, (clean_fn, df_raw) in datasets.items():
        df_clean = clean_fn(df_raw)
        path = SILVER_PHAP_DIEN / f"{name}.parquet"
        df_clean.to_parquet(path, index=False)
        results[name] = df_clean
        logger.info("Silver %s: %d rows (bronze had %d)", name, len(df_clean), len(df_raw))

    xref = validate_cross_references(results.get("muclienquan", pd.DataFrame()),
                                     results.get("dieu", pd.DataFrame()))
    logger.info("Cross-reference integrity: %d/%d valid, %d orphans",
                xref["valid_refs"], xref["total_refs"], xref["orphan_refs"])

    logger.info("Silver Pháp Điển done")


if __name__ == "__main__":
    main()
