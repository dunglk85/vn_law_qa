"""Silver layer: Data quality checks & metrics.

Validates the silver layer data and writes a quality report
to metrics/quality.json. This is a DVC metric — it becomes a
tracked signal for pipeline health over time.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import src.settings as _settings
from src.settings import setup_logging

logger = setup_logging(__name__)


def _read_silver(subdir: Path, name: str) -> pd.DataFrame:
    path = subdir / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def check_phap_dien() -> dict:
    dfs = {
        name: _read_silver(_settings.SILVER_PHAP_DIEN, name)
        for name in ["chude", "demuc", "chuong", "dieu", "table", "file", "muclienquan"]
    }

    metrics: dict = {}

    for name, df in dfs.items():
        metrics[f"pd_{name}_count"] = len(df)
        if not df.empty:
            null_cols = [c for c in df.columns if df[c].isna().any()]
            empty_cols = [
                c for c in df.columns
                if df[c].dtype == object and (df[c].str.strip() == "").any()
            ]
            metrics[f"pd_{name}_nulls_in"] = null_cols
            metrics[f"pd_{name}_empty_in"] = empty_cols

    df_dieu = dfs["dieu"]
    if not df_dieu.empty:
        metrics["pd_dieu_empty_noidung"] = int((df_dieu["noidung"].str.strip() == "").sum())
        metrics["pd_dieu_no_vbqppl_link"] = int(df_dieu["vbqppl_link"].isna().sum())

    df_lq = dfs["muclienquan"]
    if not df_lq.empty and not df_dieu.empty:
        valid_mapc = set(df_dieu["mapc"].dropna())
        orphan1 = int((~df_lq["dieu_id1"].isin(valid_mapc)).sum())
        orphan2 = int((~df_lq["dieu_id2"].isin(valid_mapc)).sum())
        metrics["pd_lienquan_orphan_refs"] = orphan1 + orphan2

    df_chuong = dfs["chuong"]
    if not df_chuong.empty:
        metrics["pd_chuong_empty_ten"] = int((df_chuong["ten"].str.strip() == "").sum())

    return metrics


def check_vbqppl() -> dict:
    df_chimuc = _read_silver(_settings.SILVER_VBQPPL, "vb_chimuc")
    metrics: dict = {}

    if df_chimuc.empty:
        metrics["vb_chimuc_count"] = 0
        return metrics

    metrics["vb_chimuc_count"] = len(df_chimuc)
    metrics["vb_chimuc_empty_noi_dung"] = int(
        (df_chimuc["noi_dung"].str.strip() == "").sum()
    )
    chapter_count = int(df_chimuc["chi_muc_cha"].isna().sum())
    article_count = len(df_chimuc) - chapter_count
    metrics["vb_chimuc_chapters"] = chapter_count
    metrics["vb_chimuc_articles"] = article_count
    metrics["vb_chimuc_unique_docs"] = int(df_chimuc["id_vb"].nunique())

    return metrics


def main() -> None:
    logger.info("=== Silver: Quality checks ===")

    metrics = {}
    metrics.update(check_phap_dien())
    metrics.update(check_vbqppl())

    metrics["checked_at"] = pd.Timestamp.now().isoformat()

    _settings.METRICS.mkdir(parents=True, exist_ok=True)
    with open(_settings.METRICS / "quality.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Quality report written to %s", _settings.METRICS / "quality.json")

    issues = sum(
        1 for k, v in metrics.items()
        if isinstance(v, (int, float)) and v > 0
        and any(tag in k for tag in ["nulls", "empty", "orphan"])
    )
    if issues:
        logger.warning("Found %d potential quality issues — check metrics/quality.json", issues)
    else:
        logger.info("No quality issues detected")


if __name__ == "__main__":
    main()
