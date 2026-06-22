"""Tests for medallion pipeline modules.

Skipped when pandas/pyarrow are not installed (e.g. in main app's venv).
"""
import json
import tempfile
from pathlib import Path

import pytest

pandas = pytest.importorskip("pandas", reason="pandas not installed")
pd = pandas


class TestSettings:
    def test_import_settings(self):
        from src.settings import BRONZE, SILVER, GOLD, PHAP_DIEN_DIR
        assert BRONZE.name == "bronze"
        assert SILVER.name == "silver"
        assert GOLD.name == "gold"
        assert PHAP_DIEN_DIR.name == "phap-dien"

    def test_setup_logging(self):
        from src.settings import setup_logging
        logger = setup_logging("test_logger")
        assert logger.name == "test_logger"

    def test_chunk_settings_defaults(self):
        from src.settings import CHUNK_OVERLAP, CHUNK_SIZE
        assert CHUNK_SIZE == 1000
        assert CHUNK_OVERLAP == 200

    def test_chunk_settings_from_env(self, monkeypatch):
        monkeypatch.setenv("LAW_CHUNK_SIZE", "500")
        monkeypatch.setenv("LAW_CHUNK_OVERLAP", "100")
        import importlib
        import src.settings
        importlib.reload(src.settings)
        from src.settings import CHUNK_OVERLAP, CHUNK_SIZE
        assert CHUNK_SIZE == 500
        assert CHUNK_OVERLAP == 100


class TestChunking:
    def test_chunk_text_short(self):
        from src.gold.chunks import chunk_text
        result = chunk_text("Hello world")
        assert result == ["Hello world"]

    def test_chunk_text_empty(self):
        from src.gold.chunks import chunk_text
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunk_text_long(self):
        from src.gold.chunks import chunk_text
        text = "A" * 2500
        chunks = chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 3
        assert all(len(c) <= 1000 for c in chunks)
        assert "".join(chunks).count("A") < len(text) + 1000  # overlapping

    def test_chunk_text_overlap(self):
        from src.gold.chunks import chunk_text
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunk_text(text, chunk_size=10, overlap=3)
        assert chunks[0] == "ABCDEFGHIJ"
        assert chunks[1] == "HIJKLMNOPQ"
        assert chunks[2] == "OPQRSTUVWX"
        assert chunks[3] == "VWXYZ"


class TestSilverCleaning:
    def test_clean_chude_dedup(self):
        from src.silver.phap_dien import clean_chude
        df = pd.DataFrame([
            {"id": "1", "ten": "  Civil Law  ", "stt": "1"},
            {"id": "1", "ten": "Duplicate", "stt": "2"},
        ])
        result = clean_chude(df)
        assert len(result) == 1
        assert result.iloc[0]["ten"] == "Civil Law"

    def test_clean_dieu_strips_text(self):
        from src.silver.phap_dien import clean_dieu
        df = pd.DataFrame([{
            "mapc": "abc123",
            "ten": "  Article 1  ",
            "noidung": "  Some content  ",
            "chimuc": "1",
            "stt": "0",
            "vbqppl": None,
            "demuc_id": "dm1",
            "chuong_id": "ch1",
        }])
        result = clean_dieu(df)
        assert result.iloc[0]["ten"] == "Article 1"
        assert result.iloc[0]["noidung"] == "Some content"

    def test_validate_cross_references(self):
        from src.silver.phap_dien import validate_cross_references
        df_lq = pd.DataFrame([
            {"dieu_id1": "A", "dieu_id2": "B"},
            {"dieu_id1": "C", "dieu_id2": "D"},
        ])
        df_dieu = pd.DataFrame({"mapc": ["A", "B"]})
        result = validate_cross_references(df_lq, df_dieu)
        assert result["total_refs"] == 2
        assert result["valid_refs"] == 1
        assert result["orphan_refs"] == 1


class TestQuality:
    def test_quality_output_json(self):
        import src.settings as _settings
        from src.silver.quality import check_phap_dien

        with tempfile.TemporaryDirectory() as tmp:
            original_silver = _settings.SILVER_PHAP_DIEN
            _settings.SILVER_PHAP_DIEN = Path(tmp)

            pd.DataFrame({"id": ["1"], "ten": ["Civil"], "stt": [1]}).to_parquet(Path(tmp) / "chude.parquet")
            pd.DataFrame({"id": ["dm1"], "ten": ["Section 1"], "stt": [1], "chude_id": ["1"]}).to_parquet(Path(tmp) / "demuc.parquet")
            pd.DataFrame({"mapc": ["ch1"], "ten": ["Chapter 1"], "chimuc": ["I"], "stt": [1], "demuc_id": ["dm1"]}).to_parquet(Path(tmp) / "chuong.parquet")
            pd.DataFrame({
                "mapc": ["a"], "ten": ["Art 1"], "noidung": ["content"],
                "chimuc": [1], "stt": [0], "vbqppl": [""], "vbqppl_link": ["link"],
                "chuong_id": ["ch1"], "demuc_id": ["dm1"],
            }).to_parquet(Path(tmp) / "dieu.parquet")
            pd.DataFrame({"dieu_id1": ["a"], "dieu_id2": ["b"]}).to_parquet(Path(tmp) / "muclienquan.parquet")
            pd.DataFrame().to_parquet(Path(tmp) / "table.parquet")
            pd.DataFrame().to_parquet(Path(tmp) / "file.parquet")

            result = check_phap_dien()
            assert result["pd_chude_count"] == 1
            assert result["pd_dieu_count"] == 1
            assert result["pd_lienquan_orphan_refs"] >= 0

            _settings.SILVER_PHAP_DIEN = original_silver


class TestBronzeParse:
    def test_extract_item_id(self):
        from src.bronze.vbqppl import get_item_id
        assert get_item_id(None) is None
        assert get_item_id("https://vbpl.vn/Pages/vbpq-toanvan.aspx?ItemID=123#tab") == "123"
        assert get_item_id("no match") is None


class TestPipeline:
    def test_stage_order(self):
        from src.pipeline import STAGE_ORDER, STAGES
        assert STAGE_ORDER == ["bronze", "silver", "gold"]
        assert "ingest_phap_dien" in [s[0] for s in STAGES["bronze"]]
        assert "quality_checks" in [s[0] for s in STAGES["silver"]]
        assert "chunk_documents" in [s[0] for s in STAGES["gold"]]
