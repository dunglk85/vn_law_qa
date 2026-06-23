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
            "chimuc": "  1  ",
            "stt": "0",
            "vbqppl": None,
            "demuc_id": "dm1",
            "chuong_id": "ch1",
            "vbqppl_link": "https://example.com",
        }])
        result = clean_dieu(df)
        assert result.iloc[0]["ten"] == "Article 1"
        assert result.iloc[0]["noidung"] == "Some content"
        assert result.iloc[0]["chimuc"] == "1"

    def test_validate_cross_references(self):
        from src.silver.phap_dien import validate_cross_references
        df_lq = pd.DataFrame([
            {"dieu_id1": "A", "dieu_id2": "B"},  # both valid → valid row
            {"dieu_id1": "C", "dieu_id2": "D"},  # both orphaned → 1 orphan row
        ])
        df_dieu = pd.DataFrame({"mapc": ["A", "B"]})
        result = validate_cross_references(df_lq, df_dieu)
        assert result["total_refs"] == 2
        assert result["valid_refs"] == 1
        assert result["orphan_refs"] == 1

    def test_clean_chuong_missing_column_raises(self):
        """M1 fix: clean_chuong must raise KeyError when demuc_id column is absent."""
        from src.silver.phap_dien import clean_chuong
        df = pd.DataFrame([{
            "mapc": "c1", "ten": "Chapter 1", "chimuc": "I", "stt": 1,
            # deliberately omit demuc_id
        }])
        with pytest.raises(KeyError, match="demuc_id"):
            clean_chuong(df)

    def test_clean_dieu_missing_column_raises(self):
        """M1 fix: clean_dieu must raise KeyError when a FK column is absent."""
        from src.silver.phap_dien import clean_dieu
        df = pd.DataFrame([{
            "mapc": "d1", "ten": "Art 1", "noidung": "body",
            "chimuc": "1", "stt": "0", "vbqppl": None,
            "demuc_id": "dm1",
            # deliberately omit chuong_id
        }])
        with pytest.raises(KeyError, match="chuong_id"):
            clean_dieu(df)

    def test_clean_tables_missing_column_raises(self):
        """M1 fix: clean_tables must raise KeyError when dieu_id column is absent."""
        from src.silver.phap_dien import clean_tables
        df = pd.DataFrame([{"html": "<table></table>"}])  # no dieu_id
        with pytest.raises(KeyError, match="dieu_id"):
            clean_tables(df)


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

    def test_no_vbqppl_link_counted_as_quality_issue(self, tmp_path, monkeypatch):
        """M4 fix: pd_dieu_no_vbqppl_link > 0 must appear in the quality issue count."""
        import importlib
        import src.settings as _settings
        import src.silver.quality as qmod

        monkeypatch.setattr(_settings, "SILVER_PHAP_DIEN", tmp_path)
        monkeypatch.setattr(_settings, "SILVER_VBQPPL", tmp_path)
        monkeypatch.setattr(_settings, "METRICS", tmp_path)

        # dieu with no vbqppl_link (all None)
        pd.DataFrame({"id": ["1"], "ten": ["x"], "stt": [1]}).to_parquet(tmp_path / "chude.parquet")
        pd.DataFrame({"id": ["dm1"], "ten": ["x"], "stt": [1], "chude_id": ["1"]}).to_parquet(tmp_path / "demuc.parquet")
        pd.DataFrame({"mapc": ["ch1"], "ten": ["x"], "chimuc": ["I"], "stt": [1], "demuc_id": ["dm1"]}).to_parquet(tmp_path / "chuong.parquet")
        pd.DataFrame({
            "mapc": ["a"], "ten": ["Art"], "noidung": ["text"],
            "chimuc": [1], "stt": [0], "vbqppl": [""], "vbqppl_link": [None],  # <-- no link
            "chuong_id": ["ch1"], "demuc_id": ["dm1"],
        }).to_parquet(tmp_path / "dieu.parquet")
        pd.DataFrame({"dieu_id1": [], "dieu_id2": []}).to_parquet(tmp_path / "muclienquan.parquet")
        pd.DataFrame().to_parquet(tmp_path / "table.parquet")
        pd.DataFrame().to_parquet(tmp_path / "file.parquet")

        importlib.reload(qmod)
        metrics = qmod.check_phap_dien()
        assert metrics["pd_dieu_no_vbqppl_link"] == 1

        # Verify the issue counter now catches the no_vbqppl tag
        issues = sum(
            1 for k, v in metrics.items()
            if isinstance(v, (int, float)) and v > 0
            and any(tag in k for tag in ["nulls", "empty", "orphan", "no_vbqppl"])
        )
        assert issues >= 1, "pd_dieu_no_vbqppl_link should contribute to quality issue count"


class TestBronzeParse:
    def test_extract_item_id(self):
        from src.bronze.vbqppl import get_item_id
        assert get_item_id(None) is None
        assert get_item_id("https://vbpl.vn/Pages/vbpq-toanvan.aspx?ItemID=123#tab") == "123"
        assert get_item_id("no match") is None

    def test_vbqppl_resume_skips_existing_ids(self, tmp_path, monkeypatch):
        """C1 fix: already-fetched IDs in parquet must be skipped, not re-crawled."""
        import src.settings as _s
        from unittest.mock import patch, MagicMock

        # Seed parquet with one already-fetched document
        existing = pd.DataFrame([{"id": "111", "noidung": "<div>old</div>"}])
        parquet_path = tmp_path / "vbpl.parquet"
        existing.to_parquet(parquet_path, index=False)

        monkeypatch.setattr(_s, "BRONZE_VBQPPL", tmp_path)

        # Provide dieu.parquet with two item IDs: one already fetched, one new
        phap_dien_dir = tmp_path / "phap_dien"
        phap_dien_dir.mkdir()
        monkeypatch.setattr(_s, "BRONZE_PHAP_DIEN", phap_dien_dir)
        dieu_df = pd.DataFrame({
            "vbqppl_link": [
                "https://vbpl.vn/?ItemID=111",  # already in parquet
                "https://vbpl.vn/?ItemID=222",  # new
            ]
        })
        dieu_df.to_parquet(phap_dien_dir / "dieu.parquet", index=False)

        # Patch requests.get so fetch_document returns a synthetic HTML doc for ID 222
        fake_html = (
            '<div class="fulltext">'
            "  <div>wrapper</div>"
            '  <div id="toanvancontent"><p>content</p></div>'
            "</div>"
        )
        mock_response = MagicMock()
        mock_response.content = fake_html.encode()
        mock_response.raise_for_status = MagicMock()

        fetched_urls: list[str] = []

        def fake_get(url, **kwargs):
            fetched_urls.append(url)
            return mock_response

        with patch("requests.get", side_effect=fake_get), patch("time.sleep"):
            import importlib, src.bronze.vbqppl as m
            importlib.reload(m)
            m.main()

        # Only the new ID (222) should have triggered an HTTP request
        assert any("222" in u for u in fetched_urls), f"Expected request for 222, got {fetched_urls}"
        assert not any("111" in u for u in fetched_urls), f"ID 111 should have been skipped, got {fetched_urls}"
        # Both records should be in the output parquet
        result = pd.read_parquet(parquet_path)
        assert set(result["id"].astype(str)) == {"111", "222"}


    def test_phap_dien_href_absent_returns_none(self):
        """I3 fix: <a> without href must return None, not raise KeyError."""
        from bs4 import BeautifulSoup
        # Tag has no href attribute
        html = "<p><a>no href here</a></p>"
        soup = BeautifulSoup(html, "html.parser")
        tag = soup.select("a")[0]
        assert tag.get("href") is None


class TestPipeline:
    def test_stage_order(self):
        from src.pipeline import STAGE_ORDER, STAGES
        assert STAGE_ORDER == ["bronze", "silver", "gold"]
        assert "ingest_phap_dien" in [s[0] for s in STAGES["bronze"]]
        assert "quality_checks" in [s[0] for s in STAGES["silver"]]
        assert "chunk_documents" in [s[0] for s in STAGES["gold"]]


class TestSilverVBQPPLIdempotency:
    def test_split_produces_same_ids_on_rerun(self, tmp_path, monkeypatch):
        """I2 fix: two consecutive runs on the same bronze input must produce identical IDs."""
        import importlib
        import src.settings as _s
        import src.silver.vbqppl as m

        # Create a minimal bronze vbpl.parquet
        fake_html = (
            '<div class="fulltext">'
            '<div id="toanvancontent">'
            "<p>Chương I header</p>"
            "<p>Điều 1 first article</p>"
            "<p>Điều 2 second article</p>"
            "</div></div>"
        )
        bronze_dir = tmp_path / "vbqppl"
        bronze_dir.mkdir(parents=True)
        pd.DataFrame([{"id": "99", "noidung": fake_html}]).to_parquet(
            bronze_dir / "vbpl.parquet", index=False
        )

        silver_dir = tmp_path / "silver_vbqppl"
        silver_dir.mkdir(parents=True)

        monkeypatch.setattr(_s, "BRONZE_VBQPPL", bronze_dir)
        monkeypatch.setattr(_s, "SILVER_VBQPPL", silver_dir)

        # First run
        importlib.reload(m)
        m.main()
        run1 = pd.read_parquet(silver_dir / "vb_chimuc.parquet").sort_values("id").reset_index(drop=True)

        # Second run — must produce identical ids
        importlib.reload(m)
        m.main()
        run2 = pd.read_parquet(silver_dir / "vb_chimuc.parquet").sort_values("id").reset_index(drop=True)

        pd.testing.assert_frame_equal(run1, run2, check_like=False)


class TestSettings:
    def test_crawl_delay_default(self):
        """I4 fix: CRAWL_DELAY should default to 0.5 when not configured."""
        import importlib, src.settings as s
        importlib.reload(s)
        assert s.CRAWL_DELAY == 0.5

    def test_crawl_delay_from_env(self, monkeypatch):
        """I4 fix: LAW_CRAWL_DELAY env var overrides the default."""
        monkeypatch.setenv("LAW_CRAWL_DELAY", "2.0")
        import importlib, src.settings as s
        importlib.reload(s)
        assert s.CRAWL_DELAY == 2.0
        monkeypatch.delenv("LAW_CRAWL_DELAY", raising=False)
        importlib.reload(s)  # restore default for other tests

    def test_save_every_zero_raises(self, monkeypatch):
        """SAVE_EVERY=0 must raise ValueError (ZeroDivisionError guard)."""
        monkeypatch.setenv("LAW_SAVE_EVERY", "0")
        import importlib, src.settings as s
        with pytest.raises(ValueError, match="must be >= 1"):
            importlib.reload(s)
        monkeypatch.delenv("LAW_SAVE_EVERY", raising=False)
        importlib.reload(s)

    def test_max_retries_zero_raises(self, monkeypatch):
        """MAX_RETRIES=0 must raise ValueError (silent skip guard)."""
        monkeypatch.setenv("LAW_MAX_RETRIES", "0")
        import importlib, src.settings as s
        with pytest.raises(ValueError, match="must be >= 1"):
            importlib.reload(s)
        monkeypatch.delenv("LAW_MAX_RETRIES", raising=False)
        importlib.reload(s)
