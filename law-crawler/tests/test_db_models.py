"""Tests for law-crawler database and model configuration.

Skipped when peewee is not installed (e.g. in main app's venv).
"""
import os
import pytest

peewee = pytest.importorskip("peewee", reason="peewee not installed")


def test_db_env_vars(monkeypatch):
    """Database should use environment variables."""
    monkeypatch.setenv("LAW_DB_USER", "testuser")
    monkeypatch.setenv("LAW_DB_PASSWORD", "testpass")
    monkeypatch.setenv("LAW_DB_HOST", "testhost")
    monkeypatch.setenv("LAW_DB_PORT", "3307")
    monkeypatch.setenv("LAW_DB_NAME", "testdb")

    import importlib
    import db as db_module
    importlib.reload(db_module)

    assert db_module._DB_USER == "testuser"
    assert db_module._DB_PASS == "testpass"
    assert db_module._DB_HOST == "testhost"
    assert db_module._DB_PORT == 3307
    assert db_module._DB_NAME == "testdb"


def test_db_connection_string():
    """Database connection string should be properly formatted."""
    import db as db_module

    assert "mysql://" in db_module.DATABASE
    assert db_module._DB_NAME in db_module.DATABASE


def test_models_import():
    """All models should be importable."""
    from models.models import (
        PDChuDe,
        PDChuong,
        PDDeMuc,
        PDDieu,
        PDFile,
        PDMucLienQuan,
        PDTable,
    )

    assert PDChuDe is not None
    assert PDDeMuc is not None
    assert PDChuong is not None
    assert PDDieu is not None
    assert PDTable is not None
    assert PDFile is not None
    assert PDMucLienQuan is not None


def test_model_table_names():
    """Models should have correct table names."""
    from models.models import (
        PDChuDe,
        PDChuong,
        PDDeMuc,
        PDDieu,
        PDFile,
        PDMucLienQuan,
        PDTable,
    )

    assert PDChuDe._meta.table_name == "pdchude"
    assert PDDeMuc._meta.table_name == "pdemuc"
    assert PDChuong._meta.table_name == "pdchuong"
    assert PDDieu._meta.table_name == "pddieu"
    assert PDTable._meta.table_name == "pdtable"
    assert PDFile._meta.table_name == "pdfile"
    assert PDMucLienQuan._meta.table_name == "pdmuclienquan"
