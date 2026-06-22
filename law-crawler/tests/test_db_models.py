"""Tests for law-crawler database and model configuration.

Skipped when peewee is not installed (e.g. in main app's venv).
"""
import pytest

peewee = pytest.importorskip("peewee", reason="peewee not installed")


def test_mysql_config_respects_env(monkeypatch):
    """mysql_config() should use environment variables."""
    monkeypatch.setenv("LAW_DB_USER", "testuser")
    monkeypatch.setenv("LAW_DB_PASSWORD", "testpass")
    monkeypatch.setenv("LAW_DB_HOST", "testhost")
    monkeypatch.setenv("LAW_DB_PORT", "3307")
    monkeypatch.setenv("LAW_DB_NAME", "testdb")

    from db import mysql_config

    cfg = mysql_config()
    assert cfg["user"] == "testuser"
    assert cfg["password"] == "testpass"
    assert cfg["host"] == "testhost"
    assert cfg["port"] == 3307
    assert cfg["database"] == "testdb"


def test_get_connection_url():
    """Connection URL should be properly formatted."""
    from db import get_connection_url

    url = get_connection_url()
    assert "mysql://" in url


def test_get_db_returns_instance():
    """get_db() should return a MySQLDatabase instance."""
    from db import get_db
    from peewee import MySQLDatabase

    db_instance = get_db()
    assert isinstance(db_instance, MySQLDatabase)


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
    assert PDDeMuc._meta.table_name == "pddemuc"
    assert PDChuong._meta.table_name == "pdchuong"
    assert PDDieu._meta.table_name == "pddieu"
    assert PDTable._meta.table_name == "pdtable"
    assert PDFile._meta.table_name == "pdfile"
    assert PDMucLienQuan._meta.table_name == "pdmuclienquan"
