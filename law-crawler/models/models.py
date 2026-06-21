"""Peewee ORM models for Pháp Điển data."""
from peewee import (
    CharField,
    ForeignKeyField,
    IntegerField,
    Model,
    TextField,
)

from db import db


class BaseModel(Model):
    """Base model with shared database configuration."""

    class Meta:
        database = db


class PDChuDe(BaseModel):
    """Subject/topic (Chủ đề) of law documents."""

    id = CharField(max_length=128, primary_key=True)
    ten = TextField()
    stt = IntegerField()


class PDDeMuc(BaseModel):
    """Table of contents (Đề mục) entry."""

    id = CharField(max_length=128, primary_key=True)
    ten = TextField()
    stt = IntegerField()
    chude_id = ForeignKeyField(PDChuDe, backref="demucs")


class PDChuong(BaseModel):
    """Chapter (Chương) of a law document."""

    mapc = CharField(max_length=128, primary_key=True)
    ten = TextField()
    demuc_id = ForeignKeyField(PDDeMuc, backref="chuongs")
    chimuc = TextField()
    stt = IntegerField()


class PDDieu(BaseModel):
    """Article/Clause (Điều) of a law document."""

    ten = TextField()
    demuc_id = ForeignKeyField(PDDeMuc, backref="dieus")
    chuong_id = ForeignKeyField(PDChuong, backref="dieus")
    chude_id = ForeignKeyField(PDChuDe, backref="dieus", null=True)
    mapc = CharField(max_length=128, primary_key=True)
    noidung = TextField()
    chimuc = IntegerField()
    vbqppl = TextField()
    vbqppl_link = TextField(null=True)
    stt = IntegerField()


class PDTable(BaseModel):
    """Table within an article."""

    dieu_id = ForeignKeyField(PDDieu, backref="tables")
    html = TextField()


class PDFile(BaseModel):
    """Attached file/link within an article."""

    dieu_id = ForeignKeyField(PDDieu, backref="files")
    link = TextField()
    path = TextField()


class PDMucLienQuan(BaseModel):
    """Cross-reference between articles."""

    dieu_id1 = ForeignKeyField(PDDieu)
    dieu_id2 = ForeignKeyField(PDDieu)
