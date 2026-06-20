"""initial_schema

Revision ID: 2c5cecf6f448
Revises: 
Create Date: 2026-06-20 23:13:12.541619

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID, JSONB

# revision identifiers, used by Alembic.
revision: str = '2c5cecf6f448'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "langchain_pg_embedding",
        sa.Column("langchain_id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("langchain_metadata", JSONB, nullable=True),
        sa.Column("category", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("langchain_pg_embedding")
    op.execute("DROP EXTENSION IF EXISTS vector")
