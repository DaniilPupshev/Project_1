"""add pgvector embeddings

Revision ID: 897039a988a4
Revises: dbdf72fb9367
Create Date: 2026-07-08

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '897039a988a4'
down_revision: Union[str, Sequence[str], None] = 'dbdf72fb9367'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    op.alter_column(
        'persons',
        'face_embedding',
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=VECTOR(512),
        existing_nullable=True,
        postgresql_using='face_embedding::text::vector(512)',
    )

    op.alter_column(
        'faces',
        'embedding',
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=VECTOR(512),
        existing_nullable=True,
        postgresql_using='embedding::text::vector(512)',
    )

    op.create_index(
        'ix_persons_face_embedding_hnsw',
        'persons',
        ['face_embedding'],
        postgresql_using='hnsw',
        postgresql_ops={'face_embedding': 'vector_cosine_ops'},
    )

    op.alter_column(
        'staff_members',
        'face_embedding',
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        type_=VECTOR(512),
        existing_nullable=True,
        postgresql_using='face_embedding::text::vector(512)',
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(
        'ix_persons_face_embedding_hnsw',
        table_name='persons',
    )

    op.alter_column(
        'faces',
        'embedding',
        existing_type=VECTOR(512),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
        postgresql_using='to_jsonb(embedding::text)',
    )

    op.alter_column(
        'persons',
        'face_embedding',
        existing_type=VECTOR(512),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
        postgresql_using='to_jsonb(face_embedding::text)',
    )

    op.alter_column(
        'staff_members',
        'face_embedding',
        existing_type=VECTOR(512),
        type_=postgresql.JSONB(astext_type=sa.Text()),
        existing_nullable=True,
        postgresql_using='to_jsonb(face_embedding::text)',
    )