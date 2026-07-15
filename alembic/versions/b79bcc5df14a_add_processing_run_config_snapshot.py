"""add processing run config snapshot

Revision ID: b79bcc5df14a
Revises: 897039a988a4
Create Date: 2026-07-15 10:46:53.364606
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b79bcc5df14a'
down_revision: str | Sequence[str] | None = '897039a988a4'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Добавляет снимок конфигурации и диапазон обработки."""

    op.add_column(
        'processing_runs',
        sa.Column(
            'config_snapshot',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        'processing_runs',
        sa.Column(
            'config_hash',
            sa.String(length=64),
            nullable=True,
        ),
    )
    op.add_column(
        'processing_runs',
        sa.Column(
            'start_frame',
            sa.Integer(),
            nullable=False,
            server_default=sa.text('0'),
        ),
    )
    op.add_column(
        'processing_runs',
        sa.Column(
            'frame_limit',
            sa.Integer(),
            nullable=True,
        ),
    )

    op.create_index(
        'ix_processing_runs_config_hash',
        'processing_runs',
        ['config_hash'],
        unique=False,
    )

    op.alter_column(
        'processing_runs',
        'start_frame',
        existing_type=sa.Integer(),
        server_default=None,
    )


def downgrade() -> None:
    """Удаляет снимок конфигурации и диапазон обработки."""

    op.drop_index(
        'ix_processing_runs_config_hash',
        table_name='processing_runs',
    )
    op.drop_column('processing_runs', 'frame_limit')
    op.drop_column('processing_runs', 'start_frame')
    op.drop_column('processing_runs', 'config_hash')
    op.drop_column('processing_runs', 'config_snapshot')