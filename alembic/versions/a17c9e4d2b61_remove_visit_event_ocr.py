"""remove visit event ocr

Revision ID: a17c9e4d2b61
Revises: f1a40b7ae73b
Create Date: 2026-07-28

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a17c9e4d2b61'
down_revision: str | Sequence[str] | None = 'f1a40b7ae73b'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Удаляет OCR-время отдельных событий визита."""

    op.drop_column('visits', 'ocr_left_at')
    op.drop_column('visits', 'ocr_entered_at')


def downgrade() -> None:
    """Возвращает OCR-время отдельных событий визита."""

    op.add_column(
        'visits',
        sa.Column(
            'ocr_entered_at',
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        'visits',
        sa.Column(
            'ocr_left_at',
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
