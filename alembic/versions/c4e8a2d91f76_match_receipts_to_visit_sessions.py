"""match receipts to visit sessions

Revision ID: c4e8a2d91f76
Revises: a17c9e4d2b61
Create Date: 2026-07-28

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c4e8a2d91f76'
down_revision: str | Sequence[str] | None = 'a17c9e4d2b61'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Переводит сопоставление чеков с Visit на VisitSession."""

    op.drop_table('receipt_matches')
    op.create_table(
        'receipt_matches',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column(
            'visit_session_id',
            sa.Integer(),
            nullable=False,
        ),
        sa.Column('receipt_id', sa.Integer(), nullable=False),
        sa.Column(
            'matched_at',
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            'time_delta_seconds',
            sa.Float(),
            nullable=False,
        ),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column(
            'is_ambiguous',
            sa.Boolean(),
            server_default=sa.text('false'),
            nullable=False,
        ),
        sa.Column(
            'is_group_match',
            sa.Boolean(),
            server_default=sa.text('false'),
            nullable=False,
        ),
        sa.Column(
            'policy',
            sa.String(length=50),
            server_default='closest_with_flag',
            nullable=False,
        ),
        sa.Column(
            'metadata',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['receipt_id'],
            ['receipts.id'],
        ),
        sa.ForeignKeyConstraint(
            ['visit_session_id'],
            ['visit_sessions.id'],
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'visit_session_id',
            name='uq_receipt_matches_visit_session_id',
        ),
    )


def downgrade() -> None:
    """Возвращает сопоставление чеков с отдельными Visit."""

    op.drop_table('receipt_matches')
    op.create_table(
        'receipt_matches',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('visit_id', sa.Integer(), nullable=False),
        sa.Column('receipt_id', sa.Integer(), nullable=False),
        sa.Column(
            'matched_at',
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            'time_delta_seconds',
            sa.Float(),
            nullable=False,
        ),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column(
            'is_ambiguous',
            sa.Boolean(),
            server_default=sa.text('false'),
            nullable=False,
        ),
        sa.Column(
            'is_group_match',
            sa.Boolean(),
            server_default=sa.text('false'),
            nullable=False,
        ),
        sa.Column(
            'policy',
            sa.String(length=50),
            server_default='closest_with_flag',
            nullable=False,
        ),
        sa.Column(
            'metadata',
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ['receipt_id'],
            ['receipts.id'],
        ),
        sa.ForeignKeyConstraint(
            ['visit_id'],
            ['visits.id'],
        ),
        sa.PrimaryKeyConstraint('id'),
    )
