"""add visit sessions

Revision ID: f1a40b7ae73b
Revises: df2cc488f30e
Create Date: 2026-07-23 16:05:32.014563

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f1a40b7ae73b'
down_revision: str | Sequence[str] | None = 'df2cc488f30e'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Добавляет логические сессии посещений."""

    op.create_table(
        'visit_sessions',
        sa.Column(
            'id',
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            'session_key',
            sa.String(length=128),
            nullable=False,
        ),
        sa.Column(
            'video_id',
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            'person_id',
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            'visit_date',
            sa.Date(),
            nullable=True,
        ),
        sa.Column(
            'entered_at',
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            'left_at',
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            'duration_seconds',
            sa.Float(),
            nullable=True,
        ),
        sa.Column(
            'entry_frame_index',
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            'exit_frame_index',
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            'segments_count',
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            'time_is_estimated',
            sa.Boolean(),
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
            ['person_id'],
            ['persons.id'],
            name='fk_visit_sessions_person_id_persons',
        ),
        sa.ForeignKeyConstraint(
            ['video_id'],
            ['videos.id'],
            name='fk_visit_sessions_video_id_videos',
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'session_key',
            name='uq_visit_sessions_session_key',
        ),
    )

    op.create_index(
        'ix_visit_sessions_person_entered_at',
        'visit_sessions',
        ['person_id', 'entered_at'],
        unique=False,
    )
    op.create_index(
        'ix_visit_sessions_video_entered_at',
        'visit_sessions',
        ['video_id', 'entered_at'],
        unique=False,
    )

    op.add_column(
        'visits',
        sa.Column(
            'visit_session_id',
            sa.Integer(),
            nullable=True,
        ),
    )
    op.create_index(
        'ix_visits_visit_session_id',
        'visits',
        ['visit_session_id'],
        unique=False,
    )
    op.create_foreign_key(
        'fk_visits_visit_session_id_visit_sessions',
        'visits',
        'visit_sessions',
        ['visit_session_id'],
        ['id'],
        ondelete='SET NULL',
    )


def downgrade() -> None:
    """Удаляет логические сессии посещений."""

    op.drop_constraint(
        'fk_visits_visit_session_id_visit_sessions',
        'visits',
        type_='foreignkey',
    )
    op.drop_index(
        'ix_visits_visit_session_id',
        table_name='visits',
    )
    op.drop_column(
        'visits',
        'visit_session_id',
    )

    op.drop_index(
        'ix_visit_sessions_video_entered_at',
        table_name='visit_sessions',
    )
    op.drop_index(
        'ix_visit_sessions_person_entered_at',
        table_name='visit_sessions',
    )
    op.drop_table('visit_sessions')