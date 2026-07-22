"""add face identity references

Revision ID: df2cc488f30e
Revises: d037dd01d9c9
Create Date: 2026-07-17 10:27:04.015913

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'df2cc488f30e'
down_revision: str | Sequence[str] | None = 'd037dd01d9c9'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Добавляет признаки пригодности и эталонного использования лица."""

    op.add_column(
        'faces',
        sa.Column(
            'identity_quality_score',
            sa.Float(),
            nullable=True,
        ),
    )
    op.add_column(
        'faces',
        sa.Column(
            'is_identity_eligible',
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    op.add_column(
        'faces',
        sa.Column(
            'is_identity_reference',
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )
    op.add_column(
        'faces',
        sa.Column(
            'pose_category',
            sa.String(length=20),
            nullable=True,
        ),
    )

    op.create_index(
        'ix_faces_person_identity_reference',
        'faces',
        ['person_id', 'is_identity_reference'],
        unique=False,
    )

    op.alter_column(
        'faces',
        'is_identity_eligible',
        existing_type=sa.Boolean(),
        server_default=None,
    )
    op.alter_column(
        'faces',
        'is_identity_reference',
        existing_type=sa.Boolean(),
        server_default=None,
    )


def downgrade() -> None:
    """Удаляет признаки пригодности и эталонного использования лица."""

    op.drop_index(
        'ix_faces_person_identity_reference',
        table_name='faces',
    )
    op.drop_column('faces', 'pose_category')
    op.drop_column('faces', 'is_identity_reference')
    op.drop_column('faces', 'is_identity_eligible')
    op.drop_column('faces', 'identity_quality_score')