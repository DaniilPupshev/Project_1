from datetime import date, datetime
from decimal import Decimal

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from pechvision.db.base import Base


class TimestampMixin:
    '''Вспомогательные колонки create/update'''

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )


class Video(TimestampMixin, Base):
    '''Таблица обработанных видео'''

    __tablename__ = 'videos'
    __table_args__ = (
        UniqueConstraint(
            'file_sha256',
            name='uq_videos_file_sha256',
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False
    )
    stored_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True
    )
    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )
    camera_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    fps: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )
    width: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )
    height: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )
    frame_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )
    duration_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    recorded_start_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    recorded_end_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    file_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True
    )
    file_size_bytes: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True
    )

    extra_data: Mapped[dict | None] = mapped_column(
        'metadata',
        JSONB,
        nullable=True
    )

    processing_runs: Mapped[list['ProcessingRun']] = relationship(
        back_populates='video',
        cascade='all, delete-orphan',
    )
    visits: Mapped[list['Visit']] = relationship(back_populates='video')


class ProcessingRun(TimestampMixin, Base):
    '''Таблица для статистики запуска обработки'''
     
    __tablename__ = 'processing_runs'

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey('videos.id'),
        nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default='pending'
    )
    config_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(
        String,
        nullable=True
    )

    config_snapshot: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True
    )
    config_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True
    )
    start_frame: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0
    )
    frame_limit: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    stats: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True
    )

    video: Mapped['Video'] = relationship(back_populates='processing_runs')
    visits: Mapped[list['Visit']] = relationship(
        back_populates='processing_run',
        cascade='all, delete-orphan',
    )


class Person(TimestampMixin, Base):
    '''Таблица людей'''

    __tablename__ = 'persons'
    __table_args__ = (
        Index(
            'ix_persons_face_embedding_hnsw',
            'face_embedding',
            postgresql_using='hnsw',
            postgresql_ops={
                'face_embedding': 'vector_cosine_ops',
            },
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    external_person_key: Mapped[str] = mapped_column(
        String(64),
        unique=True, # P_<uuid7>
        nullable=False
    )

    first_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    best_face_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True
    )
    face_embedding: Mapped[list[float] | None] = mapped_column(
        VECTOR(512),
        nullable=True
    )

    extra_data: Mapped[dict | None] = mapped_column(
        'metadata',
        JSONB,
        nullable=True
    )

    visits: Mapped[list['Visit']] = relationship(back_populates='person')
    faces: Mapped[list['Face']] = relationship(back_populates='person')


class Visit(TimestampMixin, Base):
    '''Таблица посещений людей'''

    __tablename__ = 'visits'

    __table_args__ = (
        UniqueConstraint('event_key', name='uq_visits_event_key'),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    event_key: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    video_id: Mapped[int] = mapped_column(
        ForeignKey('videos.id'),
        nullable=False
    )
    processing_run_id: Mapped[int] = mapped_column(
        ForeignKey('processing_runs.id'),
        nullable=False
    )
    person_id: Mapped[int | None] = mapped_column(
        ForeignKey('persons.id'),
        nullable=True
    )

    track_id: Mapped[str] = mapped_column(
        String(55), # 'id_video'_'count_id_track'
        nullable=False
    )

    visit_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True
    )
    entered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    left_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    duration_seconds: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    entry_frame_index: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )
    exit_frame_index: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    ocr_entered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    ocr_left_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    time_is_estimated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False
    )
    is_staff: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False
    )
    staff_id: Mapped[int | None] = mapped_column(
        ForeignKey('staff_members.id'),
        nullable=True,
    )

    is_group: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False
    )

    extra_data: Mapped[dict | None] = mapped_column(
        'metadata',
        JSONB,
        nullable=True
    )

    video: Mapped['Video'] = relationship(back_populates='visits')
    processing_run: Mapped['ProcessingRun'] = relationship(back_populates='visits')
    person: Mapped['Person | None'] = relationship(back_populates='visits')
    faces: Mapped[list['Face']] = relationship(back_populates='visit')
    staff_member: Mapped['Staff | None'] = relationship(back_populates='visits')
    receipt_matches: Mapped[list['ReceiptMatch']] = relationship(back_populates='visit')


class Face(TimestampMixin, Base):
    '''Таблица характеристик лиц людей'''

    __tablename__ = 'faces'

    id: Mapped[int] = mapped_column(primary_key=True)
    person_id: Mapped[int | None] = mapped_column(
        ForeignKey('persons.id'),
        nullable=True
    )
    visit_id: Mapped[int] = mapped_column(
        ForeignKey('visits.id'),
        nullable=False
    )

    image_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True
    )
    frame_index: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    quality_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )
    embedding: Mapped[list[float] | None] = mapped_column(
        VECTOR(512),
        nullable=True
    )

    gender: Mapped[str | None] = mapped_column(
        String(15),
        nullable=True
    )
    gender_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )
    age_estimate: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )
    age_bucket: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True
    )

    is_best: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False
    )

    extra_data: Mapped[dict | None] = mapped_column(
        'metadata',
        JSONB,
        nullable=True
    )

    visit: Mapped['Visit'] = relationship(back_populates='faces')
    person: Mapped['Person | None'] = relationship(back_populates='faces')


class Receipt(TimestampMixin, Base):
    '''Таблица чеков'''

    __tablename__ = 'receipts'

    id: Mapped[int] = mapped_column(primary_key=True)

    external_receipt_id: Mapped[str] = mapped_column(
        String(128),
        unique=True,
        nullable=False
    )
    tt: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True
    )
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    closed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    amount: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2),
        nullable=True
    )
    table_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    client_external_id: Mapped[str | None] = mapped_column(
        String(220),
        nullable=True
    )

    source_file: Mapped[str] = mapped_column(
        String(1024),
        nullable=False
    )
    
    raw_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    receipt_matches: Mapped[list['ReceiptMatch']] = relationship(back_populates='receipt')


class ReceiptMatch(TimestampMixin, Base):
    '''Таблица сводки по чекам и посещениям людей'''

    __tablename__ = 'receipt_matches'

    id: Mapped[int] = mapped_column(primary_key=True)
    
    visit_id: Mapped[int] = mapped_column(
        ForeignKey('visits.id'),
        nullable=False
    )
    receipt_id: Mapped[int] = mapped_column(
        ForeignKey('receipts.id'),
        nullable=False
    )

    matched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False
    )
    time_delta_seconds: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True
    )

    is_ambiguous: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False
    )
    is_group_match: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False
    )

    policy: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default='closest_with_flag'
    )

    extra_data: Mapped[dict | None] = mapped_column(
        'metadata',
        JSONB,
        nullable=True
    )

    visit: Mapped['Visit'] = relationship(back_populates='receipt_matches')
    receipt: Mapped['Receipt'] = relationship(back_populates='receipt_matches')


class Staff(TimestampMixin, Base):
    '''Таблица-справочник персонала'''

    __tablename__ = 'staff_members'

    id: Mapped[int] = mapped_column(primary_key=True)

    external_staff_key: Mapped[str | None] = mapped_column(
        String(128),
        unique=True,
        nullable=True
    )

    full_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )
    photo_path: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True
    )

    face_embedding: Mapped[list[float] | None] = mapped_column(
        VECTOR(512),
        nullable=True
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True
    )

    extra_data: Mapped[dict | None] = mapped_column(
        'metadata',
        JSONB,
        nullable=True
    )

    visits: Mapped[list['Visit']] = relationship(back_populates='staff_member')