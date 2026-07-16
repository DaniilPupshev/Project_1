from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from pechvision.config.schema import AppConfig
from pechvision.config.snapshot import (
    build_config_snapshot,
    build_processing_config_payload,
    calculate_config_hash,
)
from pechvision.db.models import ProcessingRun, Video


def find_existing_processing_run(
    session: Session,
    video_id: int,
    config: AppConfig,
    start_frame: int,
    frame_limit: int | None,
) -> ProcessingRun | None:
    '''Поиск уже выполняющегося или завершенного запуска обработки'''

    _, config_hash = _build_processing_run_config(config=config)

    conditions = [
        ProcessingRun.video_id == video_id,
        ProcessingRun.config_hash == config_hash,
        ProcessingRun.start_frame == start_frame,
        ProcessingRun.status.in_(('running', 'finished')),
    ]

    if frame_limit is None:
        conditions.append(ProcessingRun.frame_limit.is_(None))
    else:
        conditions.append(ProcessingRun.frame_limit == frame_limit)

    statement = (
        select(ProcessingRun)
        .where(*conditions)
        .order_by(ProcessingRun.id.desc())
    )

    return session.scalar(statement)


def _build_processing_run_config(
    config: AppConfig,
) -> tuple[dict, str]:
    '''Подготовка снимка конфигурации и хэша обработки'''

    snapshot = build_config_snapshot(config=config)

    processing_config_payload = build_processing_config_payload(
        config_snapshot=snapshot,
    )

    config_hash = calculate_config_hash(
        processing_config=processing_config_payload,
    )

    return snapshot, config_hash


def create_processing_run(
    session: Session,
    video: Video,
    config_path: str | Path,
    config: AppConfig,
    start_frame: int = 0,
    frame_limit: int | None = None
) -> ProcessingRun:
    '''Создание записи процесса обработки в БД'''

    snapshot, config_hash = _build_processing_run_config(config=config)

    process_run = ProcessingRun(
        video_id=video.id,
        status='pending',
        config_path=str(config_path),
        config_snapshot=snapshot,
        config_hash=config_hash,
        start_frame=start_frame,
        frame_limit=frame_limit,
    )

    session.add(process_run)
    session.commit()
    session.refresh(process_run)

    return process_run