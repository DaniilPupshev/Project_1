from pathlib import Path

from sqlalchemy.orm import Session

from pechvision.config.schema import AppConfig
from pechvision.config.snapshot import (
    build_config_snapshot,
    build_processing_config_payload,
    calculate_config_hash,
)
from pechvision.db.models import ProcessingRun, Video


def create_processing_run(
    session: Session,
    video: Video,
    config_path: str | Path,
    config: AppConfig,
    start_frame: int = 0,
    frame_limit: int | None = None
) -> ProcessingRun:
    '''Создание записи процесса обработки в БД'''

    snapshot = build_config_snapshot(config=config)

    processing_config_payload = build_processing_config_payload(
        config_snapshot=snapshot
    )

    config_hash = calculate_config_hash(
        processing_config=processing_config_payload
    )

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