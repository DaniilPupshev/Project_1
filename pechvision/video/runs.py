from pathlib import Path

from sqlalchemy.orm import Session

from pechvision.db.models import ProcessingRun, Video


def create_processing_run(
    session: Session,
    video: Video,
    config_path: str | Path
) -> ProcessingRun:
    '''Создание записи процесса обработки в БД'''

    process_run = ProcessingRun(
        video_id=video.id,
        status='pending',
        config_path=str(config_path),
    )

    session.add(process_run)
    session.commit()
    session.refresh(process_run)

    return process_run