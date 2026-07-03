from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from pechvision.db.models import Video
from pechvision.video.metadata import read_video_metadata


def register_video(session: Session, path: str | Path) -> tuple[Video, bool]:
    '''Регистрация обрабатываемого видео в БД'''

    metadata = read_video_metadata(path)

    existing_video = session.scalar(
        select(Video).where(Video.source_path == metadata['source_path'])
    )

    if existing_video:
        return existing_video, False
    
    video = Video(**metadata)

    session.add(video)
    session.commit()
    session.refresh(video)
    return video, True