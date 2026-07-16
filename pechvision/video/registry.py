from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from pechvision.db.models import Video
from pechvision.video.fingerprint import build_video_file_identity
from pechvision.video.metadata import read_video_metadata


def register_video(session: Session, path: str | Path) -> tuple[Video, bool]:
    '''Регистрация обрабатываемого видео в БД'''

    identity = build_video_file_identity(
        path=path
    )

    check_video = session.scalar(
        select(Video).where(Video.file_sha256 == identity['file_sha256'])
    )

    if check_video:
        return check_video, False

    metadata = read_video_metadata(path)

    existing_video = session.scalar(
        select(Video)
        .where(Video.source_path == metadata['source_path'])
        .where(Video.file_sha256.is_(None))
    )

    if existing_video:
        existing_video.file_sha256 = identity['file_sha256']
        existing_video.file_size_bytes = identity['file_size_bytes']

        session.commit()
        session.refresh(existing_video)

        return existing_video, False
    
    video = Video(**metadata, **identity)

    session.add(video)
    session.commit()
    session.refresh(video)
    return video, True