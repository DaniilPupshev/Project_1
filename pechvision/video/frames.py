from collections.abc import Iterator
from pathlib import Path

import cv2

from pechvision.video.metadata import read_video_metadata


def iter_video_frames(
    path: str | Path,
    frame_step: int,
    metadata: dict | None = None,
) -> Iterator[dict]:
    '''Хелпер-функция чтения кадров видео'''

    if metadata is None:
        metadata = read_video_metadata(path=path)

    if frame_step < 1:
        raise ValueError(
            f'''
            Параметр frame_step должен быть >= 1\n
            Сейчас: frame_step={frame_step}
            '''
        )

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError('Не удалось прочитать видео')
    
    fps = metadata['fps']

    frame_index = 0

    try:
        while True:   
            success, frame = cap.read()

            if not success:
                break

            if frame_index % frame_step == 0:
                timestamp_seconds = frame_index / fps if fps > 0 else None

                yield {
                    'frame_index': frame_index,
                    'timestamp_seconds': timestamp_seconds,
                    'frame': frame
                }
            
            frame_index += 1
    finally:
        cap.release()