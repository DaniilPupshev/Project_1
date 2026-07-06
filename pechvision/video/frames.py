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


def iter_video_frames_range(
    path: str | Path,
    frame_step: int,
    start_frame: int = 0,
    limit: int | None = None,
    metadata: dict | None = None,
) -> Iterator[dict]:
    '''Читает выбранные кадры видео начиная с заданного frame_index'''

    if metadata is None:
        metadata = read_video_metadata(path=path)

    if frame_step < 1:
        raise ValueError(
            f'''
            Параметр frame_step должен быть >= 1\n
            Сейчас: frame_step={frame_step}
            '''
        )

    if start_frame < 0:
        raise ValueError(
            f'''
            Параметр start_frame должен быть >= 0\n
            Сейчас: start_frame={start_frame}
            '''
        )

    if limit is not None and limit < 1:
        raise ValueError(
            f'''
            Параметр limit должен быть >= 1 или None\n
            Сейчас: limit={limit}
            '''
        )

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError('Не удалось прочитать видео')

    fps = metadata['fps']
    frame_index = start_frame
    yielded = 0

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while True:
            success, frame = cap.read()

            if not success:
                break

            if (frame_index - start_frame) % frame_step == 0:
                timestamp_seconds = frame_index / fps if fps > 0 else None

                yield {
                    'frame_index': frame_index,
                    'timestamp_seconds': timestamp_seconds,
                    'frame': frame,
                }

                yielded += 1

                if limit is not None and yielded >= limit:
                    break

            frame_index += 1
    finally:
        cap.release()