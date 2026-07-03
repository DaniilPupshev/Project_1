from pathlib import Path

import cv2


def read_video_metadata(path: str | Path) -> dict:
    '''Получение мета-данных обрабатываемого видео'''

    video_path = Path(path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f'Не удалось прочитать видео: {path}')

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_name = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    finally:
        cap.release()

    duration_seconds = None
    if fps > 0:
        duration_seconds = frame_count / fps

    return {
        'source_path': str(path),
        'stored_path': None,
        'filename': str(video_path.name),
        'camera_name': None,
        'fps': float(fps) if fps else None,
        'width': width if width else None,
        'height': height if height else None,
        'frame_count': int(frame_count) if frame_count else None,
        'duration_seconds': duration_seconds,
        'recorded_start_at': None,
        'recorded_end_at': None,
        'extra_data': {
            'resolution': (
                f'{width}x{height}'
                if width and height
                else None
            ),
            'type_format': str(video_path.suffix.lower()),
            'codec': str(codec_name),
        }
    }