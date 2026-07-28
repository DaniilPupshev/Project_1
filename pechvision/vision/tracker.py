from pathlib import Path
from typing import Any

from pechvision.config.schema import DetectionConfig, TrackingConfig
from pechvision.vision.person_detector import (
    get_person_tracker,
    resolve_yolo_device,
)


def track_people(
    frame,
    detection_config: DetectionConfig,
    tracking_config: TrackingConfig
) -> list[dict[str, Any]]:
    '''Трекинг людей на кадре'''

    model_path = Path(detection_config.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f'Файл модели детекции не найден: {model_path}')
    
    model = get_person_tracker(str(model_path))

    results = model.track(
        source=frame,
        device=resolve_yolo_device(detection_config.device),
        conf=detection_config.confidence_threshold,
        iou=detection_config.iou_threshold,
        classes=[detection_config.person_class_id],
        tracker=f'{tracking_config.tracker_type}.yaml',
        persist=True,
        verbose=False,
    )

    tracks = []

    if not results:
        return tracks
    
    result = results[0]

    if result.boxes is None or result.boxes.id is None:
        return tracks
    
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        track_id = int(box.id[0])

        tracks.append(
            {
                'track_id': track_id,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'class_id': class_id
            }
        )
    return tracks


def reset_person_tracker(model_path: str | Path) -> None:
    '''Сбрасывает состояние трекера перед новой активной последовательностью.'''

    model = get_person_tracker(str(model_path))
    model.predictor = None
