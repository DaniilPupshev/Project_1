from functools import lru_cache
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from pechvision.config.schema import DetectionConfig


@lru_cache(maxsize=4)
def get_person_detector(model_path: str) -> YOLO:
    '''Загрузка и переиспользование YOLO модели'''

    return YOLO(model_path)


def detect_people(frame, config: DetectionConfig) -> list[dict[str, Any]]:
    '''Детекция людей на кадре'''

    model_path = Path(config.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f'Файл модели детекции не найден: {model_path}')
    
    model = get_person_detector(str(model_path))

    results = model.predict(
        source=frame,
        conf=config.confidence_threshold,
        iou=config.iou_threshold,
        classes=[config.person_class_id],
        verbose=False,
    )

    detections = []

    if not results:
        return detections
    
    result = results[0]

    if result.boxes is None:
        return detections
    
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        detections.append(
            {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': confidence,
                'class_id': class_id
            }
        )
    return detections