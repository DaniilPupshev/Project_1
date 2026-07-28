from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from pechvision.config.schema import DetectionConfig


@lru_cache(maxsize=3)
def resolve_yolo_device(requested_device: str) -> str:
    '''Определяет доступное устройство для запуска YOLO.'''

    if requested_device == 'cpu':
        return 'cpu'

    mps_available = (
        torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )

    if requested_device == 'mps':
        if not mps_available:
            raise RuntimeError(
                'YOLO device=mps, но Apple Metal недоступен '
                'в текущей установке PyTorch'
            )

        return 'mps'

    if requested_device == 'auto':
        return 'mps' if mps_available else 'cpu'

    raise ValueError(
        f'Неизвестное устройство YOLO: {requested_device}'
    )


@lru_cache(maxsize=4)
def get_person_detector(model_path: str) -> YOLO:
    '''Загрузка и переиспользование YOLO модели'''

    return YOLO(model_path)


@lru_cache(maxsize=4)
def get_person_tracker(model_path: str) -> YOLO:
    '''Загружает отдельный экземпляр YOLO для трекинга.'''

    return YOLO(model_path)


def detect_people(frame, config: DetectionConfig) -> list[dict[str, Any]]:
    '''Детекция людей на кадре'''

    model_path = Path(config.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f'Файл модели детекции не найден: {model_path}')
    
    model = get_person_detector(str(model_path))

    results = model.predict(
        source=frame,
        device=resolve_yolo_device(config.device),
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
