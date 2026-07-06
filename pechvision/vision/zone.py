from typing import Any

import cv2
import numpy as np

from pechvision.config.schema import CashierZoneConfig


def get_bbox_point(bbox: list[int], point_policy: str) -> tuple[int, int]:
    '''Возвращает контрольную точку bbox для проверки кассовой зоны'''

    if point_policy != 'bottom_center':
        raise ValueError(f'Неподдерживаемая point_policy: {point_policy}')
    
    x1, _y1, x2, y2 = bbox
    return (x1 + x2) // 2, y2


def is_point_in_polygon(
    point: tuple[int, int],
    polygon: list[tuple[int, int]]
) -> bool:
    '''Проверка вхождения точки в полигон (кассовую зону)'''

    polygon_array = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(polygon_array, point, False)
    return result >= 0


def filter_detections_in_zone(
    detections: list[dict[str, Any]],
    zone_config: CashierZoneConfig
) -> list[dict[str, Any]]:
    '''Оставляет только детекции внутри кассовой зоны'''

    filtered_detections = []

    for detection in detections:
        point = get_bbox_point(
            bbox=detection['bbox'],
            point_policy=zone_config.point_policy
        )

        if not is_point_in_polygon(point, zone_config.polygon):
            continue

        filtered_detection = detection.copy()
        filtered_detection['zone_point'] = [point[0], point[1]]
        filtered_detections.append(filtered_detection)
        
    return filtered_detections
