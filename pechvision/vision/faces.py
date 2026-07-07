from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pechvision.config.schema import AppConfig, FacesConfig


def clip_bbox_to_frame(
    frame: np.ndarray,
    bbox: list[int] | tuple[float, float, float, float] | np.ndarray | None,
) -> list[int] | None:
    '''Обрезает bbox по границам кадра.'''

    if bbox is None or len(bbox) != 4:
        return None

    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def crop_bbox_from_frame(
    frame: np.ndarray,
    bbox: list[int] | tuple[float, float, float, float] | np.ndarray | None,
) -> np.ndarray | None:
    '''Вырезает bbox из кадра.'''

    clipped_bbox = clip_bbox_to_frame(frame, bbox)

    if clipped_bbox is None:
        return None

    x1, y1, x2, y2 = clipped_bbox

    return frame[y1:y2, x1:x2].copy()


def expand_bbox(
    frame: np.ndarray,
    bbox: list[int],
    x_scale: float = 0.10,
    y_top_scale: float = 0.40,
    y_bottom_scale: float = 0.10,
) -> list[int] | None:
    '''Расширяет bbox, чтобы лицо не терялось на краях детекции человека.'''

    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    expanded_bbox = [
        int(x1 - bbox_width * x_scale),
        int(y1 - bbox_height * y_top_scale),
        int(x2 + bbox_width * x_scale),
        int(y2 + bbox_height * y_bottom_scale),
    ]

    return clip_bbox_to_frame(frame, expanded_bbox)


def is_bbox_center_inside(
    inner_bbox: list[int],
    outer_bbox: list[int],
) -> bool:
    '''Проверяет, что центр внутреннего bbox попадает во внешний bbox.'''

    inner_x1, inner_y1, inner_x2, inner_y2 = inner_bbox
    outer_x1, outer_y1, outer_x2, outer_y2 = outer_bbox

    center_x = (inner_x1 + inner_x2) / 2
    center_y = (inner_y1 + inner_y2) / 2

    return (
        outer_x1 <= center_x <= outer_x2
        and outer_y1 <= center_y <= outer_y2
    )


def calculate_bbox_center_distance(
    first_bbox: list[int],
    second_bbox: list[int],
) -> float:
    '''Считает расстояние между центрами bbox.'''

    first_center_x = (first_bbox[0] + first_bbox[2]) / 2
    first_center_y = (first_bbox[1] + first_bbox[3]) / 2
    second_center_x = (second_bbox[0] + second_bbox[2]) / 2
    second_center_y = (second_bbox[1] + second_bbox[3]) / 2

    return float(
        (
            (first_center_x - second_center_x) ** 2
            + (first_center_y - second_center_y) ** 2
        ) ** 0.5
    )


def is_face_position_valid(
    face_bbox: list[int],
    person_bbox: list[int],
) -> bool:
    '''Отбрасывает ложные лица в верхнем фоне bbox человека.'''

    face_x1, face_y1, face_x2, face_y2 = face_bbox
    person_x1, person_y1, person_x2, person_y2 = person_bbox

    face_center_x = (face_x1 + face_x2) / 2
    face_center_y = (face_y1 + face_y2) / 2
    person_height = person_y2 - person_y1

    if person_height <= 0:
        return False

    center_is_inside_person = (
        person_x1 <= face_center_x <= person_x2
        and person_y1 <= face_center_y <= person_y2
    )

    if not center_is_inside_person:
        return True

    relative_face_center_y = (face_center_y - person_y1) / person_height

    return relative_face_center_y >= 0.12


def calculate_face_quality(face_crop: np.ndarray | None) -> float:
    '''Считает простую оценку качества face crop.'''

    if face_crop is None:
        return .0

    height, width = face_crop.shape[:2]

    if height == 0 or width == 0:
        return .0

    area_score = height * width
    gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    sharpness_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return float(area_score + sharpness_score)


def normalize_embedding(embedding: np.ndarray | None) -> list[float] | None:
    '''Готовит embedding InsightFace для JSONB.'''

    if embedding is None:
        return None

    return [float(value) for value in embedding.tolist()]


def normalize_gender(gender: Any) -> str | None:
    '''Приводит gender InsightFace к строке.'''

    if gender is None:
        return None

    gender_value = int(gender)

    if gender_value == 1:
        return 'male'

    if gender_value == 0:
        return 'female'

    return None


def normalize_age(age: Any) -> float | None:
    '''Приводит age InsightFace к float для БД.'''

    if age is None:
        return None

    return float(age)


@lru_cache(maxsize=1)
def get_insightface_app(
    model_name: str,
    model_root: str,
    providers: tuple[str, ...],
    det_size: tuple[int, int],
    det_threshold: float,
):
    '''Ленивая инициализация InsightFace FaceAnalysis.'''

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name=model_name,
        root=model_root,
        providers=list(providers),
    )
    app.prepare(
        ctx_id=-1,
        det_size=det_size,
        det_thresh=det_threshold,
    )

    return app


def build_insightface_app(config: FacesConfig):
    '''Создает или возвращает кэшированный InsightFace app из конфига.'''

    model_root = str(Path(config.model_root))
    providers = tuple(config.providers)
    det_size = tuple(config.det_size)

    return get_insightface_app(
        model_name=config.model_name,
        model_root=model_root,
        providers=providers,
        det_size=det_size,
        det_threshold=config.det_threshold,
    )


def build_face_candidate_from_insightface_face(
    frame: np.ndarray,
    insightface_face,
    config: FacesConfig,
    source: str,
) -> dict[str, Any] | None:
    '''Преобразует объект InsightFace Face в внутренний face candidate.'''

    bbox = clip_bbox_to_frame(frame, insightface_face.bbox)

    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    face_width = x2 - x1
    face_height = y2 - y1

    if face_width < config.min_face_size or face_height < config.min_face_size:
        return None

    face_crop_bgr = crop_bbox_from_frame(frame, bbox)

    if face_crop_bgr is None:
        return None

    face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    embedding = getattr(insightface_face, 'normed_embedding', None)

    if embedding is None:
        embedding = getattr(insightface_face, 'embedding', None)

    return {
        'face_crop': face_crop_rgb,
        'bbox': bbox,
        'confidence': float(insightface_face.det_score),
        'quality_score': calculate_face_quality(face_crop_rgb),
        'source': source,
        'detector': 'insightface',
        'model_name': config.model_name,
        'embedding': normalize_embedding(embedding),
        'gender': normalize_gender(getattr(insightface_face, 'gender', None)),
        'age_estimate': normalize_age(getattr(insightface_face, 'age', None)),
    }


def detect_faces_in_frame(
    frame: np.ndarray,
    config: FacesConfig,
    source: str = 'full_frame',
) -> list[dict[str, Any]]:
    '''Ищет лица на полном кадре через InsightFace.'''

    app = build_insightface_app(config)
    detected_faces = app.get(frame)
    face_candidates = []

    for detected_face in detected_faces:
        face_candidate = build_face_candidate_from_insightface_face(
            frame=frame,
            insightface_face=detected_face,
            config=config,
            source=source,
        )

        if face_candidate is None:
            continue

        face_candidates.append(face_candidate)

    return face_candidates


def detect_faces_in_person_crop(
    frame: np.ndarray,
    person_bbox: list[int],
    config: FacesConfig,
) -> list[dict[str, Any]]:
    '''Ищет лица внутри crop человека через InsightFace.'''

    clipped_person_bbox = clip_bbox_to_frame(frame, person_bbox)

    if clipped_person_bbox is None:
        return []

    person_crop = crop_bbox_from_frame(frame, clipped_person_bbox)

    if person_crop is None:
        return []

    crop_face_candidates = detect_faces_in_frame(
        frame=person_crop,
        config=config,
        source='person_crop',
    )
    person_x1, person_y1, _, _ = clipped_person_bbox
    face_candidates = []

    for crop_face_candidate in crop_face_candidates:
        candidate = crop_face_candidate.copy()
        x1, y1, x2, y2 = candidate['bbox']
        candidate['bbox'] = [
            x1 + person_x1,
            y1 + person_y1,
            x2 + person_x1,
            y2 + person_y1,
        ]

        if not is_face_position_valid(
            face_bbox=candidate['bbox'],
            person_bbox=clipped_person_bbox,
        ):
            continue

        face_candidates.append(candidate)

    return face_candidates


def find_best_associated_face(
    frame: np.ndarray,
    person_bbox: list[int],
    face_candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    '''Выбирает лицо, которое геометрически лучше всего относится к человеку.'''

    clipped_person_bbox = clip_bbox_to_frame(frame, person_bbox)

    if clipped_person_bbox is None:
        return None

    expanded_person_bbox = expand_bbox(frame, clipped_person_bbox)

    if expanded_person_bbox is None:
        return None

    associated_faces = []

    for face_candidate in face_candidates:
        face_bbox = face_candidate.get('bbox')

        if face_bbox is None:
            continue

        if not is_bbox_center_inside(
            inner_bbox=face_bbox,
            outer_bbox=expanded_person_bbox,
        ):
            continue

        if not is_face_position_valid(
            face_bbox=face_bbox,
            person_bbox=clipped_person_bbox,
        ):
            continue

        candidate = face_candidate.copy()
        candidate['association_distance'] = calculate_bbox_center_distance(
            first_bbox=face_bbox,
            second_bbox=clipped_person_bbox,
        )
        associated_faces.append(candidate)

    if not associated_faces:
        return None

    return max(
        associated_faces,
        key=lambda candidate: (
            candidate.get('quality_score') or 0,
            -(candidate.get('association_distance') or 0),
        ),
    )


def detect_faces_for_tracks(
    frame: np.ndarray,
    tracks: list[dict[str, Any]],
    config: AppConfig,
) -> dict[int, dict[str, Any]]:
    '''Ищет лица на кадре и связывает лучшие candidates с track_id.'''

    if not config.faces.save_best_face or not tracks:
        return {}

    faces_by_track_id = {}
    frame_face_candidates = detect_faces_in_frame(
        frame=frame,
        config=config.faces,
        source='full_frame',
    )

    for track in tracks:
        track_id = int(track['track_id'])
        person_bbox = track.get('bbox')

        if person_bbox is None:
            continue

        best_face = find_best_associated_face(
            frame=frame,
            person_bbox=person_bbox,
            face_candidates=frame_face_candidates,
        )

        if best_face is None:
            track_face_candidates = detect_faces_in_person_crop(
                frame=frame,
                person_bbox=person_bbox,
                config=config.faces,
            )

            if track_face_candidates:
                best_face = max(
                    track_face_candidates,
                    key=lambda candidate: candidate.get('quality_score') or 0,
                )

        if best_face is None:
            continue

        candidate = best_face.copy()
        candidate['track_id'] = track_id
        candidate['person_bbox'] = person_bbox
        candidate['track_confidence'] = track.get('confidence')

        faces_by_track_id[track_id] = candidate

    return faces_by_track_id
