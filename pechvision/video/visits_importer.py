from pathlib import Path
from typing import Any

import cv2
from sqlalchemy.orm import Session

from pechvision.db.models import Face, Visit


def build_best_face_extra_data(best_face: dict[str, Any] | None) -> dict[str, Any] | None:
    '''Собирает JSON-safe metadata по лучшему лицу визита.'''

    if not best_face:
        return None

    embedding = best_face.get('embedding')

    return {
        'frame_index': best_face.get('frame_index'),
        'timestamp_seconds': best_face.get('timestamp_seconds'),
        'bbox': best_face.get('bbox'),
        'person_bbox': best_face.get('person_bbox'),
        'confidence': best_face.get('confidence'),
        'quality_score': best_face.get('quality_score'),
        'source': best_face.get('source'),
        'detector': best_face.get('detector'),
        'model_name': best_face.get('model_name'),
        'track_confidence': best_face.get('track_confidence'),
        'association_distance': best_face.get('association_distance'),
        'embedding_dim': len(embedding) if embedding is not None else None,
        'gender': best_face.get('gender'),
        'age_estimate': best_face.get('age_estimate'),
    }


def build_visit_extra_data(visit: dict[str, Any]) -> dict[str, Any]:
    '''Собирает технические данные визита для JSONB metadata.'''

    return {
        'source_track_id': visit.get('track_id'),
        'ocr_entry_text': visit.get('ocr_entry_text'),
        'ocr_exit_text': visit.get('ocr_exit_text'),
        'entry_timestamp_seconds': visit.get('entry_timestamp_seconds'),
        'exit_timestamp_seconds': visit.get('exit_timestamp_seconds'),
        'observations_count': visit.get('observations_count'),
        'best_confidence': visit.get('best_confidence'),
        'best_face': build_best_face_extra_data(visit.get('best_face')),
        'track_observation_samples': visit.get('track_observation_samples'),
        'last_bbox': visit.get('last_bbox'),
    }


def save_best_face_for_visit(
    session: Session,
    visit_id: int,
    track_id: str,
    best_face: dict[str, Any] | None,
    faces_dir: str | Path | None,
) -> bool:
    '''Сохраняет лучший face crop визита в файл и таблицу faces.'''

    if not best_face or faces_dir is None:
        return False

    face_crop_rgb = best_face.get('face_crop')

    if face_crop_rgb is None:
        return False

    output_dir = Path(faces_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_index = best_face.get('frame_index')
    output_path = output_dir / f'visit_{visit_id}_track_{track_id}_face.jpg'
    face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
    saved = cv2.imwrite(str(output_path), face_crop_bgr)

    if not saved:
        return False

    db_face = Face(
        person_id=None,
        visit_id=visit_id,
        image_path=str(output_path),
        frame_index=frame_index,
        quality_score=best_face.get('quality_score'),
        embedding=best_face.get('embedding'),
        gender=best_face.get('gender'),
        gender_confidence=None,
        age_estimate=best_face.get('age_estimate'),
        age_bucket=None,
        is_best=True,
        extra_data=build_best_face_extra_data(best_face),
    )

    session.add(db_face)

    return True


def save_visits(
    session: Session,
    video_id: int,
    processing_run_id: int,
    visits: list[dict[str, Any]],
    faces_dir: str | Path | None = None,
) -> dict[str, int]:
    '''Запись запуска обработки видео и визитов в БД'''

    if not visits:
        return {
            'total': 0,
            'created': 0,
            'faces_created': 0,
        }
    
    created = 0
    faces_created = 0

    for visit in visits:
        ocr_entered_at = visit.get('ocr_entered_at')
        track_id = f'{video_id}_{visit["track_id"]}'

        db_visit = Visit(
            video_id=video_id,
            processing_run_id=processing_run_id,
            person_id=None,
            track_id=track_id,
            visit_date=ocr_entered_at.date() if ocr_entered_at else None,
            entered_at=ocr_entered_at,
            left_at=visit.get('ocr_left_at'),
            duration_seconds=visit.get('duration_seconds'),
            entry_frame_index=visit.get('entry_frame_index'),
            exit_frame_index=visit.get('exit_frame_index'),
            ocr_entered_at=ocr_entered_at,
            ocr_left_at=visit.get('ocr_left_at'),
            time_is_estimated=visit.get('time_is_estimated', True),
            is_staff=False,
            staff_id=None,
            is_group=False,
            extra_data=build_visit_extra_data(visit),
        )

        session.add(db_visit)
        session.flush()

        face_created = save_best_face_for_visit(
            session=session,
            visit_id=db_visit.id,
            track_id=track_id,
            best_face=visit.get('best_face'),
            faces_dir=faces_dir,
        )

        if face_created:
            faces_created += 1

        created += 1

    session.commit()

    return {
        'total': len(visits),
        'created': created,
        'faces_created': faces_created,
    }
