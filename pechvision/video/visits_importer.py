from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
from sqlalchemy import select
from sqlalchemy.orm import Session

from pechvision.db.models import Face, Visit
from pechvision.identity.person_matcher import get_or_create_person_for_face


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


def build_visit_event_key(video_id: int, visit: dict[str, Any]) -> str:
    '''Создание event_key для визитов'''

    track_id = visit['track_id']
    entry_frame_index = visit.get('entry_frame_index')
    exit_frame_index = visit.get('exit_frame_index')

    return f'video:{video_id}:track:{track_id}:entry:{entry_frame_index}:exit:{exit_frame_index}'


def build_visit_extra_data(visit: dict[str, Any]) -> dict[str, Any]:
    '''Собирает технические данные визита для JSONB metadata.'''

    return {
        'source_track_id': visit.get('track_id'),
        'ocr_entry_text': visit.get('ocr_entry_text'),
        'ocr_exit_text': visit.get('ocr_exit_text'),
        'ocr_entry_frame_index': visit.get('ocr_entry_frame_index'),
        'ocr_exit_frame_index': visit.get('ocr_exit_frame_index'),
        'ocr_entry_candidates_count': visit.get('ocr_entry_candidates_count'),
        'ocr_exit_candidates_count': visit.get('ocr_exit_candidates_count'),
        'ocr_duration_difference_seconds': visit.get(
            'ocr_duration_difference_seconds'
        ),
        'ocr_rejection_reason': visit.get('ocr_rejection_reason'),
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
    db_visit: Visit,
    track_id: str,
    best_face: dict[str, Any] | None,
    faces_dir: str | Path | None,
    recognition_threshold: float,
) -> dict[str, int]:
    '''Сохраняет лучший face crop визита в файл и таблицу faces.'''

    if not best_face or faces_dir is None:
        return {
            'faces_created': 0,
            'persons_created': 0,
            'persons_matched': 0,
            'persons_best_face_updated': 0,
        }

    face_crop_rgb = best_face.get('face_crop')

    if face_crop_rgb is None:
        return {
            'faces_created': 0,
            'persons_created': 0,
            'persons_matched': 0,
            'persons_best_face_updated': 0,
        }

    output_dir = Path(faces_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_index = best_face.get('frame_index')
    output_path = output_dir / f'visit_{db_visit.id}_track_{track_id}_face.jpg'
    face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
    saved = cv2.imwrite(str(output_path), face_crop_bgr)

    if not saved:
        return {
            'faces_created': 0,
            'persons_created': 0,
            'persons_matched': 0,
            'persons_best_face_updated': 0,
        }
    
    embedding = best_face.get('embedding')
    seen_at = db_visit.entered_at or db_visit.ocr_entered_at

    face_quality_score = best_face.get('quality_score')

    person, person_created, person_similarity, best_face_updated = get_or_create_person_for_face(
        session=session,
        embedding=embedding,
        face_image_path=str(output_path),
        seen_at=seen_at,
        threshold=recognition_threshold,
        face_quality_score=face_quality_score
    )

    if person is not None:
        db_visit.person_id = person.id

    face_extra_data = build_best_face_extra_data(best_face) or {}
    face_extra_data['person_created'] = person_created
    face_extra_data['person_similarity'] = person_similarity
    face_extra_data['person_best_face_updated'] = best_face_updated

    db_face = Face(
        person_id=person.id if person is not None else None,
        visit_id=db_visit.id,
        image_path=str(output_path),
        frame_index=frame_index,
        quality_score=best_face.get('quality_score'),
        embedding=best_face.get('embedding'),
        gender=best_face.get('gender'),
        gender_confidence=None,
        age_estimate=best_face.get('age_estimate'),
        age_bucket=None,
        is_best=True,
        extra_data=face_extra_data,
    )

    session.add(db_face)

    return {
        'faces_created': 1,
        'persons_created': 1 if person_created else 0,
        'persons_matched': 1 if person is not None and not person_created else 0,
        'persons_best_face_updated': 1 if best_face_updated else 0,
    }


def save_visits(
    session: Session,
    video_id: int,
    processing_run_id: int,
    visits: list[dict[str, Any]],
    recognition_threshold: float,
    faces_dir: str | Path | None = None,
    progress_callback: (
        Callable[[str, int, int | None, dict[str, Any] | None], None] | None
    ) = None,
) -> dict[str, int]:
    '''Запись запуска обработки видео и визитов в БД'''

    if not visits:
        if progress_callback is not None:
            progress_callback('save', 1, 1, None)

        return {
            'total': 0,
            'created': 0,
            'faces_created': 0,
            'skipped_existing': 0,
            'persons_created': 0,
            'persons_matched': 0,
            'persons_best_face_updated': 0,
        }
    
    created = 0
    faces_created = 0
    skipped_existing = 0
    persons_created = 0
    persons_matched = 0
    persons_best_face_updated = 0

    total_visits = len(visits)

    for visit_index, visit in enumerate(visits, start=1):
        entered_at = visit.get('entered_at') or visit.get('ocr_entered_at')
        left_at = visit.get('left_at') or visit.get('ocr_left_at')
        track_id = f'{video_id}_{visit["track_id"]}'

        event_key = build_visit_event_key(video_id, visit)

        existing_visit = session.scalar(
            select(Visit).where(Visit.event_key == event_key)
        )

        if existing_visit is not None:
            skipped_existing += 1

            if progress_callback is not None:
                progress_callback(
                    'save',
                    visit_index,
                    total_visits,
                    {'visit_id': existing_visit.id, 'skipped': True},
                )

            continue

        db_visit = Visit(
            video_id=video_id,
            event_key=event_key,
            processing_run_id=processing_run_id,
            person_id=None,
            track_id=track_id,
            visit_date=entered_at.date() if entered_at else left_at.date() if left_at else None,
            entered_at=entered_at,
            left_at=left_at,
            duration_seconds=visit.get('duration_seconds'),
            entry_frame_index=visit.get('entry_frame_index'),
            exit_frame_index=visit.get('exit_frame_index'),
            ocr_entered_at=visit.get('ocr_entered_at'),
            ocr_left_at=visit.get('ocr_left_at'),
            time_is_estimated=visit.get('time_is_estimated', True),
            is_staff=False,
            staff_id=None,
            is_group=False,
            extra_data=build_visit_extra_data(visit),
        )

        session.add(db_visit)
        session.flush()

        face_stats = save_best_face_for_visit(
            session=session,
            db_visit=db_visit,
            track_id=track_id,
            best_face=visit.get('best_face'),
            faces_dir=faces_dir,
            recognition_threshold=recognition_threshold,
        )

        created += 1
        persons_best_face_updated += face_stats['persons_best_face_updated']
        faces_created += face_stats['faces_created']
        persons_created += face_stats['persons_created']
        persons_matched += face_stats['persons_matched']

        if progress_callback is not None:
            progress_callback(
                'save',
                visit_index,
                total_visits,
                {'visit_id': db_visit.id, 'skipped': False},
            )

    session.commit()

    return {
        'total': len(visits),
        'created': created,
        'faces_created': faces_created,
        'persons_created': persons_created,
        'persons_matched': persons_matched,
        'skipped_existing': skipped_existing,
        'persons_best_face_updated': persons_best_face_updated
    }
