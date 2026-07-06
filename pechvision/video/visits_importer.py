from typing import Any

from sqlalchemy.orm import Session

from pechvision.db.models import Visit


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
        'last_bbox': visit.get('last_bbox'),
    }


def save_visits(
    session: Session,
    video_id: int,
    processing_run_id: int,
    visits: list[dict[str, Any]]
) -> dict[str, int]:
    '''Запись запуска обработки видео и визитов в БД'''

    if not visits:
        return {
            'total': 0,
            'created': 0
        }
    
    created = 0

    for visit in visits:
        ocr_entered_at = visit.get('ocr_entered_at')

        db_visit = Visit(
            video_id=video_id,
            processing_run_id=processing_run_id,
            person_id=None,
            track_id=f'{video_id}_{visit["track_id"]}',
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
        created += 1

    session.commit()

    return {
        'total': len(visits),
        'created': created
    }