from collections.abc import Sequence
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from pechvision.db.models import Visit, VisitSession
from pechvision.visits.session_builder import (
    VisitSessionData,
    build_visit_session_data,
    group_visits_into_session_candidates,
)

VisitSessionWriteStatus = Literal[
    'created',
    'updated',
    'unchanged',
]

def load_visits_for_session_building(
    session: Session,
    video_id: int
) -> list[Visit]:
    '''Загрузка исходных визитов видео для построения сессий'''

    select_visits = (
        select(Visit)
        .where(Visit.video_id == video_id)
        .order_by(
            Visit.entry_frame_index.asc().nullslast(),
            Visit.id.asc()
        )
    )
    return list(session.scalars(select_visits).all())


def build_session_data_from_visits(
    visits: Sequence[Visit],
    merge_timeout_seconds: int
) -> list[VisitSessionData]:
    '''Сборка данных логического визита'''

    session_candidates = group_visits_into_session_candidates(
        visits=visits,
        merge_timeout_seconds=merge_timeout_seconds
    )

    visit_session_data = []

    for candidate in session_candidates:
        session_data = build_visit_session_data(
            candidate=candidate
        )

        visit_session_data.append(session_data)
    return visit_session_data


def build_session_data_for_video(
    session: Session,
    video_id: int,
    merge_timeout_seconds: int
) -> list[VisitSessionData]:
    '''Построение данных сессии по video_id'''

    visits = load_visits_for_session_building(
        session=session,
        video_id=video_id
    )

    visit_session_data = build_session_data_from_visits(
        visits=visits,
        merge_timeout_seconds=merge_timeout_seconds
    )
    return visit_session_data


def build_visit_session_metadata(
    data: VisitSessionData,
    merge_timeout_seconds: int
) -> dict[str, object]:
    '''Формирование технического словаря логического визита'''

    if merge_timeout_seconds <= 0:
        raise ValueError('merge_timeout_seconds должен быть > 0')

    return {
        'source_visit_ids': list(data.source_visit_ids),
        'source_event_keys': list(data.source_event_keys),
        'merge_timeout_seconds': merge_timeout_seconds
    }


def upsert_visit_session(
    session: Session,
    data: VisitSessionData,
    merge_timeout_seconds: int,
) -> tuple[VisitSession, VisitSessionWriteStatus]:
    '''Создание или обновление логической сессии.'''

    existing_visit_session = session.scalar(
        select(VisitSession).where(
            VisitSession.session_key == data.session_key
        )
    )

    generated_metadata = build_visit_session_metadata(
        data=data,
        merge_timeout_seconds=merge_timeout_seconds,
    )

    if existing_visit_session is None:
        db_visit_session = VisitSession(
            session_key=data.session_key,
            video_id=data.video_id,
            person_id=data.person_id,
            visit_date=data.visit_date,
            entered_at=data.entered_at,
            left_at=data.left_at,
            duration_seconds=data.duration_seconds,
            entry_frame_index=data.entry_frame_index,
            exit_frame_index=data.exit_frame_index,
            segments_count=data.segments_count,
            time_is_estimated=data.time_is_estimated,
            extra_data=generated_metadata,
        )

        session.add(db_visit_session)
        session.flush()

        return db_visit_session, 'created'

    if existing_visit_session.video_id != data.video_id:
        raise ValueError(
            'Существующая сессия с таким session_key '
            'относится к другому видео'
        )

    updated_metadata = dict(
        existing_visit_session.extra_data or {}
    )
    updated_metadata.update(generated_metadata)

    field_values: dict[str, object] = {
        'person_id': data.person_id,
        'visit_date': data.visit_date,
        'entered_at': data.entered_at,
        'left_at': data.left_at,
        'duration_seconds': data.duration_seconds,
        'entry_frame_index': data.entry_frame_index,
        'exit_frame_index': data.exit_frame_index,
        'segments_count': data.segments_count,
        'time_is_estimated': data.time_is_estimated,
        'extra_data': updated_metadata,
    }

    has_changes = False

    for field_name, new_value in field_values.items():
        current_value = getattr(
            existing_visit_session,
            field_name,
        )

        if current_value == new_value:
            continue

        setattr(
            existing_visit_session,
            field_name,
            new_value,
        )
        has_changes = True

    if has_changes:
        return existing_visit_session, 'updated'

    return existing_visit_session, 'unchanged'


def synchronize_visit_sessions_for_video(
    session: Session,
    video_id: int,
    merge_timeout_seconds: int,
) -> dict[str, int]:
    '''Синхронизация логических сессий одного видео.'''

    if video_id <= 0:
        raise ValueError('video_id должен быть > 0')

    if merge_timeout_seconds <= 0:
        raise ValueError(
            'merge_timeout_seconds должен быть > 0'
        )

    visits = load_visits_for_session_building(
        session=session,
        video_id=video_id,
    )

    session_data_items = build_session_data_from_visits(
        visits=visits,
        merge_timeout_seconds=merge_timeout_seconds,
    )

    visits_by_id: dict[int, Visit] = {}

    for visit in visits:
        if visit.id is None:
            raise ValueError(
                'Нельзя синхронизировать Visit без ID'
            )

        if visit.id in visits_by_id:
            raise ValueError(
                f'Обнаружен повторяющийся Visit ID: {visit.id}'
            )

        visits_by_id[visit.id] = visit

    existing_sessions = list(
        session.scalars(
            select(VisitSession).where(
                VisitSession.video_id == video_id
            )
        ).all()
    )

    existing_by_key = {
        item.session_key: item
        for item in existing_sessions
    }

    desired_keys = {
        data.session_key
        for data in session_data_items
    }

    if len(desired_keys) != len(session_data_items):
        raise ValueError(
            'Обнаружены повторяющиеся session_key '
            'в рассчитанных сессиях'
        )

    for visit in visits:
        visit.visit_session = None

    created = 0
    updated = 0
    unchanged = 0
    linked_visit_ids: set[int] = set()

    for data in session_data_items:
        if data.video_id != video_id:
            raise ValueError(
                'Рассчитанная сессия относится '
                'к другому видео'
            )

        db_visit_session, status = upsert_visit_session(
            session=session,
            data=data,
            merge_timeout_seconds=merge_timeout_seconds,
        )

        if status == 'created':
            created += 1
        elif status == 'updated':
            updated += 1
        elif status == 'unchanged':
            unchanged += 1
        else:
            raise ValueError(
                f'Неизвестный статус записи сессии: {status}'
            )

        for source_visit_id in data.source_visit_ids:
            if source_visit_id in linked_visit_ids:
                raise ValueError(
                    'Один Visit попал в несколько сессий: '
                    f'{source_visit_id}'
                )

            source_visit = visits_by_id.get(
                source_visit_id
            )

            if source_visit is None:
                raise ValueError(
                    'Исходный Visit не найден: '
                    f'{source_visit_id}'
                )

            if source_visit.video_id != video_id:
                raise ValueError(
                    'Исходный Visit относится '
                    'к другому видео: '
                    f'{source_visit_id}'
                )

            source_visit.visit_session = db_visit_session
            linked_visit_ids.add(source_visit_id)

    stale_keys = set(existing_by_key) - desired_keys

    for stale_key in stale_keys:
        session.delete(existing_by_key[stale_key])

    if not linked_visit_ids.issubset(
        set(visits_by_id)
    ):
        raise ValueError(
            'Найдены связи с неизвестными Visit'
        )

    session.flush()

    return {
        'visits_total': len(visits),
        'sessions_total': len(session_data_items),
        'created': created,
        'updated': updated,
        'unchanged': unchanged,
        'deleted': len(stale_keys),
        'linked_visits': len(linked_visit_ids),
        'unlinked_visits': (
            len(visits) - len(linked_visit_ids)
        ),
    }