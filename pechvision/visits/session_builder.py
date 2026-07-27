import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from pechvision.db.models import Visit


@dataclass(slots=True)
class VisitSessionCandidate:
    '''Описание сырых строк для входа в логическую сессию'''

    video_id: int
    person_id: int | None
    visits: list[Visit]


@dataclass(slots=True)
class VisitSessionData:
    '''Описание данных логической сессии'''

    session_key: str
    video_id: int
    person_id: int | None
    visit_date: date | None
    entered_at: datetime | None
    left_at: datetime | None
    duration_seconds: float | None
    entry_frame_index: int | None
    exit_frame_index: int | None
    segments_count: int
    time_is_estimated: bool
    source_visit_ids: list[int]
    source_event_keys: list[str]


def build_visit_session_key(
    candidate: VisitSessionCandidate
) -> str:
    '''Формирование уникального ключа логического визита'''

    if len(candidate.visits) == 0:
        raise ValueError('Нельзя создать ключ для пустой сессии')

    sorted_event_keys = sorted(
        visit.event_key
        for visit in candidate.visits
    )

    sorted_value = '|'.join(sorted_event_keys)

    digest = hashlib.sha256(
        sorted_value.encode('utf-8')
    ).hexdigest()

    return f'session_{digest}'


def resolve_visit_date(visit: Visit) -> date | None:
    '''Определение календарной даты исходного визита'''

    if visit.visit_date is not None:
        return visit.visit_date

    if visit.entered_at is not None:
        return visit.entered_at.date()

    return None


def calculate_visit_gap_seconds(
    previous_left_at: datetime | None,
    current_entered_at: datetime | None
) -> float | None:
    '''Расчет разницы вход-выход визита'''

    if previous_left_at is None or current_entered_at is None:
        return None

    return (current_entered_at - previous_left_at).total_seconds()


def can_merge_visits(
    current_visits: Sequence[Visit],
    candidate: Visit,
    merge_timeout_seconds: int
) -> bool:
    '''Возможность объединения визитов'''

    if not current_visits:
        return False

    current_person_id = current_visits[0].person_id
    current_video_id = current_visits[0].video_id

    has_another_person = any(
        visit.person_id != current_person_id
        for visit in current_visits
    )

    has_another_video = any(
        visit.video_id != current_video_id
        for visit in current_visits
    )

    if has_another_video or has_another_person:
        return False

    current_visit_date = resolve_visit_date(current_visits[0])
    candidate_visit_date = resolve_visit_date(candidate)

    has_another_date = any(
        resolve_visit_date(visit) != current_visit_date
        for visit in current_visits
    )

    if (
        current_visit_date is None
        or candidate_visit_date is None
        or has_another_date
        or candidate_visit_date != current_visit_date
    ):
        return False

    person_mismatch = candidate.person_id != current_person_id
    video_mismatch = candidate.video_id != current_video_id

    check_left_at = max(
        (
            visit.left_at
            for visit in current_visits
            if visit.left_at is not None
        ),
        default=None,
    )

    gap_seconds = calculate_visit_gap_seconds(
        check_left_at,
        candidate.entered_at,
    )

    if gap_seconds is None:
        return False

    if (
        candidate.person_id is None
        or
        person_mismatch
        or
        video_mismatch
        or
        candidate.is_staff
        or
        any(visit.is_staff for visit in current_visits)
        or
        gap_seconds > merge_timeout_seconds
    ):
        return False
    return True


def group_visits_into_session_candidates(
    visits: Sequence[Visit],
    merge_timeout_seconds: int
) -> list[VisitSessionCandidate]:
    '''Группировка визитов в сессии'''

    grouped_visits = defaultdict(list)
    session_candidates = []

    for visit in visits:
        if visit.is_staff:
            continue

        if visit.person_id is None:
            session_candidates.append(
                VisitSessionCandidate(
                    video_id=visit.video_id,
                    person_id=None,
                    visits=[visit]
                )
            )
            continue

        key = (visit.video_id, visit.person_id)
        grouped_visits[key].append(visit)

    for (video_id, person_id), person_visits in grouped_visits.items():
        sorted_visits = sorted(
            person_visits,
            key=lambda visit: (
                (
                    visit.entry_frame_index
                    if visit.entry_frame_index is not None
                    else float('inf')
                ),
                visit.id if visit.id is not None else float('inf'),
            ),
        )

        current_visits = []

        for visit in sorted_visits:
            if not current_visits:
                current_visits.append(visit)
                continue

            if can_merge_visits(
                current_visits=current_visits,
                candidate=visit,
                merge_timeout_seconds=merge_timeout_seconds,
            ):
                current_visits.append(visit)
                continue

            session_candidates.append(
                VisitSessionCandidate(
                    video_id=video_id,
                    person_id=person_id,
                    visits=current_visits,
                )
            )

            current_visits = [visit]

        if current_visits:
            session_candidates.append(
                VisitSessionCandidate(
                    video_id=video_id,
                    person_id=person_id,
                    visits=current_visits,
                )
            )

    session_candidates.sort(
        key=lambda candidate: (
            candidate.video_id,
            (
                candidate.visits[0].entry_frame_index
                if candidate.visits[0].entry_frame_index is not None
                else float('inf')
            ),
            (
                candidate.visits[0].id
                if candidate.visits[0].id is not None
                else float('inf')
            ),
        )
    )

    return session_candidates


def build_visit_session_data(
    candidate: VisitSessionCandidate,
) -> VisitSessionData:
    '''Построение данных логического визита для БД'''

    if not candidate.visits:
        raise ValueError('Нельзя создать запись без визитов')

    current_visits = candidate.visits

    if any(
        visit.video_id != candidate.video_id
        for visit in current_visits
    ):
        raise ValueError(
            'video_id кандидата не соответствует исходным визитам'
        )

    if any(
        visit.person_id != candidate.person_id
        for visit in current_visits
    ):
        raise ValueError(
            'person_id кандидата не соответствует исходным визитам'
        )

    if any(visit.is_staff for visit in current_visits):
        raise ValueError(
            'Нельзя создать логический визит для персонала'
        )

    if any(visit.id is None for visit in current_visits):
        raise ValueError(
            'Нельзя создать логический визит без ID исходного визита'
        )

    entered_at = min(
        (
            visit.entered_at
            for visit in current_visits
            if visit.entered_at is not None
        ),
        default=None,
    )

    left_at = max(
        (
            visit.left_at
            for visit in current_visits
            if visit.left_at is not None
        ),
        default=None,
    )

    entry_frame_index = min(
        (
            visit.entry_frame_index
            for visit in current_visits
            if visit.entry_frame_index is not None
        ),
        default=None,
    )

    exit_frame_index = max(
        (
            visit.exit_frame_index
            for visit in current_visits
            if visit.exit_frame_index is not None
        ),
        default=None,
    )

    visit_date = min(
        (
            resolved_date
            for visit in current_visits
            if (resolved_date := resolve_visit_date(visit)) is not None
        ),
        default=None,
    )

    duration_seconds = None

    if (
        entered_at is not None
        and left_at is not None
        and left_at >= entered_at
    ):
        duration_seconds = (
            left_at - entered_at
        ).total_seconds()

    source_visit_ids = [
        visit.id
        for visit in current_visits
        if visit.id is not None
    ]

    return VisitSessionData(
        session_key=build_visit_session_key(candidate),
        video_id=candidate.video_id,
        person_id=candidate.person_id,
        visit_date=visit_date,
        entered_at=entered_at,
        left_at=left_at,
        duration_seconds=duration_seconds,
        entry_frame_index=entry_frame_index,
        exit_frame_index=exit_frame_index,
        segments_count=len(current_visits),
        time_is_estimated=any(
            bool(visit.time_is_estimated)
            for visit in current_visits
        ),
        source_visit_ids=source_visit_ids,
        source_event_keys=sorted(
            visit.event_key
            for visit in current_visits
        ),
    )