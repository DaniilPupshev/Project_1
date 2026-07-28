from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pechvision.config.schema import AppConfig
from pechvision.db.models import Receipt, ReceiptMatch, VisitSession


def build_receipt_match_window(
    visit_session: VisitSession,
    window_before_seconds: int,
    window_after_seconds: int,
) -> tuple[datetime, datetime]:
    '''Создает окно поиска чеков для логической сессии.'''

    entered_at = visit_session.entered_at
    left_at = visit_session.left_at

    if entered_at is None or left_at is None:
        raise ValueError('entered_at и left_at не могут быть None')

    return (
        entered_at - timedelta(seconds=window_before_seconds),
        left_at + timedelta(seconds=window_after_seconds),
    )


def calculate_time_delta_seconds(
    visit_session: VisitSession,
    receipt: Receipt,
) -> float:
    '''Рассчитывает расстояние между выходом и закрытием чека.'''

    if visit_session.left_at is None:
        raise ValueError('left_at не может быть None')

    return abs(
        (receipt.closed_at - visit_session.left_at).total_seconds()
    )


def find_candidate_receipts(
    session: Session,
    visit_session: VisitSession,
    window_before_seconds: int,
    window_after_seconds: int,
) -> list[Receipt]:
    '''Возвращает чеки в разрешенном временном окне.'''

    window_start, window_end = build_receipt_match_window(
        visit_session=visit_session,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds,
    )

    return list(
        session.scalars(
            select(Receipt)
            .where(Receipt.closed_at >= window_start)
            .where(Receipt.closed_at <= window_end)
            .order_by(Receipt.closed_at, Receipt.id)
        ).all()
    )


def choose_best_receipt_for_session(
    visit_session: VisitSession,
    receipts: list[Receipt],
) -> tuple[Receipt | None, float | None, bool]:
    '''Выбирает ближайший чек и отмечает неоднозначное окно.'''

    if not receipts:
        return None, None, False

    best_receipt = min(
        receipts,
        key=lambda receipt: (
            calculate_time_delta_seconds(
                visit_session,
                receipt,
            ),
            receipt.id,
        ),
    )

    return (
        best_receipt,
        calculate_time_delta_seconds(
            visit_session,
            best_receipt,
        ),
        len(receipts) > 1,
    )


def load_visit_sessions_for_matching(
    session: Session,
    video_id: int | None,
) -> list[VisitSession]:
    '''Загружает логические сессии с полными временными границами.'''

    statement = (
        select(VisitSession)
        .where(VisitSession.entered_at.is_not(None))
        .where(VisitSession.left_at.is_not(None))
        .order_by(
            VisitSession.entered_at,
            VisitSession.id,
        )
    )

    if video_id is not None:
        statement = statement.where(
            VisitSession.video_id == video_id
        )

    return list(session.scalars(statement).all())


def synchronize_group_match_flags(
    session: Session,
) -> int:
    '''Обновляет флаг группового чека по числу связанных сессий.'''

    group_counts = dict(
        session.execute(
            select(
                ReceiptMatch.receipt_id,
                func.count(ReceiptMatch.id),
            )
            .group_by(ReceiptMatch.receipt_id)
        ).all()
    )
    updated = 0

    for match in session.scalars(select(ReceiptMatch)):
        expected = group_counts.get(match.receipt_id, 0) > 1

        if match.is_group_match == expected:
            continue

        match.is_group_match = expected
        updated += 1

    return updated


def match_receipts_to_visit_sessions(
    session: Session,
    config: AppConfig,
    video_id: int | None = None,
) -> dict[str, int]:
    '''Синхронизирует чеки с логическими сессиями посещений.'''

    visit_sessions = load_visit_sessions_for_matching(
        session=session,
        video_id=video_id,
    )
    created = 0
    updated = 0
    unchanged = 0
    deleted = 0
    sessions_without_receipt = 0
    ambiguous_matches = 0

    for visit_session in visit_sessions:
        candidates = find_candidate_receipts(
            session=session,
            visit_session=visit_session,
            window_before_seconds=(
                config.matching.window_before_seconds
            ),
            window_after_seconds=(
                config.matching.window_after_seconds
            ),
        )
        best_receipt, best_delta, is_ambiguous = (
            choose_best_receipt_for_session(
                visit_session=visit_session,
                receipts=candidates,
            )
        )
        existing_match = session.scalar(
            select(ReceiptMatch).where(
                ReceiptMatch.visit_session_id
                == visit_session.id
            )
        )

        if best_receipt is None or best_delta is None:
            sessions_without_receipt += 1

            if existing_match is not None:
                session.delete(existing_match)
                deleted += 1

            continue

        match_data = {
            'receipt_id': best_receipt.id,
            'matched_at': datetime.now(UTC),
            'time_delta_seconds': best_delta,
            'confidence': None,
            'is_ambiguous': is_ambiguous,
            'policy': config.matching.ambiguity_policy,
            'extra_data': {
                'candidate_receipt_ids': [
                    receipt.id
                    for receipt in candidates
                ],
                'candidates_count': len(candidates),
                'person_id': visit_session.person_id,
                'segments_count': (
                    visit_session.segments_count
                ),
            },
        }

        if existing_match is None:
            session.add(
                ReceiptMatch(
                    visit_session_id=visit_session.id,
                    is_group_match=False,
                    **match_data,
                )
            )
            created += 1
        else:
            comparable_fields = {
                key: value
                for key, value in match_data.items()
                if key != 'matched_at'
            }
            has_changes = any(
                getattr(existing_match, key) != value
                for key, value in comparable_fields.items()
            )

            if has_changes:
                for key, value in match_data.items():
                    setattr(existing_match, key, value)

                updated += 1
            else:
                unchanged += 1

        if is_ambiguous:
            ambiguous_matches += 1

    session.flush()
    group_flags_updated = synchronize_group_match_flags(
        session=session,
    )
    session.commit()

    return {
        'sessions_checked': len(visit_sessions),
        'matches_created': created,
        'matches_updated': updated,
        'matches_unchanged': unchanged,
        'matches_deleted': deleted,
        'sessions_without_receipt': sessions_without_receipt,
        'ambiguous_matches': ambiguous_matches,
        'group_flags_updated': group_flags_updated,
    }
