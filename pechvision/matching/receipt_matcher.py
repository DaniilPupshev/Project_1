from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from pechvision.config.schema import AppConfig
from pechvision.db.models import Receipt, ReceiptMatch, Visit


def build_receipt_match_window(
    visit: Visit,
    window_before_seconds: int,
    window_after_seconds: int
) -> tuple[datetime, datetime]:
    '''Создание окна поиска чеков'''

    entered_at = visit.entered_at
    left_at = visit.left_at

    if entered_at is None or left_at is None:
        raise ValueError('entered_at и left_at не могут быть None')
    
    window_start = entered_at - timedelta(seconds=window_before_seconds)
    window_end = left_at + timedelta(seconds=window_after_seconds)

    return window_start, window_end


def calculate_time_delta_seconds(
    visit: Visit,
    receipt: Receipt
) -> float:
    '''Расчет близости чека к визиту'''

    if visit.left_at is None:
        raise ValueError('left_at не может быть None')

    delta = receipt.closed_at - visit.left_at

    return abs(delta.total_seconds())


def find_candidate_receipts(
    session: Session,
    visit: Visit,
    window_before_seconds: int,
    window_after_seconds: int
) -> list[Receipt]:
    '''Поиск кандидатов по чекам'''

    window_start, window_end = build_receipt_match_window(
        visit=visit,
        window_before_seconds=window_before_seconds,
        window_after_seconds=window_after_seconds
    )

    return list(
        session.scalars(
            select(Receipt)
            .where(Receipt.closed_at >= window_start)
            .where(Receipt.closed_at <= window_end)
            .order_by(Receipt.closed_at)
        ).all()
    )


def choose_best_receipt_for_visit(
    visit: Visit,
    receipts: list[Receipt]
) -> tuple[Receipt | None, float | None, bool]:
    '''Выбор лучшего чека для matching'''

    if not receipts:
        return None, None, False

    is_ambiguous = len(receipts) > 1
    best_receipt = min(
        receipts,
        key=lambda receipt: calculate_time_delta_seconds(visit, receipt),
    )
    best_delta = calculate_time_delta_seconds(visit, best_receipt)

    return best_receipt, best_delta, is_ambiguous


def receipt_match_exists(
    session: Session,
    visit_id: int,
    receipt_id: int
) -> bool:
    '''Проверка дублирования matches'''

    check = session.scalar(
        select(ReceiptMatch.id)
        .where(ReceiptMatch.visit_id == visit_id)
        .where(ReceiptMatch.receipt_id == receipt_id)
    )

    return check is not None
        

def match_receipts_to_visits(
    session: Session,
    config: AppConfig,
    video_id: int | None = None
) -> dict[str, int]:
    '''Основная функция matching чеков и person'''

    window_before_seconds = config.matching.window_before_seconds
    window_after_seconds = config.matching.window_after_seconds
    ambiguity_policy = config.matching.ambiguity_policy

    if video_id is None:
        visits = list(
            session.scalars(
                select(Visit)
                .where(Visit.entered_at.is_not(None))
                .where(Visit.left_at.is_not(None))
                .where(Visit.is_staff.is_(False))
            )
        )

    else:
        visits = list(
            session.scalars(
                select(Visit)
                .where(Visit.video_id == video_id)
                .where(Visit.entered_at.is_not(None))
                .where(Visit.left_at.is_not(None))
                .where(Visit.is_staff.is_(False))
            )
        )

    visits_checked = 0
    matches_created = 0
    matches_skipped_existing = 0
    visits_without_receipt = 0
    ambiguous_matches = 0

    for visit in visits:
        candidate = find_candidate_receipts(
            session=session,
            visit=visit,
            window_before_seconds=window_before_seconds,
            window_after_seconds=window_after_seconds
        )

        visits_checked += 1

        if not candidate:
            visits_without_receipt += 1
            continue

        best_receipt, best_delta, is_ambiguous = choose_best_receipt_for_visit(
            visit=visit,
            receipts=candidate
        )

        if best_receipt is None or best_delta is None:
            visits_without_receipt += 1
            continue

        double_check = receipt_match_exists(
            session=session,
            visit_id=visit.id,
            receipt_id=best_receipt.id
        )

        if double_check:
            matches_skipped_existing += 1
            continue

        match = ReceiptMatch(
            visit_id=visit.id,
            receipt_id=best_receipt.id,
            matched_at=datetime.now(UTC),
            time_delta_seconds=best_delta,
            confidence=None,
            is_ambiguous=is_ambiguous,
            is_group_match=False,
            policy=ambiguity_policy,
            extra_data={
                'candidate_receipt_ids': [receipt.id for receipt in candidate],
                'candidates_count': len(candidate),
            },
        )

        session.add(match)
        matches_created += 1

        if is_ambiguous:
            ambiguous_matches += 1
    
    session.commit()

    return {
        'visits_checked': visits_checked,
        'matches_created': matches_created,
        'matches_skipped_existing': matches_skipped_existing,
        'visits_without_receipt': visits_without_receipt,
        'ambiguous_matches': ambiguous_matches,
    }