from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from pechvision.db.models import Receipt
from pechvision.receipts.reader import normalize_receipts_file


def import_receipts(session: Session, path: str | Path) -> dict:
    '''Сервис импорта файлов чеков и запись данных в БД'''

    data_file = normalize_receipts_file(path)

    external_ids = [
        row['external_receipt_id']
        for row in data_file
        if row['external_receipt_id']
    ]

    existing_ids = set(
        session.scalars(
            select(Receipt.external_receipt_id).where(
                Receipt.external_receipt_id.in_(external_ids)
            )
        ).all()
    )

    skipped = 0
    created = 0

    for row in data_file:
        external_receipt_id = row.get('external_receipt_id')

        if not external_receipt_id:
            skipped += 1
            continue

        if external_receipt_id in existing_ids:
            skipped += 1
            continue

        session.add(Receipt(**row))
        existing_ids.add(external_receipt_id)
        created += 1

    session.commit()

    return {
        'all_rows': len(data_file),
        'created': created,
        'skipped': skipped
    }