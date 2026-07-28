from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from pechvision.db.models import Face, Person, Staff, Visit


def has_active_staff(session: Session) -> bool:
    '''Проверяет наличие хотя бы одного активного эталона сотрудника'''

    staff_id = session.scalar(
        select(Staff.id)
        .where(Staff.is_active.is_(True))
        .where(Staff.face_embedding.is_not(None))
        .limit(1)
    )
    return staff_id is not None


def repair_persons_after_staff_classification(
    session: Session,
    person_ids: set[int],
) -> tuple[int, int]:
    '''Удаляет осиротевших person и пересобирает представителей смешанных person'''

    deleted = 0
    rebuilt = 0

    for person_id in sorted(person_ids):
        person = session.get(Person, person_id)

        if person is None:
            continue

        remaining_faces = list(
            session.scalars(
                select(Face)
                .where(Face.person_id == person_id)
                .where(Face.embedding.is_not(None))
                .order_by(Face.quality_score.desc().nullslast(), Face.id)
            ).all()
        )
        remaining_visits = list(
            session.scalars(
                select(Visit)
                .where(Visit.person_id == person_id)
                .order_by(Visit.entered_at, Visit.id)
            ).all()
        )

        if not remaining_faces and not remaining_visits:
            session.delete(person)
            deleted += 1
            continue

        if remaining_faces:
            best_face = remaining_faces[0]
            person.face_embedding = best_face.embedding
            person.best_face_path = best_face.image_path

            person_extra_data = dict(person.extra_data or {})
            person_extra_data['best_face_quality_score'] = best_face.quality_score
            person_extra_data['rebuilt_after_staff_classification'] = True
            person.extra_data = person_extra_data
        else:
            person.face_embedding = None
            person.best_face_path = None

        seen_times = [
            visit.entered_at
            for visit in remaining_visits
            if visit.entered_at is not None
        ]

        if seen_times:
            person.first_seen_at = min(seen_times)
            person.last_seen_at = max(seen_times)

        rebuilt += 1

    session.flush()
    return deleted, rebuilt


def find_matching_staff(
    session: Session,
    embedding: list[float] | None,
    threshold: float,
) -> tuple[Staff | None, float | None]:
    '''Ищет ближайшего активного сотрудника по cosine similarity'''

    if embedding is None or len(embedding) != 512:
        return None, None

    embedding_array = np.asarray(embedding, dtype=np.float32)

    if not np.isfinite(embedding_array).all():
        return None, None

    distance = Staff.face_embedding.cosine_distance(embedding)
    row = session.execute(
        select(Staff, distance.label('distance'))
        .where(Staff.is_active.is_(True))
        .where(Staff.face_embedding.is_not(None))
        .order_by(distance)
        .limit(1)
    ).first()

    if row is None:
        return None, None

    staff_member, distance_value = row
    similarity = 1 - float(distance_value)

    if similarity < threshold:
        return None, similarity

    return staff_member, similarity


def classify_existing_staff(
    session: Session,
    threshold: float,
    video_id: int | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    '''Находит сотрудников среди сохраненных лиц и опционально обновляет связи'''

    query = (
        select(Face, Visit)
        .join(Visit, Visit.id == Face.visit_id)
        .where(Face.embedding.is_not(None))
    )

    if video_id is not None:
        query = query.where(Visit.video_id == video_id)

    rows = session.execute(query.order_by(Visit.id, Face.id)).all()
    matches = []
    updated = 0
    affected_person_ids = set()

    for face, visit in rows:
        staff_member, similarity = find_matching_staff(
            session=session,
            embedding=face.embedding,
            threshold=threshold,
        )

        if staff_member is None or similarity is None:
            continue

        matches.append(
            {
                'visit_id': visit.id,
                'face_id': face.id,
                'staff_id': staff_member.id,
                'external_staff_key': staff_member.external_staff_key,
                'full_name': staff_member.full_name,
                'similarity': similarity,
                'already_classified': (
                    visit.is_staff and visit.staff_id == staff_member.id
                ),
            }
        )

        if not apply:
            continue

        already_classified = (
            visit.is_staff
            and visit.staff_id == staff_member.id
            and visit.person_id is None
            and face.person_id is None
        )

        if already_classified:
            continue

        previous_person_id = visit.person_id or face.person_id

        if previous_person_id is not None:
            affected_person_ids.add(previous_person_id)

        visit.is_staff = True
        visit.staff_id = staff_member.id
        visit.person_id = None
        face.person_id = None

        face_extra_data = dict(face.extra_data or {})
        face_extra_data.update(
            {
                'staff_id': staff_member.id,
                'external_staff_key': staff_member.external_staff_key,
                'staff_similarity': similarity,
                'previous_person_id': previous_person_id,
                'staff_reclassified': True,
            }
        )
        face.extra_data = face_extra_data
        updated += 1

    persons_deleted = 0
    persons_rebuilt = 0

    if apply:
        session.flush()
        persons_deleted, persons_rebuilt = repair_persons_after_staff_classification(
            session=session,
            person_ids=affected_person_ids,
        )

    return {
        'faces_checked': len(rows),
        'matches_found': len(matches),
        'visits_updated': updated,
        'persons_deleted': persons_deleted,
        'persons_rebuilt': persons_rebuilt,
        'matches': matches,
    }
