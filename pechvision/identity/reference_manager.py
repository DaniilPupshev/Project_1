from sqlalchemy import select
from sqlalchemy.orm import Session

from pechvision.db.models import Face


def select_identity_reference_faces(
    faces: list[Face],
    max_references_per_person: int,
    max_references_per_pose: int,
) -> list[Face]:
    '''Выбирает ограниченный набор эталонных лиц персоны'''

    if max_references_per_person < 1:
        raise ValueError(
            'max_references_per_person должен быть >= 1'
        )

    if max_references_per_pose < 1:
        raise ValueError(
            'max_references_per_pose должен быть >= 1'
        )

    eligible_faces = [
        face
        for face in faces
        if (
            face.id is not None
            and face.person_id is not None
            and face.is_identity_eligible is True
            and face.embedding is not None
            and face.identity_quality_score is not None
        )
    ]

    person_ids = {
        face.person_id
        for face in eligible_faces
    }

    if len(person_ids) > 1:
        raise ValueError(
            'Все лица должны принадлежать одной персоне'
        )

    sorted_faces = sorted(
        eligible_faces,
        key=lambda face: (
            -float(face.identity_quality_score),
            int(face.id),
        ),
    )

    selected_faces: list[Face] = []
    pose_counts: dict[str, int] = {}

    for face in sorted_faces:
        if len(selected_faces) >= max_references_per_person:
            break

        pose_category = face.pose_category or 'unknown'
        pose_count = pose_counts.get(pose_category, 0)

        if pose_count >= max_references_per_pose:
            continue

        selected_faces.append(face)
        pose_counts[pose_category] = pose_count + 1

    return selected_faces


def refresh_person_identity_references(
    session: Session,
    person_id: int,
    max_references_per_person: int,
    max_references_per_pose: int,
) -> list[int]:
    '''Поддержка актуальной галереи эталонов персоны'''

    if person_id < 1:
        raise ValueError('Ошибка в person_id')

    session.flush()

    all_faces_person = list(
        session.scalars(
            select(Face)
            .where(Face.person_id == person_id)
            .order_by(Face.id)
        )
    )

    selected_faces = select_identity_reference_faces(
        faces=all_faces_person,
        max_references_per_person=max_references_per_person,
        max_references_per_pose=max_references_per_pose,
    )

    selected_ids = {
        face.id
        for face in selected_faces
    }

    for face in all_faces_person:
        face.is_identity_reference = face.id in selected_ids

    return [
        int(face.id)
        for face in selected_faces
        if face.id is not None
    ]