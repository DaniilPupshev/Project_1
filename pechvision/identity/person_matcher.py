import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from pechvision.db.models import Face, Person
from pechvision.identity.match_result import (
    IdentityMatchResult,
    IdentityMatchStatus,
    PersonMatchCandidate,
    select_identity_match_result,
)


def find_person_reference_candidates(
    session: Session,
    embedding: list[float],
    candidate_limit: int,
) -> list[PersonMatchCandidate]:
    '''Ищет ближайшие эталонные лица разных персон'''

    if len(embedding) != 512:
        raise ValueError(
            'Embedding должен содержать 512 значений'
        )

    if candidate_limit < 1:
        raise ValueError(
            'candidate_limit должен быть >= 1'
        )

    distance = Face.embedding.cosine_distance(embedding)

    reference_rank = func.row_number().over(
        partition_by=Face.person_id,
        order_by=(distance, Face.id),
    ).label('reference_rank')

    ranked_references = (
        select(
            Face.person_id.label('person_id'),
            Face.id.label('reference_face_id'),
            distance.label('distance'),
            reference_rank,
        )
        .where(
            Face.person_id.is_not(None),
            Face.embedding.is_not(None),
            Face.is_identity_eligible.is_(True),
            Face.is_identity_reference.is_(True),
        )
        .subquery()
    )

    query = (
        select(
            ranked_references.c.person_id,
            ranked_references.c.reference_face_id,
            ranked_references.c.distance,
        )
        .where(
            ranked_references.c.reference_rank == 1
        )
        .order_by(
            ranked_references.c.distance,
            ranked_references.c.person_id,
        )
        .limit(candidate_limit)
    )

    rows = session.execute(query).mappings().all()

    candidates = []

    for row in rows:
        distance_value = float(row['distance'])

        candidates.append(
            PersonMatchCandidate(
                person_id=int(row['person_id']),
                reference_face_id=int(row['reference_face_id']),
                similarity=1.0 - distance_value,
            )
        )

    return candidates


def find_matching_person(
    session: Session,
    embedding: list[float] | None,
    recognition_threshold: float,
    candidate_limit: int,
    ambiguity_margin: float,
) -> tuple[Person | None, IdentityMatchResult]:
    '''Поиск схожего person'''

    if embedding is None or len(embedding) != 512:
        return (
            None,
            IdentityMatchResult(
                status=IdentityMatchStatus.INVALID_EMBEDDING,
                person_id=None,
                similarity=None,
                reference_face_id=None,
                second_best_similarity=None,
                similarity_margin=None,
                candidates=[]
            )
        )

    person_reference_candidates = find_person_reference_candidates(
        session=session,
        embedding=embedding,
        candidate_limit=candidate_limit,
    )

    match_result = select_identity_match_result(
        candidates=person_reference_candidates,
        recognition_threshold=recognition_threshold,
        ambiguity_margin=ambiguity_margin
    )

    if match_result.status != IdentityMatchStatus.MATCHED:
        return None, match_result

    if match_result.person_id is None:
        raise RuntimeError('Статус MATCHED получен без person_id')

    person = session.get(
        Person,
        match_result.person_id
    )

    if person is None:
        raise RuntimeError(
            'Эталонное лицо связано с отсутствующей персоной: '
            f'person_id={match_result.person_id}'
        )
    
    return person, match_result


def create_person_from_face(
    session: Session,
    embedding: list[float],
    face_image_path: str | None,
    seen_at: datetime | None,
    similarity_threshold: float,
    face_quality_score: float | None
) -> Person:
    '''Создание person по face'''

    external_person_key = f'P_{uuid.uuid4().hex}'

    person = Person(
        external_person_key=external_person_key,
        first_seen_at=seen_at,
        last_seen_at=seen_at,
        best_face_path=face_image_path,
        face_embedding=embedding,
        extra_data={
            'created_from': 'face_embedding',
            'similarity_threshold': similarity_threshold,
            'best_face_quality_score': face_quality_score
        }
    )

    session.add(person)
    session.flush()
    return person


def update_person_best_face_if_better(
    person: Person,
    embedding: list[float] | None,
    face_image_path: str | None,
    face_quality_score: float | None
) -> bool:
    '''Обновление лучшего face'''

    if embedding is None or face_image_path is None or face_quality_score is None:
        return False

    extra_data = person.extra_data or {}
    current_score = extra_data.get('best_face_quality_score')

    if current_score is None or face_quality_score > current_score:
        person.best_face_path = face_image_path
        person.face_embedding = embedding
        extra_data['best_face_quality_score'] = face_quality_score
        person.extra_data = extra_data
        return True

    return False


def get_or_create_person_for_face(
    session: Session,
    embedding: list[float] | None,
    face_image_path: str | None,
    seen_at: datetime | None,
    recognition_threshold: float,
    candidate_limit: int,
    ambiguity_margin: float,
    face_quality_score: float | None
) -> tuple[
    Person | None,
    bool,
    IdentityMatchResult,
    bool,
]:
    '''Определяет создание/изменение person'''
    
    person, match_result = find_matching_person(
        session=session,
        embedding=embedding,
        recognition_threshold=recognition_threshold,
        candidate_limit=candidate_limit,
        ambiguity_margin=ambiguity_margin
    )

    if match_result.status == IdentityMatchStatus.MATCHED:
        if person is not None:
            best_face_updated = update_person_best_face_if_better(
                person=person,
                embedding=embedding,
                face_image_path=face_image_path,
                face_quality_score=face_quality_score
            )

            if seen_at is not None and (
                person.last_seen_at is None or seen_at > person.last_seen_at
            ):
                person.last_seen_at = seen_at

            return (person, False, match_result, best_face_updated)

        raise RuntimeError(
            'Статус MATCHED получен без объекта Person'
        )

    if match_result.status == IdentityMatchStatus.NO_MATCH:
        if embedding is None or len(embedding) != 512:
            raise RuntimeError('Параметры embedding не соответствуют требованиям')

        person = create_person_from_face(
            session=session,
            embedding=embedding,
            face_image_path=face_image_path,
            seen_at=seen_at,
            similarity_threshold=recognition_threshold,
            face_quality_score=face_quality_score
        )

        return (person, True, match_result, False)

    return None, False, match_result, False