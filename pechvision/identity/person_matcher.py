import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from pechvision.db.models import Person


def find_matching_person(
    session: Session,
    embedding: list[float],
    threshold: float
) -> tuple[Person | None, float | None]:
    '''Поиск схожего person'''

    if embedding == [] or len(embedding) != 512:
        return (None, None)
    
    distance = Person.face_embedding.cosine_distance(embedding)

    row = session.execute(
        select(Person, distance.label('distance'))
        .where(Person.face_embedding.is_not(None))
        .order_by(distance)
        .limit(1)
    ).first()

    if row is None:
        return None, None

    person, distance_value = row

    if person is None:
        return (None, None)
    
    similarity = 1 - float(distance_value)

    if similarity < threshold:
        return None, similarity
    return person, similarity


def create_person_from_face(
    session: Session,
    embedding: list[float],
    face_image_path: str | None,
    seen_at: datetime | None,
    similarity_threshold: float
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
        }
    )

    session.add(person)
    session.flush()
    return person


def get_or_create_person_for_face(
    session: Session,
    embedding: list[float] | None,
    face_image_path: str | None,
    seen_at: datetime | None,
    threshold: float
) -> tuple[Person | None, bool, float | None]:
    '''Определяет создание/изменение person'''

    if embedding is None or embedding == []:
        return (None, False, None)
    
    person, similarity = find_matching_person(
        session=session,
        embedding=embedding,
        threshold=threshold
    )

    if person is not None:
        if seen_at is not None and (
            person.last_seen_at is None or seen_at > person.last_seen_at
        ):
            person.last_seen_at = seen_at

        return (person, False, similarity)
    
    person = create_person_from_face(
        session=session,
        embedding=embedding,
        face_image_path=face_image_path,
        seen_at=seen_at,
        similarity_threshold=threshold
    )

    return (person, True, similarity)