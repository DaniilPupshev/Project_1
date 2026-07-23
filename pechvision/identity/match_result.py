from dataclasses import dataclass, field
from enum import StrEnum


class IdentityMatchStatus(StrEnum):
    '''Статусы идентификации кандидатов'''

    MATCHED = 'matched'
    NO_MATCH = 'no_match'
    AMBIGUOUS = 'ambiguous'
    INVALID_EMBEDDING = 'invalid_embedding'


@dataclass
class PersonMatchCandidate:
    '''Описание лучшей эталонной фотографии персоны'''

    person_id: int
    reference_face_id: int
    similarity: float


@dataclass
class IdentityMatchResult:
    '''Результат сопоставления лица с персоной'''

    status: IdentityMatchStatus
    person_id: int | None
    similarity: float | None
    reference_face_id: int | None
    second_best_similarity: float | None
    similarity_margin: float | None
    candidates: list[PersonMatchCandidate] = field(default_factory=list)


def select_identity_match_result(
    candidates: list[PersonMatchCandidate],
    recognition_threshold: float,
    ambiguity_margin: float,
) -> IdentityMatchResult:
    '''Выбирает результат идентификации среди кандидатов'''

    if not 0.0 <= recognition_threshold <= 1.0:
        raise ValueError(
            'recognition_threshold должен находиться в диапазоне 0.0–1.0'
        )

    if not 0.0 <= ambiguity_margin <= 1.0:
        raise ValueError(
            'ambiguity_margin должен находиться в диапазоне 0.0–1.0'
        )

    if not candidates:
        return IdentityMatchResult(
            status=IdentityMatchStatus.NO_MATCH,
            person_id=None,
            similarity=None,
            reference_face_id=None,
            second_best_similarity=None,
            similarity_margin=None,
        )

    person_ids = [
        candidate.person_id
        for candidate in candidates
    ]

    if len(person_ids) != len(set(person_ids)):
        raise ValueError(
            'Каждая персона должна быть представлена одним кандидатом'
        )

    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.similarity,
            candidate.person_id,
        ),
    )

    best_candidate = sorted_candidates[0]
    second_candidate = (
        sorted_candidates[1]
        if len(sorted_candidates) > 1
        else None
    )

    second_best_similarity = (
        second_candidate.similarity
        if second_candidate is not None
        else None
    )

    similarity_margin = (
        best_candidate.similarity - second_candidate.similarity
        if second_candidate is not None
        else None
    )

    if best_candidate.similarity < recognition_threshold:
        status = IdentityMatchStatus.NO_MATCH
        matched_person_id = None
    elif (
        similarity_margin is not None
        and similarity_margin < ambiguity_margin
    ):
        status = IdentityMatchStatus.AMBIGUOUS
        matched_person_id = None
    else:
        status = IdentityMatchStatus.MATCHED
        matched_person_id = best_candidate.person_id

    return IdentityMatchResult(
        status=status,
        person_id=matched_person_id,
        similarity=best_candidate.similarity,
        reference_face_id=best_candidate.reference_face_id,
        second_best_similarity=second_best_similarity,
        similarity_margin=similarity_margin,
        candidates=sorted_candidates,
    )