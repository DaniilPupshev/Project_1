from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from pechvision.config.schema import FacesConfig
from pechvision.db.models import Staff
from pechvision.vision.faces import detect_faces_in_frame

REQUIRED_STAFF_COLUMNS = {
    'id_staff',
    'position',
    'first_name',
    'last_name',
    'gender',
    'age',
}


def build_external_staff_key(raw_staff_id: Any) -> str:
    '''Формирует стабильный ключ сотрудника из значения CSV.'''

    if raw_staff_id is None or pd.isna(raw_staff_id):
        raise ValueError('Не заполнено поле id_staff')

    try:
        staff_id = int(raw_staff_id)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f'Некорректное значение id_staff: {raw_staff_id}'
        ) from exc

    if staff_id <= 0:
        raise ValueError('id_staff должен быть положительным числом')

    return f'cashier_{staff_id:04d}'


def extract_staff_face(
    photo_path: str | Path,
    faces_config: FacesConfig,
    supported_extensions: list[str],
) -> dict[str, Any]:
    '''Получает лицо и embedding с эталонной фотографии сотрудника.'''

    photo = Path(photo_path)

    if not photo.is_file():
        raise ValueError(f'Файла не существует: {photo_path}')

    allowed_extensions = {
        extension.lower()
        for extension in supported_extensions
    }
    file_extension = photo.suffix.lower()

    if file_extension not in allowed_extensions:
        raise ValueError(
            f'Файлы такого формата не поддерживаются: {file_extension}'
        )

    image = cv2.imread(str(photo))

    if image is None:
        raise RuntimeError(f'Не удалось загрузить файл: {photo_path}')

    faces = detect_faces_in_frame(
        frame=image,
        config=faces_config,
        source='staff_reference',
    )

    if not faces:
        raise ValueError(f'Лица не найдены: {photo_path}')

    if len(faces) > 1:
        raise ValueError(f'На фото слишком много лиц: {photo_path}')

    face = faces[0]

    embedding = face.get('embedding')
    quality_score = face.get('quality_score')

    if embedding is None or len(embedding) != 512:
        raise RuntimeError(f'Неправильный embedding: {photo_path}')

    if quality_score is None:
        raise RuntimeError(f'Не определен quality_score: {photo_path}')

    face['photo_path'] = str(photo)
    return face


def build_staff_reference_embedding(
    embeddings: list[list[float]]
) -> list[float]:
    '''Построение embedding персонала'''

    if not embeddings:
        raise ValueError('embeddings не были получены')

    for index, embedding in enumerate(embeddings):
        if len(embedding) != 512:
            raise ValueError(
                f'Embedding с индексом {index} должен содержать 512 значений, '
                f'получено: {len(embedding)}'
            )

    embedding_matrix = np.asarray(embeddings, dtype=np.float32)

    if not np.isfinite(embedding_matrix).all():
        raise ValueError('Матрица получила не конечные значения')

    mean_embedding = embedding_matrix.mean(axis=0)

    embedding_norm = np.linalg.norm(mean_embedding)
    if embedding_norm <= 0 or not np.isfinite(embedding_norm):
        raise ValueError('Невозможно нормализовать embedding сотрудника')

    normalized_embedding = mean_embedding / embedding_norm

    return [float(value) for value in normalized_embedding]


def read_csv_cashier_data(
    file_path: str | Path,
) -> list[dict[str, Any]]:
    '''Читает и проверяет CSV-справочник сотрудников.'''

    path = Path(file_path)

    if not path.is_file():
        raise ValueError(
            f'Файл с данными персонала неисправен или не найден: {path}'
        )

    df = pd.read_csv(
        path,
        sep=',',
        skipinitialspace=True,
    )

    df.columns = df.columns.str.strip()
    missing_columns = REQUIRED_STAFF_COLUMNS - set(df.columns)

    if missing_columns:
        missing_columns_text = ', '.join(sorted(missing_columns))
        raise ValueError(
            f'В CSV отсутствуют обязательные колонки: {missing_columns_text}'
        )

    if df.empty:
        raise ValueError('CSV-справочник сотрудников пуст')

    if df['id_staff'].isna().any():
        raise ValueError('В CSV есть строки без id_staff')

    duplicated_ids = df.loc[
        df['id_staff'].duplicated(keep=False),
        'id_staff',
    ].tolist()

    if duplicated_ids:
        raise ValueError(f'В CSV повторяются id_staff: {duplicated_ids}')

    df = df.astype(object).where(pd.notna(df), None)

    return df.to_dict(orient='records')


def find_staff_photo_paths(
    cashiers_dir: str | Path,
    external_staff_key: str,
    supported_extensions: list[str],
) -> list[Path]:
    '''Находит эталонные фотографии сотрудника в его папке.'''

    staff_dir = Path(cashiers_dir) / external_staff_key

    if not staff_dir.is_dir():
        raise ValueError(f'Не найдена папка сотрудника: {staff_dir}')

    allowed_extensions = {
        extension.lower()
        for extension in supported_extensions
    }
    photo_paths = sorted(
        path
        for path in staff_dir.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    )

    if not photo_paths:
        raise ValueError(f'Не найдены фотографии сотрудника: {staff_dir}')

    return photo_paths


def register_staff_member(
    session: Session,
    data_cashier: dict[str, Any],
    photo_paths: list[str | Path],
    faces_config: FacesConfig,
    supported_extensions: list[str],
) -> tuple[Staff, bool]:
    '''Создает или обновляет сотрудника по данным CSV и эталонным фотографиям.'''

    if not photo_paths:
        raise ValueError('Не обнаружены фотографии сотрудника')

    raw_staff_id = data_cashier.get('id_staff')
    external_staff_key = build_external_staff_key(raw_staff_id)
    staff_id = int(raw_staff_id)

    first_name_value = data_cashier.get('first_name')
    last_name_value = data_cashier.get('last_name')

    if first_name_value is None or pd.isna(first_name_value):
        raise ValueError('Не заполнено поле first_name')

    if last_name_value is None or pd.isna(last_name_value):
        raise ValueError('Не заполнено поле last_name')

    first_name = str(first_name_value).strip()
    last_name = str(last_name_value).strip()

    if not first_name:
        raise ValueError('Поле first_name не может быть пустым')

    if not last_name:
        raise ValueError('Поле last_name не может быть пустым')

    full_name = f'{first_name} {last_name}'

    reference_faces = []

    for photo_path in photo_paths:
        reference_face = extract_staff_face(
            photo_path=str(photo_path),
            faces_config=faces_config,
            supported_extensions=supported_extensions,
        )
        reference_faces.append(reference_face)

    reference_embeddings = [
        reference_face['embedding']
        for reference_face in reference_faces
    ]

    reference_embedding = build_staff_reference_embedding(
        reference_embeddings
    )

    staff_member = session.scalar(
        select(Staff).where(
            Staff.external_staff_key == external_staff_key
        )
    )

    created = staff_member is None

    if staff_member is None:
        staff_member = Staff(
            external_staff_key=external_staff_key,
        )

    position_value = data_cashier.get('position')
    gender_value = data_cashier.get('gender')
    age_value = data_cashier.get('age')

    position = (
        None
        if position_value is None or pd.isna(position_value)
        else str(position_value).strip() or None
    )
    declared_gender = (
        None
        if gender_value is None or pd.isna(gender_value)
        else str(gender_value).strip() or None
    )
    declared_age = (
        None
        if age_value is None or pd.isna(age_value)
        else int(age_value)
    )

    reference_photo_paths = [
        str(reference_face['photo_path'])
        for reference_face in reference_faces
    ]
    reference_quality_scores = [
        float(reference_face['quality_score'])
        for reference_face in reference_faces
    ]

    extra_data = dict(staff_member.extra_data or {})
    extra_data.update(
        {
            'source_csv_id': staff_id,
            'position': position,
            'declared_gender': declared_gender,
            'declared_age': declared_age,
            'reference_photo_paths': reference_photo_paths,
            'reference_faces_count': len(reference_faces),
            'reference_quality_scores': reference_quality_scores,
            'embedding_strategy': 'normalized_mean',
            'model_name': faces_config.model_name,
        }
    )

    staff_member.external_staff_key = external_staff_key
    staff_member.full_name = full_name
    staff_member.photo_path = reference_photo_paths[0]
    staff_member.face_embedding = reference_embedding
    staff_member.is_active = True
    staff_member.extra_data = extra_data

    session.add(staff_member)
    session.flush()

    return staff_member, created


def register_staff_from_registry(
    session: Session,
    registry_path: str | Path,
    cashiers_dir: str | Path,
    faces_config: FacesConfig,
    supported_extensions: list[str],
) -> dict[str, int]:
    '''Регистрирует всех сотрудников из CSV в одной транзакции.'''

    staff_rows = read_csv_cashier_data(registry_path)
    created = 0
    updated = 0

    for staff_row in staff_rows:
        external_staff_key = build_external_staff_key(
            staff_row.get('id_staff')
        )
        photo_paths = find_staff_photo_paths(
            cashiers_dir=cashiers_dir,
            external_staff_key=external_staff_key,
            supported_extensions=supported_extensions,
        )
        _, staff_created = register_staff_member(
            session=session,
            data_cashier=staff_row,
            photo_paths=photo_paths,
            faces_config=faces_config,
            supported_extensions=supported_extensions,
        )

        if staff_created:
            created += 1
        else:
            updated += 1

    return {
        'total': len(staff_rows),
        'created': created,
        'updated': updated,
    }
