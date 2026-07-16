import hashlib
from pathlib import Path

FILE_HASH_CHUNK_SIZE = 4 * 1024 * 1024


def calculate_file_sha256(
    path: str | Path,
    chunk_size: int = FILE_HASH_CHUNK_SIZE
) -> str:
    '''Расчет хэша импортируемого видеофайла'''

    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f'Не удалось найти видеофайл: {str(file_path)}')
    
    if chunk_size <= 0:
        raise ValueError('chunk_size должен быть > 0')
    
    hasher = hashlib.sha256()

    with file_path.open('rb') as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def build_video_file_identity(
    path: str | Path
) -> dict[str, str | int]:
    '''Защитная проверка изменения видеофайла'''

    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f'Не удалось найти видеофайл: {str(file_path)}')

    stat_before = file_path.stat()

    file_sha256 = calculate_file_sha256(
        path=file_path,
        chunk_size=FILE_HASH_CHUNK_SIZE
    )

    stat_after = file_path.stat()

    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
    ):
        raise RuntimeError(f'Файл изменился во время вычисления fingerprint: {str(file_path)}')
    
    return {
        'file_sha256': file_sha256,
        'file_size_bytes': stat_before.st_size
    }