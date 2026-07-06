import re
from datetime import datetime

import cv2
from paddleocr import PaddleOCR

from pechvision.config.schema import OCRConfig

_ocr_engine: PaddleOCR | None = None


def get_ocr_engine() -> PaddleOCR:
    '''Создание движка PaddleOCR для переиспользования'''

    global _ocr_engine

    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            lang='en',
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    return _ocr_engine


def normalize_ocr_text(text: str | None) -> str:
    '''Нормализация базового OCR-текста'''

    if not text:
        return ''

    return ' '.join(text.strip().split())


def normalize_camera_datetime_text(text: str | None) -> str:
    '''Нормализация OCR-текста даты и времени камеры'''

    normalized_text = normalize_ocr_text(text)

    replacements = {
        'Mony': 'Mon ',
        'Mon:': 'Mon ',
        'Mon0': 'Mon 0',
        'Mon1': 'Mon 1',
        'Mon2': 'Mon 2',
        'Mon3': 'Mon 3',
        'Mon4': 'Mon 4',
        'Mon5': 'Mon 5',
        'Mon6': 'Mon 6',
        'Mon7': 'Mon 7',
        'Mon8': 'Mon 8',
        'Mon9': 'Mon 9',
    }

    for old, new in replacements.items():
        normalized_text = normalized_text.replace(old, new)

    return normalized_text


def parse_ocr_datetime(text: str | None, datetime_format: str) -> datetime | None:
    '''Получение datetime из OCR-текста камеры.'''

    normalized_text = normalize_camera_datetime_text(text)

    date_match = re.search(r'\d{2}-\d{2}-\d{4}', normalized_text)
    time_match = re.search(r'\d{1,2}[:.]\d{2}[:.]\d{2}', normalized_text)

    if date_match is None or time_match is None:
        return None

    date_text = date_match.group(0)
    time_text = time_match.group(0).replace('.', ':')

    try:
        return datetime.strptime(
            f'{date_text} {time_text}',
            '%m-%d-%Y %H:%M:%S',
        )
    except ValueError:
        return None

def preprocess_ocr_crop(crop):
    '''Подготовка OCR crop к распознаванию'''

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    upscaled_gray = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC,
    )

    return cv2.cvtColor(upscaled_gray, cv2.COLOR_GRAY2BGR)


def extract_text_from_ocr_result(result) -> str:
    '''Достает текст из результата PaddleOCR'''

    if not result:
        return ''

    parts = []

    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                texts = item.get('rec_texts') or item.get('texts') or []
                parts.extend(str(text) for text in texts)
                continue

            if isinstance(item, list):
                for line in item:
                    if len(line) >= 2 and isinstance(line[1], tuple):
                        parts.append(str(line[1][0]))

        return normalize_ocr_text(' '.join(parts))

    if isinstance(result, dict):
        texts = result.get('rec_texts') or result.get('texts') or []
        return normalize_ocr_text(' '.join(str(text) for text in texts))

    return normalize_ocr_text(str(result))


def recognize_text_from_image(image) -> str:
    '''Распознавание текста с изображения'''

    ocr_engine = get_ocr_engine()
    result = ocr_engine.ocr(image)

    return extract_text_from_ocr_result(result)


def recognize_datetime_from_crop(crop, config: OCRConfig) -> tuple[str, datetime | None]:
    '''Распознавание даты и времени из OCR crop'''

    prepared_crop = preprocess_ocr_crop(crop)
    text = recognize_text_from_image(prepared_crop)
    parsed_datetime = parse_ocr_datetime(text, config.datetime_format)

    return text, parsed_datetime