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


def extract_camera_date_text(text: str) -> str | None:
    '''Извлекает дату камеры и исправляет типичные ошибки OCR.'''

    date_patterns = (
        r'(?<!\d)(\d{2})-(\d{2})-(\d{4})(?!\d)',
        r'(?<!\d)(\d{2})[\s.-]+(\d{2})[\s.-]+(\d{4})(?!\d)',
    )

    for pattern in date_patterns:
        match = re.search(pattern, text)

        if match is None:
            continue

        month_text, day_text, year_text = match.groups()

        try:
            parsed_date = datetime.strptime(
                f'{month_text}-{day_text}-{year_text}',
                '%m-%d-%Y',
            )
        except ValueError:
            continue

        return parsed_date.strftime('%m-%d-%Y')

    return None


def extract_camera_time_text(text: str) -> str | None:
    '''Извлекает время камеры из поврежденного OCR-текста.'''

    weekday_match = re.search(
        r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b',
        text,
        flags=re.IGNORECASE,
    )

    search_text = text[weekday_match.end():] if weekday_match else text

    time_patterns = (
        (
            r'(?<!\d)([0-2]?\d)[:.]([0-5]\d)[:.]([0-5]\d)'
            r'(?!\d)(?![:.\s-]+\d)'
        ),
        (
            r'(?<!\d)([0-2]?\d)[:.\s-]+([0-5]\d)[:.\s-]+'
            r'([0-5]\d)(?!\d)(?![:.\s-]+\d)'
        ),
        (
            r'(?<!\d)([0-2]\d)([0-5]\d)([0-5]\d)'
            r'(?!\d)(?![:.\s-]+\d)'
        ),
        (
            r'(?<!\d)([0-2]\d)([0-5]\d)[:.]([0-5]\d)'
            r'(?!\d)(?![:.\s-]+\d)'
        ),
        (
            r'(?<!\d)([0-2]?\d)[:.\s-]+([0-5]\d)[:.\s-]+'
            r'(\d)(?!\d)(?![:.\s-]+\d)'
        ),
    )

    for pattern in time_patterns:
        matches = list(re.finditer(pattern, search_text))

        for match in reversed(matches):
            hour_text, minute_text, second_text = match.groups()

            hour = int(hour_text)
            minute = int(minute_text)
            second = int(second_text)

            if hour > 23 or minute > 59 or second > 59:
                continue

            return f'{hour:02d}:{minute:02d}:{second:02d}'

    return None


def normalize_camera_datetime_text(text: str | None) -> str:
    '''Нормализует типичные ошибки OCR в дате и времени камеры.'''

    normalized_text = normalize_ocr_text(text)

    normalized_text = re.sub(
        r'(?<=202)[fF](?=\D|$)',
        '6',
        normalized_text,
    )

    weekdays = r'Mon|Tue|Wed|Thu|Fri|Sat|Sun'

    normalized_text = re.sub(
        rf'(\d{{4}})\s*[/\\-]?\s*({weekdays})',
        r'\1 \2 ',
        normalized_text,
        flags=re.IGNORECASE,
    )
    normalized_text = re.sub(
        rf'\b({weekdays})[yY]?',
        lambda match: match.group(1).title(),
        normalized_text,
        flags=re.IGNORECASE,
    )
    normalized_text = re.sub(
        rf'\b({weekdays})(?=\d)',
        r'\1 ',
        normalized_text,
        flags=re.IGNORECASE,
    )

    return normalize_ocr_text(normalized_text)


def parse_ocr_datetime(
    text: str | None,
    datetime_format: str,
) -> datetime | None:
    '''Получение datetime из OCR-текста камеры.'''

    normalized_text = normalize_camera_datetime_text(text)
    date_text = extract_camera_date_text(normalized_text)
    time_text = extract_camera_time_text(normalized_text)

    if date_text is None or time_text is None:
        return None

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
