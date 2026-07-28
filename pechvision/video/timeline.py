import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from math import isfinite
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class TimeAnchorSource(StrEnum):
    '''Источники получения исходного времени'''

    OCR = 'ocr'
    MANUAL = 'manual'


CAMERA_WEEKDAYS = (
    'Mon',
    'Tue',
    'Wed',
    'Thu',
    'Fri',
    'Sat',
    'Sun',
)

CAMERA_WEEKDAYS_PATTERN = '|'.join(
    re.escape(weekday)
    for weekday in CAMERA_WEEKDAYS
)

STRICT_CAMERA_DATETIME_PATTERN = re.compile(
    rf'\A'
    rf'(?P<month>[0-9]{{2}})-'
    rf'(?P<day>[0-9]{{2}})-'
    rf'(?P<year>[0-9]{{4}}) '
    rf'(?P<weekday>{CAMERA_WEEKDAYS_PATTERN}) '
    rf'(?P<hour>[0-9]{{2}}):'
    rf'(?P<minute>[0-9]{{2}}):'
    rf'(?P<second>[0-9]{{2}})'
    rf'\Z'
)


@dataclass(frozen=True, slots=True)
class VideoTimeAnchor:
    '''Информация о времени начала видео'''

    reference_datetime: datetime
    reference_timestamp_seconds: float
    source: TimeAnchorSource
    reference_frame_index: int | None = None
    raw_text: str | None = None

    def __post_init__(self) -> None:
        if (
            self.reference_datetime.tzinfo is None
            or
            self.reference_datetime.utcoffset() is None
        ):
            raise ValueError('Получено время reference_datetime без часового пояса')

        if (
            not isfinite(self.reference_timestamp_seconds)
            or self.reference_timestamp_seconds < 0
        ):
            raise ValueError(
                'reference_timestamp_seconds должен быть конечным числом >= 0'
            )

        if self.reference_frame_index is not None:
            if self.reference_frame_index < 0:
                raise ValueError('reference_frame_index должен быть >= 0')

    @property
    def recorded_start_at(self) -> datetime:
        return self.reference_datetime - timedelta(
            seconds=self.reference_timestamp_seconds,
        )

    def datetime_at(self, timestamp_seconds: float) -> datetime:
        '''Возвращает абсолютное время в указанной позиции видео'''

        if (
            not isfinite(timestamp_seconds)
            or timestamp_seconds < 0
        ):
            raise ValueError(
                'timestamp_seconds должен быть конечным числом >= 0'
            )

        return self.recorded_start_at + timedelta(seconds=timestamp_seconds)

    def recorded_end_at(self, duration_seconds: float) -> datetime:
        '''Абсолютное время окончания видео'''

        if (
            not isfinite(duration_seconds)
            or duration_seconds < 0
        ):
            raise ValueError('duration_seconds должен быть конечным числом >= 0')

        return self.datetime_at(duration_seconds)


@dataclass(frozen=True, slots=True)
class VideoTimeline:
    '''Преобразует медиапозицию видео в абсолютное время камеры.'''

    start_anchor: VideoTimeAnchor
    duration_seconds: float
    calibration_anchor: VideoTimeAnchor | None = None

    def __post_init__(self) -> None:
        if (
            not isfinite(self.duration_seconds)
            or self.duration_seconds <= 0
        ):
            raise ValueError(
                'duration_seconds должен быть конечным числом > 0'
            )

        if self.calibration_anchor is not None:
            media_elapsed = (
                self.calibration_anchor.reference_timestamp_seconds
                - self.start_anchor.reference_timestamp_seconds
            )
            clock_elapsed = (
                self.calibration_anchor.reference_datetime
                - self.start_anchor.reference_datetime
            ).total_seconds()

            if media_elapsed <= 0:
                raise ValueError(
                    'Калибровочная опора должна находиться позже стартовой'
                )

            if clock_elapsed <= 0:
                raise ValueError(
                    'Время калибровочной опоры должно быть позже стартового'
                )

    @property
    def time_scale(self) -> float:
        if self.calibration_anchor is None:
            return 1.0

        media_elapsed = (
            self.calibration_anchor.reference_timestamp_seconds
            - self.start_anchor.reference_timestamp_seconds
        )
        clock_elapsed = (
            self.calibration_anchor.reference_datetime
            - self.start_anchor.reference_datetime
        ).total_seconds()

        return clock_elapsed / media_elapsed

    @property
    def is_calibrated(self) -> bool:
        return self.calibration_anchor is not None

    @property
    def recorded_start_at(self) -> datetime:
        return self.datetime_at(0.0)

    @property
    def recorded_end_at(self) -> datetime:
        return self.datetime_at(self.duration_seconds)

    def datetime_at(self, timestamp_seconds: float) -> datetime:
        '''Возвращает время камеры для медиапозиции видео.'''

        if (
            not isfinite(timestamp_seconds)
            or timestamp_seconds < 0
        ):
            raise ValueError(
                'timestamp_seconds должен быть конечным числом >= 0'
            )

        media_delta = (
            timestamp_seconds
            - self.start_anchor.reference_timestamp_seconds
        )

        return self.start_anchor.reference_datetime + timedelta(
            seconds=media_delta * self.time_scale,
        )


def parse_manual_start_datetime(
    value: str,
    timezone_name: str,
) -> datetime:
    '''Парсинг начального времени видео, введённого вручную.'''

    match = STRICT_CAMERA_DATETIME_PATTERN.fullmatch(value.strip())

    if match is None:
        raise ValueError(
            'Начальное время должно соответствовать формату '
            'MM-DD-YYYY Day HH:MM:SS, например '
            '07-27-2026 Mon 14:30:00'
        )

    parts = match.groupdict()

    try:
        parsed_datetime = datetime(
            year=int(parts['year']),
            month=int(parts['month']),
            day=int(parts['day']),
            hour=int(parts['hour']),
            minute=int(parts['minute']),
            second=int(parts['second']),
        )
    except ValueError as exc:
        raise ValueError(
            f'Указана некорректная дата или время: {value.strip()}'
        ) from exc

    expected_weekday = CAMERA_WEEKDAYS[
        parsed_datetime.weekday()
    ]
    provided_weekday = parts['weekday']

    if provided_weekday != expected_weekday:
        raise ValueError(
            f'Неверный день недели: для '
            f'{parsed_datetime:%m-%d-%Y} ожидается '
            f'{expected_weekday}, получено {provided_weekday}'
        )

    try:
        timezone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f'Неизвестный часовой пояс: {timezone_name}'
        ) from exc

    return parsed_datetime.replace(tzinfo=timezone)


def build_manual_time_anchor(
    value: str,
    timezone_name: str,
) -> VideoTimeAnchor:
    '''Получение времени начала видео'''

    manual_start_datetime = parse_manual_start_datetime(
        value=value,
        timezone_name=timezone_name,
    )

    return VideoTimeAnchor(
        reference_datetime=manual_start_datetime,
        reference_timestamp_seconds=0.0,
        source=TimeAnchorSource.MANUAL,
        reference_frame_index=0,
        raw_text=value.strip(),
    )
