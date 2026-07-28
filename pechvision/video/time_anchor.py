from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

import cv2

from pechvision.config.schema import AppConfig
from pechvision.video.ocr import recognize_datetime_from_crop
from pechvision.video.roi import crop_frame
from pechvision.video.timeline import (
    TimeAnchorSource,
    VideoTimeAnchor,
    build_manual_time_anchor,
)


@dataclass(frozen=True, slots=True)
class OCRTimeAnchorAttempt:
    '''Описание временного якоря OCR'''

    requested_offset_seconds: float
    frame_index: int
    timestamp_seconds: float
    raw_text: str
    parsed_datetime: datetime | None
    anchor: VideoTimeAnchor | None

    @property
    def succeeded(self) -> bool:
        return self.anchor is not None


class TimeAnchorResolutionStatus(StrEnum):
    '''Статусы определения временной опоры видео.'''

    OCR = 'ocr'
    MANUAL = 'manual'
    MANUAL_REQUIRED = 'manual_required'


class EndTimeValidationStatus(StrEnum):
    '''Статусы независимой проверки временной шкалы.'''

    VALID = 'valid'
    CALIBRATED = 'calibrated'
    UNAVAILABLE = 'unavailable'
    MISMATCH = 'mismatch'


@dataclass(frozen=True, slots=True)
class TimeAnchorResolution:
    '''Итог определения временной опоры видео.'''

    status: TimeAnchorResolutionStatus
    anchor: VideoTimeAnchor | None
    attempts: tuple[OCRTimeAnchorAttempt, ...]
    consistent_attempts: tuple[OCRTimeAnchorAttempt, ...]
    failure_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.anchor is not None


@dataclass(frozen=True, slots=True)
class EndTimeValidationResult:
    '''Итог OCR-проверки временной шкалы около конца видео.'''

    status: EndTimeValidationStatus
    calculated_end_at: datetime
    attempts: tuple[OCRTimeAnchorAttempt, ...]
    consistent_attempts: tuple[OCRTimeAnchorAttempt, ...]
    observed_start_at: datetime | None = None
    difference_seconds: float | None = None
    reference_anchor: VideoTimeAnchor | None = None
    failure_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.status in (
            EndTimeValidationStatus.VALID,
            EndTimeValidationStatus.CALIBRATED,
        )


def calculate_anchor_frame_position(
    offset_seconds: float,
    fps: float,
    frame_count: int | None,
) -> tuple[int, float]:
    '''Расчет позиции кадра'''

    if (
        not isfinite(offset_seconds)
        or
        offset_seconds < 0
    ):
        raise ValueError('offset_seconds должен быть конечным и >= 0')

    if (
        not isfinite(fps)
        or
        fps <= 0
    ):
        raise ValueError('fps должен быть конечным и > 0')

    frame_index = round(offset_seconds * fps)

    if frame_count is not None:
        if frame_count <= 0:
            raise ValueError('frame_count должен быть > 0')

        if frame_index >= frame_count:
            raise ValueError(
                f'Кадр {frame_index} выходит за пределы видео: '
                f'frame_count={frame_count}'
            )

    timestamp_seconds = frame_index / fps

    return frame_index, timestamp_seconds


def recognize_ocr_anchor_attempt(
    capture: cv2.VideoCapture,
    video_path: str | Path,
    offset_seconds: float,
    metadata: dict[str, Any],
    config: AppConfig,
) -> OCRTimeAnchorAttempt:
    '''Выполнение одной OCR-попытки определения времени'''

    if not capture.isOpened():
        raise RuntimeError(f'Видео закрыто или недоступно: {video_path}')

    fps = metadata.get('fps')
    frame_count = metadata.get('frame_count')

    if fps is None:
        raise ValueError(
            f'В метаданных отсутствует FPS видео: {video_path}'
        )

    frame_index, timestamp_seconds = calculate_anchor_frame_position(
        offset_seconds=offset_seconds,
        fps=fps,
        frame_count=frame_count,
    )

    position_set = capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    if not position_set:
        raise RuntimeError(
            f'Не удалось установить позицию кадра '
            f'{frame_index}: {video_path}'
        )

    success, frame = capture.read()

    if not success:
        raise RuntimeError(
            f'Не удалось прочитать кадр {frame_index}: {video_path}'
        )

    ocr_crop = crop_frame(
        frame=frame,
        crop_config=config.ocr.crop,
    )

    text, parsed_datetime = recognize_datetime_from_crop(
        crop=ocr_crop,
        config=config.ocr,
    )

    time_anchor = None

    if parsed_datetime is not None:
        timezone = ZoneInfo(config.project.timezone)

        if (
            parsed_datetime.tzinfo is None
            or parsed_datetime.utcoffset() is None
        ):
            parsed_datetime = parsed_datetime.replace(
                tzinfo=timezone,
            )
        else:
            parsed_datetime = parsed_datetime.astimezone(
                timezone,
            )

        time_anchor = VideoTimeAnchor(
            reference_datetime=parsed_datetime,
            reference_timestamp_seconds=timestamp_seconds,
            source=TimeAnchorSource.OCR,
            reference_frame_index=frame_index,
            raw_text=text,
        )

    return OCRTimeAnchorAttempt(
        requested_offset_seconds=offset_seconds,
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        raw_text=text,
        parsed_datetime=parsed_datetime,
        anchor=time_anchor,
    )


def get_attempt_recorded_start_at(
    attempt: OCRTimeAnchorAttempt,
) -> datetime:
    '''Возвращает рассчитанное время начала видео из OCR-попытки.'''

    if attempt.anchor is None:
        raise ValueError(
            'OCR-попытка не содержит временной опоры'
        )

    return attempt.anchor.recorded_start_at


def find_largest_consistent_attempt_group(
    attempts: list[OCRTimeAnchorAttempt],
    tolerance_seconds: float,
) -> list[OCRTimeAnchorAttempt]:
    '''Находит крупнейшую согласованную группу OCR-попыток.'''

    if (
        not isfinite(tolerance_seconds)
        or tolerance_seconds < 0
    ):
        raise ValueError(
            'tolerance_seconds должен быть конечным числом >= 0'
        )

    successful_attempts = [
        attempt
        for attempt in attempts
        if attempt.succeeded
    ]

    if not successful_attempts:
        return []

    sorted_attempts = sorted(
        successful_attempts,
        key=get_attempt_recorded_start_at,
    )

    left = 0
    best_group: list[OCRTimeAnchorAttempt] = []
    best_rank: tuple[int, float, float] | None = None

    for right, right_attempt in enumerate(sorted_attempts):
        right_start_at = get_attempt_recorded_start_at(
            right_attempt
        )

        while left <= right:
            left_start_at = get_attempt_recorded_start_at(
                sorted_attempts[left]
            )
            current_spread = (
                right_start_at - left_start_at
            ).total_seconds()

            if current_spread <= tolerance_seconds:
                break

            left += 1

        candidate_group = sorted_attempts[left:right + 1]

        candidate_start_at = get_attempt_recorded_start_at(
            candidate_group[0]
        )
        candidate_end_at = get_attempt_recorded_start_at(
            candidate_group[-1]
        )
        candidate_spread = (
            candidate_end_at - candidate_start_at
        ).total_seconds()
        minimum_requested_offset = min(
            attempt.requested_offset_seconds
            for attempt in candidate_group
        )

        candidate_rank = (
            -len(candidate_group),
            candidate_spread,
            minimum_requested_offset,
        )

        if best_rank is None or candidate_rank < best_rank:
            best_rank = candidate_rank
            best_group = list(candidate_group)

    return sorted(
        best_group,
        key=lambda attempt: attempt.requested_offset_seconds,
    )


def select_representative_attempt(
    attempts: list[OCRTimeAnchorAttempt],
) -> OCRTimeAnchorAttempt:
    '''Выбирает OCR-попытку, ближайшую к медианному началу видео.'''

    if not attempts:
        raise ValueError(
            'Нельзя выбрать временную опору из пустой группы'
        )

    start_timestamps = [
        get_attempt_recorded_start_at(attempt).timestamp()
        for attempt in attempts
    ]
    median_timestamp = median(start_timestamps)

    return min(
        attempts,
        key=lambda attempt: (
            abs(
                get_attempt_recorded_start_at(attempt).timestamp()
                - median_timestamp
            ),
            attempt.requested_offset_seconds,
        ),
    )


def collect_ocr_anchor_attempts(
    video_path: str | Path,
    offsets_seconds: list[float] | tuple[float, ...],
    metadata: dict[str, Any],
    config: AppConfig,
) -> list[OCRTimeAnchorAttempt]:
    '''Выполняет OCR-попытки на заданных позициях одним capture.'''

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f'Не удалось открыть видео: {video_path}')

    attempts = []

    try:
        for offset_seconds in offsets_seconds:
            attempts.append(
                recognize_ocr_anchor_attempt(
                    capture=capture,
                    video_path=video_path,
                    offset_seconds=offset_seconds,
                    metadata=metadata,
                    config=config,
                )
            )
    finally:
        capture.release()

    return attempts


def resolve_ocr_time_anchor(
    video_path: str | Path,
    metadata: dict[str, Any],
    config: AppConfig,
) -> TimeAnchorResolution:
    '''Определяет стартовую временную опору по нескольким OCR-кадрам.'''

    if not config.ocr.enabled:
        return TimeAnchorResolution(
            status=TimeAnchorResolutionStatus.MANUAL_REQUIRED,
            anchor=None,
            attempts=(),
            consistent_attempts=(),
            failure_reason='OCR отключен в конфигурации',
        )

    attempts = collect_ocr_anchor_attempts(
        video_path=video_path,
        offsets_seconds=config.timeline.anchor_search_offsets_seconds,
        metadata=metadata,
        config=config,
    )
    consistent_attempts = find_largest_consistent_attempt_group(
        attempts=attempts,
        tolerance_seconds=(
            config.timeline.anchor_consistency_tolerance_seconds
        ),
    )

    if (
        len(consistent_attempts)
        < config.timeline.minimum_consistent_anchors
    ):
        return TimeAnchorResolution(
            status=TimeAnchorResolutionStatus.MANUAL_REQUIRED,
            anchor=None,
            attempts=tuple(attempts),
            consistent_attempts=tuple(consistent_attempts),
            failure_reason=(
                'Недостаточно согласованных OCR-результатов: '
                f'{len(consistent_attempts)} из '
                f'{config.timeline.minimum_consistent_anchors}'
            ),
        )

    representative = select_representative_attempt(
        attempts=consistent_attempts,
    )

    return TimeAnchorResolution(
        status=TimeAnchorResolutionStatus.OCR,
        anchor=representative.anchor,
        attempts=tuple(attempts),
        consistent_attempts=tuple(consistent_attempts),
    )


def resolve_video_time_anchor(
    video_path: str | Path,
    metadata: dict[str, Any],
    config: AppConfig,
    manual_start_time: str | None = None,
) -> TimeAnchorResolution:
    '''Определяет временную опору через OCR или ручной fallback.'''

    ocr_resolution = resolve_ocr_time_anchor(
        video_path=video_path,
        metadata=metadata,
        config=config,
    )

    if ocr_resolution.succeeded:
        return ocr_resolution

    if manual_start_time is None:
        return ocr_resolution

    return apply_manual_time_anchor_fallback(
        resolution=ocr_resolution,
        manual_start_time=manual_start_time,
        timezone_name=config.project.timezone,
    )


def apply_manual_time_anchor_fallback(
    resolution: TimeAnchorResolution,
    manual_start_time: str,
    timezone_name: str,
) -> TimeAnchorResolution:
    '''Добавляет ручную опору к уже выполненной OCR-диагностике.'''

    manual_anchor = build_manual_time_anchor(
        value=manual_start_time,
        timezone_name=timezone_name,
    )

    return TimeAnchorResolution(
        status=TimeAnchorResolutionStatus.MANUAL,
        anchor=manual_anchor,
        attempts=resolution.attempts,
        consistent_attempts=resolution.consistent_attempts,
        failure_reason=resolution.failure_reason,
    )


def validate_time_anchor_near_video_end(
    video_path: str | Path,
    metadata: dict[str, Any],
    config: AppConfig,
    anchor: VideoTimeAnchor,
) -> EndTimeValidationResult:
    '''Сравнивает стартовую опору с OCR около конца видео.'''

    duration_seconds = metadata.get('duration_seconds')

    if (
        duration_seconds is None
        or not isfinite(duration_seconds)
        or duration_seconds <= 0
    ):
        raise ValueError(
            f'Некорректная длительность видео: {duration_seconds}'
        )

    if not config.ocr.enabled:
        return EndTimeValidationResult(
            status=EndTimeValidationStatus.UNAVAILABLE,
            calculated_end_at=anchor.recorded_end_at(
                duration_seconds
            ),
            attempts=(),
            consistent_attempts=(),
            failure_reason='OCR отключен в конфигурации',
        )

    end_offsets = []

    for seconds_before_end in (
        config.timeline.end_validation_offsets_before_end_seconds
    ):
        offset_seconds = duration_seconds - seconds_before_end

        if offset_seconds < 0:
            continue

        end_offsets.append(offset_seconds)

    if not end_offsets:
        return EndTimeValidationResult(
            status=EndTimeValidationStatus.UNAVAILABLE,
            calculated_end_at=anchor.recorded_end_at(
                duration_seconds
            ),
            attempts=(),
            consistent_attempts=(),
            failure_reason=(
                'Видео короче всех настроенных позиций проверки конца'
            ),
        )

    attempts = collect_ocr_anchor_attempts(
        video_path=video_path,
        offsets_seconds=end_offsets,
        metadata=metadata,
        config=config,
    )
    consistent_attempts = find_largest_consistent_attempt_group(
        attempts=attempts,
        tolerance_seconds=(
            config.timeline.anchor_consistency_tolerance_seconds
        ),
    )
    calculated_end_at = anchor.recorded_end_at(duration_seconds)

    if (
        len(consistent_attempts)
        < config.timeline.minimum_consistent_end_anchors
    ):
        return EndTimeValidationResult(
            status=EndTimeValidationStatus.UNAVAILABLE,
            calculated_end_at=calculated_end_at,
            attempts=tuple(attempts),
            consistent_attempts=tuple(consistent_attempts),
            failure_reason=(
                'Недостаточно согласованных OCR-результатов около конца: '
                f'{len(consistent_attempts)} из '
                f'{config.timeline.minimum_consistent_end_anchors}'
            ),
        )

    representative = select_representative_attempt(
        attempts=consistent_attempts,
    )
    observed_start_at = get_attempt_recorded_start_at(
        representative
    )
    difference_seconds = abs(
        (
            observed_start_at - anchor.recorded_start_at
        ).total_seconds()
    )
    if (
        difference_seconds
        <= config.timeline.end_validation_tolerance_seconds
    ):
        status = EndTimeValidationStatus.VALID
    elif (
        difference_seconds
        <= config.timeline.max_end_calibration_seconds
    ):
        status = EndTimeValidationStatus.CALIBRATED
    else:
        status = EndTimeValidationStatus.MISMATCH

    return EndTimeValidationResult(
        status=status,
        calculated_end_at=calculated_end_at,
        attempts=tuple(attempts),
        consistent_attempts=tuple(consistent_attempts),
        observed_start_at=observed_start_at,
        difference_seconds=difference_seconds,
        reference_anchor=representative.anchor,
        failure_reason=(
            None
            if status
            in (
                EndTimeValidationStatus.VALID,
                EndTimeValidationStatus.CALIBRATED,
            )
            else (
                'OCR около конца видео не согласуется со стартовой '
                'временной опорой'
            )
        ),
    )


def serialize_ocr_attempt(
    attempt: OCRTimeAnchorAttempt,
) -> dict[str, Any]:
    '''Преобразует OCR-попытку в JSON-совместимую диагностику.'''

    return {
        'requested_offset_seconds': attempt.requested_offset_seconds,
        'frame_index': attempt.frame_index,
        'timestamp_seconds': attempt.timestamp_seconds,
        'raw_text': attempt.raw_text,
        'parsed_datetime': (
            attempt.parsed_datetime.isoformat()
            if attempt.parsed_datetime is not None
            else None
        ),
        'recorded_start_at': (
            attempt.anchor.recorded_start_at.isoformat()
            if attempt.anchor is not None
            else None
        ),
        'succeeded': attempt.succeeded,
    }
