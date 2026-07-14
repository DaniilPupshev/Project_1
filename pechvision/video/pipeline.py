import math
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2

from pechvision.config.schema import AppConfig
from pechvision.video.frames import iter_video_frames_range, read_video_frame
from pechvision.video.metadata import read_video_metadata
from pechvision.video.ocr import recognize_datetime_from_crop
from pechvision.video.roi import crop_frame
from pechvision.vision.faces import detect_faces_for_tracks
from pechvision.vision.tracker import track_people
from pechvision.vision.visits_builder import VisitsBuilder
from pechvision.vision.zone import filter_detections_in_zone

ProgressCallback = Callable[
    [str, int, int | None, dict[str, Any] | None],
    None,
]


def apply_project_timezone(
    parsed_datetime: datetime | None,
    timezone_name: str,
) -> datetime | None:
    '''Принятие timezone проекта'''

    if parsed_datetime is None:
        return None
    
    if parsed_datetime.tzinfo is not None:
        return parsed_datetime
    return parsed_datetime.replace(tzinfo=ZoneInfo(timezone_name))


def recognize_ocr_time_for_frame(
    video_path: str | Path,
    frame_index: int,
    config: AppConfig,
    capture: cv2.VideoCapture | None = None,
    ocr_cache: dict[int, tuple[str, datetime | None]] | None = None,
) -> tuple[str, datetime | None]:
    '''Распознает OCR-время кадра с поддержкой capture и кэша.'''

    if ocr_cache is not None and frame_index in ocr_cache:
        return ocr_cache[frame_index]

    if capture is None:
        frame = read_video_frame(
            path=video_path,
            frame_index=frame_index,
        )
    else:
        if not capture.isOpened():
            raise RuntimeError(f'Видео закрыто или недоступно: {video_path}')

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()

        if not success:
            raise RuntimeError(
                f'Не удалось прочитать кадр {frame_index}: {video_path}'
            )

    crop = crop_frame(
        frame=frame,
        crop_config=config.ocr.crop,
    )
    text, parsed_datetime = recognize_datetime_from_crop(
        crop,
        config.ocr,
    )

    parsed_datetime = apply_project_timezone(
        parsed_datetime=parsed_datetime,
        timezone_name=config.project.timezone,
    )

    result = text, parsed_datetime

    if ocr_cache is not None:
        ocr_cache[frame_index] = result

    return result


def build_ocr_search_frame_indices(
    target_frame_index: int,
    fps: float,
    search_seconds: int,
    frame_count: int | None,
) -> list[int]:
    '''Формирует кадры для OCR-поиска от ближайших к удаленным.'''

    frame_indices = [target_frame_index]

    if search_seconds <= 0 or fps <= 0:
        return frame_indices

    for offset_seconds in range(1, search_seconds + 1):
        frame_offset = max(1, round(fps * offset_seconds))

        previous_frame = target_frame_index - frame_offset
        next_frame = target_frame_index + frame_offset

        if previous_frame >= 0:
            frame_indices.append(previous_frame)

        if frame_count is None or next_frame < frame_count:
            frame_indices.append(next_frame)

    return frame_indices


def recognize_ocr_time_candidates_near_frame(
    video_path: str | Path,
    target_frame_index: int,
    config: AppConfig,
    metadata: dict[str, Any],
    capture: cv2.VideoCapture | None = None,
    ocr_cache: dict[int, tuple[str, datetime | None]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    '''Собирает OCR-кандидатов на заданном и соседних кадрах.'''

    fps = metadata.get('fps') or 0.0
    frame_count = metadata.get('frame_count')

    frame_indices = build_ocr_search_frame_indices(
        target_frame_index=target_frame_index,
        fps=fps,
        search_seconds=config.ocr.event_frame_search_seconds,
        frame_count=frame_count,
    )

    target_frame_text = ''
    candidates = []

    for frame_index in frame_indices:
        try:
            text, parsed_datetime = recognize_ocr_time_for_frame(
                video_path=video_path,
                frame_index=frame_index,
                config=config,
                capture=capture,
                ocr_cache=ocr_cache
            )
        except RuntimeError:
            continue

        if frame_index == target_frame_index:
            target_frame_text = text

        if parsed_datetime is not None:
            frame_offset_seconds = (
                (frame_index - target_frame_index) / fps
                if fps > 0
                else 0.0
            )
            event_datetime = parsed_datetime - timedelta(seconds=frame_offset_seconds)

            candidates.append(
                {
                    'text': text,
                    'parsed_datetime': parsed_datetime,
                    'event_datetime': event_datetime,
                    'frame_index': frame_index,
                    'frame_offset_seconds': frame_offset_seconds,
                }
            )

    return target_frame_text, candidates


def calculate_ocr_duration_tolerance(track_duration_seconds: float) -> float:
    '''Считает допустимое расхождение OCR и длительности трека.'''

    return max(5.0, track_duration_seconds * 0.2)


def select_consensus_ocr_candidate(
    candidates: list[dict[str, Any]],
    consensus_seconds: float = 3.0,
) -> tuple[dict[str, Any] | None, int]:
    '''Выбирает OCR-кандидата, подтвержденного соседними кадрами.'''

    if not candidates:
        return None, 0

    ranked_candidates = []

    for candidate in candidates:
        support = sum(
            1
            for other_candidate in candidates
            if abs(
                (
                    other_candidate['event_datetime']
                    - candidate['event_datetime']
                ).total_seconds()
            ) <= consensus_seconds
        )

        ranked_candidates.append(
            (
                support,
                -abs(candidate['frame_offset_seconds']),
                candidate,
            )
        )

    support, _, candidate = max(
        ranked_candidates,
        key=lambda item: (item[0], item[1]),
    )

    if support < 2:
        return None, support

    return candidate, support


def select_ocr_candidates_for_visit(
    entry_candidates: list[dict[str, Any]],
    exit_candidates: list[dict[str, Any]],
    track_duration_seconds: float | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, float | None]:
    '''Выбирает согласованные OCR-кандидаты входа и выхода.'''

    if track_duration_seconds is None:
        return None, None, None

    tolerance_seconds = calculate_ocr_duration_tolerance(track_duration_seconds)
    valid_pairs = []

    for entry_candidate in entry_candidates:
        for exit_candidate in exit_candidates:
            ocr_duration_seconds = (
                exit_candidate['event_datetime']
                - entry_candidate['event_datetime']
            ).total_seconds()
            duration_difference_seconds = abs(
                ocr_duration_seconds - track_duration_seconds
            )

            if ocr_duration_seconds < 0:
                continue

            if duration_difference_seconds > tolerance_seconds:
                continue

            frame_distance_seconds = (
                abs(entry_candidate['frame_offset_seconds'])
                + abs(exit_candidate['frame_offset_seconds'])
            )
            score = duration_difference_seconds + frame_distance_seconds * 0.1

            valid_pairs.append(
                (
                    score,
                    duration_difference_seconds,
                    entry_candidate,
                    exit_candidate,
                )
            )

    if valid_pairs:
        _, difference, entry_candidate, exit_candidate = min(
            valid_pairs,
            key=lambda item: item[0],
        )
        return entry_candidate, exit_candidate, difference

    entry_candidate, entry_support = select_consensus_ocr_candidate(entry_candidates)
    exit_candidate, exit_support = select_consensus_ocr_candidate(exit_candidates)

    if entry_candidate is not None and exit_candidate is None:
        return entry_candidate, None, None

    if exit_candidate is not None and entry_candidate is None:
        return None, exit_candidate, None

    if entry_candidate is not None and exit_candidate is not None:
        if entry_support > exit_support:
            return entry_candidate, None, None

        if exit_support > entry_support:
            return None, exit_candidate, None

    return None, None, None


def resolve_visit_times(
    fallback_duration_seconds: float | None,
    ocr_entered_at: datetime | None,
    ocr_left_at: datetime | None,
    ocr_time_is_estimated: bool = False,
) -> tuple[datetime | None, datetime | None, float | None, bool]:
    '''Восстанавливает время входа/выхода визита по OCR и длительности трека.'''

    if ocr_entered_at is not None and ocr_left_at is not None:
        ocr_duration_seconds = (ocr_left_at - ocr_entered_at).total_seconds()

        if fallback_duration_seconds is None and ocr_duration_seconds >= 0:
            return (
                ocr_entered_at,
                ocr_left_at,
                ocr_duration_seconds,
                ocr_time_is_estimated,
            )

        if fallback_duration_seconds is not None:
            tolerance_seconds = calculate_ocr_duration_tolerance(
                fallback_duration_seconds
            )
            duration_difference_seconds = abs(
                ocr_duration_seconds - fallback_duration_seconds
            )

            if (
                ocr_duration_seconds >= 0
                and duration_difference_seconds <= tolerance_seconds
            ):
                return (
                    ocr_entered_at,
                    ocr_left_at,
                    fallback_duration_seconds,
                    ocr_time_is_estimated,
                )

        return None, None, fallback_duration_seconds, True

    if fallback_duration_seconds is None:
        return ocr_entered_at, ocr_left_at, None, True

    if ocr_entered_at is not None:
        estimated_left_at = ocr_entered_at + timedelta(seconds=fallback_duration_seconds)
        return ocr_entered_at, estimated_left_at, fallback_duration_seconds, True

    if ocr_left_at is not None:
        estimated_entered_at = ocr_left_at - timedelta(seconds=fallback_duration_seconds)
        return estimated_entered_at, ocr_left_at, fallback_duration_seconds, True

    return None, None, fallback_duration_seconds, True


def calculate_visit_duration(
    fallback_duration_seconds: float | None,
    ocr_entered_at: datetime | None,
    ocr_left_at: datetime | None
) -> tuple[float | None, bool]:
    '''Считает длительность визита и признак приблизительности времени'''

    if ocr_entered_at is None or ocr_left_at is None:
        return fallback_duration_seconds, True
    
    duration_seconds = (ocr_left_at - ocr_entered_at).total_seconds()

    if duration_seconds < 0:
        return fallback_duration_seconds, True
    
    return duration_seconds, False


def enrich_visit_with_ocr_time(
    visit: dict[str, Any],
    video_path: str | Path,
    config: AppConfig,
    metadata: dict[str, Any],
    capture: cv2.VideoCapture | None = None,
    ocr_cache: dict[int, tuple[str, datetime | None]] | None = None,
) -> dict[str, Any]:
    '''Добавление к визиту OCR-времени входа/выхода'''

    entry_frame_index = visit['entry_frame_index']
    exit_frame_index = visit['exit_frame_index']

    entry_target_text, entry_candidates = recognize_ocr_time_candidates_near_frame(
        video_path=video_path,
        target_frame_index=entry_frame_index,
        config=config,
        metadata=metadata,
        capture=capture,
        ocr_cache=ocr_cache
    )
    exit_target_text, exit_candidates = recognize_ocr_time_candidates_near_frame(
        video_path=video_path,
        target_frame_index=exit_frame_index,
        config=config,
        metadata=metadata,
        capture=capture,
        ocr_cache=ocr_cache
    )

    entry_candidate, exit_candidate, ocr_duration_difference_seconds = (
        select_ocr_candidates_for_visit(
            entry_candidates=entry_candidates,
            exit_candidates=exit_candidates,
            track_duration_seconds=visit['duration_seconds'],
        )
    )

    entry_text = (
        entry_candidate['text']
        if entry_candidate is not None
        else entry_target_text
    )
    exit_text = (
        exit_candidate['text']
        if exit_candidate is not None
        else exit_target_text
    )
    ocr_entered_at = (
        entry_candidate['event_datetime']
        if entry_candidate is not None
        else None
    )
    ocr_left_at = (
        exit_candidate['event_datetime']
        if exit_candidate is not None
        else None
    )
    ocr_entry_frame_index = (
        entry_candidate['frame_index']
        if entry_candidate is not None
        else None
    )
    ocr_exit_frame_index = (
        exit_candidate['frame_index']
        if exit_candidate is not None
        else None
    )

    ocr_time_is_estimated = (
        (
            ocr_entry_frame_index is not None
            and ocr_entry_frame_index != entry_frame_index
        )
        or (
            ocr_exit_frame_index is not None
            and ocr_exit_frame_index != exit_frame_index
        )
    )

    entered_at, left_at, _, time_is_estimated = resolve_visit_times(
        fallback_duration_seconds=visit['duration_seconds'],
        ocr_entered_at=ocr_entered_at,
        ocr_left_at=ocr_left_at,
        ocr_time_is_estimated=ocr_time_is_estimated,
    )

    enriched_visit = visit.copy()
    enriched_visit['ocr_entry_text'] = entry_text
    enriched_visit['ocr_exit_text'] = exit_text
    enriched_visit['ocr_entry_frame_index'] = ocr_entry_frame_index
    enriched_visit['ocr_exit_frame_index'] = ocr_exit_frame_index
    enriched_visit['ocr_entry_candidates_count'] = len(entry_candidates)
    enriched_visit['ocr_exit_candidates_count'] = len(exit_candidates)
    enriched_visit['ocr_duration_difference_seconds'] = (
        ocr_duration_difference_seconds
    )
    enriched_visit['ocr_entered_at'] = ocr_entered_at
    enriched_visit['ocr_left_at'] = ocr_left_at
    enriched_visit['duration_seconds'] = visit['duration_seconds']
    enriched_visit['entered_at'] = entered_at
    enriched_visit['left_at'] = left_at
    enriched_visit['time_is_estimated'] = time_is_estimated
    enriched_visit['ocr_rejection_reason'] = (
        'no_consistent_candidates'
        if (
            entered_at is None
            and left_at is None
            and (entry_candidates or exit_candidates)
        )
        else None
    )

    return enriched_visit


def reject_visit_ocr_time(
    visit: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    '''Удаляет недостоверное OCR-время, сохраняя диагностический текст.'''

    rejected_visit = visit.copy()
    rejected_visit['ocr_entered_at'] = None
    rejected_visit['ocr_left_at'] = None
    rejected_visit['entered_at'] = None
    rejected_visit['left_at'] = None
    rejected_visit['time_is_estimated'] = True
    rejected_visit['ocr_rejection_reason'] = reason

    return rejected_visit


def validate_visits_ocr_chronology(
    visits: list[dict[str, Any]],
    tolerance_seconds: float = 5.0,
) -> list[dict[str, Any]]:
    '''Отклоняет OCR-время, нарушающее порядок кадров видео.'''

    known_time_indices = [
        index
        for index, visit in enumerate(visits)
        if visit.get('entered_at') is not None
    ]
    local_outlier_indices = set()

    for position, current_index in enumerate(known_time_indices):
        current_time = visits[current_index]['entered_at']
        previous_index = (
            known_time_indices[position - 1]
            if position > 0
            else None
        )
        next_index = (
            known_time_indices[position + 1]
            if position + 1 < len(known_time_indices)
            else None
        )
        previous_time = (
            visits[previous_index]['entered_at']
            if previous_index is not None
            else None
        )
        next_time = (
            visits[next_index]['entered_at']
            if next_index is not None
            else None
        )
        next_next_time = (
            visits[known_time_indices[position + 2]]['entered_at']
            if position + 2 < len(known_time_indices)
            else None
        )

        if (
            next_time is not None
            and next_time < current_time - timedelta(seconds=tolerance_seconds)
            and (
                (
                    previous_time is not None
                    and next_time
                    >= previous_time - timedelta(seconds=tolerance_seconds)
                )
                or (
                    previous_time is None
                    and next_next_time is not None
                    and next_next_time
                    >= next_time - timedelta(seconds=tolerance_seconds)
                )
            )
        ):
            local_outlier_indices.add(current_index)

    rejected_indices = set(local_outlier_indices)
    previous_accepted_time = None

    for current_index in known_time_indices:
        if current_index in local_outlier_indices:
            continue

        current_time = visits[current_index]['entered_at']

        if (
            previous_accepted_time is not None
            and current_time
            < previous_accepted_time - timedelta(seconds=tolerance_seconds)
        ):
            rejected_indices.add(current_index)
            continue

        previous_accepted_time = current_time

    return [
        reject_visit_ocr_time(visit, 'non_monotonic_video_timeline')
        if index in rejected_indices
        else visit
        for index, visit in enumerate(visits)
    ]


def calculate_processing_total_frames(
    metadata: dict[str, Any],
    frame_step: int,
    start_frame: int = 0,
    limit: int | None = None
) -> int | None:
    '''Считает количество кадров, которые будут реально обработаны.'''

    if limit is not None:
        return limit

    frame_count = metadata.get('frame_count')

    if frame_count is None:
        return None

    available_frames = max(0, int(frame_count) - start_frame)

    if available_frames == 0:
        return 0

    return math.ceil(available_frames / frame_step)


def build_visits_from_video(
    config: AppConfig,
    video_path: str | Path,
    start_frame: int = 0,
    limit: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    '''Строит визиты по видео и определяет OCR-время'''

    metadata = read_video_metadata(video_path)

    total_frames = calculate_processing_total_frames(
        metadata=metadata,
        frame_step=config.video.frame_step,
        start_frame=start_frame,
        limit=limit,
    )

    processed_frames = 0

    visits_builder = VisitsBuilder(
        max_missing_seconds=config.tracking.max_missing_seconds,
        min_visit_seconds=config.video.min_visit_seconds,
    )

    for frame_data in iter_video_frames_range(
        path=video_path,
        frame_step=config.video.frame_step,
        start_frame=start_frame,
        limit=limit,
        metadata=metadata,
    ):
        tracks = track_people(
            frame=frame_data['frame'],
            detection_config=config.detection,
            tracking_config=config.tracking,
        )

        tracks_in_zone = filter_detections_in_zone(
            detections=tracks,
            zone_config=config.cashier_zone,
        )
        should_search_faces = (
            config.faces.save_best_face
            and processed_frames % config.faces.search_every_processed_frames == 0
        )
        faces_by_track_id = {}

        if should_search_faces:
            faces_by_track_id = detect_faces_for_tracks(
                frame=frame_data['frame'],
                tracks=tracks_in_zone,
                config=config,
            )

        visits_builder.update(
            frame_index=frame_data['frame_index'],
            timestamp_seconds=frame_data['timestamp_seconds'],
            tracks_in_zone=tracks_in_zone,
            faces_by_track_id=faces_by_track_id,
        )

        processed_frames += 1

        if progress_callback is not None:
            progress_callback(
                'video',
                processed_frames,
                total_frames,
                {
                    'frame_index': frame_data['frame_index'],
                    'timestamp_seconds': frame_data['timestamp_seconds'],
                },
            )

    visits = visits_builder.finish_all()

    ocr_capture = cv2.VideoCapture(str(video_path))
    ocr_cache: dict[int, tuple[str, datetime | None]] = {}

    try:
        if not ocr_capture.isOpened():
            raise RuntimeError(f'Не удалось открыть видео: {video_path}')

        enriched_visits = []
        total_visits = len(visits)

        if total_visits == 0 and progress_callback is not None:
            progress_callback('ocr', 1, 1, {'cached_frames': 0})

        for visit_index, visit in enumerate(visits, start=1):
            enriched_visit = enrich_visit_with_ocr_time(
                visit=visit,
                video_path=video_path,
                config=config,
                metadata=metadata,
                capture=ocr_capture,
                ocr_cache=ocr_cache,
            )
            enriched_visits.append(enriched_visit)

            if progress_callback is not None:
                progress_callback(
                    'ocr',
                    visit_index,
                    total_visits,
                    {
                        'track_id': visit.get('track_id'),
                        'cached_frames': len(ocr_cache),
                    },
                )

    finally:
        ocr_capture.release()

    return validate_visits_ocr_chronology(enriched_visits)
