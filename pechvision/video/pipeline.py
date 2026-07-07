import math
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from pechvision.config.schema import AppConfig
from pechvision.video.frames import iter_video_frames_range, read_video_frame
from pechvision.video.metadata import read_video_metadata
from pechvision.video.ocr import recognize_datetime_from_crop
from pechvision.video.roi import crop_frame
from pechvision.vision.faces import detect_faces_for_tracks
from pechvision.vision.tracker import track_people
from pechvision.vision.visits_builder import VisitsBuilder
from pechvision.vision.zone import filter_detections_in_zone


def recognize_ocr_time_for_frame(
    video_path: str | Path,
    frame_index: int,
    config: AppConfig
) -> tuple[str, datetime | None]:
    '''Распознавание OCR-времени для одного кадра'''

    frame = read_video_frame(
        path=video_path,
        frame_index=frame_index
    )

    crop = crop_frame(
        frame=frame,
        crop_config=config.ocr.crop
    )
    text, parsed_datetime = recognize_datetime_from_crop(crop, config.ocr)

    return text, parsed_datetime


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
    config: AppConfig
) -> dict[str, Any]:
    '''Добавление к визиту OCR-времени входа/выхода'''

    entry_text, ocr_entered_at = recognize_ocr_time_for_frame(
        video_path=video_path,
        frame_index=visit['entry_frame_index'],
        config=config,
    )
    exit_text, ocr_left_at = recognize_ocr_time_for_frame(
        video_path=video_path,
        frame_index=visit['exit_frame_index'],
        config=config,
    )

    duration_seconds, time_is_estimated = calculate_visit_duration(
        fallback_duration_seconds=visit['duration_seconds'],
        ocr_entered_at=ocr_entered_at,
        ocr_left_at=ocr_left_at,
    )

    enriched_visit = visit.copy()
    enriched_visit['ocr_entry_text'] = entry_text
    enriched_visit['ocr_exit_text'] = exit_text
    enriched_visit['ocr_entered_at'] = ocr_entered_at
    enriched_visit['ocr_left_at'] = ocr_left_at
    enriched_visit['duration_seconds'] = duration_seconds
    enriched_visit['time_is_estimated'] = time_is_estimated

    return enriched_visit


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
    progress_callback: Callable[[int, int | None, int, float | None], None] | None = None
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
                processed_frames,
                total_frames,
                frame_data['frame_index'],
                frame_data['timestamp_seconds'],
            )

    visits = visits_builder.finish_all()

    return [
        enrich_visit_with_ocr_time(
            visit=visit,
            video_path=video_path,
            config=config,
        )
        for visit in visits
    ]
