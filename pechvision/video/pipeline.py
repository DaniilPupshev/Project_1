from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import cv2

from pechvision.config.schema import AppConfig
from pechvision.video.metadata import read_video_metadata
from pechvision.video.timeline import TimeAnchorSource, VideoTimeline
from pechvision.vision.faces import detect_faces_for_tracks
from pechvision.vision.person_detector import detect_people
from pechvision.vision.tracker import reset_person_tracker, track_people
from pechvision.vision.visits_builder import VisitsBuilder
from pechvision.vision.zone import filter_detections_in_zone

ProgressCallback = Callable[
    [str, int, int | None, dict[str, Any] | None],
    None,
]


class ProcessingMode(StrEnum):
    '''Режим анализа кадров видео.'''

    ACTIVE = 'active'
    IDLE = 'idle'


@dataclass(slots=True)
class VideoProcessingStats:
    '''Счетчики адаптивной обработки видео.'''

    inference_frames: int = 0
    active_frames: int = 0
    idle_frames: int = 0
    idle_transitions: int = 0
    wakeups: int = 0
    rewound_frames: int = 0
    maximum_frame_index: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class VideoPipelineResult:
    '''Результат построения визитов из видео.'''

    visits: list[dict[str, Any]]
    stats: VideoProcessingStats


def enrich_visit_with_timeline(
    visit: dict[str, Any],
    timeline: VideoTimeline,
) -> dict[str, Any]:
    '''Рассчитывает абсолютное время визита по шкале видео.'''

    entry_timestamp_seconds = visit.get(
        'entry_timestamp_seconds'
    )
    exit_timestamp_seconds = visit.get(
        'exit_timestamp_seconds'
    )

    if entry_timestamp_seconds is None:
        raise ValueError(
            f'У визита {visit.get("track_id")} отсутствует '
            'entry_timestamp_seconds'
        )

    if exit_timestamp_seconds is None:
        raise ValueError(
            f'У визита {visit.get("track_id")} отсутствует '
            'exit_timestamp_seconds'
        )

    if exit_timestamp_seconds < entry_timestamp_seconds:
        raise ValueError(
            f'Визит {visit.get("track_id")} содержит '
            'отрицательный временной интервал'
        )

    entered_at = timeline.datetime_at(entry_timestamp_seconds)
    left_at = timeline.datetime_at(exit_timestamp_seconds)
    enriched_visit = visit.copy()
    enriched_visit['entered_at'] = entered_at
    enriched_visit['left_at'] = left_at
    enriched_visit['visit_date'] = entered_at.date()
    enriched_visit['duration_seconds'] = (
        left_at - entered_at
    ).total_seconds()
    enriched_visit['time_is_estimated'] = (
        timeline.start_anchor.source == TimeAnchorSource.MANUAL
    )
    enriched_visit['time_source'] = (
        f'video_timeline_{timeline.start_anchor.source.value}'
    )
    enriched_visit['timeline_calibrated'] = (
        timeline.is_calibrated
    )
    enriched_visit['timeline_time_scale'] = timeline.time_scale

    return enriched_visit


def build_visits_from_video(
    config: AppConfig,
    video_path: str | Path,
    timeline: VideoTimeline,
    start_frame: int = 0,
    limit: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> VideoPipelineResult:
    '''Строит визиты с адаптивной частотой анализа кадров.'''

    metadata = read_video_metadata(video_path)
    fps = float(metadata['fps'])
    frame_count = int(metadata['frame_count'])

    if fps <= 0:
        raise ValueError(f'Некорректный FPS видео: {fps}')

    if start_frame < 0:
        raise ValueError('start_frame должен быть >= 0')

    if start_frame >= frame_count:
        raise ValueError(
            f'start_frame={start_frame} выходит за пределы '
            f'видео из {frame_count} кадров'
        )

    if limit is not None and limit < 1:
        raise ValueError('limit должен быть >= 1 или None')

    active_stride_frames = max(
        1,
        round(config.processing.active_interval_seconds * fps),
    )
    idle_stride_frames = max(
        active_stride_frames + 1,
        round(config.processing.idle_interval_seconds * fps),
    )
    rewind_frames = round(
        config.processing.wakeup_rewind_seconds * fps
    )
    visits_builder = VisitsBuilder(
        max_missing_seconds=config.tracking.max_missing_seconds,
        min_visit_seconds=config.video.min_visit_seconds,
    )
    stats = VideoProcessingStats(
        maximum_frame_index=start_frame,
    )
    mode = ProcessingMode.ACTIVE
    last_people_timestamp = start_frame / fps
    next_selected_frame = start_frame
    current_frame_index = start_frame
    active_face_frame_count = 0
    maximum_frame_index = start_frame
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f'Не удалось открыть видео: {video_path}')

    reset_person_tracker(config.detection.model_path)

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while current_frame_index < frame_count:
            if limit is not None and stats.inference_frames >= limit:
                break

            success = capture.grab()

            if not success:
                break

            if current_frame_index < next_selected_frame:
                current_frame_index += 1
                continue

            success, frame = capture.retrieve()

            if not success:
                break

            timestamp_seconds = current_frame_index / fps
            maximum_frame_index = max(
                maximum_frame_index,
                current_frame_index,
            )
            stats.maximum_frame_index = maximum_frame_index
            stats.inference_frames += 1

            if mode == ProcessingMode.ACTIVE:
                tracks = track_people(
                    frame=frame,
                    detection_config=config.detection,
                    tracking_config=config.tracking,
                )
                stats.active_frames += 1

                if tracks:
                    last_people_timestamp = timestamp_seconds

                tracks_in_zone = filter_detections_in_zone(
                    detections=tracks,
                    zone_config=config.cashier_zone,
                )
                should_search_faces = (
                    config.faces.save_best_face
                    and active_face_frame_count
                    % config.faces.search_every_processed_frames
                    == 0
                )
                faces_by_track_id = {}

                if should_search_faces and tracks_in_zone:
                    faces_by_track_id = detect_faces_for_tracks(
                        frame=frame,
                        tracks=tracks_in_zone,
                        config=config,
                    )

                visits_builder.update(
                    frame_index=current_frame_index,
                    timestamp_seconds=timestamp_seconds,
                    tracks_in_zone=tracks_in_zone,
                    faces_by_track_id=faces_by_track_id,
                )
                active_face_frame_count += 1

                idle_seconds = (
                    timestamp_seconds - last_people_timestamp
                )

                if (
                    not tracks
                    and idle_seconds
                    >= config.processing.idle_after_seconds
                ):
                    visits_builder.finish_active()
                    reset_person_tracker(config.detection.model_path)
                    mode = ProcessingMode.IDLE
                    stats.idle_transitions += 1

                stride_frames = (
                    idle_stride_frames
                    if mode == ProcessingMode.IDLE
                    else active_stride_frames
                )
                next_selected_frame = (
                    current_frame_index + stride_frames
                )
            else:
                detections = detect_people(
                    frame=frame,
                    config=config.detection,
                )
                stats.idle_frames += 1

                if detections:
                    rewind_frame = max(
                        start_frame,
                        current_frame_index - rewind_frames,
                    )
                    actual_rewind_frames = (
                        current_frame_index - rewind_frame
                    )
                    stats.wakeups += 1
                    stats.rewound_frames += actual_rewind_frames
                    mode = ProcessingMode.ACTIVE
                    last_people_timestamp = timestamp_seconds
                    active_face_frame_count = 0
                    reset_person_tracker(config.detection.model_path)
                    capture.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        rewind_frame,
                    )
                    current_frame_index = rewind_frame
                    next_selected_frame = rewind_frame

                    if progress_callback is not None:
                        progress_callback(
                            'video',
                            (
                                stats.inference_frames
                                if limit is not None
                                else maximum_frame_index - start_frame + 1
                            ),
                            (
                                limit
                                if limit is not None
                                else frame_count - start_frame
                            ),
                            {
                                'frame_index': maximum_frame_index,
                                'timestamp_seconds': (
                                    maximum_frame_index / fps
                                ),
                                'mode': mode.value,
                                **stats.as_dict(),
                            },
                        )

                    continue

                next_selected_frame = (
                    current_frame_index + idle_stride_frames
                )

            if progress_callback is not None:
                progress_callback(
                    'video',
                    (
                        stats.inference_frames
                        if limit is not None
                        else maximum_frame_index - start_frame + 1
                    ),
                    (
                        limit
                        if limit is not None
                        else frame_count - start_frame
                    ),
                    {
                        'frame_index': maximum_frame_index,
                        'timestamp_seconds': (
                            maximum_frame_index / fps
                        ),
                        'mode': mode.value,
                        **stats.as_dict(),
                    },
                )

            current_frame_index += 1
    finally:
        capture.release()

    visits = visits_builder.finish_all()
    enriched_visits = [
        enrich_visit_with_timeline(
            visit=visit,
            timeline=timeline,
        )
        for visit in visits
    ]

    return VideoPipelineResult(
        visits=enriched_visits,
        stats=stats,
    )
