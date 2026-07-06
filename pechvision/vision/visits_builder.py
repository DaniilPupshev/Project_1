from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActiveVisit:
    '''Активный визит трека в зоне кассы'''

    track_id: int
    entry_frame_index: int
    last_seen_frame_index: int
    entry_timestamp_seconds: float | None
    last_seen_timestamp_seconds: float | None
    observations_count: int = 0
    best_confidence: float = 0.0
    last_bbox: list[int] | None = None
    extra_data: dict[str, Any] = field(default_factory=dict)


class VisitsBuilder:
    '''Собирает визиты из последовательности треков в зоне'''

    def __init__(
        self,
        max_missing_seconds: float,
        min_visit_seconds: float
    ) -> None:
        self.max_missing_seconds = max_missing_seconds
        self.min_visit_seconds = min_visit_seconds
        self.active_visits: dict[int, ActiveVisit] = {}
        self.finished_visits: list[dict[str, Any]] = []

    
    def update(
        self,
        frame_index: int,
        timestamp_seconds: float | None,
        tracks_in_zone: list[dict[str, Any]]
    ) -> None:
        '''Обновляет визиты по одному кадру'''

        seen_track_ids = set()

        for track in tracks_in_zone:
            track_id = int(track['track_id'])
            seen_track_ids.add(track_id)

            if track_id not in self.active_visits:
                self.active_visits[track_id] = ActiveVisit(
                    track_id=track_id,
                    entry_frame_index=frame_index,
                    last_seen_frame_index=frame_index,
                    entry_timestamp_seconds=timestamp_seconds,
                    last_seen_timestamp_seconds=timestamp_seconds,
                    observations_count=1,
                    best_confidence=float(track['confidence']),
                    last_bbox=track['bbox'],
                )
                continue

            active_visit = self.active_visits[track_id]
            active_visit.last_seen_frame_index = frame_index
            active_visit.last_seen_timestamp_seconds = timestamp_seconds
            active_visit.observations_count += 1
            active_visit.best_confidence = max(
                active_visit.best_confidence,
                float(track['confidence']),
            )
            active_visit.last_bbox = track['bbox']
        
        self._close_missing_visits(
            current_timestamp_seconds=timestamp_seconds,
            seen_track_ids=seen_track_ids,
        )


    def _close_missing_visits(
        self,
        current_timestamp_seconds: float | None,
        seen_track_ids: set[int],
    ) -> None:
        '''Закрытие визитов, которые давно не наблюдались'''

        if current_timestamp_seconds is None:
            return

        track_ids_to_close = []

        for track_id, active_visit in self.active_visits.items():
            if track_id in seen_track_ids:
                continue

            if active_visit.last_seen_timestamp_seconds is None:
                continue

            missing_seconds = (
                current_timestamp_seconds
                - active_visit.last_seen_timestamp_seconds
            )

            if missing_seconds > self.max_missing_seconds:
                track_ids_to_close.append(track_id)

        for track_id in track_ids_to_close:
            active_visit = self.active_visits.pop(track_id)
            self._finish_visit(active_visit)


    def _finish_visit(self, active_visit: ActiveVisit) -> None:
        '''Перевод активных визитов в завершенные'''

        duration_seconds = self._calculate_duration(active_visit)

        if duration_seconds is not None and duration_seconds < self.min_visit_seconds:
            return

        self.finished_visits.append(
            {
                'track_id': active_visit.track_id,
                'entry_frame_index': active_visit.entry_frame_index,
                'exit_frame_index': active_visit.last_seen_frame_index,
                'entry_timestamp_seconds': active_visit.entry_timestamp_seconds,
                'exit_timestamp_seconds': active_visit.last_seen_timestamp_seconds,
                'duration_seconds': duration_seconds,
                'observations_count': active_visit.observations_count,
                'best_confidence': active_visit.best_confidence,
                'last_bbox': active_visit.last_bbox,
            }
        )

    
    def _calculate_duration(self, active_visit: ActiveVisit) -> float | None:
        '''Расчет длительности визита в секундах'''

        if (
            active_visit.entry_timestamp_seconds is None
            or active_visit.last_seen_timestamp_seconds is None
        ):
            return None

        return (
            active_visit.last_seen_timestamp_seconds
            - active_visit.entry_timestamp_seconds
        )
    

    def finish_all(self) -> list[dict[str, Any]]:
        '''Завершение всех активных визитов и сохранение результатов'''

        for track_id in list(self.active_visits):
            active_visit = self.active_visits.pop(track_id)
            self._finish_visit(active_visit)

        return self.finished_visits