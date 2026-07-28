from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)


class ProjectConfig(BaseModel):
    name: str
    timezone: str


class DatabaseConfig(BaseModel):
    url: str


class PathsConfig(BaseModel):
    videos_dir: Path
    receipts_dir: Path
    faces_dir: Path
    reports_dir: Path
    runs_dir: Path
    models_dir: Path
    cashiers_dir: Path


class VideoConfig(BaseModel):
    frame_step: PositiveInt
    min_visit_seconds: PositiveInt
    supported_extensions: list[str]


class ProcessingConfig(BaseModel):
    active_interval_seconds: PositiveFloat
    idle_interval_seconds: PositiveFloat
    idle_after_seconds: PositiveFloat
    wakeup_rewind_seconds: NonNegativeFloat

    @model_validator(mode='after')
    def validate_intervals(self) -> Self:
        if self.idle_interval_seconds <= self.active_interval_seconds:
            raise ValueError(
                'idle_interval_seconds должен быть больше '
                'active_interval_seconds'
            )

        if self.wakeup_rewind_seconds >= self.idle_after_seconds:
            raise ValueError(
                'wakeup_rewind_seconds должен быть меньше '
                'idle_after_seconds'
            )

        return self


class CropConfig(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: PositiveInt
    height: PositiveInt


class OCRConfig(BaseModel):
    enabled: bool
    datetime_format: str
    crop: CropConfig


class CashierZoneConfig(BaseModel):
    point_policy: Literal['bottom_center']
    polygon: list[tuple[int, int]]


class DetectionConfig(BaseModel):
    model_path: Path
    device: Literal['auto', 'cpu', 'mps']
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    iou_threshold: float = Field(ge=0.0, le=1.0)
    person_class_id: int = Field(ge=0)


class TrackingConfig(BaseModel):
    tracker_type: Literal['bytetrack', 'botsort']
    max_missing_seconds: PositiveInt


class FacesConfig(BaseModel):
    model_name: str
    model_root: Path
    providers: list[str]
    det_size: tuple[PositiveInt, PositiveInt]
    det_threshold: float = Field(ge=0.0, le=1.0)
    recognition_threshold: float = Field(ge=0.0, le=1.0)
    min_face_size: PositiveInt
    min_identity_sharpness: float = Field(ge=0.0)
    min_identity_confidence: float = Field(ge=0.0, le=1.0)
    identity_frame_margin: int = Field(ge=0)
    max_identity_yaw: float = Field(ge=0.0, le=90.0)
    max_identity_pitch: float = Field(ge=0.0, le=90.0)
    max_identity_roll: float = Field(ge=0.0, le=90.0)
    max_identity_references_per_person: PositiveInt
    max_identity_references_per_pose: PositiveInt
    identity_frontal_yaw_threshold: float = Field(ge=0.0, le=90.0)
    identity_candidate_limit: PositiveInt
    identity_ambiguity_margin: float = Field(ge=0.0, le=1.0)
    search_every_processed_frames: PositiveInt
    save_best_face: bool


class DemographicsConfig(BaseModel):
    enabled: bool
    provider: str
    age_bucket_size: PositiveInt


class CashiersConfig(BaseModel):
    enabled: bool
    registry_path: Path
    supported_extensions: list[str]
    similarity_threshold: float = Field(ge=0.0, le=1.0)


class ReceiptsConfig(BaseModel):
    time_column: str
    open_time_column: str
    id_column: str
    amount_column: str
    timezone: str


class MatchingConfig(BaseModel):
    window_before_seconds: int = Field(ge=0)
    window_after_seconds: int = Field(ge=0)
    ambiguity_policy: Literal['closest_with_flag']
    group_policy: Literal['attach_all_with_group_flag']


class ReportsConfig(BaseModel):
    formats: list[Literal['csv', 'xlsx']]


class VisitSessionsConfig(BaseModel):
    merge_timeout_seconds: PositiveInt


class TimelineConfig(BaseModel):
    anchor_search_offsets_seconds: list[NonNegativeInt] = Field(
        min_length=1,
    )
    minimum_consistent_anchors: PositiveInt
    anchor_consistency_tolerance_seconds: float = Field(ge=0.0)
    end_validation_offsets_before_end_seconds: list[
        NonNegativeInt
    ] = Field(
        min_length=1,
    )
    minimum_consistent_end_anchors: PositiveInt
    end_validation_tolerance_seconds: float = Field(ge=0.0)
    max_end_calibration_seconds: float = Field(ge=0.0)

    @field_validator('anchor_search_offsets_seconds')
    @classmethod
    def validate_anchor_search_offsets_seconds(
        cls,
        value: list[int],
    ) -> list[int]:
        if value[0] != 0:
            raise ValueError(
                'Первое смещение поиска временной опоры должно быть 0'
            )

        if value != sorted(value):
            raise ValueError(
                'Смещения поиска временной опоры должны идти по возрастанию'
            )

        if len(value) != len(set(value)):
            raise ValueError(
                'Смещения поиска временной опоры не должны повторяться'
            )

        return value

    @field_validator('end_validation_offsets_before_end_seconds')
    @classmethod
    def validate_end_validation_offsets(
        cls,
        value: list[int],
    ) -> list[int]:
        if value != sorted(value, reverse=True):
            raise ValueError(
                'Смещения проверки конца должны идти по убыванию'
            )

        if len(value) != len(set(value)):
            raise ValueError(
                'Смещения проверки конца не должны повторяться'
            )

        return value

    @model_validator(mode='after')
    def validate_consistent_anchors_count(self) -> Self:
        available_anchors = len(
            self.anchor_search_offsets_seconds
        )

        if self.minimum_consistent_anchors > available_anchors:
            raise ValueError(
                'minimum_consistent_anchors не может быть больше '
                'количества anchor_search_offsets_seconds'
            )

        available_end_anchors = len(
            self.end_validation_offsets_before_end_seconds
        )

        if (
            self.minimum_consistent_end_anchors
            > available_end_anchors
        ):
            raise ValueError(
                'minimum_consistent_end_anchors не может быть больше '
                'количества end_validation_offsets_before_end_seconds'
            )

        return self


class AppConfig(BaseModel):
    project: ProjectConfig
    database: DatabaseConfig
    paths: PathsConfig
    video: VideoConfig
    processing: ProcessingConfig
    ocr: OCRConfig
    timeline: TimelineConfig
    cashier_zone: CashierZoneConfig
    detection: DetectionConfig
    tracking: TrackingConfig
    faces: FacesConfig
    demographics: DemographicsConfig
    cashiers: CashiersConfig
    receipts: ReceiptsConfig
    matching: MatchingConfig
    reports: ReportsConfig
    visit_sessions: VisitSessionsConfig
