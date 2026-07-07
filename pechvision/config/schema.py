from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, PositiveInt


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


class CropConfig(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: PositiveInt
    height: PositiveInt


class OCRConfig(BaseModel):
    enabled: bool
    datetime_format: str
    crop: CropConfig
    read_on_events_only: bool
    max_interpolation_gap_seconds: PositiveInt


class CashierZoneConfig(BaseModel):
    point_policy: Literal['bottom_center']
    polygon: list[tuple[int, int]]


class DetectionConfig(BaseModel):
    model_path: Path
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
    search_every_processed_frames: PositiveInt
    save_best_face: bool


class DemographicsConfig(BaseModel):
    enabled: bool
    provider: str
    age_bucket_size: PositiveInt


class CashiersConfig(BaseModel):
    enabled: bool
    embeddings_path: Path
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


class AppConfig(BaseModel):
    project: ProjectConfig
    database: DatabaseConfig
    paths: PathsConfig
    video: VideoConfig
    ocr: OCRConfig
    cashier_zone: CashierZoneConfig
    detection: DetectionConfig
    tracking: TrackingConfig
    faces: FacesConfig
    demographics: DemographicsConfig
    cashiers: CashiersConfig
    receipts: ReceiptsConfig
    matching: MatchingConfig
    reports: ReportsConfig
