import click
import cv2
import numpy as np
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from pechvision.config.loader import load_config
from pechvision.db.session import make_engine, make_session_factory
from pechvision.receipts.importer import import_receipts
from pechvision.video.frames import iter_video_frames, iter_video_frames_range
from pechvision.video.metadata import read_video_metadata
from pechvision.video.ocr import preprocess_ocr_crop, recognize_datetime_from_crop
from pechvision.video.registry import register_video
from pechvision.video.roi import crop_frame
from pechvision.video.runs import create_processing_run
from pechvision.vision.person_detector import detect_people
from pechvision.vision.tracker import track_people
from pechvision.vision.zone import filter_detections_in_zone, get_bbox_point


@click.group()
def cli() -> None:
    '''PechVision интерфейс командной строки.'''


@cli.command('version')
def version() -> None:
    '''Версия проекта'''

    click.echo('PechVision 0.1.0')


@cli.command('config-check')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
def config_check(config_path: str) -> None:
    '''
    Проверка файла конфигурации;
    [Arg]: config_path
    '''

    config = load_config(config_path)

    click.echo('\nCONFIG IS VALID')
    click.echo('-' * 20)
    click.echo(f'Project: {config.project.name}')
    click.echo(f'Timezone: {config.project.timezone}')
    click.echo(f'Database URL: {config.database.url}')
    click.echo(f'Frame step: {config.video.frame_step}')
    click.echo(f'Min visit seconds: {config.video.min_visit_seconds}')
    click.echo(f'Detection model: {config.detection.model_path}\n')


@cli.command('db-check')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
def db_check(config_path: str) -> None:
    '''
    Проверка подключения к БД;
    [Arg]: config_path
    '''

    config = load_config(config_path)
    engine = make_engine(config)

    try:
        with engine.connect() as connection:
            result = connection.execute(text('SELECT 1')).scalar_one()
    except SQLAlchemyError as exc:
        raise click.ClickException(f'Неудачное подключение к БД: {exc}') from exc
    
    click.echo('DATABASE CONNECTION IS VALID')
    click.echo('-' * 20)
    click.echo(f'DATABASE_URL: {engine.url}')
    click.echo(f'Result: {result}')


@cli.command('import-receipts')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('receipts_path', type=click.Path(exists=True, dir_okay=False))
def import_receipts_command(config_path: str, receipts_path: str) -> None:
    '''
    Импорт чеков в БД;
    [Arg]: config_path, receipts_path
    '''

    config = load_config(config_path)
    engine = make_engine(config)
    session_factory = make_session_factory(engine)

    session = session_factory()

    try:
        stats = import_receipts(
            session=session,
            path=receipts_path
        )
    except Exception as exc:
        session.rollback()
        raise click.ClickException(f'Ошибка импорта чеков: {exc}') from exc
    finally:
        session.close()

    click.echo('RECEIPTS IMPORT FINISHED')
    click.echo('-' * 20)
    click.echo(f'File: {receipts_path}')
    click.echo(f'Total rows: {stats["all_rows"]}')
    click.echo(f'Created: {stats["created"]}')
    click.echo(f'Skipped: {stats["skipped"]}')


@cli.command('register-video')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
def registry_videos_command(config_path: str, video_path: str) -> None:
    '''
    Регистрация видео в БД;
    [Arg]: config_path, video_path
    '''

    config = load_config(config_path)
    engine = make_engine(config)
    session_factory = make_session_factory(engine)

    session = session_factory()

    try:
        video, created = register_video(
            session=session,
            path=video_path
        )
    except Exception as exc:
        session.rollback()
        raise click.ClickException(f'Ошибка регистрации видео: {exc}') from exc
    finally:
        session.close()

    click.echo('VIDEO REGISTRATION FINISHED')
    click.echo('-' * 20)
    click.echo(f'File: {video_path}')
    click.echo(f'Video ID: {video.id}')
    click.echo(f'Filename: {video.filename}')
    click.echo(f'Created: {created}')


@cli.command('create-run')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
def create_run_command(config_path: str, video_path: str) -> None:
    '''
    Создание процесса обработки
    [Arg]: config_path, video_path
    '''

    config = load_config(config_path)
    engine = make_engine(config)
    session_factory = make_session_factory(engine)

    session = session_factory()

    try:
        video, created = register_video(
            session=session,
            path=video_path
        )
        processing_run = create_processing_run(
            session=session,
            video=video,
            config_path=config_path
        )

        video_id = video.id
        run_id = processing_run.id
        run_status = processing_run.status
    except Exception as exc:
        session.rollback()
        raise click.ClickException(f'Ошибка создания процесса обработки: {exc}') from exc
    finally:
        session.close()

    click.echo('PROCESSING_RUN CREATE FINISHED')
    click.echo('-' * 20)
    click.echo(f'File: {video_path}')
    click.echo(f'Video ID: {video_id}')
    click.echo(f'Run ID: {run_id}')
    click.echo(f'Status: {run_status}')
    click.echo(f'Created video: {created}')


@cli.command('video-frames-check')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--limit',
    type=int,
    default=5,
    show_default=True,
    help='Сколько выбранных кадров вывести',
)
def video_frames_check_command(
    config_path: str,
    video_path: str,
    limit: int,
) -> None:
    '''
    Проверка чтения кадров видео;
    [Arg]: config_path, video_path, limit
    '''

    if limit < 1:
        raise click.ClickException('limit должен быть >= 1')

    config = load_config(config_path)
    metadata = read_video_metadata(video_path)

    click.echo('VIDEO FRAMES CHECK FINISHED')
    click.echo('-' * 20)
    click.echo(f'File: {video_path}')
    click.echo(f'Frame step: {config.video.frame_step}')
    click.echo(f'Limit: {limit}')

    printed = 0

    for frame_data in iter_video_frames(
        path=video_path,
        frame_step=config.video.frame_step,
        metadata=metadata,
    ):
        frame = frame_data['frame']

        click.echo(
            f'Frame index: {frame_data["frame_index"]}; '
            f'Timestamp seconds: {frame_data["timestamp_seconds"]}; '
            f'Shape: {frame.shape}'
        )

        printed += 1

        if printed >= limit:
            break

    click.echo(f'Printed: {printed}')


@cli.command('ocr-crop-check')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--frame-index',
    type=int,
    default=0,
    show_default=True,
    help='Номер кадра для проверки',
)
def ocr_crop_check_command(
    config_path: str,
    video_path: str,
    frame_index: int,
) -> None:
    '''
    Проверка области OCR ROI;
    [Arg]: config_path, video_path, frame_index
    '''

    if frame_index < 0:
        raise click.ClickException('frame_index должен быть >= 0')

    config = load_config(config_path)
    output_dir = config.paths.runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'ocr_crop_check_frame_{frame_index}.jpg'

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise click.ClickException(f'Не удалось открыть видео: {video_path}')

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if not success:
            raise click.ClickException(f'Не удалось прочитать кадр: {frame_index}')

        crop = crop_frame(frame, config.ocr.crop)
        saved = cv2.imwrite(str(output_path), crop)

        if not saved:
            raise click.ClickException(f'Не удалось сохранить OCR crop: {output_path}')
    finally:
        cap.release()

    click.echo('OCR CROP CHECK FINISHED')
    click.echo('-' * 20)
    click.echo(f'Video: {video_path}')
    click.echo(f'Frame index: {frame_index}')
    click.echo(f'Crop shape: {crop.shape}')
    click.echo(f'Output: {output_path}')


@cli.command('ocr-time-check')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--frame-index',
    type=int,
    default=0,
    show_default=True,
    help='Номер кадра для проверки OCR времени',
)
def ocr_time_check_command(
    config_path: str,
    video_path: str,
    frame_index: int,
) -> None:
    '''
    Проверка распознавания времени через OCR;
    [Arg]: config_path, video_path, frame_index
    '''

    if frame_index < 0:
        raise click.ClickException('frame_index должен быть >= 0')

    config = load_config(config_path)
    output_dir = config.paths.runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_output_path = output_dir / f'ocr_time_check_frame_{frame_index}_raw.jpg'
    prepared_output_path = output_dir / f'ocr_time_check_frame_{frame_index}_prepared.jpg'

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise click.ClickException(f'Не удалось открыть видео: {video_path}')

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if not success:
            raise click.ClickException(f'Не удалось прочитать кадр: {frame_index}')

        crop = crop_frame(frame, config.ocr.crop)
        prepared_crop = preprocess_ocr_crop(crop)

        raw_saved = cv2.imwrite(str(raw_output_path), crop)
        prepared_saved = cv2.imwrite(str(prepared_output_path), prepared_crop)

        if not raw_saved:
            raise click.ClickException(f'Не удалось сохранить OCR raw crop: {raw_output_path}')

        if not prepared_saved:
            raise click.ClickException(
                f'Не удалось сохранить OCR prepared crop: {prepared_output_path}'
            )

        raw_text, parsed_datetime = recognize_datetime_from_crop(crop, config.ocr)
    finally:
        cap.release()

    click.echo('OCR TIME CHECK FINISHED')
    click.echo('-' * 20)
    click.echo(f'Video: {video_path}')
    click.echo(f'Frame index: {frame_index}')
    click.echo(f'Crop shape: {crop.shape}')
    click.echo(f'Prepared crop shape: {prepared_crop.shape}')
    click.echo(f'OCR text: {raw_text}')
    click.echo(f'Parsed datetime: {parsed_datetime}')
    click.echo(f'Output raw: {raw_output_path}')
    click.echo(f'Output prepared: {prepared_output_path}')


@cli.command('detect-people-check')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--frame-index',
    type=int,
    default=0,
    show_default=True,
    help='Номер кадра для проверки детекции людей',
)
def detect_people_check_command(
    config_path: str,
    video_path: str,
    frame_index: int,
) -> None:
    '''
    Проверка детекции людей на одном кадре;
    [Arg]: config_path, video_path, frame_index
    '''

    if frame_index < 0:
        raise click.ClickException('frame_index должен быть >= 0')

    config = load_config(config_path)
    output_dir = config.paths.runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'detect_people_frame_{frame_index}.jpg'

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise click.ClickException(f'Не удалось открыть видео: {video_path}')

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if not success:
            raise click.ClickException(f'Не удалось прочитать кадр: {frame_index}')

        detections = detect_people(frame, config.detection)
        annotated_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )
            cv2.putText(
                annotated_frame,
                f'person {confidence:.2f}',
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        saved = cv2.imwrite(str(output_path), annotated_frame)

        if not saved:
            raise click.ClickException(f'Не удалось сохранить кадр: {output_path}')
    finally:
        cap.release()

    click.echo('DETECT PEOPLE CHECK FINISHED')
    click.echo('-' * 20)
    click.echo(f'Video: {video_path}')
    click.echo(f'Frame index: {frame_index}')
    click.echo(f'Detections: {len(detections)}')
    click.echo(f'Output: {output_path}')

    for detection in detections:
        click.echo(
            f'BBox: {detection["bbox"]}; '
            f'confidence: {detection["confidence"]:.4f}; '
            f'class_id: {detection["class_id"]}'
        )


@cli.command('zone-check')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--frame-index',
    type=int,
    default=0,
    show_default=True,
    help='Номер кадра для проверки зоны кассы',
)
def zone_check_command(
    config_path: str,
    video_path: str,
    frame_index: int,
) -> None:
    '''
    Проверка попадания найденных людей в зону кассы;
    [Arg]: config_path, video_path, frame_index
    '''

    if frame_index < 0:
        raise click.ClickException('frame_index должен быть >= 0')

    config = load_config(config_path)
    output_dir = config.paths.runs_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'zone_check_frame_{frame_index}.jpg'

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise click.ClickException(f'Не удалось открыть видео: {video_path}')

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()

        if not success:
            raise click.ClickException(f'Не удалось прочитать кадр: {frame_index}')

        detections = detect_people(frame, config.detection)
        detections_in_zone = filter_detections_in_zone(
            detections=detections,
            zone_config=config.cashier_zone,
        )

        in_zone_bboxes = {
            tuple(detection['bbox'])
            for detection in detections_in_zone
        }

        annotated_frame = frame.copy()

        polygon_points = np.array(config.cashier_zone.polygon, dtype=np.int32)

        cv2.polylines(
            annotated_frame,
            [polygon_points],
            isClosed=True,
            color=(255, 0, 0),
            thickness=3,
        )

        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox

            point = get_bbox_point(
                bbox=bbox,
                point_policy=config.cashier_zone.point_policy,
            )
            is_in_zone = tuple(bbox) in in_zone_bboxes

            color = (0, 255, 0) if is_in_zone else (0, 0, 255)
            label = 'IN_ZONE' if is_in_zone else 'OUT_ZONE'

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                color,
                3,
            )
            cv2.circle(
                annotated_frame,
                point,
                8,
                color,
                -1,
            )
            cv2.putText(
                annotated_frame,
                f'{label} {detection["confidence"]:.2f}',
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA,
            )

        saved = cv2.imwrite(str(output_path), annotated_frame)

        if not saved:
            raise click.ClickException(f'Не удалось сохранить кадр: {output_path}')
    finally:
        cap.release()

    click.echo('ZONE CHECK FINISHED')
    click.echo('-' * 20)
    click.echo(f'Video: {video_path}')
    click.echo(f'Frame index: {frame_index}')
    click.echo(f'Detections: {len(detections)}')
    click.echo(f'In zone: {len(detections_in_zone)}')
    click.echo(f'Output: {output_path}')

    for detection in detections:
        point = get_bbox_point(
            bbox=detection['bbox'],
            point_policy=config.cashier_zone.point_policy,
        )
        status = 'IN_ZONE' if tuple(detection['bbox']) in in_zone_bboxes else 'OUT_ZONE'

        click.echo(
            f'{status}; '
            f'BBox: {detection["bbox"]}; '
            f'Point: {[point[0], point[1]]}; '
            f'confidence: {detection["confidence"]:.4f}'
        )


@cli.command('scan-zone-check')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--start-frame',
    type=int,
    default=0,
    show_default=True,
    help='Первый кадр диапазона проверки',
)
@click.option(
    '--limit',
    type=int,
    default=300,
    show_default=True,
    help='Сколько выбранных кадров обработать',
)
@click.option(
    '--save-limit',
    type=int,
    default=10,
    show_default=True,
    help='Сколько диагностических кадров сохранить',
)
def scan_zone_check_command(
    config_path: str,
    video_path: str,
    start_frame: int,
    limit: int,
    save_limit: int,
) -> None:
    '''
    Сканирование диапазона кадров с детекцией людей и фильтрацией зоны;
    [Arg]: config_path, video_path, start_frame, limit, save_limit
    '''

    if start_frame < 0:
        raise click.ClickException('start_frame должен быть >= 0')

    if limit < 1:
        raise click.ClickException('limit должен быть >= 1')

    if save_limit < 0:
        raise click.ClickException('save_limit должен быть >= 0')

    config = load_config(config_path)
    metadata = read_video_metadata(video_path)

    output_dir = config.paths.runs_dir / f'scan_zone_start_{start_frame}'
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_frames = 0
    frames_with_people = 0
    frames_with_people_in_zone = 0
    total_detections = 0
    total_detections_in_zone = 0
    max_people_in_zone = 0
    saved_frames = 0

    for frame_data in iter_video_frames_range(
        path=video_path,
        frame_step=config.video.frame_step,
        start_frame=start_frame,
        limit=limit,
        metadata=metadata,
    ):
        processed_frames += 1

        frame_index = frame_data['frame_index']
        frame = frame_data['frame']

        detections = detect_people(frame, config.detection)
        detections_in_zone = filter_detections_in_zone(
            detections=detections,
            zone_config=config.cashier_zone,
        )

        detections_count = len(detections)
        detections_in_zone_count = len(detections_in_zone)

        total_detections += detections_count
        total_detections_in_zone += detections_in_zone_count
        max_people_in_zone = max(max_people_in_zone, detections_in_zone_count)

        if detections_count > 0:
            frames_with_people += 1

        if detections_in_zone_count > 0:
            frames_with_people_in_zone += 1

            if saved_frames < save_limit:
                annotated_frame = frame.copy()
                polygon_points = np.array(config.cashier_zone.polygon, dtype=np.int32)

                cv2.polylines(
                    annotated_frame,
                    [polygon_points],
                    isClosed=True,
                    color=(255, 0, 0),
                    thickness=3,
                )

                in_zone_bboxes = {
                    tuple(detection['bbox'])
                    for detection in detections_in_zone
                }

                for detection in detections:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox

                    point = get_bbox_point(
                        bbox=bbox,
                        point_policy=config.cashier_zone.point_policy,
                    )
                    is_in_zone = tuple(bbox) in in_zone_bboxes

                    color = (0, 255, 0) if is_in_zone else (0, 0, 255)
                    label = 'IN_ZONE' if is_in_zone else 'OUT_ZONE'

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        3,
                    )
                    cv2.circle(
                        annotated_frame,
                        point,
                        8,
                        color,
                        -1,
                    )
                    cv2.putText(
                        annotated_frame,
                        f'{label} {detection["confidence"]:.2f}',
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                output_path = output_dir / f'frame_{frame_index}.jpg'
                saved = cv2.imwrite(str(output_path), annotated_frame)

                if not saved:
                    raise click.ClickException(f'Не удалось сохранить кадр: {output_path}')

                saved_frames += 1

    click.echo('SCAN ZONE CHECK FINISHED')
    click.echo('-' * 20)
    click.echo(f'Video: {video_path}')
    click.echo(f'Start frame: {start_frame}')
    click.echo(f'Frame step: {config.video.frame_step}')
    click.echo(f'Limit: {limit}')
    click.echo(f'Processed frames: {processed_frames}')
    click.echo(f'Frames with people: {frames_with_people}')
    click.echo(f'Frames with people in zone: {frames_with_people_in_zone}')
    click.echo(f'Total detections: {total_detections}')
    click.echo(f'Total detections in zone: {total_detections_in_zone}')
    click.echo(f'Max people in zone: {max_people_in_zone}')
    click.echo(f'Saved diagnostic frames: {saved_frames}')
    click.echo(f'Output dir: {output_dir}')


@cli.command('tracking-check')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--start-frame',
    type=int,
    default=0,
    show_default=True,
    help='Первый кадр диапазона проверки',
)
@click.option(
    '--limit',
    type=int,
    default=300,
    show_default=True,
    help='Сколько выбранных кадров обработать',
)
@click.option(
    '--save-limit',
    type=int,
    default=10,
    show_default=True,
    help='Сколько диагностических кадров сохранить',
)
def tracking_check_command(
    config_path: str,
    video_path: str,
    start_frame: int,
    limit: int,
    save_limit: int,
) -> None:
    '''
    Проверка трекинга людей по диапазону кадров;
    [Arg]: config_path, video_path, start_frame, limit, save_limit
    '''

    if start_frame < 0:
        raise click.ClickException('start_frame должен быть >= 0')

    if limit < 1:
        raise click.ClickException('limit должен быть >= 1')

    if save_limit < 0:
        raise click.ClickException('save_limit должен быть >= 0')

    config = load_config(config_path)
    metadata = read_video_metadata(video_path)

    output_dir = config.paths.runs_dir / f'tracking_start_{start_frame}'
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_frames = 0
    frames_with_tracks = 0
    frames_with_tracks_in_zone = 0
    total_tracks = 0
    total_tracks_in_zone = 0
    max_tracks_in_zone = 0
    saved_frames = 0
    unique_track_ids = set()

    for frame_data in iter_video_frames_range(
        path=video_path,
        frame_step=config.video.frame_step,
        start_frame=start_frame,
        limit=limit,
        metadata=metadata,
    ):
        processed_frames += 1

        frame_index = frame_data['frame_index']
        frame = frame_data['frame']

        tracks = track_people(
            frame=frame,
            detection_config=config.detection,
            tracking_config=config.tracking,
        )
        tracks_in_zone = filter_detections_in_zone(
            detections=tracks,
            zone_config=config.cashier_zone,
        )

        tracks_count = len(tracks)
        tracks_in_zone_count = len(tracks_in_zone)

        total_tracks += tracks_count
        total_tracks_in_zone += tracks_in_zone_count
        max_tracks_in_zone = max(max_tracks_in_zone, tracks_in_zone_count)

        for track in tracks:
            unique_track_ids.add(track['track_id'])

        if tracks_count > 0:
            frames_with_tracks += 1

        if tracks_in_zone_count > 0:
            frames_with_tracks_in_zone += 1

        if tracks_in_zone_count > 0 and saved_frames < save_limit:
            annotated_frame = frame.copy()
            polygon_points = np.array(config.cashier_zone.polygon, dtype=np.int32)

            cv2.polylines(
                annotated_frame,
                [polygon_points],
                isClosed=True,
                color=(255, 0, 0),
                thickness=3,
            )

            in_zone_track_ids = {
                track['track_id']
                for track in tracks_in_zone
            }

            for track in tracks:
                bbox = track['bbox']
                x1, y1, x2, y2 = bbox

                point = get_bbox_point(
                    bbox=bbox,
                    point_policy=config.cashier_zone.point_policy,
                )
                is_in_zone = track['track_id'] in in_zone_track_ids

                color = (0, 255, 0) if is_in_zone else (0, 0, 255)
                label = 'IN_ZONE' if is_in_zone else 'OUT_ZONE'

                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    3,
                )
                cv2.circle(
                    annotated_frame,
                    point,
                    8,
                    color,
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    f'{label} id={track["track_id"]} {track["confidence"]:.2f}',
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            output_path = output_dir / f'frame_{frame_index}.jpg'
            saved = cv2.imwrite(str(output_path), annotated_frame)

            if not saved:
                raise click.ClickException(f'Не удалось сохранить кадр: {output_path}')

            saved_frames += 1

    click.echo('TRACKING CHECK FINISHED')
    click.echo('-' * 20)
    click.echo(f'Video: {video_path}')
    click.echo(f'Start frame: {start_frame}')
    click.echo(f'Frame step: {config.video.frame_step}')
    click.echo(f'Limit: {limit}')
    click.echo(f'Processed frames: {processed_frames}')
    click.echo(f'Frames with tracks: {frames_with_tracks}')
    click.echo(f'Frames with tracks in zone: {frames_with_tracks_in_zone}')
    click.echo(f'Total tracks: {total_tracks}')
    click.echo(f'Total tracks in zone: {total_tracks_in_zone}')
    click.echo(f'Max tracks in zone: {max_tracks_in_zone}')
    click.echo(f'Unique track ids: {sorted(unique_track_ids)}')
    click.echo(f'Unique track count: {len(unique_track_ids)}')
    click.echo(f'Saved diagnostic frames: {saved_frames}')
    click.echo(f'Output dir: {output_dir}')