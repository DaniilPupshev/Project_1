from datetime import UTC, datetime
from time import monotonic

import click
import cv2
import numpy as np
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from tqdm import tqdm

from pechvision.config.loader import load_config
from pechvision.db.models import ProcessingRun
from pechvision.db.session import make_engine, make_session_factory
from pechvision.identity.staff_matcher import classify_existing_staff, has_active_staff
from pechvision.identity.staff_registry import register_staff_from_registry
from pechvision.matching.receipt_matcher import match_receipts_to_visits
from pechvision.receipts.importer import import_receipts
from pechvision.video.frames import iter_video_frames, iter_video_frames_range, read_video_frame
from pechvision.video.metadata import read_video_metadata
from pechvision.video.ocr import preprocess_ocr_crop, recognize_datetime_from_crop
from pechvision.video.pipeline import build_visits_from_video
from pechvision.video.registry import register_video
from pechvision.video.roi import crop_frame
from pechvision.video.runs import (
    create_processing_run,
    find_existing_processing_run,
)
from pechvision.video.visits_importer import save_visits
from pechvision.vision.person_detector import detect_people
from pechvision.vision.tracker import track_people
from pechvision.vision.visits_builder import VisitsBuilder
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
    click.echo(f'Face model: InsightFace {config.faces.model_name}')
    click.echo(f'Face model root: {config.faces.model_root}\n')


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
def create_run_command(
    config_path: str,
    video_path: str
) -> None:
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
            config=config,
            config_path=config_path,
            start_frame=0,
            frame_limit=None,
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


@cli.command('process-video')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--start-frame',
    type=int,
    default=0,
    show_default=True,
    help='Первый кадр обработки',
)
@click.option(
    '--limit',
    type=int,
    default=None,
    help='Сколько выбранных кадров обработать. Если не задано, обрабатывается все видео',
)
def process_video_command(
    config_path: str,
    video_path: str,
    start_frame: int,
    limit: int | None,
) -> None:
    '''
    Полная обработка видео: регистрация, запуск pipeline, сохранение визитов;
    [Arg]: config_path, video_path, start_frame, limit
    '''

    if start_frame < 0:
        raise click.ClickException('start_frame должен быть >= 0')

    if limit is not None and limit < 1:
        raise click.ClickException('limit должен быть >= 1')

    config = load_config(config_path)
    engine = make_engine(config)
    session_factory = make_session_factory(engine)
    session = session_factory()

    try:
        if config.cashiers.enabled and not has_active_staff(session):
            raise click.ClickException(
                'Исключение персонала включено, но в БД нет активных сотрудников. '
                'Сначала выполните register-staff или установите '
                'cashiers.enabled: false для запуска без фильтрации персонала.'
            )

        video, video_created = register_video(
            session=session,
            path=video_path,
        )

        existing_run = find_existing_processing_run(
            session=session,
            video_id=video.id,
            config=config,
            start_frame=start_frame,
            frame_limit=limit,
        )

        if existing_run is not None:
            if existing_run.status == 'running':
                reason = 'Обработка с такими параметрами уже выполняется'
            else:
                reason = 'Видео с такими параметрами уже обработано'

            raise click.ClickException(
                f'{reason}. '
                f'Video ID: {video.id}. '
                f'Run ID: {existing_run.id}.'
            )

    except click.ClickException:
        session.rollback()
        session.close()
        raise
    except Exception as exc:
        session.rollback()
        session.close()
        raise click.ClickException(
            f'Ошибка подготовки обработки видео: {exc}'
        ) from exc

    processing_run_id = None
    progress_bar = tqdm(
        total=100.0,
        desc='PechVision: preparing',
        unit='%',
        dynamic_ncols=True,
        mininterval=0.2,
        bar_format='{desc}: {percentage:6.2f}%|{bar}| {elapsed}<{remaining}',
    )

    stage_ranges = {
        'registration': (0.0, 1.0),
        'video': (1.0, 93.0),
        'ocr': (93.0, 99.0),
        'save': (99.0, 99.8),
        'completed': (99.8, 100.0),
    }
    stage_labels = {
        'registration': 'PechVision: registration',
        'video': 'PechVision: video analysis',
        'ocr': 'PechVision: OCR',
        'save': 'PechVision: saving results',
        'completed': 'PechVision: completed',
    }
    current_progress_stage = None
    video_progress_started_at = None

    def update_progress(
        stage: str,
        completed: int,
        total: int | None,
        details: dict | None,
    ) -> None:
        nonlocal current_progress_stage, video_progress_started_at

        stage_range = stage_ranges.get(stage)

        if stage_range is None:
            return

        start_percent, end_percent = stage_range

        if total is None:
            ratio = 0.0
        elif total <= 0:
            ratio = 1.0
        else:
            ratio = completed / total

        ratio = min(1.0, max(0.0, ratio))
        target_percent = start_percent + (
            end_percent - start_percent
        ) * ratio
        progress_delta = target_percent - progress_bar.n

        stage_changed = stage != current_progress_stage

        if stage_changed:
            progress_bar.set_description(
                stage_labels[stage],
                refresh=False,
            )
            current_progress_stage = stage

            if stage == 'video':
                video_progress_started_at = monotonic()

        should_refresh_details = (
            stage != 'video'
            or completed % 100 == 0
            or total is not None and completed >= total
        )

        if should_refresh_details:
            if stage == 'video' and details is not None:
                timestamp_seconds = details.get('timestamp_seconds')
                postfix = {
                    'frame': details.get('frame_index'),
                }

                if video_progress_started_at is not None:
                    elapsed_seconds = (
                        monotonic() - video_progress_started_at
                    )

                    if elapsed_seconds > 0:
                        postfix['speed'] = (
                            f'{completed / elapsed_seconds:.1f} frame/s'
                        )

                if timestamp_seconds is not None:
                    postfix['video_time'] = (
                        f'{timestamp_seconds:.1f}s'
                    )

                progress_bar.set_postfix(
                    postfix,
                    refresh=False,
                )

            elif stage == 'ocr' and details is not None:
                progress_bar.set_postfix(
                    ocr=f'{completed}/{total}',
                    cache=details.get('cached_frames'),
                    refresh=False,
                )

            elif stage == 'save':
                progress_bar.set_postfix(
                    visits=f'{completed}/{total}',
                    refresh=False,
                )

            elif stage == 'completed':
                progress_bar.set_postfix(refresh=False)

        if progress_delta > 0:
            progress_bar.update(progress_delta)
        elif stage_changed or should_refresh_details:
            progress_bar.refresh()

    try:
        processing_run = create_processing_run(
            session=session,
            video=video,
            config=config,
            config_path=config_path,
            start_frame=start_frame,
            frame_limit=limit,
        )

        video_id = video.id
        processing_run_id = processing_run.id

        processing_run.status = 'running'
        processing_run.started_at = datetime.now(UTC)
        session.commit()

        update_progress('registration', 1, 1, None)

        visits = build_visits_from_video(
            config=config,
            video_path=video_path,
            start_frame=start_frame,
            limit=limit,
            progress_callback=update_progress,
        )

        save_stats = save_visits(
            session=session,
            video_id=video_id,
            processing_run_id=processing_run_id,
            visits=visits,
            faces_dir=config.paths.faces_dir,
            recognition_threshold=config.faces.recognition_threshold,
            staff_matching_enabled=config.cashiers.enabled,
            identity_candidate_limit=config.faces.identity_candidate_limit,
            identity_ambiguity_margin=config.faces.identity_ambiguity_margin,
            max_identity_references_per_person=(
                config.faces.max_identity_references_per_person
            ),
            max_identity_references_per_pose=(
                config.faces.max_identity_references_per_pose
            ),
            staff_similarity_threshold=config.cashiers.similarity_threshold,
            progress_callback=update_progress,
        )

        finished_run = session.get(ProcessingRun, processing_run_id)

        if finished_run is None:
            raise RuntimeError(
                f'Processing run не найден: {processing_run_id}'
            )

        finished_run.status = 'finished'
        finished_run.finished_at = datetime.now(UTC)
        finished_run.stats = {
            'start_frame': start_frame,
            'limit': limit,
            'video_created': video_created,
            'visits_found': len(visits),
            'visits_created': save_stats['created'],
            'visits_skipped_existing': save_stats['skipped_existing'],
            'faces_created': save_stats['faces_created'],
            'persons_created': save_stats['persons_created'],
            'persons_matched': save_stats['persons_matched'],
            'persons_ambiguous': save_stats['persons_ambiguous'],
            'persons_best_face_updated': save_stats[
                'persons_best_face_updated'
            ],
            'staff_visits_matched': save_stats['staff_visits_matched'],
        }

        session.commit()
        update_progress('completed', 1, 1, None)

    except Exception as exc:
        session.rollback()
        progress_bar.set_description('PechVision: failed')

        if processing_run_id is not None:
            failed_run = session.get(ProcessingRun, processing_run_id)

            if failed_run is not None:
                failed_run.status = 'failed'
                failed_run.finished_at = datetime.now(UTC)
                failed_run.error_message = str(exc)
                session.commit()

        raise click.ClickException(
            f'Ошибка обработки видео: {exc}'
        ) from exc

    finally:
        progress_bar.close()
        session.close()

    click.echo('VIDEO PROCESSING FINISHED')
    click.echo('-' * 20)
    click.echo(f'Video: {video_path}')
    click.echo(f'Video ID: {video_id}')
    click.echo(f'Run ID: {processing_run_id}')
    click.echo(f'Start frame: {start_frame}')
    click.echo(f'Limit: {limit}')
    click.echo(f'Visits found: {len(visits)}')
    click.echo(f'Visits created: {save_stats["created"]}')
    click.echo(f'Visits skipped existing: {save_stats["skipped_existing"]}')
    click.echo(f'Faces created: {save_stats["faces_created"]}')
    click.echo(f'Persons created: {save_stats["persons_created"]}')
    click.echo(f'Persons matched: {save_stats["persons_matched"]}')
    click.echo(
        f'Persons ambiguous: {save_stats["persons_ambiguous"]}'
    )
    click.echo(
        'Persons best face updated: '
        f'{save_stats["persons_best_face_updated"]}'
    )
    click.echo(f'Staff visits matched: {save_stats["staff_visits_matched"]}')
    click.echo(f'Video created: {video_created}')


@cli.command('run-mvp')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.argument('video_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--start-frame',
    type=int,
    default=0,
    show_default=True,
    help='Первый кадр обработки',
)
@click.option(
    '--limit',
    type=int,
    default=None,
    help='Сколько выбранных кадров обработать. Если не задано, обрабатывается все видео',
)
@click.pass_context
def run_mvp_command(
    ctx: click.Context,
    config_path: str,
    video_path: str,
    start_frame: int,
    limit: int | None,
) -> None:
    '''
    MVP-запуск полного текущего pipeline;
    [Arg]: config_path, video_path, start_frame, limit
    '''

    ctx.invoke(
        process_video_command,
        config_path=config_path,
        video_path=video_path,
        start_frame=start_frame,
        limit=limit,
    )


@cli.command('register-staff')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
def register_staff_command(config_path: str) -> None:
    '''Регистрирует или обновляет сотрудников из CSV-справочника.'''

    config = load_config(config_path)
    engine = make_engine(config)
    session_factory = make_session_factory(engine)
    session = session_factory()

    try:
        stats = register_staff_from_registry(
            session=session,
            registry_path=config.cashiers.registry_path,
            cashiers_dir=config.paths.cashiers_dir,
            faces_config=config.faces,
            supported_extensions=config.cashiers.supported_extensions,
        )
        session.commit()
    except Exception as exc:
        session.rollback()
        raise click.ClickException(
            f'Ошибка регистрации персонала: {exc}'
        ) from exc
    finally:
        session.close()

    click.echo('STAFF REGISTRATION FINISHED')
    click.echo('-' * 20)
    click.echo(f'Total: {stats["total"]}')
    click.echo(f'Created: {stats["created"]}')
    click.echo(f'Updated: {stats["updated"]}')


@cli.command('classify-staff')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--video-id',
    type=int,
    default=None,
    help='ID видео. Если не указан, проверяются все сохраненные лица',
)
@click.option(
    '--apply',
    'apply_changes',
    is_flag=True,
    help='Применить найденную классификацию. Без флага выполняется dry-run',
)
@click.option(
    '--show-limit',
    type=int,
    default=20,
    show_default=True,
    help='Максимальное количество найденных совпадений в выводе',
)
def classify_staff_command(
    config_path: str,
    video_id: int | None,
    apply_changes: bool,
    show_limit: int,
) -> None:
    '''Ищет сотрудников среди уже сохраненных лиц.'''

    if video_id is not None and video_id < 1:
        raise click.ClickException('video-id должен быть >= 1')

    if show_limit < 0:
        raise click.ClickException('show-limit должен быть >= 0')

    config = load_config(config_path)
    engine = make_engine(config)
    session_factory = make_session_factory(engine)
    session = session_factory()

    try:
        stats = classify_existing_staff(
            session=session,
            threshold=config.cashiers.similarity_threshold,
            video_id=video_id,
            apply=apply_changes,
        )

        if apply_changes:
            session.commit()
        else:
            session.rollback()
    except Exception as exc:
        session.rollback()
        raise click.ClickException(
            f'Ошибка классификации персонала: {exc}'
        ) from exc
    finally:
        session.close()

    mode = 'APPLY' if apply_changes else 'DRY-RUN'
    click.echo(f'STAFF CLASSIFICATION {mode} FINISHED')
    click.echo('-' * 20)
    click.echo(f'Faces checked: {stats["faces_checked"]}')
    click.echo(f'Matches found: {stats["matches_found"]}')
    click.echo(f'Visits updated: {stats["visits_updated"]}')
    click.echo(f'Orphan persons deleted: {stats["persons_deleted"]}')
    click.echo(f'Persons rebuilt: {stats["persons_rebuilt"]}')

    for match in stats['matches'][:show_limit]:
        click.echo(
            f'Visit {match["visit_id"]}: '
            f'{match["external_staff_key"]} '
            f'({match["full_name"]}), '
            f'similarity={match["similarity"]:.3f}, '
            f'already_classified={match["already_classified"]}'
        )


@cli.command('match-receipts')
@click.argument('config_path', type=click.Path(exists=True, dir_okay=False))
@click.option(
    '--video-id',
    type=int,
    default=None,
    help='ID видео. Если не задано, сопоставляются визиты по всем видео',
)
def match_receipts_command(config_path: str, video_id: int | None) -> None:
    '''
    Сопоставление визитов с чеками;
    [Arg]: config_path, video_id
    '''

    config = load_config(config_path)
    engine = make_engine(config)
    session_factory = make_session_factory(engine)

    session = session_factory()

    try:
        stats = match_receipts_to_visits(
            session=session,
            config=config,
            video_id=video_id,
        )
    except Exception as exc:
        session.rollback()
        raise click.ClickException(f'Ошибка сопоставления чеков: {exc}') from exc
    finally:
        session.close()

    video_label = video_id if video_id is not None else 'all'

    click.echo('RECEIPT MATCHING FINISHED')
    click.echo('-' * 20)
    click.echo(f'Video ID: {video_label}')
    click.echo(f'Visits checked: {stats["visits_checked"]}')
    click.echo(f'Matches created: {stats["matches_created"]}')
    click.echo(f'Matches skipped existing: {stats["matches_skipped_existing"]}')
    click.echo(f'Visits without receipt: {stats["visits_without_receipt"]}')
    click.echo(f'Ambiguous matches: {stats["ambiguous_matches"]}')


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


@cli.command('visits-check')
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
def visits_check_command(
    config_path: str,
    video_path: str,
    start_frame: int,
    limit: int,
) -> None:
    '''
    Проверка формирования визитов из треков в зоне кассы;
    [Arg]: config_path, video_path, start_frame, limit
    '''

    if start_frame < 0:
        raise click.ClickException('start_frame должен быть >= 0')

    if limit < 1:
        raise click.ClickException('limit должен быть >= 1')

    config = load_config(config_path)
    metadata = read_video_metadata(video_path)

    visits_builder = VisitsBuilder(
        max_missing_seconds=config.tracking.max_missing_seconds,
        min_visit_seconds=config.video.min_visit_seconds,
    )

    processed_frames = 0
    frames_with_tracks = 0
    frames_with_tracks_in_zone = 0
    total_tracks = 0
    total_tracks_in_zone = 0
    unique_track_ids = set()
    unique_track_ids_in_zone = set()

    for frame_data in iter_video_frames_range(
        path=video_path,
        frame_step=config.video.frame_step,
        start_frame=start_frame,
        limit=limit,
        metadata=metadata,
    ):
        processed_frames += 1

        frame_index = frame_data['frame_index']
        timestamp_seconds = frame_data['timestamp_seconds']
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

        total_tracks += len(tracks)
        total_tracks_in_zone += len(tracks_in_zone)

        if tracks:
            frames_with_tracks += 1

        if tracks_in_zone:
            frames_with_tracks_in_zone += 1

        for track in tracks:
            unique_track_ids.add(track['track_id'])

        for track in tracks_in_zone:
            unique_track_ids_in_zone.add(track['track_id'])

        visits_builder.update(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            tracks_in_zone=tracks_in_zone,
        )

    visits = visits_builder.finish_all()

    click.echo('VISITS CHECK FINISHED')
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
    click.echo(f'Unique track ids: {sorted(unique_track_ids)}')
    click.echo(f'Unique track ids in zone: {sorted(unique_track_ids_in_zone)}')
    click.echo(f'Visits found: {len(visits)}')
    click.echo('')

    for visit in visits:
        click.echo(
            f'Track ID: {visit["track_id"]}; '
            f'entry_frame: {visit["entry_frame_index"]}; '
            f'exit_frame: {visit["exit_frame_index"]}; '
            f'entry_ts: {visit["entry_timestamp_seconds"]}; '
            f'exit_ts: {visit["exit_timestamp_seconds"]}; '
            f'duration: {visit["duration_seconds"]}; '
            f'observations: {visit["observations_count"]}; '
            f'best_confidence: {visit["best_confidence"]:.4f}'
        )


@cli.command('visits-ocr-check')
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
def visits_ocr_check_command(
    config_path: str,
    video_path: str,
    start_frame: int,
    limit: int,
) -> None:
    '''
    Проверка формирования визитов с OCR-временем входа и выхода;
    [Arg]: config_path, video_path, start_frame, limit
    '''

    if start_frame < 0:
        raise click.ClickException('start_frame должен быть >= 0')

    if limit < 1:
        raise click.ClickException('limit должен быть >= 1')

    config = load_config(config_path)
    metadata = read_video_metadata(video_path)

    visits_builder = VisitsBuilder(
        max_missing_seconds=config.tracking.max_missing_seconds,
        min_visit_seconds=config.video.min_visit_seconds,
    )

    processed_frames = 0
    frames_with_tracks = 0
    frames_with_tracks_in_zone = 0

    for frame_data in iter_video_frames_range(
        path=video_path,
        frame_step=config.video.frame_step,
        start_frame=start_frame,
        limit=limit,
        metadata=metadata,
    ):
        processed_frames += 1

        frame_index = frame_data['frame_index']
        timestamp_seconds = frame_data['timestamp_seconds']
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

        if tracks:
            frames_with_tracks += 1

        if tracks_in_zone:
            frames_with_tracks_in_zone += 1

        visits_builder.update(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            tracks_in_zone=tracks_in_zone,
        )

    visits = visits_builder.finish_all()

    visits_with_ocr = []

    for visit in visits:
        entry_frame = read_video_frame(
            path=video_path,
            frame_index=visit['entry_frame_index'],
        )
        entry_crop = crop_frame(entry_frame, config.ocr.crop)
        entry_text, ocr_entered_at = recognize_datetime_from_crop(
            entry_crop,
            config.ocr,
        )

        exit_frame = read_video_frame(
            path=video_path,
            frame_index=visit['exit_frame_index'],
        )
        exit_crop = crop_frame(exit_frame, config.ocr.crop)
        exit_text, ocr_left_at = recognize_datetime_from_crop(
            exit_crop,
            config.ocr,
        )

        visit_with_ocr = visit.copy()
        visit_with_ocr['ocr_entry_text'] = entry_text
        visit_with_ocr['ocr_exit_text'] = exit_text
        visit_with_ocr['ocr_entered_at'] = ocr_entered_at
        visit_with_ocr['ocr_left_at'] = ocr_left_at

        visits_with_ocr.append(visit_with_ocr)

    click.echo('VISITS OCR CHECK FINISHED')
    click.echo('-' * 20)
    click.echo(f'Video: {video_path}')
    click.echo(f'Start frame: {start_frame}')
    click.echo(f'Frame step: {config.video.frame_step}')
    click.echo(f'Limit: {limit}')
    click.echo(f'Processed frames: {processed_frames}')
    click.echo(f'Frames with tracks: {frames_with_tracks}')
    click.echo(f'Frames with tracks in zone: {frames_with_tracks_in_zone}')
    click.echo(f'Visits found: {len(visits)}')
    click.echo(f'Visits with OCR: {len(visits_with_ocr)}')
    click.echo('')

    for visit in visits_with_ocr:
        click.echo(
            f'Track ID: {visit["track_id"]}; '
            f'entry_frame: {visit["entry_frame_index"]}; '
            f'exit_frame: {visit["exit_frame_index"]}; '
            f'duration: {visit["duration_seconds"]}; '
            f'ocr_entered_at: {visit["ocr_entered_at"]}; '
            f'ocr_left_at: {visit["ocr_left_at"]}'
        )
        click.echo(
            f'  OCR entry text: {visit["ocr_entry_text"]}; '
            f'OCR exit text: {visit["ocr_exit_text"]}'
        )
