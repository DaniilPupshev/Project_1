import click
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from pechvision.config.loader import load_config
from pechvision.db.session import make_engine, make_session_factory
from pechvision.receipts.importer import import_receipts
from pechvision.video.registry import register_video
from pechvision.video.runs import create_processing_run


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