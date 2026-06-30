import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from pechvision.config.schema import AppConfig


def get_database_url(config: AppConfig) -> str:
    '''Получение URL базы данных'''

    return os.getenv('DATABASE_URL') or config.database.url


def make_engine(config: AppConfig) -> Engine:
    '''Создание engine базы данных'''

    return create_engine(
        get_database_url(config),
        echo=False,
        pool_pre_ping=True
    )


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    '''Создание сессии базы данных'''

    return sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )