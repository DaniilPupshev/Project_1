import hashlib
import json
from typing import Any

from sqlalchemy import make_url

from pechvision.config.schema import AppConfig


def build_config_snapshot(config: AppConfig) -> dict[str, Any]:
    '''Создание снимка конфигурации'''

    snapshot = config.model_dump(mode='json')
    snapshot['database']['url'] = make_url(config.database.url).render_as_string(hide_password=True)
    return snapshot


def build_processing_config_payload(
    config_snapshot: dict[str, Any]
) -> dict[str, Any]:
    '''Выбирает параметры конфигурации, влияющие на обработку видео'''

    PROCESSING_CONFIG_SECTIONS = (
        'project',
        'video',
        'ocr',
        'cashier_zone',
        'detection',
        'tracking',
        'faces',
        'demographics',
        'cashiers',
    )

    return {
        section: config_snapshot[section]
        for section in PROCESSING_CONFIG_SECTIONS
    }

def calculate_config_hash(
    processing_config: dict[str, Any]
) -> str:

    canonical_json = json.dumps(
        processing_config,
        sort_keys=True,
        separators=(',', ':')
    ).encode('utf-8')

    config_payload = hashlib.sha256(canonical_json).hexdigest()
    return config_payload
