from pathlib import Path

import yaml

from pechvision.config.schema import AppConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)

    with config_path.open('r', encoding='utf-8') as file:
        raw_config = yaml.safe_load(file)

    return AppConfig.model_validate(raw_config)