import sys
from typing import Optional
from pathlib import Path
import logging

ROOT_LOGGER = 'ml_lai'


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f'{ROOT_LOGGER}.{name}')


logger = get_logger(__name__)


def setup_logging(level, log_dir: Optional[Path]):
    root_logger = logging.getLogger(ROOT_LOGGER)
    root_logger.setLevel(level)

    handlers = [logging.StreamHandler(stream=sys.stdout)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / 'ml_lai.log'))
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        root_logger.addHandler(handler)
