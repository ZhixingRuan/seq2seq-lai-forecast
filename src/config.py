from typing import Callable
from functools import cached_property, wraps
import yaml
import json
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Annotated, Any

from pydantic import PlainSerializer
from pydantic_settings import (
    BaseSettings,
)
from torch.profiler import ProfilerActivity

from util import slurm_job_id
from data import DatasetConfig


ROOT_LOGGER = 'ml_lai'
THIS_DIR = Path(__file__).parent
DEFAULT_CONFIG_PATH = THIS_DIR / 'default_config.yaml'


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f'{ROOT_LOGGER}.{name}')


logger = get_logger(__name__)


def ensure_dirs(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        path = f(*args, **kwargs)
        path.mkdir(parents=True, exist_ok=True)
        return path

    return wrapper


Path = Annotated[Path, PlainSerializer(lambda path: str(path), return_type=str)]


class ProfilerConfig(BaseSettings):
    activities: list[ProfilerActivity] = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True
    on_trace_ready: Optional[Callable] = None
    schedule: Optional[Callable] = None


class RunConfig(BaseSettings):
    sample: Optional[int] = None
    train_test_split: float
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    val_start: str = '2022-01-01'
    dataset: DatasetConfig
    train_loader_params: dict[str, Any]
    val_loader_params: dict[str, Any]
    train_params: dict[str, Any]
    root_dir: Path

    @cached_property
    @ensure_dirs
    def run_dir(self) -> Path:
        if job_id := slurm_job_id():
            run_dir = self.root_dir / f'slurm_{job_id}'
        else:
            run_dir = (
                self.root_dir / f'local_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
        return run_dir

    @property
    @ensure_dirs
    def log_dir(self) -> Path:
        return self.run_dir / 'logs'

    @property
    @ensure_dirs
    def checkpoint_dir(self) -> Path:
        return self.run_dir / 'checkpoints'

    @property
    @ensure_dirs
    def tensorboard_dir(self) -> Path:
        return self.run_dir / 'tensorboard'

    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'RunConfig':
        path = DEFAULT_CONFIG_PATH if path is None else path
        logger.info(f'Loading config from {path}')
        with open(path, 'r') as yaml_file:
            config_dict = yaml.safe_load(yaml_file)

        return cls(**config_dict)

    @property
    def yaml(self) -> str:
        return yaml.dump(self.model_dump())

    @property
    def json(self) -> str:
        return json.dumps(self.model_dump(), indent=4)


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
