import logging
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer
import torch
from torch.utils.data import DataLoader, Subset
import xarray as xr

from data import SequenceDataset
from train import train_convLSTM
from config import RunConfig
from log import setup_logging, get_logger

logger = get_logger(__name__)


def main(
    config_path: Annotated[
        Path,
        typer.Option(
            '--config',
            exists=True,
            dir_okay=False,
            file_okay=True,
            readable=True,
        ),
    ],
    profile: bool = False,
    checkpoint_epochs: int = typer.Option(5, help="Save a checkpoint every N epochs"),
    resume: bool = typer.Option(False, "--resume/--no-resume", help="Resume from latest checkpoint"),
    resume_checkpoint_dir: Optional[Path] = typer.Option(
        None,
        "--resume-checkpoint-dir",
        help="Directory to look for checkpoints to resume from",
    ),
    accumulation_steps: int = typer.Option(1, help="Gradient accumulation steps"),
):
    config = RunConfig.load(config_path)
    setup_logging(logging.DEBUG, config.log_dir)

    logger.info(f'Config: \n{config.json}')
    with open(config.run_dir / 'config.yaml', 'w') as f:
        f.write(config.yaml)
    logger.info(f'Run directory: {config.run_dir}')

    logger.info('Loading data')
    data = xr.open_zarr(config.dataset.data.path, consolidated=False)
    if config.time_start is not None or config.time_end is not None:
        data = data.sel(time=slice(config.time_start, config.time_end))
        logger.info(f'Time slice: {config.time_start} to {config.time_end}')

    train_end  = pd.Timestamp(config.val_start) - pd.Timedelta(days=1)
    train_data = data.sel(time=slice(None, train_end))
    val_data   = data.sel(time=slice(config.val_start, None))
    logger.info(f'Train data: up to {train_end.date()} ({len(train_data.time)} time steps), '
                f'Val data: from {config.val_start} ({len(val_data.time)} time steps)')

    train_dataset = SequenceDataset(train_data, config.dataset)
    val_dataset   = SequenceDataset(val_data,   config.dataset)

    if config.sample is not None:
        n_train = int(config.sample * config.train_test_split)
        n_val   = config.sample - n_train
        rng = np.random.default_rng(seed=33)
        train_indices = rng.choice(len(train_dataset), size=min(n_train, len(train_dataset)), replace=False)
        val_indices   = rng.choice(len(val_dataset),   size=min(n_val,   len(val_dataset)),   replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset   = Subset(val_dataset,   val_indices)
        logger.info(f'Subsampled to {len(train_dataset)} train, {len(val_dataset)} val samples '
                    f'(target {n_train}/{n_val} from sample={config.sample}, split={config.train_test_split:.0%})')

    logger.info(f'Train dataset length: {len(train_dataset)}, Val dataset length: {len(val_dataset)}')

    train_loader = DataLoader(train_dataset, **config.train_loader_params)
    val_loader   = DataLoader(val_dataset,   **config.val_loader_params)

    logger.info('Beginning training')

    best_model = train_convLSTM(
        train_loader,
        val_loader,
        input_dim=config.dataset.n_input_channels,
        **config.train_params,
        tensorboard_dir=config.tensorboard_dir,
        checkpoint_dir=config.checkpoint_dir,
        checkpoint_epochs=checkpoint_epochs,
        profile=profile,
        resume=resume,
        resume_checkpoint_dir=resume_checkpoint_dir,
        accumulation_steps=accumulation_steps,
    )

    torch.save(best_model, config.run_dir / 'best_model.pth')


if __name__ == '__main__':
    typer.run(main)
