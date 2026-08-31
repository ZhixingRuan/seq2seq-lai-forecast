import logging
from pathlib import Path
from typing import Optional, Annotated

import typer
import zarr
import xarray as xr
from xarray.core.coordinates import DatasetCoordinates
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
import dask.array as da

from config import RunConfig
from model import ConvLSTMSeq2SeqDOY, ConvLSTMSeq2SeqDOY2Stacked
from data import SequenceDataset
from data_additional_laiflag import replace_lai_flag_input
from log import setup_logging, get_logger

logger = get_logger(__name__)


NUM_WORKERS = 1
PREDICT_BATCH_SIZE = 2
OUTPUT_DTYPE = 'float16'  # float16 (~2x smaller) or float32 (more precision)


def load_model(path: Path, config: RunConfig) -> torch.nn.Module:
    result = torch.load(path, weights_only=False)
    # Is it a pickled model?
    if isinstance(result, torch.nn.Module):
        return result

    model_kwargs = {
        'input_dim': config.dataset.n_input_channels,
        'target_dim': 1,
        'hidden_dim': config.train_params['hidden_dim'],
        'kernel_size': config.train_params['kernel_size'],
    }

    state_dict = result['model_state_dict'] if 'model_state_dict' in result else result
    if any(k.startswith('encoder_1') for k in state_dict):
        model = ConvLSTMSeq2SeqDOY2Stacked(**model_kwargs)
    else:
        model = ConvLSTMSeq2SeqDOY(**model_kwargs)

    model.load_state_dict(state_dict)

    return model


def initialize_zarr(
    coords: DatasetCoordinates,
    max_leadtime: int,
    input_seq_length: int,
    features: list[str],
    output_path: Path,
    data: xr.Dataset,
    dtype: str = OUTPUT_DTYPE,
) -> None:
    """Initialize a (mostly empty) zarr dataset for predictions to be written into."""
    dims = ['time', 'leadtime', 'lat', 'lon']
    leadtimes = range(max_leadtime + 1)
    coord_leadtime = xr.DataArray(
        leadtimes, dims='leadtime', name='leadtime', coords={'leadtime': leadtimes}
    )
    coords = {
        'time': coords.get('time').isel(time=slice(input_seq_length - 1, None)),
        'leadtime': coord_leadtime,
        'lat': coords.get('lat'),
        'lon': coords.get('lon'),
    }

    shape = tuple(len(coords[dim]) for dim in dims)

    # Initialize data_vars with Dask arrays and auto-chunking for spatial dimensions
    data_vars = {
        var: (dims, da.zeros(shape, dtype=dtype, chunks=(1, max_leadtime, "auto", "auto")))
        for var in features
    }

    # Create the xarray Dataset
    ds = xr.Dataset(data_vars, coords)

    # Use the same chunking as in data_vars for encoding
    compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=5, shuffle='bitshuffle')
    encoding = {
        var: {'dtype': dtype, 'compressor': compressor}
        for var in features
    }

    # Write to Zarr using Dask auto-chunking and matching encoding
    ds.to_zarr(output_path, mode='w', encoding=encoding, compute=False)

    # Save data['LAI_norm'] at leadtime=0 for each time point
    logger.info('Save LAI_norm at leadtime=0 for each time point')
    zarr_ds = xr.open_zarr(output_path)
    lai_values = data['LAI_norm'].isel(time=slice(input_seq_length - 1, None)).values

    expected_shape = zarr_ds['LAI_norm'].sel(leadtime=0).shape
    if lai_values.shape != expected_shape:
        raise ValueError(f"Shape mismatch: Expected {expected_shape}, got {lai_values.shape}")

    zarr_ds['LAI_norm'].loc[dict(leadtime=0)] = lai_values
    zarr_ds.to_zarr(output_path, mode='a')


def append_zarr(
    coords: DatasetCoordinates,
    max_leadtime: int,
    input_seq_length: int,
    features: list[str],
    output_path: Path,
    data: xr.Dataset,
    dtype: str = OUTPUT_DTYPE,
) -> int:
    """Extend an existing zarr with new time steps. Returns the existing time count (write offset)."""
    existing_ds = xr.open_zarr(output_path)
    time_start_offset = len(existing_ds['time'])

    dims = ['time', 'leadtime', 'lat', 'lon']
    leadtimes = range(max_leadtime + 1)
    coord_leadtime = xr.DataArray(
        leadtimes, dims='leadtime', name='leadtime', coords={'leadtime': leadtimes}
    )
    new_coords = {
        'time': coords.get('time').isel(time=slice(input_seq_length - 1, None)),
        'leadtime': coord_leadtime,
        'lat': coords.get('lat'),
        'lon': coords.get('lon'),
    }
    shape = tuple(len(new_coords[dim]) for dim in dims)
    data_vars = {
        var: (dims, da.zeros(shape, dtype=dtype, chunks=(1, max_leadtime, "auto", "auto")))
        for var in features
    }
    ds = xr.Dataset(data_vars, new_coords)
    ds.to_zarr(output_path, mode='a', append_dim='time')

    # Fill leadtime=0 with observed LAI for the new time steps
    logger.info('Save LAI_norm at leadtime=0 for new time steps')
    store = zarr.open_group(output_path, mode='r+')
    lai_values = data['LAI_norm'].isel(time=slice(input_seq_length - 1, None)).values
    store['LAI_norm'][time_start_offset:, 0, :, :] = lai_values

    return time_start_offset


def predict_whole_area(
    model: torch.nn.Module,
    data: xr.Dataset,
    config_path,
    output_path,
    device: torch.device,
    dtype: str = OUTPUT_DTYPE,
    time_start_offset: int = 0,
) -> xr.Dataset:
    """
    Generate predictions for the whole dataset without patches and write to a zarr dataset.
    """
    config = RunConfig.load(config_path)
    max_leadtime = config.dataset.target_seq_length

    # Create dataset for inference — always stride=1 to predict every timestep
    inference_config = config.dataset.model_copy(update={'stride': 1, 'patch_size': None})
    dataset = SequenceDataset(data, inference_config)
    data_loader = DataLoader(
        dataset, batch_size=PREDICT_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    store = zarr.open_group(output_path, mode='r+')

    leadtime_slice = slice(1, None)  # Skip leadtime=0 (filled by initialize_zarr)

    model.eval()
    with torch.no_grad():
        time_offset = time_start_offset
        for x, _, doy_x, doy_y, _ in data_loader:
            x = x.to(device, non_blocking=True)
            doy_x = doy_x.to(device, non_blocking=True)
            doy_y = doy_y.to(device, non_blocking=True)

            with autocast(device.type):
                pred = model(x, doy_x, doy_y, max_leadtime)

            # pred shape: (B, max_leadtime, 1, H, W) -> (B, max_leadtime, H, W)
            pred = pred.squeeze(2).cpu().numpy().astype(dtype)
            batch_size = pred.shape[0]

            store['LAI_norm'][time_offset:time_offset + batch_size, leadtime_slice, :, :] = pred
            time_offset += batch_size

            if time_offset % (PREDICT_BATCH_SIZE * 10) == 0:
                logger.info('Processed %d/%d time steps', time_offset, len(dataset))

    logger.info('Processed %d/%d time steps', time_offset, len(dataset))


def initialize_zarr_range(
    coords: DatasetCoordinates,
    leadtime_start: int,
    leadtime_end: int,
    input_seq_length: int,
    features: list[str],
    output_path: Path,
    dtype: str = OUTPUT_DTYPE,
) -> None:
    """Initialize an empty zarr dataset covering only the given leadtime range [leadtime_start, leadtime_end]."""
    dims = ['time', 'leadtime', 'lat', 'lon']
    leadtimes = list(range(leadtime_start, leadtime_end + 1))
    coord_leadtime = xr.DataArray(
        leadtimes, dims='leadtime', name='leadtime', coords={'leadtime': leadtimes}
    )
    out_coords = {
        'time': coords.get('time').isel(time=slice(input_seq_length - 1, None)),
        'leadtime': coord_leadtime,
        'lat': coords.get('lat'),
        'lon': coords.get('lon'),
    }

    shape = tuple(len(out_coords[dim]) for dim in dims)
    n_leadtimes = len(leadtimes)

    data_vars = {
        var: (dims, da.zeros(shape, dtype=dtype, chunks=(1, n_leadtimes, "auto", "auto")))
        for var in features
    }

    ds = xr.Dataset(data_vars, out_coords)

    compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=5, shuffle='bitshuffle')
    encoding = {
        var: {'dtype': dtype, 'compressor': compressor}
        for var in features
    }

    ds.to_zarr(output_path, mode='w', encoding=encoding, compute=False)


def predict_whole_area_range(
    model: torch.nn.Module,
    data: xr.Dataset,
    config_path,
    output_path,
    device: torch.device,
    leadtime_start: int,
    leadtime_end: int,
    dtype: str = OUTPUT_DTYPE,
    time_start_offset: int = 0,
) -> None:
    """
    Generate predictions for the whole dataset and write only the specified leadtime range
    [leadtime_start, leadtime_end] to the output zarr.  The model still runs for all
    leadtimes; only the requested subset is saved.
    """
    config = RunConfig.load(config_path)
    max_leadtime = config.dataset.target_seq_length

    if leadtime_start < 1 or leadtime_end > max_leadtime or leadtime_start > leadtime_end:
        raise ValueError(
            f"leadtime range [{leadtime_start}, {leadtime_end}] is invalid "
            f"(model max_leadtime={max_leadtime}, minimum leadtime=1)"
        )

    inference_config = config.dataset.model_copy(update={'stride': 1, 'patch_size': None})
    dataset = SequenceDataset(data, inference_config)
    data_loader = DataLoader(
        dataset, batch_size=PREDICT_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    store = zarr.open_group(output_path, mode='r+')

    # pred is 0-indexed where index i corresponds to leadtime i+1;
    # slice out only the requested range before writing.
    pred_slice = slice(leadtime_start - 1, leadtime_end)

    model.eval()
    with torch.no_grad():
        time_offset = time_start_offset
        for x, _, doy_x, doy_y, _ in data_loader:
            x = x.to(device, non_blocking=True)
            doy_x = doy_x.to(device, non_blocking=True)
            doy_y = doy_y.to(device, non_blocking=True)

            with autocast(device.type):
                pred = model(x, doy_x, doy_y, max_leadtime)

            # pred shape: (B, max_leadtime, 1, H, W) -> (B, max_leadtime, H, W)
            pred = pred.squeeze(2).cpu().numpy().astype(dtype)
            batch_size = pred.shape[0]

            store['LAI_norm'][time_offset:time_offset + batch_size, :, :, :] = pred[:, pred_slice, :, :]
            time_offset += batch_size

            if time_offset % (PREDICT_BATCH_SIZE * 10) == 0:
                logger.info('Processed %d/%d time steps', time_offset, len(dataset))

    logger.info('Processed %d/%d time steps', time_offset, len(dataset))



def main(
    model_path: Annotated[Path, typer.Option('--model')],
    input_path: Annotated[Path, typer.Option('--input')],
    output_path: Annotated[Optional[Path], typer.Option('--output')] = None,
    config_path: Annotated[Optional[Path], typer.Option('--config')] = None,
    start_date: Annotated[str, typer.Option(help="Start date for slicing input data (format: YYYY-MM-DD)")] = "2021-01-01",
    end_date: Annotated[str, typer.Option(help="End date for slicing input data (format: YYYY-MM-DD)")] = "2023-12-31",
    initialize: Annotated[bool, typer.Option(help='Initialize the output zarr (overwrites existing)')] = True,
    append: Annotated[bool, typer.Option(help='Append new time steps to an existing zarr')] = False,
    device_type: Annotated[Optional[str], typer.Option()] = None,
    leadtime_start: Annotated[Optional[int], typer.Option(help='First leadtime day to save (inclusive). If set, --leadtime-end must also be set.')] = None,
    leadtime_end: Annotated[Optional[int], typer.Option(help='Last leadtime day to save (inclusive). If set, --leadtime-start must also be set.')] = None,
    flag_path: Annotated[
        Optional[Path],
        typer.Option(
            '--flag-path',
            help="Zarr store whose LAI_flag replaces the input data's LAI_flag "
                 "(used as an input feature, not the loss mask -- predict.py never computes loss)",
        ),
    ] = None,
):
    """Command line tool to generate predictions using a trained model.

    Usage:
        predict.py --model model.pth --input input.zarr --output output.zarr
        predict.py --model model.pth --input input.zarr --output output.zarr --append --start-date 2024-01-01 --end-date 2024-12-31
        predict.py --model model.pth --input input.zarr --output output.zarr --leadtime-start 31 --leadtime-end 60
        predict.py --model model.pth --input input.zarr --output output.zarr --flag-path other_flags.zarr
    """
    config = RunConfig.load(config_path)
    setup_logging(logging.DEBUG, None)

    use_range = leadtime_start is not None or leadtime_end is not None
    if use_range and (leadtime_start is None or leadtime_end is None):
        raise typer.BadParameter('--leadtime-start and --leadtime-end must both be provided together')

    output_path = (
        input_path.with_suffix('.pred.zarr') if output_path is None else output_path
    )

    device: torch.device
    if device_type is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_type)
    logger.info('Using device: %s', device)

    logger.info('Loading model: %s', model_path)
    model = load_model(model_path, config).to(device)

    logger.info('Loading input data: %s', input_path)
    input_data = xr.open_zarr(input_path, consolidated=False)
    input_data = input_data.sel(time=slice(start_date, end_date))

    if flag_path is not None:
        logger.info("Replacing LAI_flag input feature with: %s", flag_path)
        flag_data = xr.open_zarr(flag_path, consolidated=False)
        flag_data = flag_data.sel(time=slice(start_date, end_date))
        input_data = replace_lai_flag_input(input_data, flag_data)

    max_leadtime = config.dataset.target_seq_length
    input_seq_length = config.dataset.input_seq_length
    logger.debug('Max leadtime: %s', max_leadtime)

    time_start_offset = 0
    if use_range:
        if initialize:
            logger.info('Initializing output (leadtime %d–%d): %s', leadtime_start, leadtime_end, output_path)
            initialize_zarr_range(
                input_data.coords,
                leadtime_start, leadtime_end, input_seq_length,
                ['LAI_norm'],
                output_path)
        logger.info('Generating predictions (leadtime %d–%d)', leadtime_start, leadtime_end)
        predict_whole_area_range(
            model=model,
            data=input_data,
            config_path=config_path,
            output_path=output_path,
            device=device,
            leadtime_start=leadtime_start,
            leadtime_end=leadtime_end,
            time_start_offset=time_start_offset,
        )
    else:
        if append:
            logger.info('Appending to existing output: %s', output_path)
            time_start_offset = append_zarr(
                input_data.coords,
                max_leadtime, input_seq_length,
                ['LAI_norm'],
                output_path, input_data)
        elif initialize:
            logger.info('Initializing output: %s', output_path)
            initialize_zarr(
                input_data.coords,
                max_leadtime, input_seq_length,
                ['LAI_norm'],
                output_path, input_data)

        logger.info('Generating predictions as a whole image')
        predict_whole_area(
            model=model,
            data=input_data,
            config_path=config_path,
            output_path=output_path,
            device=device,
            time_start_offset=time_start_offset,
        )


if __name__ == '__main__':
    typer.run(main)
