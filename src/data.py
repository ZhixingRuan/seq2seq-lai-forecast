from typing import Any, Optional

import numpy as np
import dask
import torch
from torch.utils.data import Dataset
import xarray as xr
from pydantic_settings import BaseSettings

from log import get_logger

# Set dask to synchronous to avoid issues with dask and multiprocessing / workers > 0
dask.config.set(scheduler='synchronous')


logger = get_logger(__name__)


class DataConfig(BaseSettings):
    path: str
    lat_dim: str = 'lat'
    lon_dim: str = 'lon'
    time_dim: str = 'time'


class DatasetConfig(BaseSettings):
    data: DataConfig = DataConfig()
    features: list[str] = ['tmax_norm', 'tmin_norm', 'tp_norm', 'LAI_norm', 'LAI_flag', 'lat_norm', 'lon_norm']
    lc_features: list[str] = []  # e.g. ['lc_crops', 'lc_grass', 'lc_others', 'lc_shrubs', 'lc_trees']
    target: str = 'LAI_norm'
    mask_var: str = 'LAI_flag'  # variable used for flag_y (loss mask); defaults to LAI_flag
    input_seq_length: int = 15
    target_seq_length: int = 15
    patch_size: Optional[int] = None
    stride: int = 1
    min_valid_fraction: float = 0.0  # minimum fraction of non-zero LAI_flag pixels over the full sequence window (input+target); 0.0 disables filtering

    @property
    def target_index(self):
        return self.features.index(self.target)
    
    @property
    def n_input_channels(self):
        return len(self.features) + len(self.lc_features)


class SequenceDataset(Dataset):
    def __init__(self, data: xr.Dataset, config: DatasetConfig):
        self.data = data
        self.time = data[config.data.time_dim]
        self.lat = data[config.data.lat_dim]
        self.lon = data[config.data.lon_dim]

        self.patch_size = config.patch_size
        self.stride = config.stride
        self.input_seq_length = config.input_seq_length
        self.target_seq_length = config.target_seq_length

        self.features = config.features
        self.target = [config.target]
        self.mask_var = config.mask_var
        self._has_mask_var = self.mask_var in data.variables
        if not self._has_mask_var:
            logger.warning(
                f"mask_var '{self.mask_var}' not found in data; flag_y will be all-zero "
                f"(fine for inference, but training loss masking requires it)."
            )

        self.output_dims = (
            config.data.time_dim,
            'variable',
            config.data.lat_dim,
            config.data.lon_dim,
        )

        if config.lc_features:
            self.lc_vars = config.lc_features
            self.lc_annual = np.stack(
                [data[v].values for v in self.lc_vars], axis=0
            ).astype(np.uint8)  # (n_lc, n_ann, n_lat, n_lon)
            self.lc_years = np.array(data[self.lc_vars[0]].lc_years, dtype="datetime64[Y]")
            logger.info(f"Loaded LC vars {self.lc_vars}, shape: {self.lc_annual.shape}")
        else:
            self.lc_annual = None
            self.lc_vars = []
        
        # drop lc vars BEFORE self.data is used anywhere else
        self.data = data.drop_vars(self.lc_vars, errors='ignore')

        # Precompute DOY encoding for all timesteps once
        self._doy_encoded = self.get_encoded_doy_from_time(self.time.values)  # (T, 2)

        # Cache which weather flag variables are present so __getitem__ doesn't
        if 'weather_flag' in data:
            self._weather_flag_vars = ['weather_flag']
        elif 'temp_flag' in data and 'prcp_flag' in data:
            self._weather_flag_vars = ['temp_flag', 'prcp_flag']
        elif 'temp_flag' in data:
            self._weather_flag_vars = ['temp_flag']
        elif 'prcp_flag' in data:
            self._weather_flag_vars = ['prcp_flag']
        else:
            self._weather_flag_vars = []
        
        # All variables needed for the x window in one read
        self._x_read_vars = list(dict.fromkeys(self.features + self._weather_flag_vars))

        # All variables needed for the full sequence window (input + target) in one read
        mask_vars = [self.mask_var] if self._has_mask_var else []
        self._all_read_vars = list(dict.fromkeys(self._x_read_vars + self.target + mask_vars))

        # Precompute valid indices by checking LAI_flag over the full sequence window
        if config.min_valid_fraction > 0.0 and config.patch_size is not None:
            self._valid_indices = self._compute_valid_indices(config.min_valid_fraction)
            n_patches = self.n_lat_patch * self.n_lon_patch
            logger.info(
                f'Valid patch filtering (min_valid_fraction={config.min_valid_fraction}): '
                f'{len(self._valid_indices) // self.n_seq} / {n_patches} spatial patches retained '
                f'({len(self._valid_indices)} / {self.n_seq * n_patches} total samples).'
            )
        else:
            self._valid_indices = None

        # Warn if patch_size doesn't evenly divide the spatial dimensions
        if self.patch_size is not None:
            lat_rem = len(self.lat) % self.patch_size
            lon_rem = len(self.lon) % self.patch_size
            if lat_rem:
                logger.warning(
                    f'lat size {len(self.lat)} is not divisible by patch_size {self.patch_size}; '
                    f'last {lat_rem} lat pixel(s) will never be sampled.'
                )
            if lon_rem:
                logger.warning(
                    f'lon size {len(self.lon)} is not divisible by patch_size {self.patch_size}; '
                    f'last {lon_rem} lon pixel(s) will never be sampled.'
                )

    @property
    def total_seq_length(self):
        return self.input_seq_length + self.target_seq_length

    @property
    def n_seq(self):
        # floor division is intentional: last valid start is (n_seq-1)*stride,
        # giving end index (n_seq-1)*stride + total_seq_length <= len(time).
        # With stride > 1, up to (stride - 1) tail timesteps are unused.
        return (len(self.time) - self.total_seq_length) // self.stride + 1

    @property
    def n_lat_patch(self):
        return 1 if self.patch_size is None else len(self.lat) // self.patch_size

    @property
    def n_lon_patch(self):
        return 1 if self.patch_size is None else len(self.lon) // self.patch_size

    def __len__(self):
        if self._valid_indices is not None:
            return len(self._valid_indices)
        return self.n_seq * self.n_lon_patch * self.n_lat_patch

    def __getitem__(self, idx):
        if self._valid_indices is not None:
            idx = self._valid_indices[idx]
        num_pixels = self.n_lon_patch * self.n_lat_patch
        seq_idx = (idx // num_pixels) * self.stride

        x_isel, _ = self.index_to_isel(idx)

        # Single zarr read covering the full sequence window (input + target)
        full_isel = {
            self.time.name: slice(seq_idx, seq_idx + self.total_seq_length),
            self.lat.name: x_isel[self.lat.name],
            self.lon.name: x_isel[self.lon.name],
        }
        full_data = self.data[self._all_read_vars].isel(full_isel).load()

        # Stack features/target directly from numpy — no xarray intermediate objects.
        # Static variables (e.g. lat_norm, lon_norm) have dims (lat, lon) with no time
        # axis; broadcast them to (input_seq_length, lat, lon) before stacking.
        feature_arrays = []
        for v in self.features:
            arr = full_data[v].to_numpy()
            if arr.ndim == 2:  # static (lat, lon) — no time dim
                arr = np.broadcast_to(arr[np.newaxis], (self.input_seq_length,) + arr.shape)
            else:
                arr = arr[:self.input_seq_length]
            feature_arrays.append(arr)
        x = torch.as_tensor(
            np.stack(feature_arrays, axis=1),
            dtype=torch.float,
        )  # (seq_x, variable, lat, lon)

        # ── Append LC channels ────────────────────────────────────────────────
        if self.lc_annual is not None:
            lc = self._get_lc(
                seq_idx,
                x_isel[self.lat.name],
                x_isel[self.lon.name],
            )  # (n_lc, lat, lon), float32
            lc_tensor = (
                torch.as_tensor(lc, dtype=torch.float)
                .unsqueeze(0)
                .expand(self.input_seq_length, -1, -1, -1)
                .contiguous()
            )  # (seq_x, n_lc, lat, lon)
            x = torch.cat([x, lc_tensor], dim=1)
            # x is now (seq_x, n_features + n_lc, lat, lon)

        y = torch.as_tensor(
            np.stack([full_data[v].to_numpy()[self.input_seq_length:] for v in self.target], axis=1),
            dtype=torch.float,
        )  # (seq_y, variable, lat, lon)

        if self._has_mask_var:
            mask_flag_np = full_data[self.mask_var].to_numpy()[self.input_seq_length:]  # (seq_y, lat, lon)
            flag_y = torch.as_tensor(mask_flag_np, dtype=torch.uint8)
        else:
            flag_y = torch.zeros((self.target_seq_length,) + x.shape[-2:], dtype=torch.uint8)

        # Slice precomputed DOY encoding
        doy_encoded_x = self._doy_encoded[seq_idx: seq_idx + self.input_seq_length]
        doy_encoded_y = self._doy_encoded[seq_idx + self.input_seq_length: seq_idx + self.total_seq_length]

        _, _, height, width = x.size()
        doy_x = torch.tensor(doy_encoded_x, dtype=torch.float).permute(1, 0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        doy_y = torch.tensor(doy_encoded_y, dtype=torch.float).permute(1, 0).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

        return x, y, doy_x, doy_y, flag_y

    def _compute_valid_indices(self, min_valid_fraction: float) -> list[int]:
        """Return flat indices whose spatial patch meets the min valid fraction.

        Missing pixels (LAI_flag == 0) due to reprojection gaps and ocean areas are
        spatially static — they don't vary across time.  We therefore check the
        patch validity from a single timestep and apply the result to all sequences.
        """
        logger.info('Computing valid indices — checking spatial LAI_flag mask...')
        # flag=2 timesteps are all fill-forward (no 0s), so they don't reveal the
        # true ocean/reprojection missing pattern.  Find the first flag=1 timestep.
        # Load the full time axis in one read and find the first flag=1 in numpy.
        lai_flag_all = self.data['LAI_flag'].load().to_numpy()  # (T, lat, lon)
        flag1_times = np.where(np.any(lai_flag_all == 1, axis=(1, 2)))[0]
        if flag1_times.size == 0:
            raise ValueError('No flag=1 timestep found in LAI_flag — cannot determine spatial missing mask.')
        lai_flag_spatial = lai_flag_all[flag1_times[0]]
        spatial_valid = (lai_flag_spatial != 0).astype(np.float32)  # (lat, lon)

        num_pixels = self.n_lat_patch * self.n_lon_patch
        valid_patch_pixel_ids = []

        for lat_i in range(self.n_lat_patch):
            for lon_i in range(self.n_lon_patch):
                if self.patch_size is None:
                    patch = spatial_valid
                else:
                    lat_sl = slice(lat_i * self.patch_size, (lat_i + 1) * self.patch_size)
                    lon_sl = slice(lon_i * self.patch_size, (lon_i + 1) * self.patch_size)
                    patch = spatial_valid[lat_sl, lon_sl]

                if patch.mean() >= min_valid_fraction:
                    valid_patch_pixel_ids.append(lat_i + lon_i * self.n_lat_patch)

        # Every sequence is valid for spatially-valid patches
        valid_indices = sorted(
            seq_i * num_pixels + pixel_idx
            for seq_i in range(self.n_seq)
            for pixel_idx in valid_patch_pixel_ids
        )

        return valid_indices

    def index_to_isel(self, idx) -> tuple[dict[str, Any], dict[str, Any]]:
        """Converts the index to the isel dictionaries for time, lat and lon

        Args:
            idx (int): Index of the dataset

        Returns:
            tuple(dict[str, Any], dict[str, Any]): Tuple of xarray isel dictionaries for x and y sequences
        """

        num_pixels = self.n_lon_patch * self.n_lat_patch
        lat_patch_idx = (idx % num_pixels) % self.n_lat_patch  # Latitude index first
        lon_patch_idx = (idx % num_pixels) // self.n_lat_patch  # Longitude index next
        seq_idx = (idx // num_pixels) * self.stride  # Apply stride to time index

        if self.patch_size is None:
            lat_slice = slice(None)
            lon_slice = slice(None)
        else:
            start_lon_idx = lon_patch_idx * self.patch_size
            lon_slice = slice(start_lon_idx, start_lon_idx + self.patch_size)

            start_lat_idx = lat_patch_idx * self.patch_size
            lat_slice = slice(start_lat_idx, start_lat_idx + self.patch_size)

        x_isel = {
            self.time.name: slice(seq_idx, seq_idx + self.input_seq_length),
            self.lat.name: lat_slice,
            self.lon.name: lon_slice,
        }
        y_isel = {
            self.time.name: slice(
                seq_idx + self.input_seq_length, seq_idx + self.total_seq_length
            ),
            self.lat.name: lat_slice,
            self.lon.name: lon_slice,
        }

        return x_isel, y_isel

    def get_xarray(self, idx) -> tuple[xr.Dataset, xr.Dataset]:
        """Get the xarray dataset for the given index"""
        x_isel, y_isel = self.index_to_isel(idx)
        return self.data.isel(x_isel), self.data.isel(y_isel)

    def get_encoded_doy_from_time(self, time_arr):
        # Vectorized DOY computation — no Python loop over timesteps
        ts_ns = time_arr.astype("datetime64[ns]").astype(np.int64)
        jan1_ns = time_arr.astype("datetime64[Y]").astype("datetime64[ns]").astype(np.int64)
        ns_per_day = 24 * 3600 * 1_000_000_000
        day_of_year = (ts_ns - jan1_ns) // ns_per_day + 1

        # Cyclical encoding
        sine_component = np.sin(2 * np.pi * day_of_year / 365)
        cosine_component = np.cos(2 * np.pi * day_of_year / 365)

        # Stack the sine and cosine components along a new dimension
        day_of_year_encoded = np.stack([sine_component, cosine_component], axis=-1)

        return day_of_year_encoded
    
    def _get_lc(self, t_start: int, lat_sl: slice, lon_sl: slice) -> np.ndarray:
        """Return LC (n_lc, lat, lon) as float32 for the year at t_start."""
        date = self.time.values[t_start].astype("datetime64[Y]")
        ai = int(np.searchsorted(self.lc_years, date, side="right") - 1)
        ai = np.clip(ai, 0, self.lc_annual.shape[1] - 1)
        return self.lc_annual[:, ai, lat_sl, lon_sl].astype(np.float32)



