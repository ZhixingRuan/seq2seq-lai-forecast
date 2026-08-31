import xarray as xr

from log import get_logger

logger = get_logger(__name__)


def replace_lai_flag_input(
    data: xr.Dataset,
    flag_data: xr.Dataset,
    flag_var: str = 'LAI_flag',
) -> xr.Dataset:
    """Replace `data`'s `flag_var` (used as an input feature) with `flag_data`'s `flag_var`.

    Overwrites the variable in place -- intended for swapping the input feature
    itself at inference time (see predict.py's --flag-path).

    Assumes `flag_data` already shares `data`'s time/lat/lon coordinates;
    raises if shared dimension sizes or coordinate values don't line up.
    """
    if flag_var not in flag_data:
        raise ValueError(f"'{flag_var}' not found in flag_data")
    if flag_var not in data:
        raise ValueError(f"'{flag_var}' not found in data")

    shared_dims = set(data.dims) & set(flag_data[flag_var].dims)
    for dim in shared_dims:
        if data.sizes[dim] != flag_data.sizes[dim]:
            raise ValueError(
                f"Dimension '{dim}' size mismatch: data={data.sizes[dim]}, "
                f"flag_data={flag_data.sizes[dim]}"
            )
        if dim in data.coords and dim in flag_data.coords and not data[dim].equals(flag_data[dim]):
            raise ValueError(f"Coordinate '{dim}' values differ between data and flag_data")

    logger.info(f"Replacing '{flag_var}' in data with values from flag_data (input feature)")
    return data.assign({flag_var: flag_data[flag_var]})
