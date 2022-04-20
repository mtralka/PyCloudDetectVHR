from os import path
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from pycvhr.classes import CombinationMethod

try:
    import gdal
    import gdalconst
    from gdal import Dataset
except ImportError:
    from osgeo import gdal, gdalconst
    from osgeo.gdal import Dataset


def _determine_array_remainder(target: int, maximum: int, window: int) -> int:
    """Returns the maximum value within bounds for `target`"""

    # if ending number is greater than RGB array size,
    # use RGB array size
    ending_target: int = target + window

    if ending_target > maximum:
        return maximum
    return ending_target


def number_of_bands(target: Union[Path, str]) -> int:
    """Returns number of raster bands in `target` or 0 if `target` is unreadable"""
    ds = gdal.Open(str(target))

    if ds is None:
        return 0

    return int(ds.RasterCount)


def open_raster_as_array(
    file_path: Union[Path, str], bands: Union[List[int], int] = 1
) -> np.ndarray:
    """Opens and returns raster as NumPy array

    Uses GDAL to open raster `band` of raster at `file_path` as NumPy array.
    Verifies the existance of `file_path` and the validity of `band`

    Parameters
    ----------
    file_path : str or Path
        Full file path to target raster file
    bands : List[int] or int
        Band(s) to extract from raster file at `file_path`

    Returns
    -------
    np.ndarray
        n * len(bands) dimensional array of raster `bands` from `file_path`
    """

    file_path: Path = Path(file_path) if isinstance(file_path, str) else file_path
    target_bands: List[int] = [bands] if isinstance(bands, int) else bands

    # Verify that `file_path` is a file before continuing
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"{file_path} does not exist")

    # Open `file_path` with GDAL
    ds = gdal.Open(str(file_path))

    if ds is None:
        raise TypeError(f"`ds` is None. GDAL was unable to open {file_path}")

    # Count number of bands
    number_bands: int = ds.RasterCount

    band_array_list: List[np.ndarray] = []

    for target_band in target_bands:
        if target_band > number_bands or target_band <= 0:
            raise ValueError(
                f"target band {target_band} is outside `file_path` band scope"
            )

        band_array_list.append(ds.GetRasterBand(target_band).ReadAsArray())

    array: np.ndarray = np.stack(band_array_list, axis=2)

    # Close file
    ds = None

    return array


def extract_window_from_array(
    array: np.ndarray,
    col_offset: int,
    row_offset: int,
    window_size: Tuple[int, int, int],
) -> np.ndarray:
    """Returns window with shape `window_size` offset by `{row,col}_offset to `array`"""

    number_rows: int = array.shape[0]
    number_cols: int = array.shape[1]

    window_ending_col: int = _determine_array_remainder(
        target=col_offset, maximum=number_cols, window=window_size[0]
    )
    window_ending_row: int = _determine_array_remainder(
        target=row_offset, maximum=number_rows, window=window_size[1]
    )

    window: np.ndarray = array[
        row_offset:window_ending_row, col_offset:window_ending_col
    ]

    return window


def pad_array(array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Bottom and right Pad 3D input array - `array` -
    to match 2D dimensions of `target_shape`"""

    x_pad: int = target_shape[0] - array.shape[0]
    y_pad: int = target_shape[1] - array.shape[1]

    # ((top, bottom), (left, right), (3rd dimension))
    return np.pad(array, ((0, x_pad), (0, y_pad), (0, 0)))


def reconcile_window_to_array(
    windowed_array: np.ndarray,
    master_array: np.ndarray,
    col_offset: int,
    row_offset: int,
    combination_method: CombinationMethod.MAX,
):
    """Takes input array `windowed_array` which is subset from `master_array` by `{col,row}_offset.
    Reconciles values using `combination_method`"""

    window_ending_col: int = _determine_array_remainder(
        target=col_offset, maximum=master_array.shape[1], window=windowed_array.shape[0]
    )
    window_ending_row: int = _determine_array_remainder(
        target=row_offset, maximum=master_array.shape[0], window=windowed_array.shape[1]
    )

    current_mask_value: np.ndarray = master_array[
        row_offset:window_ending_row,
        col_offset:window_ending_col,
    ]

    windowed_array = windowed_array[
        : current_mask_value.shape[0],
        : current_mask_value.shape[1],
    ]

    merged_results: np.ndarray
    if combination_method.MAX:
        merged_results = np.maximum(current_mask_value, windowed_array)
    elif combination_method.MIN:
        windowed_array = np.where(windowed_array == 0, 1, -1)
        merged_results = np.minimum(windowed_array, current_mask_value)
    elif combination_method.REPLACE:
        merged_results = windowed_array
    else:
        raise ValueError(f"{combination_method.name} not supported")

    master_array[
        row_offset:window_ending_row, col_offset:window_ending_col
    ] = merged_results

    return master_array


def create_outfile_dataset(
    file_path: str,
    x_size: int,
    y_size: int,
    wkt_projection: str,
    geo_transform: tuple,
    number_bands: int,
    driver: str = "GTiff",
    data_type=gdalconst.GDT_Int16,
    outfile_options: list = ["COMPRESS=DEFLATE"],
) -> Dataset:

    """Creates outfile dataset

    Uses GDAL to create an outfile Dataset to `file_path` using given metadata
    parameters

    Parameters
    ----------
    file_path : str
        Full file path to target raster file
    x_size : int
        Desired outfile dataset X size
    y_size : int
        Desired outfile dataset Y size
    wkt_projection : str
        WKT formated projection for outfile dataset
    geo_transform : tuple
        Geographic transformation for outfile dataset
    number_bands : int
        Number of bands for outfile dataset
    driver : str
        Outfile driver type. Default `GTiff`
    data_type : gdalconst.*
        Outfile data type. Default gdalconst.GDT_Int16
    outfile_options : list
        List of GDAL outfile options. Default ['COMPRESS=DEFLATE']

    Returns
    -------
    osgeo.gdal.Dataset
        GDAl dataset with given metdata parameters
    """
    # Create outfile driver
    driver = gdal.GetDriverByName(driver)

    # Create outfile dataset
    ds = driver.Create(
        file_path, x_size, y_size, number_bands, data_type, outfile_options
    )

    # Confirm successful `ds` creation
    if ds is None:
        raise TypeError(f"`ds` is None. GDAL was unable to create {file_path}")

    # Set outfile projection in WKT format
    ds.SetProjection(wkt_projection)

    # Set outfile geo transform
    ds.SetGeoTransform(geo_transform)

    return ds


def write_array_to_ds(
    ds: Dataset, array: np.ndarray, band: int = 1, no_data_value: int = -9999
) -> Dataset:

    """Writes NumPy array to GDAL Dataset band

    Uses GDAL to write `array` to `ds` `band` using given metadata parameters

    Parameters
    ----------
    ds : osgeo.gdal.Dataset
        GDAL dataset
    array : np.ndarray
        Full file path to target raster file
    band : int
        Target DS band to write `array`
    no_data_value : int
        No data value for `band`. Default -9999

    Returns
    -------
    osgeo.gdal.Dataset
        GDAl dataset with `array` written to `band`

    """
    # Confirm `ds` is valid
    if ds is None:
        raise ValueError("`ds` is None")

    number_bands: int = ds.RasterCount

    # Verify `band` is within the number of bands in `file_path` and
    # greater than zero
    if band > number_bands or band <= 0:
        raise ValueError(f"target band {band} is outside `ds` band scope")

    # Write `array` to outfile dataset
    ds.GetRasterBand(band).WriteArray(array)

    # Set outfile `no_data_value`
    ds.GetRasterBand(band).SetNoDataValue(no_data_value)

    return


def get_raster_metadata(file_path: str) -> dict:
    """Opens and returns raster metadata

    Uses GDAL to open raster at `file_path` and returns raster metadata. Band
    agnostic

    Parameters
    ----------
    file_path : str
        Full file path to target raster file

    Returns
    -------
    dict
        dictionary of metadata from `file_path` raster
    """
    # Verify that `file_path` is a file before continuing
    if not path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")

    # Open `file_path` with GDAL
    ds = gdal.Open(str(file_path))

    if ds is None:
        raise TypeError(f"`ds` is None. GDAL was unable to open {file_path}")

    # Create `stats` dictionary
    stats: dict = {}
    stats["total_bands"] = ds.RasterCount
    stats["x_size"] = ds.RasterXSize
    stats["y_size"] = ds.RasterYSize
    stats["wkt_projection"] = ds.GetProjectionRef()
    stats["geo_transform"] = ds.GetGeoTransform()
    stats["xmin"] = ds.GetGeoTransform()[0]
    stats["xmax"] = stats["xmin"] + stats["x_size"] * ds.GetGeoTransform()[1]
    stats["ymax"] = ds.GetGeoTransform()[3]
    stats["ymin"] = stats["ymax"] + stats["y_size"] * ds.GetGeoTransform()[5]

    # Close file
    ds = None

    return stats
