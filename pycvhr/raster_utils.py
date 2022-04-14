from pathlib import Path
from typing import List
from typing import Tuple
from typing import Union

import numpy as np


try:
    import gdal
except ImportError:
    from osgeo import gdal


def determine_array_remainder(target: int, maximum: int, window: int) -> int:

    # if ending number is greater than RGB array size,
    # use RGB array size
    ending_target: int = target + window

    if ending_target > maximum:
        print("Hit Max")
        return maximum
    return ending_target


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
    self,
    array: np.ndarray,
    col_offset: int,
    row_offset: int,
    window_size: Tuple[int, int, int],
) -> np.ndarray:

    number_rows: int = array.shape[0]
    number_cols: int = array.shape[1]

    window_ending_col: int = determine_array_remainder(
        target=col_offset, maximum=number_cols, window=window_size[0]
    )
    window_ending_row: int = determine_array_remainder(
        target=row_offset, maximum=number_rows, window=window_size[1]
    )

    window: np.ndarray = array[
        row_offset:window_ending_row, col_offset:window_ending_col
    ]

    return window


def pad_array(array: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    x_pad: int = target_shape[0] - array.shape[0]
    y_pad: int = target_shape[1] - array.shape[1]

    # ((top, bottom), (left, right), (3rd dimension))
    return np.pad(array, ((0, x_pad), (0, y_pad), (0, 0)))
