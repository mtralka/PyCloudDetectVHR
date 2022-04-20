from datetime import datetime
from enum import IntEnum
from itertools import product as iter_product
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from tensorflow.keras.models import load_model

from pycvhr.classes import (
    CombinationMethod,
    Optimizers,
    SupportedImageTypes,
    SupportedPlatforms,
    WV2Bands,
    WV3Bands,
)
from pycvhr.core.losses import (
    accuracy,
    dice_coef,
    dice_loss,
    sensitivity,
    specificity,
    tversky,
)
from pycvhr.raster_utils import (
    create_outfile_dataset,
    extract_window_from_array,
    get_raster_metadata,
    number_of_bands,
    open_raster_as_array,
    pad_array,
    reconcile_window_to_array,
    write_array_to_ds,
)
from pycvhr.utils import validate_enum, validate_path, yield_batch

try:
    import gdal
    import gdalconst
except ImportError:
    from osgeo import gdal, gdalconst


class VHRCloudDetector:
    """Very High Resolution (VHR) Cloud Detector using UNet

    Attributes
    ----------
    `input` : Union[Path, str]
        Path to input file or dirctory of input files
    `model_path`: Union[Path, str]
        Path to pretrained UNet model
    `input_type` : Union[`SupportedImageTypes`, str]
        Filter files not of type `input_type` found in `input` if `input` is a directory, default `TIF`
    `optimizer`: Union[`Optimizers`, str]
        Desired optimizer to use with model, default `Optimizers.ADADELTA`
    `platform` : Union[`SuppportedPlatforms`, str]
        Platform {WV2 or WV3} of `input`, default `SupportedPlatforms.AUTO` autodetects platform type
    `output_dir` : Optional[Path, str]]
        Path of output predictions. If not given and `input` is a file - saves to `input` parent directory. If not given
        and `input` is a directory - saves to `input` directory.
    `save_prefix` : str
        Prefix of saved output mask, default `pred_`
    normalize: bool
        If True, normalizes input arrays before prediction, default True
    `batch_size`: int
        Number of input selections to que together before prediction, default 200
    `threshold` : float
        Binary cloud mask detection threshold, default .50
    `stride` : float
        Length of window stride for model batching. Higher provides model prediction overlap at cost of compue.
        Default 256
    `combination_method` : Union[`CombinationMethod`, str]
        Method to use to combine model batch results, default `CombinationMethod.MAX`
    `window` : Tuple[int, int]
        Size of moving window. Must be the same size as used to train model. Default (256,256)
    `auto_run` : bool
        If True, runs `cls.run()` upon model initialization, default True
    `auto_save` : bool
        If True, runs `cls.save()` after each input file. Always `True` when multiple files are processed
        Default, True

    Methods
    -------
    `run()`
        Run cloud detection routine
    `save(array: Optional[np.ndarray], name: Optional[str], path: Optional[Union[Path, str]], threshold: Optional[float] )`
        Save cloud mask

    Example
    -------
    >>>
    """

    def __init__(
        self,
        input: Union[Path, str],
        model_path: Union[Path, str],
        recursive_input: bool = False,
        input_type: Union[SupportedImageTypes, str] = SupportedImageTypes.TIF,
        optimizer: Union[str, Optimizers] = Optimizers.ADADELTA,
        platform: Union[str, SupportedPlatforms] = SupportedPlatforms.AUTO,
        output_dir: Optional[Union[Path, str]] = None,
        save_prefix: str = "pred_",
        normalize: bool = True,
        batch_size: int = 200,
        threshold: float = 0.5,
        stride: int = 256,
        combination_method: Union[str, CombinationMethod] = CombinationMethod.MAX,
        window: Tuple[int, int] = (256, 256),
        auto_run: bool = True,
        auto_save: bool = False,
    ):

        self.input: Path = validate_path(input, check_exists=True, check_is_dir=False)
        self.recursive_input: bool = recursive_input

        self.model_path: Path = validate_path(
            model_path, check_exists=True, check_is_file=True
        )

        self.output_dir: Path
        if output_dir is not None:
            self.output_dir = validate_path(
                output_dir, check_exists=True, check_is_dir=True
            )
        elif self.input.is_file():
            self.output_dir = self.input.parent
        elif self.input.is_dir():
            self.output_dir = self.input
        else:
            raise ValueError("`output_dir` must not be `None`")

        self.input_type: SupportedImageTypes = validate_enum(
            SupportedImageTypes, input_type
        )
        self.platform: SupportedPlatforms = validate_enum(SupportedPlatforms, platform)
        self.optimizer: Optimizers = validate_enum(Optimizers, optimizer)
        self.combination_method: CombinationMethod = validate_enum(
            CombinationMethod, combination_method
        )

        self._batch_size: int = batch_size - 1

        self.normalize: bool = normalize
        self.threshold: float = threshold

        self.window: Tuple(int, int) = window
        self.stride: int = stride

        self.save_prefix = save_prefix

        self.custom_objects: Dict[str, Any] = {
            "tversky": tversky,
            "dice_coef": dice_coef,
            "dice_loss": dice_loss,
            "accuracy": accuracy,
            "specificity": specificity,
            "sensitivity": sensitivity,
        }

        self.model: Optional[Any] = None

        self.auto_save: bool = auto_save

        if auto_run:
            self.run()

    @property
    def batch_size(self) -> int:
        return self._batch_size + 1

    def run(self) -> None:
        """Run cloud detection routine"""

        ##
        # Find input file(s)
        ##
        input_files: List[Path]
        if self.input.is_dir():
            input_files = self._find_input_files()
        elif self.input.is_file():
            input_files = [self.input]
        else:
            raise FileNotFoundError(f"{self.input} is not a directory or file")

        # results must be saved if processing multiple files
        if len(input_files) > 1:
            self.auto_save = True

        ##
        # Load & prepare trained model
        ##
        self._prepare_model()

        ##
        # Predict for each found input file
        ##
        for input_file in input_files:

            target_bands: IntEnum
            if self.platform is SupportedPlatforms.AUTO:
                num_bands: int = number_of_bands(input_file)
                if num_bands > 8:
                    target_bands = WV3Bands
                else:
                    target_bands = WV2Bands
            else:
                target_bands = self.platform.value

            rgb_array: np.ndarray = open_raster_as_array(
                input_file, list(map(int, target_bands))
            )

            self.input_file_metadata: dict = get_raster_metadata(input_file)

            self.mask = self._batch_predict_array(rgb_array)

            ##
            # Save predicted mask
            ##
            if self.auto_save:
                self.save(array=self.mask, name=f"{self.save_prefix}{input_file.name}")

    def save(
        self,
        array: Optional[np.ndarray] = None,
        name: Optional[str] = None,
        path: Optional[Union[Path, str]] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """Save binary cloud mask output as a GeoTiff.

        If `name` or `path` are not given: saves to directory `self.output_dir`,
        with the last processed input file name or the current timestamp.

        Paramters
        ----------
        `array` : Optional[np.ndarray]
            Override cloud mask array. If not given, uses the last generated cloud mask - `self.mask`
        `name`: Optional[str]
            Override out file name. If not given, saves to `cls.output_dir` / $timestamp
        `path` : Optional[Union[Path, str]]
            Overides `name` to provide full outfile save path.
        `threshold` : Optional[float]
            Overrides given `self.threshold` prediction level to generate mask
        """

        array: np.ndarray = array if array is not None else self.mask
        threshold: float = threshold if threshold is not None else self.threshold

        outfile_path: Path
        if path:
            outfile_path = path
        elif name:
            outfile_path = self.output_dir / name
        else:
            name: str = self.input_file_metadata.get(
                "file_name", str(datetime.now().timestamp())
            )
            outfile_path = self.output_dir / f"{self.save_prefix}{name}"

        cloud_mask: np.ndarray = np.where(array > threshold, 1, 0)

        ds = create_outfile_dataset(
            file_path=str(outfile_path),
            x_size=self.input_file_metadata["x_size"],
            y_size=self.input_file_metadata["y_size"],
            wkt_projection=self.input_file_metadata["wkt_projection"],
            geo_transform=self.input_file_metadata["geo_transform"],
            number_bands=1,
            data_type=gdalconst.GDT_Byte,
        )

        ds = write_array_to_ds(ds, cloud_mask)
        ds = None

    def _find_input_files(self) -> List[Path]:
        """Searches input directory (recursive if `self.recursive_input`) to find
        files with type `self.input_type`
        """

        search_glob: str = "**/*" if self.recursive_input else "*"
        file_generator: Optional[Generator[Path]] = self.input.glob(
            f"{search_glob}.{self.input_type.value}"
        )

        if file_generator is None:
            raise FileNotFoundError(
                f"No {self.input_type.value} files found at {self.input}"
            )

        return list(file_generator)

    def _prepare_model(self) -> None:
        """Loads `self.model_path` UNet model"""

        self.model = load_model(
            self.model_path, custom_objects=self.custom_objects, compile=False
        )
        self.model.compile(
            optimizer=self.optimizer.value,
            loss=tversky,
            metrics=[dice_coef, dice_loss, accuracy, specificity, sensitivity],
        )

    def _batch_predict_array(self, array: np.ndarray) -> np.ndarray:
        """Read and predict `array` using a sliding window of stride `self.stride`.
        Batches window predictions in size `self.batch_size` to reduce compute requirements
        """

        number_rows: int = array.shape[0]
        number_cols: int = array.shape[1]

        offsets: List[tuple[int, int]] = list(
            iter_product(
                range(0, number_cols, self.stride),
                range(0, number_rows, self.stride),
            )
        )

        mask: np.ndarray = np.empty((number_rows, number_cols))

        for offsets_batch in yield_batch(offsets, self._batch_size):

            batched_arrays: List[np.ndarray] = []
            for (col_offset, row_offset) in offsets_batch:

                window_array: np.ndarray = extract_window_from_array(
                    array=array,
                    col_offset=col_offset,
                    row_offset=row_offset,
                    window_size=self.window,
                )

                if self.normalize:
                    window_array = (
                        window_array - np.mean(window_array, axis=(0, 1))
                    ) / (np.std(window_array, axis=(0, 1)) + 1e-8)

                window_array = pad_array(
                    array=window_array, target_shape=self.window[0:2]
                )

                batched_arrays.append(window_array)

            prediction = self.model.predict(np.stack(batched_arrays, axis=0))

            for idx, (col_offset, row_offset) in enumerate(offsets_batch):
                prediction_array: np.ndarray = np.squeeze(prediction[idx], axis=-1)
                mask = reconcile_window_to_array(
                    windowed_array=prediction_array,
                    master_array=mask,
                    col_offset=col_offset,
                    row_offset=row_offset,
                    combination_method=self.combination_method,
                )

        return mask
