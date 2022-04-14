from curses import window
from enum import IntEnum
from itertools import product as iter_product
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from tensorflow.keras.models import load_model

from pycvhr.classes import CombinationMethod
from pycvhr.classes import Optimizers
from pycvhr.classes import SupportedImageTypes
from pycvhr.classes import SupportedPlatforms
from pycvhr.classes import WV2Bands
from pycvhr.classes import WV3Bands
from pycvhr.core.losses import accuracy
from pycvhr.core.losses import dice_coef
from pycvhr.core.losses import dice_loss
from pycvhr.core.losses import sensitivity
from pycvhr.core.losses import specificity
from pycvhr.core.losses import tversky
from pycvhr.raster_utils import extract_window_from_array
from pycvhr.raster_utils import open_raster_as_array
from pycvhr.raster_utils import pad_array
from pycvhr.utils import validate_enum
from pycvhr.utils import validate_path


try:
    import gdal
except ImportError:
    from osgeo import gdal


class VHRCloudDetector:
    """ """

    def __init__(
        self,
        input_directory: Union[Path, str],
        model_path: Union[Path, str],
        recursive_input: bool = False,
        input_type: Union[SupportedImageTypes, str] = SupportedImageTypes.ALL,
        optimizer: Union[str, Optimizers] = Optimizers.ADADELTA,
        platform: Union[str, SupportedPlatforms] = SupportedPlatforms.WV2,
        output_prefix: str = "det",
        normalize: bool = True,
        batch_size: int = 200,
        threshold: float = 0.5,
        # width: int = 256,
        # height: int = 256,
        # stride: int = 128,
        combination_method: Union[str, CombinationMethod] = CombinationMethod.MAX,
        window: Tuple[int, int, int] = (256, 256, 128),  # w, h, s
        auto_run: bool = True,
    ):

        self.input_directory: Path = validate_path(
            input_directory, check_exists=True, check_is_dir=True
        )
        self.recursive_input: str = recursive_input

        self.model_path: Path = validate_path(
            model_path, check_exists=True, check_is_file=True
        )

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

        self.window: Tuple(int, int, int) = window
        # self.window_width: int = width
        # self.height: int = height
        # self.stride: int = stride

        self.custom_objects: Dict[str, Any] = {
            "tversky": tversky,
            "dice_coef": dice_coef,
            "dice_loss": dice_loss,
            "accuracy": accuracy,
            "specificity": specificity,
            "sensitivity": sensitivity,
        }

        self.model: Optional[Any] = None
        self.mask: np.ndarray

    @property
    def batch_size(self) -> int:
        return self._batch_size + 1

    def run(self) -> None:

        ##
        # Find all input files
        ##
        self.input_files: List[Path] = self._find_input_files()

        ##
        # Load & prepare trained model
        ##
        self._prepare_model()

        ##
        # Predict for each found input file
        ##
        for files_batch in self._yield_batch(self.input_files, self._batch_size):
            for input_file in files_batch:
                print(input_file)

                self._predict_input_file(input_file)

                self._predict_input_file_to_mask(input_file)
                # saves result to self.mask

        ##
        # Save predicted mask
        ##
        ...

    def save(self, path: Union[Path, str], threshold: Optional[float]) -> None:

        threshold: float = threshold if threshold is not None else self.threshold

        # np where to threshold

        #

        ...

    def _yield_batch(self, iterator: Iterable, n: int) -> Iterable:
        for i in range(0, len(iterator), n):
            yield iterator[i : i + n]

    def _find_input_files(self) -> List[Path]:

        search_glob: str = "**/*" if self.recursive_input else "*"
        file_generator: Optional[Generator[Path]] = self.input_directory.glob(
            f"{search_glob}.{self.input_type.value}"
        )

        if file_generator is None:
            raise FileNotFoundError(
                f"No {self.input_type.value} files found at {self.input_directory}"
            )

        return list(file_generator)

    def _prepare_model(self) -> None:

        self.model = load_model(
            self.model_path, custom_objects=self.custom_objects, compile=False
        )
        self.model.compile(
            optimizer=self.optimizer.value,
            loss=tversky,
            metrics=[dice_coef, dice_loss, accuracy, specificity, sensitivity],
        )

    def _determine_array_remainder(self, target: int, maximum: int, window: int) -> int:

        # if ending number is greater than RGB array size,
        # use RGB array size
        ending_target: int = target + window

        if ending_target > maximum:
            print("Hit Max")
            return maximum
        return ending_target

    # def _add_subset_to_mask(
    #     self, prediction_array: np.ndarray, col_offset: int, row_offset: int
    # ):

    #     window_ending_col: int = self._determine_array_remainder(
    #         target=col_offset, maximum=number_cols, window=self.window[0]
    #     )
    #     window_ending_row: int = self._determine_array_remainder(
    #         target=row_offset, maximum=number_rows, window=self.window[1]
    #     )

    #     current_mask_value: np.ndarray = self.mask[
    #             row_offset : window_ending_row,
    #             col_offset : col_offset + self.window[1],
    #         ]
    #     print("cur mask", current_mask_value.shape)
    #     print("pred shape", prediction_array.shape, np.max(prediction_array))
    #     print(self.mask.shape, " mask shape")

    #     #new_mask_value: np.ndarray = prediction_array[: self.window[0], self.window[1]]
    #     new_mask_value = prediction_array.astype(int)
    #     merged_results: np.ndarray
    #     if self.combination_method.MAX:
    #         merged_results = np.maximum(current_mask_value, new_mask_value)
    #     elif self.combination_method.MIN:
    #         new_mask_value = np.where(new_mask_value == 0, 1, -1)
    #         merged_results = np.minimum(new_mask_value, current_mask_value)
    #     elif self.combination_method.REPLACE:
    #         merged_results = new_mask_value
    #     else:
    #         raise ValueError(f"{self.combination_method.name} not supported")

    #     self.mask[
    #         row_offset : row_offset + self.window[0],
    #         col_offset : col_offset + self.window[1],
    #     ] = merged_results

    def _predict_input_file(self, input: Path) -> Any:

        if self.model is None:
            ...

        target_bands: IntEnum
        if self.platform is SupportedPlatforms.AUTO:
            # TODO auto find platform
            target_bands = SupportedPlatforms.WV2
        else:
            print("setting else")
            target_bands = self.platform.value

        rgb_array: np.ndarray = open_raster_as_array(
            input, list(map(int, target_bands))
        )

        number_rows: int = rgb_array.shape[0]
        number_cols: int = rgb_array.shape[1]
        array_depth: int = rgb_array.shape[2]

        self.mask = np.empty((number_rows, number_cols))

        offsets: List[tuple[int, int]] = list(
            iter_product(
                range(0, number_cols, self.window[2]),
                range(0, number_rows, self.window[2]),
            )
        )

        for offsets_batch in self._yield_batch(offsets, self._batch_size):

            batched_arrays: List[np.ndarray] = []
            for index, (col_offset, row_offset) in enumerate(offsets_batch):

                window_array: np.ndarray = extract_window_from_array(
                    array=rgb_array,
                    col_offset=col_offset,
                    row_offset=row_offset,
                    window_size=self.window,
                )

                if self.normalize:
                    window_array = (
                        window_array - np.mean(window_array, axis=(0, 1))
                    ) / (np.std(window_array, axis=(0, 1)) + 1e-8)

                window_array: np.ndarray = pad_array(
                    array=window_array, target_shape=self.window[0:2]
                )

                batched_arrays.append(window_array)

            tm = np.stack(batched_arrays, axis=0)
            prediction = self.model.predict(tm)
            for idx, (col, row) in enumerate(offsets_batch):
                prediction_array: np.ndarray = np.squeeze(prediction[idx], axis=-1)
                self._add_subset_to_mask(prediction_array, col, row)
        ...

    # def _predict_input_file_to_mask(self, input: Path) -> Any:

    #     if self.model is None:
    #         # Must have file
    #         return

    #     target_bands: IntEnum
    #     if self.platform is SupportedPlatforms.AUTO:
    #         # TODO auto find platform
    #         target_bands = SupportedPlatforms.WV2
    #     else:
    #         print("setting else")
    #         target_bands = self.platform.value

    #     rgb_array: np.ndarray = open_raster_as_array(
    #         input, list(map(int, target_bands))
    #     )

    #     print(rgb_array.shape)
    #     number_rows: int = rgb_array.shape[0]
    #     number_cols: int = rgb_array.shape[1]
    #     array_depth: int = rgb_array.shape[2]

    #     self.mask = np.empty((number_rows, number_cols))

    #     offsets: List[tuple[int, int]] = list(iter_product(
    #         range(0, number_cols, self.window[2]), range(0, number_rows, self.window[2])
    #     ))

    #     # print(len(list(offsets)))

    #     # batch_array: np.ndarray = np.empty(
    #     #     (self.window[0], self.window[1], self.batch_size)
    #     # )

    #     batched_arrays: list[np.ndarray] = []
    #     for index, (col_offset, row_offset) in enumerate(offsets):

    #         window_ending_col: int = self._determine_array_remainder(
    #             target=col_offset, maximum=number_cols, window=self.window[0]
    #         )
    #         window_ending_row: int = self._determine_array_remainder(
    #             target=row_offset, maximum=number_rows, window=self.window[1]
    #         )

    #         rgb_selection: np.ndarray = rgb_array[
    #             row_offset:window_ending_row, col_offset:window_ending_col
    #         ]

    #         # rgb_selection = np.squeeze(rgb_selection)

    #         # rgb_selection = np.transpose(rgb_selection, axes=(1, 2, 0))

    #         if self.normalize:
    #             rgb_selection = (
    #                 rgb_selection - np.mean(rgb_selection, axis=(0, 1))
    #             ) / (np.std(rgb_selection, axis=(0, 1)) + 1e-8)

    #         # pad rgb_selection by difference to desired window size
    #         # handles cases where window is on/by an edge
    #         x_pad: int = self.window[0] - rgb_selection.shape[0]
    #         y_pad: int = self.window[1] - rgb_selection.shape[1]

    #         # ((top, bottom), (left, right), (3rd dimension))
    #         padded_array: np.ndarray = np.pad(
    #             rgb_selection, ((0, x_pad), (0, y_pad), (0, 0))
    #         )

    #         # if index == 0:
    #         #     batch_array = padded_array
    #         # else:
    #         #     print(batch_array.shape)
    #         #     print(padded_array.shape)

    #         #     # batch_array = np.stack((batch_array, array_subset), axis=0)
    #         #     batch_array = np.append(batch_array, padded_array, axis=0)
    #         #     # batch_array = batch_array.stack(array_subset, axis=0)
    #         #     print("batch after", batch_array.shape)
    #         batched_arrays.append(padded_array)

    #         if (index % self._batch_size) == 0 and (index != 0):
    #             print("predicting", len(batched_arrays))
    #             tm = np.stack(batched_arrays, axis=0)
    #             print(tm.shape, "TM shape")
    #             prediction = self.model.predict(tm)
    #             print("max", np.max(prediction))
    #             for idx, (col, row) in enumerate(
    #                 offsets[index - self._batch_size : self._batch_size]
    #             ):
    #                 print("done", idx, col, row)
    #                 # prediction_array: np.ndarray = np.squeeze(prediction[index], axis=-1)
    #                 # self._add_subset_to_mask(prediction_array, col_offset, row_offset)
    #                 prediction_array: np.ndarray = np.squeeze(prediction[idx], axis=-1)
    #                 print("shape", prediction_array.shape)
    #                 self._add_subset_to_mask(prediction_array, col, row)
    #             break

    #     # prediction = self.model.predict(batch_array)

    #     # for index, col_offset, row_offset in enumerate(offsets):
    #     #     prediction_array: np.ndarray = np.squeeze(prediction[index], axis=-1)
    #     #     self._add_subset_to_mask(prediction_array, col_offset, row_offset)

    #     return None
