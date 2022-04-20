# Python Cloud Detection for VHR (pycvhr)

 🚧 **under construction** 🚧 

Very High Resolution (WV2/3) cloud detection using UNet

Currently supports WV2 and WV3

## Usage

Intelligent defaults
```python

from pycvhr import VHRCloudDetector

input_path: Path | str = "path/to/input/file/or/directory"
model_path: Path | str = "path/to/model.h5"

detector = VHRCloudDetector(input=input_path, model_path=model_path)
detector.save(name="cloud_mask.tif")
```

Using the `MIN` combination method, explicitly set WV2 platform, look for only .tif input files
```python

from pycvhr import VHRCloudDetector
from pycvhr import CombinationMethod, SupportedPlatforms, SupportedImageTypes

input_path: Path | str = "path/to/input/file/or/directory"
model_path: Path | str = "path/to/model.h5"

detector = VHRCloudDetector(
    input=input_path, model_path=model_path, 
    combination_method=CombinationMethod.MIN, # or str - 'min'
    platform=SupportedPlatforms.WV2, # or str - 'WV2'
    input_type=SupportedImageTypes.TIF # or str - 'tif'
  )
detector.save(name="cloud_mask.tif")
```

## Docstring

### VHRCloudDetector
```
Very High Resolution (VHR) Cloud Detector using UNet

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
```

## Dependencies

- Python 3.8 >=
- Tensorflow
- **GDAL** - not listed in `pyproject.toml`
