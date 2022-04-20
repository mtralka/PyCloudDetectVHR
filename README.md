# VHR-Cloud-Detection-UNet

 🚧 **under construction** 🚧 

Very High Resolution (WV2/3) cloud detection using UNet

## Usage

```python

from pycvhr import VHRCloudDetector

input_path: Path | str = "path/to/input/file/or/directory"
model_path: Path | str = "path/to/model.h5"

detector = VHRCloudDetector(input=input_path, model_path=model_path)
detector.save(name="cloud_mask.tif")
```
