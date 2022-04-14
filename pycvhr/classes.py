from enum import Enum
from enum import IntEnum
from enum import auto

from pycvhr.core.optimizers import adaDelta
from pycvhr.core.optimizers import adagrad
from pycvhr.core.optimizers import adam
from pycvhr.core.optimizers import nadam


class CombinationMethod(Enum):
    MIN = auto()
    MAX = auto()
    REPLACE = auto()


class SupportedImageTypes(Enum):
    ALL = "*"
    TIF = "tif"
    PNG = "png"
    JPG = "jpg"


class Optimizers(Enum):
    ADADELTA = adaDelta
    ADAGRAD = adagrad
    ADAM = adam
    NADAM = nadam


class WV2Bands(IntEnum):
    BLUE = 1
    GREEN = 2
    RED = 3


class WV3Bands(IntEnum):
    BLUE = 2
    GREEN = 3
    RED = 5


class SupportedPlatforms(Enum):
    AUTO = auto()
    WV2 = WV2Bands
    WV3 = WV3Bands
