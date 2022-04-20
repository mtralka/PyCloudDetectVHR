from enum import Enum, IntEnum, auto

from pycvhr.core.optimizers import adaDelta, adagrad, adam, nadam


class CombinationMethod(Enum):
    """Combination method used to combine prediction arrays"""

    MIN = auto()
    MAX = auto()
    REPLACE = auto()


class SupportedImageTypes(Enum):
    """Supported input types"""

    ALL = "*"
    TIF = "tif"
    # PNG = "png"
    # JPG = "jpg"


class Optimizers(Enum):
    """UNet optimizer to use"""

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
    """Supported VHR platforms"""

    AUTO = auto()
    WV2 = WV2Bands
    WV3 = WV3Bands
