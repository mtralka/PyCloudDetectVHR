from enum import Enum
from pathlib import Path
from typing import Union


def validate_path(
    path: Union[str, Path],
    check_exists: bool = False,
    check_is_file: bool = False,
    check_is_dir=False,
) -> Path:

    valid_path: Path = Path(path) if isinstance(path, str) else path

    if check_exists:
        if not valid_path.exists():
            raise FileExistsError(f"{path} must exist")

    if check_is_file:
        if not valid_path.is_file():
            raise FileNotFoundError(f"{path} must be a file")

    if check_is_dir:
        if not valid_path.is_dir():
            raise ValueError(f"{path} must be a directory")

    return valid_path


def validate_enum(enum: Enum, value: Union[str, Enum, None]) -> Enum:
    for method_type in enum.__members__.values():
        if value == method_type or value == method_type.value:
            return method_type
        elif value is None:
            return None
        elif isinstance(value, str) and value.upper().strip() == method_type.name:
            return enum[value.upper().strip()]
            break
    else:
        raise AttributeError(
            f"{value} method not recognized. Select from {','.join([item.name for item in enum.__members__.values()])}"
        )
