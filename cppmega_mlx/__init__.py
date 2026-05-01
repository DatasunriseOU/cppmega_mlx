"""MLX-native cppmega training components."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

_SUBPACKAGES = ("config", "data", "kernels", "models", "nn", "recipes", "training")

if TYPE_CHECKING:
    from . import config as config
    from . import data as data
    from . import kernels as kernels
    from . import models as models
    from . import nn as nn
    from . import recipes as recipes
    from . import training as training

__all__ = [
    "__version__",
    "config",
    "data",
    "kernels",
    "models",
    "nn",
    "recipes",
    "training",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> ModuleType:
    if name in _SUBPACKAGES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
