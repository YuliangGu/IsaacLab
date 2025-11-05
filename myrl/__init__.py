"""Expose reorganized package structure for BYOL-enhanced RL components."""

from . import models
from . import utils
from . import algorithms
from . import runners
from . import config

# Backwards-compatible aliases (legacy dotted paths still resolve)
modules = models
utils_core = utils.core
ppo_BYOL = algorithms.ppo_byol
runner_BYOL = runners.byol
rsl_cfg = config.rsl_cfg

__all__ = [
    "models",
    "utils",
    "algorithms",
    "runners",
    "config",
    "modules",
    "utils_core",
    "ppo_BYOL",
    "runner_BYOL",
    "rsl_cfg",
]
