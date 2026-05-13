"""
Lacuna - A scientific Python package for neuroimaging lesion analysis.

Main package providing unified API for lesion data loading, preprocessing, analysis,
and export.
"""

# Cap BLAS/OMP thread pools to 1 by default. Must happen BEFORE numpy / scipy /
# nilearn import (those libraries snapshot these env vars when their native
# libraries load). Prevents fork-after-init deadlocks when joblib spawns workers
# on many-core nodes. setdefault preserves explicit user overrides.
import os as _os

for _v in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _os.environ.setdefault(_v, "1")

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installations without setuptools-scm
    __version__ = "0.0.0+unknown"

# Core data structures and bundled data access (convenience imports).
# Imports placed after the thread-cap setdefault above, so any nilearn/numpy
# pulled in transitively sees the capped env vars when its native libs load.
from . import data  # noqa: E402
from .batch import batch_process  # noqa: E402
from .core.pipeline import Pipeline, analyze  # noqa: E402
from .core.subject_data import SubjectData  # noqa: E402

# Exports
__all__ = [
    "__version__",
    "SubjectData",
    "Pipeline",
    "analyze",
    "data",
    "batch_process",
]
