"""
Lacuna - A scientific Python package for neuroimaging lesion analysis.

Main package providing unified API for lesion data loading, preprocessing, analysis,
and export.
"""

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installations without setuptools-scm
    __version__ = "0.0.0+unknown"

# Core data structures
# Bundled data access (convenience imports)
from . import data
from .batch import batch_process
from .core.lesion_data import LesionData

# Exports
__all__ = [
    "__version__",
    "LesionData",
    "data",
    "batch_process",
]
