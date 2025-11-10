"""
Lacuna - A scientific Python package for neuroimaging lesion analysis.

Main package providing unified API for lesion data loading, preprocessing, analysis,
and export.
"""

__version__ = "0.1.0"

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
