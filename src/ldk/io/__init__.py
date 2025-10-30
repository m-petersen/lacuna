"""
Input/Output module for loading and saving lesion data.

Provides functions for:
- Loading lesion masks from NIfTI files
- Loading BIDS datasets
- Exporting results to BIDS derivatives format
- Saving NIfTI files
- Fetching and caching reference datasets (atlases, templates)
- Converting connectome data to LDK format
"""

from .bids import BidsError, export_bids_derivatives, load_bids_dataset, save_nifti
from .convert import gsp1000_to_ldk, trk_to_tck
from .fetch import (
    get_atlas,
    get_connectome_path,
    get_data_dir,
    get_tractogram,
    list_available_atlases,
)

__all__ = [
    "BidsError",
    "load_bids_dataset",
    "export_bids_derivatives",
    "save_nifti",
    "get_atlas",
    "get_connectome_path",
    "get_data_dir",
    "get_tractogram",
    "list_available_atlases",
    "gsp1000_to_ldk",
    "trk_to_tck",
]
