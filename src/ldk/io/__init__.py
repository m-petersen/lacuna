"""
Input/Output module for loading and saving lesion data.

Provides functions for:
- Loading lesion masks from NIfTI files
- Loading BIDS datasets
- Exporting results to BIDS derivatives format
- Saving NIfTI files
"""

from .bids import BidsError, export_bids_derivatives, load_bids_dataset, save_nifti

__all__ = [
    "BidsError",
    "load_bids_dataset",
    "export_bids_derivatives",
    "save_nifti",
]
