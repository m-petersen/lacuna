"""
Input/Output module for loading and saving lesion data.

Provides functions for:
- Loading lesion masks from NIfTI files
- Loading BIDS datasets
- Exporting results to BIDS derivatives format
- Saving NIfTI files
- Exporting analysis results to CSV/TSV/JSON
- Fetching and caching reference datasets (atlases, templates)
- Converting connectome data to LDK format
"""

from .bids import (
    BidsError,
    export_bids_derivatives,
    load_bids_dataset,
    save_nifti,
    validate_bids_derivatives,
)
from .convert import gsp1000_to_ldk, trk_to_tck
from .export import (
    batch_export_to_csv,
    batch_export_to_tsv,
    export_provenance_to_json,
    export_results_to_csv,
    export_results_to_json,
    export_results_to_tsv,
)
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
    "validate_bids_derivatives",
    "export_results_to_csv",
    "export_results_to_tsv",
    "export_results_to_json",
    "export_provenance_to_json",
    "batch_export_to_csv",
    "batch_export_to_tsv",
    "get_atlas",
    "get_connectome_path",
    "get_data_dir",
    "get_tractogram",
    "list_available_atlases",
    "gsp1000_to_ldk",
    "trk_to_tck",
]
