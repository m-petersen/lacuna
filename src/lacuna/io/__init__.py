"""
Input/Output module for loading and saving lesion data.

Provides functions for:
- Loading lesion masks from NIfTI files
- Loading BIDS datasets
- Exporting results to BIDS derivatives format
- Saving NIfTI files
- Exporting analysis results to CSV/TSV/JSON
- Fetching and caching reference datasets (atlases, templates)
- Converting connectome data to Lacuna HDF5 format
- Downloading and registering connectomes (GSP1000, dTOR985)
"""

from .bids import (
    BidsError,
    export_bids_derivatives,
    load_bids_dataset,
    save_nifti,
    validate_bids_derivatives,
)
from .convert import gsp1000_to_hdf5, trk_to_tck
from .downloaders import (
    CONNECTOME_SOURCES,
    ConnectomeSource,
    FetchConfig,
    FetchProgress,
    FetchResult,
)
from .export import (
    batch_export_to_csv,
    batch_export_to_tsv,
    export_provenance_to_json,
    export_results_to_csv,
    export_results_to_json,
    export_results_to_tsv,
)
from .fetch import (
    fetch_connectome,
    fetch_dtor985,
    fetch_gsp1000,
    get_atlas,
    get_connectome_path,
    get_data_dir,
    get_fetch_status,
    get_tractogram,
    list_available_atlases,
    list_fetchable_connectomes,
)

__all__ = [
    # BIDS
    "BidsError",
    "load_bids_dataset",
    "export_bids_derivatives",
    "save_nifti",
    "validate_bids_derivatives",
    # Export
    "export_results_to_csv",
    "export_results_to_tsv",
    "export_results_to_json",
    "export_provenance_to_json",
    "batch_export_to_csv",
    "batch_export_to_tsv",
    # Fetch - atlases & tractograms
    "get_atlas",
    "get_connectome_path",
    "get_data_dir",
    "get_tractogram",
    "list_available_atlases",
    # Fetch - connectomes
    "fetch_gsp1000",
    "fetch_dtor985",
    "fetch_connectome",
    "list_fetchable_connectomes",
    "get_fetch_status",
    # Fetch types
    "CONNECTOME_SOURCES",
    "ConnectomeSource",
    "FetchConfig",
    "FetchProgress",
    "FetchResult",
    # Convert
    "gsp1000_to_hdf5",
    "trk_to_tck",
]
