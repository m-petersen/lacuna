"""
BIDS dataset loading and derivative export functionality.

Provides functions to load mask data from BIDS-compliant datasets and
export analysis results in BIDS derivatives format.

No external BIDS validation library (pybids) is required.
"""

from __future__ import annotations

import fnmatch
import json
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..core.exceptions import LacunaError
from ..core.keys import format_bids_export_filename
from ..core.subject_data import SubjectData

if TYPE_CHECKING:
    from ..core.data_types import ConnectivityMatrix, ParcelData, VoxelMap


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class BidsError(LacunaError):
    """Raised when BIDS dataset operations fail."""

    pass


def load_bids_dataset(
    bids_root: str | Path,
    pattern: str = "*",
    suffix: str = "_mask.nii.gz",
    recursive: bool = True,
    space: str | None = None,
    resolution: float | None = None,
) -> dict[str, SubjectData]:
    """
    Load mask files from a BIDS dataset using pattern matching.

    This function finds all files matching the pattern and suffix in the BIDS
    dataset structure and loads them as SubjectData objects. No external BIDS
    validation library (pybids) is required.

    Parameters
    ----------
    bids_root : str or Path
        Path to BIDS dataset root directory (or any directory containing masks).
    pattern : str, default="*"
        Glob/fnmatch pattern to filter files. Matched against the full filename
        (without path). Examples:
        - "*" : All mask files
        - "CAS001*" : All masks for subject CAS001
        - "*ses-01*" : All session 01 masks
        - "*acuteinfarct*" : All acute infarct masks
        - "CAS001*ses-01*acuteinfarct" : Specific subject, session, and label
    suffix : str, default="_mask.nii.gz"
        File suffix to search for. Common options:
        - "_mask.nii.gz" : Standard BIDS mask suffix
        - "_mask.nii" : Uncompressed masks
        - ".nii.gz" : Any NIfTI file
    recursive : bool, default=True
        If True, search recursively in subdirectories.
    space : str or None, default=None
        Coordinate space for loaded masks. If None, attempts to detect from
        filename (_space-XXX) or sidecar JSON. If detection fails and space
        is not provided, a warning is emitted and the file is skipped.
        Supported spaces: MNI152NLin6Asym, MNI152NLin2009aAsym, MNI152NLin2009cAsym
    resolution : float or None, default=None
        Voxel resolution in mm. If None, attempts to detect from filename
        (_res-X) or sidecar JSON.

    Returns
    -------
    dict of str -> SubjectData
        Dictionary mapping filenames (without suffix) to SubjectData objects.

    Raises
    ------
    FileNotFoundError
        If bids_root doesn't exist.
    BidsError
        If no matching files are found.

    Examples
    --------
    Load all masks in a BIDS dataset:

    >>> dataset = load_bids_dataset('/data/METAVCI_PSCI_BIDS')
    >>> print(f"Loaded {len(dataset)} masks")

    Load specific subject:

    >>> dataset = load_bids_dataset(
    ...     '/data/METAVCI_PSCI_BIDS',
    ...     pattern="CAS001*"
    ... )

    Load specific session and label:

    >>> dataset = load_bids_dataset(
    ...     '/data/METAVCI_PSCI_BIDS',
    ...     pattern="CAS001*ses-01*acuteinfarct"
    ... )

    Load from a specific subject's anat folder:

    >>> dataset = load_bids_dataset(
    ...     '/data/METAVCI_PSCI_BIDS/sub-CAS001/ses-01/anat',
    ...     pattern="*WMH*"
    ... )

    Load all WMH masks across all subjects:

    >>> dataset = load_bids_dataset(
    ...     '/data/METAVCI_PSCI_BIDS',
    ...     pattern="*WMH*"
    ... )

    Load masks with explicit space (when not in filename):

    >>> dataset = load_bids_dataset(
    ...     '/data/METAVCI_PSCI_BIDS',
    ...     pattern="*CAS005*",
    ...     space="MNI152NLin6Asym",
    ...     resolution=2.0
    ... )
    """
    bids_root = Path(bids_root)

    # Check if path exists
    if not bids_root.exists():
        raise FileNotFoundError(f"Directory not found: {bids_root}")

    # Find all matching files
    if recursive:
        # Search recursively
        all_files = list(bids_root.rglob(f"*{suffix}"))
    else:
        # Search only in root
        all_files = list(bids_root.glob(f"*{suffix}"))

    # Filter by pattern - match pattern anywhere in filename
    matching_files = []
    for filepath in all_files:
        filename = filepath.name
        # Remove suffix for pattern matching
        name_without_suffix = filename
        if filename.endswith(".nii.gz"):
            name_without_suffix = filename[:-7]
        elif filename.endswith(".nii"):
            name_without_suffix = filename[:-4]

        # Match pattern (supports wildcards) - try multiple patterns
        if (
            fnmatch.fnmatch(name_without_suffix, f"*{pattern}*")
            or fnmatch.fnmatch(name_without_suffix, pattern)
            or fnmatch.fnmatch(name_without_suffix, f"{pattern}*")
            or fnmatch.fnmatch(name_without_suffix, f"*{pattern}")
        ):
            matching_files.append(filepath)

    if not matching_files:
        raise BidsError(
            f"No files matching pattern '{pattern}' with suffix '{suffix}' "
            f"found in: {bids_root}\n"
            f"Searched {'recursively' if recursive else 'non-recursively'}."
        )

    # Load each file as SubjectData
    mask_data_dict = {}

    for filepath in sorted(matching_files):
        # Create key from filename (without suffix)
        filename = filepath.name
        if filename.endswith(".nii.gz"):
            key = filename[:-7]  # Remove .nii.gz
        elif filename.endswith(".nii"):
            key = filename[:-4]  # Remove .nii
        else:
            key = filename

        # Build metadata from BIDS entities in filename
        metadata = _parse_bids_entities(filename)
        metadata["source_path"] = str(filepath)
        metadata["bids_root"] = str(bids_root)

        # Parse sidecar JSON if available
        sidecar_data = _parse_sidecar(filepath)

        # Get space: function parameter > sidecar JSON > filename entity
        file_space = (
            space  # Function parameter takes precedence
            or sidecar_data.get("Space")
            or sidecar_data.get("space")
            or metadata.get("space")
        )

        # Get resolution: function parameter > sidecar JSON > filename entity
        file_resolution = _parse_resolution(
            resolution  # Function parameter takes precedence
            or sidecar_data.get("Resolution")
            or sidecar_data.get("resolution")
            or metadata.get("resolution")
        )

        try:
            mask_data = SubjectData.from_nifti(
                mask_path=filepath,
                metadata=metadata,
                space=file_space,
                resolution=file_resolution,
            )
            mask_data_dict[key] = mask_data
        except Exception as e:
            warnings.warn(
                f"Failed to load {filepath}: {e}",
                UserWarning,
                stacklevel=2,
            )

    if not mask_data_dict:
        raise BidsError(
            f"No valid mask files could be loaded from: {bids_root}\n"
            f"Pattern: '{pattern}', Suffix: '{suffix}'"
        )

    return mask_data_dict


def _parse_bids_entities(filename: str) -> dict:
    """
    Parse BIDS entities from filename.

    Extracts subject, session, label, desc, and other BIDS key-value pairs.

    Parameters
    ----------
    filename : str
        BIDS-compliant filename

    Returns
    -------
    dict
        Extracted entities
    """
    metadata = {}

    # Remove extension
    name = filename
    for ext in [".nii.gz", ".nii", ".json"]:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break

    # Parse underscore-separated entities
    parts = name.split("_")
    for part in parts:
        if "-" in part:
            key, value = part.split("-", 1)
            # Map BIDS keys to metadata keys
            if key == "sub":
                metadata["subject_id"] = f"sub-{value}"
            elif key == "ses":
                metadata["session_id"] = f"ses-{value}"
            elif key == "label":
                metadata["label"] = value
            elif key == "desc":
                metadata["description"] = value
            elif key == "space":
                metadata["space"] = value
            elif key == "res":
                try:
                    metadata["resolution"] = float(value)
                except ValueError:
                    metadata["resolution_label"] = value
            else:
                metadata[key] = value

    return metadata


def _parse_resolution(value: str | float | int | None) -> float | None:
    """
    Parse resolution value from various formats.

    Parameters
    ----------
    value : str, float, int, or None
        Resolution value (e.g., "2mm", "2", 2, 2.0)

    Returns
    -------
    float or None
        Numeric resolution value in mm, or None if not parseable
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        import re

        # Match numeric part with optional units
        match = re.match(r"^(\d+(?:\.\d+)?)\s*(mm)?$", value.strip())
        if match:
            return float(match.group(1))

    return None


def _parse_sidecar(nifti_path: Path) -> dict:
    """
    Parse JSON sidecar file for a NIfTI image.

    Parameters
    ----------
    nifti_path : Path
        Path to NIfTI file

    Returns
    -------
    dict
        Sidecar data, or empty dict if no sidecar exists
    """
    nifti_path = Path(nifti_path)

    # Try both .nii.gz and .nii extensions
    if nifti_path.suffix == ".gz":
        sidecar_path = nifti_path.with_suffix("").with_suffix(".json")
    else:
        sidecar_path = nifti_path.with_suffix(".json")

    if sidecar_path.exists():
        try:
            with open(sidecar_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    return {}


def _extract_space_from_filename(filename: str) -> str | None:
    """
    Extract space entity from BIDS filename.

    Parameters
    ----------
    filename : str
        BIDS filename (e.g., 'sub-001_space-MNI152NLin6Asym_mask.nii.gz')

    Returns
    -------
    str or None
        Space name if found, None otherwise
    """

    match = re.search(r"space-([a-zA-Z0-9]+)", filename)
    if match:
        return match.group(1)
    return None


def export_voxelmap(
    voxelmap: VoxelMap,
    output_dir: str | Path,
    subject_id: str,
    session_id: str | None = None,
    desc: str = "map",
    space: str | None = None,
    label: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Export a VoxelMap to BIDS-compliant NIfTI with JSON sidecar.

    Parameters
    ----------
    voxelmap : VoxelMap
        VoxelMap to export
    output_dir : str or Path
        Output directory for the file
    subject_id : str
        SubjectData identifier (e.g., 'sub-001')
    session_id : str, optional
        Session identifier (e.g., 'ses-01')
    desc : str, default='map'
        BIDS-formatted key with entities and suffix (e.g.,
        'parc-Schaefer100_source-fnm_rmap_map')
    space : str, optional
        Override space from voxelmap.space
    label : str, optional
        Label entity for disambiguation (e.g., 'WMH', 'acuteinfarct', 'lesion')
    overwrite : bool, default=False
        Overwrite existing files

    Returns
    -------
    Path
        Path to the saved NIfTI file
    """
    import nibabel as nib

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build BIDS filename - desc already contains full formatted key with suffix
    space = space or voxelmap.space
    label_part = f"_label-{label}" if label else ""
    if session_id:
        base_name = f"{subject_id}_{session_id}_space-{space}{label_part}_{desc}"
    else:
        base_name = f"{subject_id}_space-{space}{label_part}_{desc}"

    nifti_path = output_dir / f"{base_name}.nii.gz"
    sidecar_path = output_dir / f"{base_name}.json"

    if nifti_path.exists() and not overwrite:
        raise FileExistsError(f"File exists: {nifti_path}. Use overwrite=True.")

    # Save NIfTI
    nib.save(voxelmap.data, nifti_path)

    # Create sidecar
    sidecar = {
        "Space": space,
        "Resolution": voxelmap.resolution,
        "Description": voxelmap.name,
    }
    if voxelmap.metadata:
        sidecar["Metadata"] = voxelmap.metadata

    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, cls=NumpyJSONEncoder)

    return nifti_path


def export_parcel_data(
    parcel_data: ParcelData,
    output_dir: str | Path,
    subject_id: str,
    session_id: str | None = None,
    desc: str = "parcels",
    label: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Export ParcelData to BIDS-compliant TSV with JSON sidecar.

    Parameters
    ----------
    parcel_data : ParcelData
        ParcelData to export
    output_dir : str or Path
        Output directory for the file
    subject_id : str
        SubjectData identifier (e.g., 'sub-001')
    session_id : str, optional
        Session identifier (e.g., 'ses-01')
    desc : str, default='parcels'
        BIDS-formatted key with entities and suffix (e.g.,
        'parc-Schaefer100_source-maskimg_maskimg_values')
    label : str, optional
        Label entity for disambiguation (e.g., 'WMH', 'acuteinfarct', 'lesion')
    overwrite : bool, default=False
        Overwrite existing files

    Returns
    -------
    Path
        Path to the saved TSV file
    """
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build BIDS filename - desc already contains full formatted key with suffix
    label_part = f"_label-{label}" if label else ""
    if session_id:
        base_name = f"{subject_id}_{session_id}{label_part}_{desc}"
    else:
        base_name = f"{subject_id}{label_part}_{desc}"

    tsv_path = output_dir / f"{base_name}.tsv"
    sidecar_path = output_dir / f"{base_name}.json"

    if tsv_path.exists() and not overwrite:
        raise FileExistsError(f"File exists: {tsv_path}. Use overwrite=True.")

    # Create DataFrame
    df = pd.DataFrame([{"region": k, "value": v} for k, v in parcel_data.data.items()])

    # Save TSV
    df.to_csv(tsv_path, sep="\t", index=False)

    # Create sidecar
    sidecar = {
        "Description": parcel_data.name,
        "Parcellation": parcel_data.parcel_names,
    }
    if hasattr(parcel_data, "aggregation_method") and parcel_data.aggregation_method:
        sidecar["AggregationMethod"] = parcel_data.aggregation_method
    if parcel_data.metadata:
        sidecar["Metadata"] = parcel_data.metadata

    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, cls=NumpyJSONEncoder)

    return tsv_path


def export_connectivity_matrix(
    matrix: ConnectivityMatrix,
    output_dir: str | Path,
    subject_id: str,
    session_id: str | None = None,
    desc: str = "connectivity",
    label: str | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Export ConnectivityMatrix to BIDS-compliant TSV with JSON sidecar.

    Parameters
    ----------
    matrix : ConnectivityMatrix
        ConnectivityMatrix to export
    output_dir : str or Path
        Output directory for the file
    subject_id : str
        SubjectData identifier (e.g., 'sub-001')
    session_id : str, optional
        Session identifier (e.g., 'ses-01')
    desc : str, default='connectivity'
        BIDS-formatted key with entities and suffix (e.g.,
        'parc-Schaefer100_source-snm_connectome_connmatrix')
    label : str, optional
        Label entity for disambiguation (e.g., 'WMH', 'acuteinfarct', 'lesion')
    overwrite : bool, default=False
        Overwrite existing files

    Returns
    -------
    Path
        Path to the saved TSV file
    """
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build BIDS filename - desc already contains full formatted key with suffix
    label_part = f"_label-{label}" if label else ""
    if session_id:
        base_name = f"{subject_id}_{session_id}{label_part}_{desc}"
    else:
        base_name = f"{subject_id}{label_part}_{desc}"

    tsv_path = output_dir / f"{base_name}.tsv"
    sidecar_path = output_dir / f"{base_name}.json"

    if tsv_path.exists() and not overwrite:
        raise FileExistsError(f"File exists: {tsv_path}. Use overwrite=True.")

    # Create DataFrame with region labels
    df = pd.DataFrame(
        matrix.matrix,
        index=matrix.region_labels,
        columns=matrix.region_labels,
    )

    # Save TSV
    df.to_csv(tsv_path, sep="\t")

    # Create sidecar
    sidecar = {
        "Description": matrix.name,
        "MatrixType": matrix.matrix_type,
        "RegionLabels": matrix.region_labels,
        "Shape": list(matrix.matrix.shape),
    }
    if matrix.metadata:
        sidecar["Metadata"] = matrix.metadata

    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, cls=NumpyJSONEncoder)

    return tsv_path


def export_bids_derivatives_batch(
    results: list,
    output_dir: str | Path,
    export_lesion_mask: bool = True,
    export_voxelmaps: bool = True,
    export_parcel_data: bool = True,
    export_connectivity: bool = True,
    export_scalars: bool = True,
    export_provenance: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Export multiple SubjectData results to BIDS derivatives format.

    Parameters
    ----------
    results : list[SubjectData]
        List of processed SubjectData objects
    output_dir : str or Path
        Root directory for derivatives
    export_lesion_mask : bool, default=True
        Save original lesion masks as NIfTI files
    export_voxelmaps : bool, default=True
        Save VoxelMap results (e.g., correlation maps) as NIfTI files
    export_parcel_data : bool, default=True
        Save ParcelData results as TSV files
    export_connectivity : bool, default=True
        Save ConnectivityMatrix results as TSV files
    export_scalars : bool, default=True
        Save ScalarMetric and other scalar results as JSON files
    export_provenance : bool, default=True
        Save processing provenance as JSON
    overwrite : bool, default=False
        Overwrite existing files

    Returns
    -------
    Path
        Path to the derivatives directory
    """
    output_dir = Path(output_dir)

    # Create dataset_description.json once
    desc_file = output_dir / "dataset_description.json"
    if not desc_file.exists():
        desc_file.parent.mkdir(parents=True, exist_ok=True)
        from .. import __version__

        dataset_description = {
            "Name": "Lacuna Derivatives",
            "BIDSVersion": "1.6.0",
            "GeneratedBy": [
                {
                    "Name": "lesion-decoding-toolkit",
                    "Version": __version__,
                    "Description": "Lesion network mapping and analysis",
                }
            ],
        }
        with open(desc_file, "w") as f:
            json.dump(dataset_description, f, indent=2)

    # Export each subject
    for subject_data in results:
        export_bids_derivatives(
            subject_data=subject_data,
            output_dir=output_dir,
            export_lesion_mask=export_lesion_mask,
            export_voxelmaps=export_voxelmaps,
            export_parcel_data=export_parcel_data,
            export_connectivity=export_connectivity,
            export_scalars=export_scalars,
            export_provenance=export_provenance,
            overwrite=overwrite,
        )

    return output_dir


def export_bids_derivatives(
    subject_data: SubjectData,
    output_dir: str | Path,
    export_lesion_mask: bool = True,
    export_voxelmaps: bool = True,
    export_parcel_data: bool = True,
    export_connectivity: bool = True,
    export_scalars: bool = True,
    export_provenance: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Export SubjectData and all its analysis results to BIDS derivatives format.

    Exports the full spectrum of results stored in a SubjectData object:
    - Lesion mask as NIfTI
    - VoxelMaps (correlation maps, disconnection maps, etc.) as NIfTI
    - ParcelData (regional values) as TSV
    - ConnectivityMatrix as TSV
    - ScalarMetric and other scalars as JSON
    - Processing provenance as JSON

    Parameters
    ----------
    subject_data : SubjectData
        Processed lesion data with analysis results.
    output_dir : str or Path
        Root directory for derivatives (e.g., 'derivatives/lacuna-v0.1.0').
    export_lesion_mask : bool, default=True
        Save the original lesion mask as NIfTI file.
    export_voxelmaps : bool, default=True
        Save VoxelMap results (e.g., correlation maps, z-maps) as NIfTI files.
    export_parcel_data : bool, default=True
        Save ParcelData results (regional aggregations) as TSV files.
    export_connectivity : bool, default=True
        Save ConnectivityMatrix results as TSV files.
    export_scalars : bool, default=True
        Save ScalarMetric and other scalar results as JSON files.
    export_provenance : bool, default=True
        Save processing provenance as JSON.
    overwrite : bool, default=False
        Overwrite existing files.

    Returns
    -------
    Path
        Path to created subject derivatives directory.

    Raises
    ------
    FileExistsError
        If output files exist and overwrite=False.
    ValueError
        If subject_data has no subject_id in metadata.

    Examples
    --------
    >>> # Export all results
    >>> output_path = export_bids_derivatives(
    ...     subject_data,
    ...     'derivatives/lacuna-v0.1.0'
    ... )
    >>> print(f"Derivatives saved to: {output_path}")
    >>>
    >>> # Export only VoxelMaps (NIfTI files)
    >>> export_bids_derivatives(
    ...     subject_data,
    ...     'derivatives/lacuna-v0.1.0',
    ...     export_lesion_mask=False,
    ...     export_parcel_data=False,
    ...     export_connectivity=False,
    ...     export_scalars=False,
    ...     export_provenance=False
    ... )
    """
    import nibabel as nib

    from ..core.data_types import (
        ConnectivityMatrix,
        ScalarMetric,
        VoxelMap,
    )
    from ..core.data_types import (
        ParcelData as ParcelDataType,
    )

    output_dir = Path(output_dir)

    # Validate metadata
    if "subject_id" not in subject_data.metadata:
        raise ValueError("SubjectData metadata must contain 'subject_id' for BIDS export")

    subject_id = subject_data.metadata["subject_id"]
    session_id = subject_data.metadata.get("session_id")

    # Determine base filename
    if session_id:
        base_name = f"{subject_id}_{session_id}"
    else:
        base_name = subject_id

    # Create subject directory
    subject_dir = output_dir / subject_id
    if session_id:
        subject_dir = subject_dir / session_id

    # Create dataset_description.json if it doesn't exist
    desc_file = output_dir / "dataset_description.json"
    if not desc_file.exists():
        desc_file.parent.mkdir(parents=True, exist_ok=True)
        from .. import __version__

        dataset_description = {
            "Name": "Lacuna Derivatives",
            "BIDSVersion": "1.6.0",
            "GeneratedBy": [
                {
                    "Name": "lacuna",
                    "Version": __version__,
                    "Description": "Lesion network mapping and analysis toolkit",
                }
            ],
        }
        with open(desc_file, "w") as f:
            json.dump(dataset_description, f, indent=2)

    # Create anat/ directory for all derivatives (BIDS compliant)
    # All lesion-derived outputs go in anat/ per BIDS derivatives spec
    anat_dir = subject_dir / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)

    # Save lesion mask - use label entity per BIDS spec
    # Preserve original label from metadata if available (e.g., WMH, acuteinfarct, lacune)
    label = subject_data.metadata.get("label", "lesion")
    if export_lesion_mask:
        coord_space = subject_data.get_coordinate_space()
        lesion_filename = f"{base_name}_space-{coord_space}_label-{label}_mask.nii.gz"
        lesion_path = anat_dir / lesion_filename

        if lesion_path.exists() and not overwrite:
            raise FileExistsError(
                f"Lesion mask already exists: {lesion_path}. Use overwrite=True to replace."
            )

        nib.save(subject_data.mask_img, lesion_path)

    # Save analysis results
    if subject_data.results:
        for _namespace, results_data in subject_data.results.items():
            if not isinstance(results_data, dict):
                continue

            for key, value in results_data.items():
                # VoxelMap -> NIfTI (goes to anat/ for spatial data)
                if isinstance(value, VoxelMap) and export_voxelmaps:
                    bids_key = format_bids_export_filename(key, "map")
                    export_voxelmap(
                        value,
                        anat_dir,
                        subject_id=subject_id,
                        session_id=session_id,
                        desc=bids_key,
                        label=label,
                        overwrite=overwrite,
                    )

                # ParcelData -> TSV (goes to anat/ for BIDS compliance)
                elif isinstance(value, ParcelDataType) and export_parcel_data:
                    bids_key = format_bids_export_filename(key, "values")
                    _export_parcel_data(
                        value,
                        anat_dir,
                        subject_id=subject_id,
                        session_id=session_id,
                        desc=bids_key,
                        label=label,
                        overwrite=overwrite,
                    )

                # ConnectivityMatrix -> TSV (goes to anat/ for BIDS compliance)
                elif isinstance(value, ConnectivityMatrix) and export_connectivity:
                    bids_key = format_bids_export_filename(key, "connmatrix")
                    export_connectivity_matrix(
                        value,
                        anat_dir,
                        subject_id=subject_id,
                        session_id=session_id,
                        desc=bids_key,
                        label=label,
                        overwrite=overwrite,
                    )

                # ScalarMetric or other serializable -> JSON (goes to anat/ for BIDS compliance)
                elif export_scalars:
                    if isinstance(value, ScalarMetric):
                        data_to_save = value.get_data()
                    else:
                        data_to_save = value

                    try:
                        bids_key = format_bids_export_filename(key, "metrics")
                        label_part = f"_label-{label}" if label else ""
                        results_filename = f"{base_name}{label_part}_{bids_key}.json"
                        results_path = anat_dir / results_filename

                        if results_path.exists() and not overwrite:
                            continue

                        with open(results_path, "w") as f:
                            json.dump(data_to_save, f, indent=2, default=str)
                    except (TypeError, ValueError):
                        # Skip non-serializable results
                        pass

    # Save provenance (goes to anat/ for BIDS compliance)
    if export_provenance and subject_data.provenance:
        prov_filename = f"{base_name}_desc-provenance.json"
        prov_path = anat_dir / prov_filename

        if prov_path.exists() and not overwrite:
            raise FileExistsError(
                f"Provenance file already exists: {prov_path}. Use overwrite=True to replace."
            )

        # Convert provenance to serializable format
        prov_data = []
        for step in subject_data.provenance:
            if hasattr(step, "to_dict"):
                prov_data.append(step.to_dict())
            elif isinstance(step, dict):
                prov_data.append(step)
            else:
                prov_data.append(str(step))

        with open(prov_path, "w") as f:
            json.dump(prov_data, f, indent=2, default=str)

    return subject_dir


# Alias to avoid name collision with ParcelData type
_export_parcel_data = export_parcel_data


def save_nifti(
    mask_data: SubjectData, output_path: str | Path, save_anatomical: bool = False
) -> None:
    """
    Save lesion mask to NIfTI file.

    Parameters
    ----------
    mask_data : SubjectData
        Lesion data to save.
    output_path : str or Path
        Path for output NIfTI file (e.g., 'lesion.nii.gz').
    save_anatomical : bool, default=False
        Also save anatomical image (if present) to adjacent file.

    Raises
    ------
    ValueError
        If output_path doesn't have .nii or .nii.gz extension.

    Examples
    --------
    >>> save_nifti(mask_data, 'output/lesion.nii.gz')
    >>> save_nifti(mask_data, 'output/lesion.nii.gz', save_anatomical=True)
    """
    import nibabel as nib

    output_path = Path(output_path)

    # Validate extension
    if output_path.suffix not in [".nii", ".gz"]:
        raise ValueError(
            f"Output path must have .nii or .nii.gz extension, got: {output_path.suffix}"
        )

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save lesion mask
    nib.save(mask_data.mask_img, output_path)


def validate_bids_derivatives(
    derivatives_dir: str | Path,
    raise_on_error: bool = True,
) -> dict[str, list[str]]:
    """
    Validate BIDS derivatives directory structure.

    Checks that a derivatives directory follows BIDS specifications:
    - Has dataset_description.json
    - SubjectData directories follow naming conventions
    - Files follow BIDS naming patterns
    - Required metadata is present

    Parameters
    ----------
    derivatives_dir : str or Path
        Path to derivatives directory (e.g., 'derivatives/lacuna-v0.1.0')
    raise_on_error : bool, default=True
        If True, raises BidsError on validation failure.
        If False, returns errors as list.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with validation results:
        - 'errors': List of error messages (MUST fix)
        - 'warnings': List of warning messages (SHOULD fix)
        Empty lists indicate passing validation.

    Raises
    ------
    BidsError
        If validation fails and raise_on_error=True
    FileNotFoundError
        If derivatives_dir doesn't exist

    Examples
    --------
    >>> from lacuna.io import validate_bids_derivatives
    >>>
    >>> # Validate after export
    >>> validate_bids_derivatives('derivatives/lacuna-v0.1.0')
    {'errors': [], 'warnings': []}
    >>>
    >>> # Check without raising exceptions
    >>> result = validate_bids_derivatives('derivatives/lacuna-v0.1.0', raise_on_error=False)
    >>> if result['errors']:
    ...     print(f"Found {len(result['errors'])} errors")

    Notes
    -----
    Validation checks:
    - dataset_description.json exists and is valid JSON
    - Contains required fields: Name, BIDSVersion, GeneratedBy
    - SubjectData directories match pattern: sub-<label>[/ses-<label>]
    - File naming follows BIDS conventions
    - No unexpected files in root directory
    """
    derivatives_dir = Path(derivatives_dir)
    errors = []
    warnings_list = []

    # Check directory exists
    if not derivatives_dir.exists():
        raise FileNotFoundError(f"Derivatives directory not found: {derivatives_dir}")

    if not derivatives_dir.is_dir():
        errors.append(f"Path is not a directory: {derivatives_dir}")
        if raise_on_error:
            raise BidsError("Validation failed:\n" + "\n".join(errors))
        return {"errors": errors, "warnings": warnings_list}

    # Check for dataset_description.json
    desc_file = derivatives_dir / "dataset_description.json"
    if not desc_file.exists():
        errors.append(
            "Missing required file: dataset_description.json\n"
            "This file is required for BIDS derivatives."
        )
    else:
        # Validate dataset_description.json content
        try:
            with open(desc_file) as f:
                desc_data = json.load(f)

            # Check required fields
            required_fields = ["Name", "BIDSVersion", "GeneratedBy"]
            for field in required_fields:
                if field not in desc_data:
                    errors.append(f"dataset_description.json missing required field: '{field}'")

            # Check GeneratedBy structure if present
            if "GeneratedBy" in desc_data:
                if not isinstance(desc_data["GeneratedBy"], list):
                    errors.append("dataset_description.json: 'GeneratedBy' must be a list")
                elif desc_data["GeneratedBy"]:
                    # Check first entry has required fields
                    gen_by = desc_data["GeneratedBy"][0]
                    if not isinstance(gen_by, dict):
                        errors.append(
                            "dataset_description.json: GeneratedBy entries must be objects"
                        )
                    elif "Name" not in gen_by:
                        warnings_list.append(
                            "dataset_description.json: GeneratedBy entry should have 'Name' field"
                        )

        except json.JSONDecodeError as e:
            errors.append(f"dataset_description.json is not valid JSON: {e}")
        except Exception as e:
            errors.append(f"Error reading dataset_description.json: {e}")

    # Check subject directories
    subject_dirs = [d for d in derivatives_dir.iterdir() if d.is_dir()]

    if not subject_dirs:
        warnings_list.append("No subject directories found in derivatives")
    else:
        for subj_dir in subject_dirs:
            subj_name = subj_dir.name

            # Check subject directory naming
            if not subj_name.startswith("sub-"):
                # Skip non-subject directories (like sourcedata, code)
                if subj_name not in ["sourcedata", "code", ".git"]:
                    warnings_list.append(
                        f"Directory '{subj_name}' doesn't follow BIDS naming "
                        f"(should start with 'sub-')"
                    )
                continue

            # Check for expected subdirectories (all outputs go to anat/ per BIDS spec)
            expected_subdirs = ["anat", "func", "dwi"]
            has_subdirs = any((subj_dir / sd).exists() for sd in expected_subdirs)

            if not has_subdirs:
                warnings_list.append(
                    f"SubjectData '{subj_name}' has no standard BIDS subdirectories "
                    f"(anat, func, dwi)"
                )

            # Check for session subdirectories
            session_dirs = [
                d for d in subj_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")
            ]
            for ses_dir in session_dirs:
                ses_name = ses_dir.name
                # Validate session naming
                if not ses_name.startswith("ses-"):
                    warnings_list.append(
                        f"Session directory '{ses_name}' in {subj_name} doesn't follow "
                        f"BIDS naming (should start with 'ses-')"
                    )

    # Check for unexpected files in root
    root_files = [f for f in derivatives_dir.iterdir() if f.is_file()]
    expected_root_files = [
        "dataset_description.json",
        "README",
        "README.md",
        "CHANGES",
        "LICENSE",
        ".bidsignore",
    ]

    for root_file in root_files:
        if root_file.name not in expected_root_files:
            warnings_list.append(
                f"Unexpected file in derivatives root: {root_file.name}\n"
                f"Consider moving to a subject directory or removing"
            )

    # Raise error if requested and errors found
    if errors and raise_on_error:
        error_msg = "BIDS derivatives validation failed:\n\nErrors:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        if warnings_list:
            error_msg += "\n\nWarnings:\n" + "\n".join(f"  - {w}" for w in warnings_list)
        raise BidsError(error_msg)

    return {"errors": errors, "warnings": warnings_list}


def aggregate_parcelstats(
    derivatives_dir: str | Path,
    output_dir: str | Path | None = None,
    pattern: str = "*_parcelstats.tsv",
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Aggregate subject-level parcelstats TSV files into group-level DataFrames.

    Scans a BIDS derivatives directory for parcelstats TSV files and combines
    them into single TSV files per output type, with each row representing a
    subject and each column representing a brain region.

    This is the "group" analysis level, similar to fMRIprep's group analysis.

    Parameters
    ----------
    derivatives_dir : str or Path
        Path to BIDS derivatives directory containing subject folders.
    output_dir : str or Path, optional
        Output directory for group-level TSV files. If None, uses
        derivatives_dir root.
    pattern : str, default="*_parcelstats.tsv"
        Glob pattern to match parcelstats files.
    overwrite : bool, default=False
        Overwrite existing group-level files.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping output type names to paths of created group TSV files.

    Raises
    ------
    BidsError
        If no parcelstats files are found or aggregation fails.

    Examples
    --------
    >>> from lacuna.io.bids import aggregate_parcelstats
    >>> # After running participant-level analysis
    >>> group_files = aggregate_parcelstats("output/lacuna")
    >>> print(group_files)
    {'atlas-schaefer2018_desc-100parcels7networks_source-fnm_desc-rmap_parcelstats':
     PosixPath('output/lacuna/group_atlas-schaefer2018_..._parcelstats.tsv')}

    Notes
    -----
    The output TSV files have the following structure:
    - First column: participant_id (e.g., sub-001)
    - Second column: session_id (if present, e.g., ses-01)
    - Third column: label (if present, e.g., acuteinfarct)
    - Remaining columns: brain region names with their values

    This enables direct loading into statistical analysis tools.
    """
    import pandas as pd

    derivatives_dir = Path(derivatives_dir)
    if not derivatives_dir.exists():
        raise BidsError(f"Derivatives directory not found: {derivatives_dir}")

    output_dir = Path(output_dir) if output_dir else derivatives_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all parcelstats files
    parcelstats_files = list(derivatives_dir.rglob(pattern))

    if not parcelstats_files:
        raise BidsError(f"No parcelstats files found matching '{pattern}' in {derivatives_dir}")

    # Group files by their output type (everything after label- entity)
    # Example: sub-CAS001_ses-01_label-acuteinfarct_atlas-schaefer2018_..._parcelstats.tsv
    # Output type: atlas-schaefer2018_..._parcelstats
    file_groups: dict[str, list[tuple[Path, dict[str, str]]]] = {}

    for tsv_file in parcelstats_files:
        filename = tsv_file.name

        # Parse BIDS entities from filename
        entities = _parse_bids_filename(filename)

        # Extract the output type (everything after participant/session/label entities)
        # This is the part that should be consistent across subjects
        output_type = _extract_output_type(filename)

        if output_type not in file_groups:
            file_groups[output_type] = []

        file_groups[output_type].append((tsv_file, entities))

    # Create group-level TSV for each output type
    created_files: dict[str, Path] = {}

    for output_type, files_with_entities in file_groups.items():
        # Build output filename
        group_filename = f"group_{output_type}.tsv"
        group_path = output_dir / group_filename

        if group_path.exists() and not overwrite:
            # Skip existing files unless overwrite is True
            created_files[output_type] = group_path
            continue

        # Collect data from all subjects
        rows = []
        for tsv_file, entities in files_with_entities:
            try:
                df = pd.read_csv(tsv_file, sep="\t")

                # Pivot the data: region -> value becomes columns
                if "region" in df.columns and "value" in df.columns:
                    # Create a row dict with metadata + region values
                    row_data = {
                        "participant_id": entities.get("sub", "unknown"),
                    }
                    if "ses" in entities:
                        row_data["session_id"] = entities["ses"]
                    if "label" in entities:
                        row_data["label"] = entities["label"]

                    # Add all region values as columns
                    for _, region_row in df.iterrows():
                        region_name = region_row["region"]
                        row_data[region_name] = region_row["value"]

                    rows.append(row_data)
            except Exception as e:
                warnings.warn(f"Failed to read {tsv_file}: {e}", stacklevel=2)
                continue

        if not rows:
            warnings.warn(f"No valid data for output type: {output_type}", stacklevel=2)
            continue

        # Create group DataFrame
        group_df = pd.DataFrame(rows)

        # Sort by participant_id, then session_id if present
        sort_cols = ["participant_id"]
        if "session_id" in group_df.columns:
            sort_cols.append("session_id")
        group_df = group_df.sort_values(sort_cols).reset_index(drop=True)

        # Save group TSV
        group_df.to_csv(group_path, sep="\t", index=False)

        # Create sidecar JSON
        sidecar_path = group_path.with_suffix(".json")
        sidecar = {
            "Description": f"Group-level aggregation of {output_type}",
            "Sources": [str(f.relative_to(derivatives_dir)) for f, _ in files_with_entities],
            "NumberOfSubjects": len(rows),
            "Columns": {
                "participant_id": "Subject identifier",
            },
        }
        if "session_id" in group_df.columns:
            sidecar["Columns"]["session_id"] = "Session identifier"
        if "label" in group_df.columns:
            sidecar["Columns"]["label"] = "Mask label (e.g., lesion type)"

        # Add region columns description
        region_cols = [
            c for c in group_df.columns if c not in ["participant_id", "session_id", "label"]
        ]
        if region_cols:
            sidecar["Columns"]["<region_name>"] = "Value for each brain region"
            sidecar["NumberOfRegions"] = len(region_cols)

        with open(sidecar_path, "w") as f:
            json.dump(sidecar, f, indent=2)

        created_files[output_type] = group_path

    return created_files


def _parse_bids_filename(filename: str) -> dict[str, str]:
    """
    Parse BIDS entities from a filename.

    Parameters
    ----------
    filename : str
        BIDS filename (e.g., sub-001_ses-01_label-lesion_atlas-X_parcelstats.tsv)

    Returns
    -------
    dict[str, str]
        Dictionary of entity key-value pairs.
    """
    entities = {}

    # Remove extension
    name = filename
    for ext in [".tsv", ".json", ".nii.gz", ".nii"]:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break

    # Parse key-value pairs
    parts = name.split("_")
    for part in parts:
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value

    return entities


def _extract_output_type(filename: str) -> str:
    """
    Extract the output type from a parcelstats filename.

    Removes subject-specific entities (sub-, ses-, label-) to get the
    consistent output type that should match across subjects.

    Parameters
    ----------
    filename : str
        BIDS filename

    Returns
    -------
    str
        Output type string (e.g., atlas-schaefer2018_desc-100parcels7networks_parcelstats)
    """
    # Remove extension
    name = filename
    for ext in [".tsv", ".json"]:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break

    # Split into parts and filter out subject-specific entities
    parts = name.split("_")
    output_parts = []

    for part in parts:
        # Skip subject-specific entities
        if part.startswith("sub-") or part.startswith("ses-") or part.startswith("label-"):
            continue
        output_parts.append(part)

    return "_".join(output_parts)
