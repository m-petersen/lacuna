"""
BIDS dataset loading and derivative export functionality.

Provides functions to load lesion data from BIDS-compliant datasets and
export analysis results in BIDS derivatives format.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.exceptions import LacunaError
from ..core.mask_data import MaskData

if TYPE_CHECKING:
    from ..core.data_types import ConnectivityMatrix, ParcelData, VoxelMap


class BidsError(LacunaError):
    """Raised when BIDS dataset operations fail."""

    pass


def load_bids_dataset(
    bids_root: str | Path,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    derivatives: bool = False,
    validate_bids: bool = True,
) -> dict[str, MaskData]:
    """
    Load lesion masks from a BIDS dataset.

    Parameters
    ----------
    bids_root : str or Path
        Path to BIDS dataset root directory.
    subjects : list of str, optional
        Specific subject IDs to load (e.g., ['sub-001', 'sub-002']).
        If None, loads all subjects with lesion masks.
    sessions : list of str, optional
        Specific session IDs to load (e.g., ['ses-01']).
        If None, loads all sessions.
    derivatives : bool, default=False
        Load from derivatives folder instead of raw data.
    validate_bids : bool, default=True
        Validate BIDS structure before loading (requires bids-validator).

    Returns
    -------
    dict of str -> MaskData
        Dictionary mapping subject IDs to MaskData objects.
        For multi-session data, keys are 'sub-XXX_ses-YYY'.

    Raises
    ------
    FileNotFoundError
        If bids_root doesn't exist.
    BidsError
        If BIDS validation fails or no lesion masks found.

    Warns
    -----
    UserWarning
        If some subjects have lesion masks but no anatomical scans.

    Examples
    --------
    >>> dataset = load_bids_dataset('/data/my_bids_dataset')
    >>> print(f"Loaded {len(dataset)} subjects")
    >>> lesion = dataset['sub-001']
    """
    bids_root = Path(bids_root)

    # Check if BIDS root exists
    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root directory not found: {bids_root}")

    # Check for dataset_description.json
    desc_file = bids_root / "dataset_description.json"
    if not desc_file.exists():
        raise BidsError(
            f"Missing dataset_description.json in BIDS root: {bids_root}\n"
            "This doesn't appear to be a valid BIDS dataset."
        )

    # Optionally validate BIDS structure
    if validate_bids:
        try:
            import bids

            # Use pybids to validate
            layout = bids.BIDSLayout(str(bids_root), validate=True, derivatives=derivatives)
        except ImportError:
            warnings.warn(
                "pybids not installed. Skipping BIDS validation. Install with: pip install pybids",
                UserWarning,
                stacklevel=2,
            )
            # Fall back to manual loading
            return _load_bids_manual(bids_root, subjects, sessions, derivatives)
        except Exception as e:
            raise BidsError(f"BIDS validation failed: {e}") from e
    else:
        # Manual loading without pybids
        return _load_bids_manual(bids_root, subjects, sessions, derivatives)

    # Load lesion masks using pybids
    mask_data_dict = {}

    # Get all lesion masks - try multiple approaches for BIDS compatibility
    # BIDS doesn't have a 'label' entity, so we search for mask/roi suffix
    # and filter by filename pattern for lesion-related files
    try:
        lesion_files = layout.get(
            suffix=["mask", "roi"],
            extension=[".nii", ".nii.gz"],
            subject=subjects,
            session=sessions,
        )
        # Filter to only include lesion-related files
        lesion_files = [
            f for f in lesion_files
            if "lesion" in f.filename.lower() or "desc-lesion" in f.filename.lower()
        ]
    except Exception:
        # Fall back to manual loading if pybids query fails
        return _load_bids_manual(bids_root, subjects, sessions, derivatives)

    if not lesion_files:
        raise BidsError(f"No lesion masks found in BIDS dataset: {bids_root}")

    # Load each lesion mask
    for lesion_file in lesion_files:
        entities = lesion_file.get_entities()
        subject_id = entities.get("subject")
        session_id = entities.get("session")

        # Create subject key
        if session_id:
            subject_key = f"sub-{subject_id}_ses-{session_id}"
        else:
            subject_key = f"sub-{subject_id}"

        # Look for corresponding anatomical image
        anat_files = layout.get(
            subject=subject_id,
            session=session_id,
            suffix="T1w",
            extension=[".nii", ".nii.gz"],
        )

        anatomical_path = None
        if anat_files:
            anatomical_path = anat_files[0].path
        else:
            warnings.warn(
                f"No anatomical image found for {subject_key}. Loading lesion mask only.",
                UserWarning,
                stacklevel=2,
            )

        # Load lesion data
        metadata = {
            "subject_id": f"sub-{subject_id}",
            "bids_root": str(bids_root),
            "lesion_path": lesion_file.path,
        }
        if session_id:
            metadata["session_id"] = f"ses-{session_id}"

        mask_data = MaskData.from_nifti(
            lesion_path=lesion_file.path, anatomical_path=anatomical_path, metadata=metadata
        )

        mask_data_dict[subject_key] = mask_data

    return mask_data_dict


def _load_bids_manual(
    bids_root: Path,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    derivatives: bool = False,
) -> dict[str, MaskData]:
    """
    Load BIDS dataset without pybids (manual parsing).

    This is a fallback implementation for when pybids is not installed.
    """
    mask_data_dict = {}

    # Get all subject directories
    subject_dirs = sorted(bids_root.glob("sub-*"))

    if derivatives:
        # Look in derivatives folder
        deriv_root = bids_root / "derivatives"
        if deriv_root.exists():
            subject_dirs = sorted(deriv_root.glob("*/sub-*"))

    # Filter subjects if specified
    if subjects:
        subject_ids = [s if s.startswith("sub-") else f"sub-{s}" for s in subjects]
        subject_dirs = [d for d in subject_dirs if d.name in subject_ids]

    found_any = False

    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        # Check for session directories
        session_dirs = sorted(subject_dir.glob("ses-*"))

        if session_dirs:
            # Multi-session dataset
            for session_dir in session_dirs:
                session_id = session_dir.name

                # Filter sessions if specified
                if sessions and session_id not in [
                    s if s.startswith("ses-") else f"ses-{s}" for s in sessions
                ]:
                    continue

                # Look for lesion mask in anat folder
                anat_dir = session_dir / "anat"
                if anat_dir.exists():
                    # Look for BIDS-compliant desc-lesion pattern
                    lesion_files = list(
                        anat_dir.glob(f"{subject_id}_{session_id}*desc-lesion*mask*.nii*")
                    )
                    # Also try legacy mask-lesion pattern
                    if not lesion_files:
                        lesion_files = list(
                            anat_dir.glob(f"{subject_id}_{session_id}*mask-lesion*.nii*")
                        )
                    lesion_files.extend(
                        anat_dir.glob(f"{subject_id}_{session_id}*roi-lesion*.nii*")
                    )

                    if lesion_files:
                        found_any = True
                        lesion_path = lesion_files[0]

                        # Look for anatomical
                        anat_files = list(anat_dir.glob(f"{subject_id}_{session_id}*T1w.nii*"))
                        anatomical_path = anat_files[0] if anat_files else None

                        if anatomical_path is None:
                            warnings.warn(
                                f"No anatomical image found for {subject_id}_{session_id}",
                                UserWarning,
                                stacklevel=2,
                            )

                        # Create MaskData
                        subject_key = f"{subject_id}_{session_id}"
                        metadata = {
                            "subject_id": subject_id,
                            "session_id": session_id,
                            "bids_root": str(bids_root),
                            "lesion_path": str(lesion_path),
                        }

                        # Parse sidecar for space/resolution
                        sidecar_data = _parse_sidecar(lesion_path)
                        space = sidecar_data.get("Space") or sidecar_data.get("space")
                        resolution = _parse_resolution(
                            sidecar_data.get("Resolution") or sidecar_data.get("resolution")
                        )

                        mask_data_dict[subject_key] = MaskData.from_nifti(
                            lesion_path=lesion_path,
                            anatomical_path=anatomical_path,
                            metadata=metadata,
                            space=space,
                            resolution=resolution,
                        )
        else:
            # Single-session dataset
            anat_dir = subject_dir / "anat"
            if anat_dir.exists():
                # Look for BIDS-compliant desc-lesion pattern
                lesion_files = list(anat_dir.glob(f"{subject_id}*desc-lesion*mask*.nii*"))
                # Also try legacy mask-lesion pattern
                if not lesion_files:
                    lesion_files = list(anat_dir.glob(f"{subject_id}*mask-lesion*.nii*"))
                lesion_files.extend(anat_dir.glob(f"{subject_id}*roi-lesion*.nii*"))

                if lesion_files:
                    found_any = True
                    lesion_path = lesion_files[0]

                    # Look for anatomical
                    anat_files = list(anat_dir.glob(f"{subject_id}*T1w.nii*"))
                    anatomical_path = anat_files[0] if anat_files else None

                    if anatomical_path is None:
                        warnings.warn(
                            f"No anatomical image found for {subject_id}",
                            UserWarning,
                            stacklevel=2,
                        )

                    # Create MaskData
                    metadata = {
                        "subject_id": subject_id,
                        "bids_root": str(bids_root),
                        "lesion_path": str(lesion_path),
                    }

                    # Parse sidecar for space/resolution
                    sidecar_data = _parse_sidecar(lesion_path)
                    space = sidecar_data.get("Space") or sidecar_data.get("space")
                    resolution = _parse_resolution(
                        sidecar_data.get("Resolution") or sidecar_data.get("resolution")
                    )

                    mask_data_dict[subject_id] = MaskData.from_nifti(
                        lesion_path=lesion_path,
                        metadata=metadata,
                        space=space,
                        resolution=resolution,
                    )

    if not found_any:
        raise BidsError(f"No lesion masks found in BIDS dataset: {bids_root}")

    return mask_data_dict


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
    import re

    match = re.search(r"space-([a-zA-Z0-9]+)", filename)
    if match:
        return match.group(1)
    return None


def export_voxelmap(
    voxelmap: "VoxelMap",
    output_dir: str | Path,
    subject_id: str,
    session_id: str | None = None,
    desc: str = "map",
    space: str | None = None,
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
        Subject identifier (e.g., 'sub-001')
    session_id : str, optional
        Session identifier (e.g., 'ses-01')
    desc : str, default='map'
        Description label for BIDS filename
    space : str, optional
        Override space from voxelmap.space
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

    # Build BIDS filename
    space = space or voxelmap.space
    if session_id:
        base_name = f"{subject_id}_{session_id}_space-{space}_desc-{desc}"
    else:
        base_name = f"{subject_id}_space-{space}_desc-{desc}"

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
        json.dump(sidecar, f, indent=2)

    return nifti_path


def export_parcel_data(
    parcel_data: "ParcelData",
    output_dir: str | Path,
    subject_id: str,
    session_id: str | None = None,
    desc: str = "parcels",
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
        Subject identifier (e.g., 'sub-001')
    session_id : str, optional
        Session identifier (e.g., 'ses-01')
    desc : str, default='parcels'
        Description label for BIDS filename
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

    # Build BIDS filename
    if session_id:
        base_name = f"{subject_id}_{session_id}_desc-{desc}_parcels"
    else:
        base_name = f"{subject_id}_desc-{desc}_parcels"

    tsv_path = output_dir / f"{base_name}.tsv"
    sidecar_path = output_dir / f"{base_name}.json"

    if tsv_path.exists() and not overwrite:
        raise FileExistsError(f"File exists: {tsv_path}. Use overwrite=True.")

    # Create DataFrame
    df = pd.DataFrame([
        {"region": k, "value": v}
        for k, v in parcel_data.data.items()
    ])

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
        json.dump(sidecar, f, indent=2)

    return tsv_path


def export_connectivity_matrix(
    matrix: "ConnectivityMatrix",
    output_dir: str | Path,
    subject_id: str,
    session_id: str | None = None,
    desc: str = "connectivity",
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
        Subject identifier (e.g., 'sub-001')
    session_id : str, optional
        Session identifier (e.g., 'ses-01')
    desc : str, default='connectivity'
        Description label for BIDS filename
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

    # Build BIDS filename
    if session_id:
        base_name = f"{subject_id}_{session_id}_desc-{desc}_connmatrix"
    else:
        base_name = f"{subject_id}_desc-{desc}_connmatrix"

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
        json.dump(sidecar, f, indent=2)

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
    Export multiple MaskData results to BIDS derivatives format.

    Parameters
    ----------
    results : list[MaskData]
        List of processed MaskData objects
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
    for mask_data in results:
        export_bids_derivatives(
            mask_data,
            output_dir,
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
    mask_data: MaskData,
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
    Export MaskData and all its analysis results to BIDS derivatives format.

    Exports the full spectrum of results stored in a MaskData object:
    - Lesion mask as NIfTI
    - VoxelMaps (correlation maps, disconnection maps, etc.) as NIfTI
    - ParcelData (regional values) as TSV
    - ConnectivityMatrix as TSV
    - ScalarMetric and other scalars as JSON
    - Processing provenance as JSON

    Parameters
    ----------
    mask_data : MaskData
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
        If mask_data has no subject_id in metadata.

    Examples
    --------
    >>> # Export all results
    >>> output_path = export_bids_derivatives(
    ...     mask_data,
    ...     'derivatives/lacuna-v0.1.0'
    ... )
    >>> print(f"Derivatives saved to: {output_path}")
    >>>
    >>> # Export only VoxelMaps (NIfTI files)
    >>> export_bids_derivatives(
    ...     mask_data,
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
        ParcelData as ParcelDataType,
        ScalarMetric,
        VoxelMap,
    )

    output_dir = Path(output_dir)

    # Validate metadata
    if "subject_id" not in mask_data.metadata:
        raise ValueError("MaskData metadata must contain 'subject_id' for BIDS export")

    subject_id = mask_data.metadata["subject_id"]
    session_id = mask_data.metadata.get("session_id")

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

    # Determine which directories we need
    needs_anat = export_lesion_mask
    needs_results = (
        export_voxelmaps or export_parcel_data or export_connectivity or
        export_scalars or export_provenance
    )

    # Create directories
    if needs_anat:
        anat_dir = subject_dir / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

    if needs_results:
        results_dir = subject_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

    # Save lesion mask
    if export_lesion_mask:
        coord_space = mask_data.get_coordinate_space()
        lesion_filename = f"{base_name}_space-{coord_space}_desc-lesion_mask.nii.gz"
        lesion_path = anat_dir / lesion_filename

        if lesion_path.exists() and not overwrite:
            raise FileExistsError(
                f"Lesion mask already exists: {lesion_path}. Use overwrite=True to replace."
            )

        nib.save(mask_data.mask_img, lesion_path)

    # Save analysis results
    if mask_data.results:
        for namespace, results_data in mask_data.results.items():
            if not isinstance(results_data, dict):
                continue

            for key, value in results_data.items():
                # VoxelMap -> NIfTI
                if isinstance(value, VoxelMap) and export_voxelmaps:
                    export_voxelmap(
                        value,
                        results_dir,
                        subject_id=subject_id,
                        session_id=session_id,
                        desc=f"{namespace.lower()}_{key}",
                        overwrite=overwrite,
                    )

                # ParcelData -> TSV
                elif isinstance(value, ParcelDataType) and export_parcel_data:
                    _export_parcel_data(
                        value,
                        results_dir,
                        subject_id=subject_id,
                        session_id=session_id,
                        desc=f"{namespace.lower()}_{key}",
                        overwrite=overwrite,
                    )

                # ConnectivityMatrix -> TSV
                elif isinstance(value, ConnectivityMatrix) and export_connectivity:
                    export_connectivity_matrix(
                        value,
                        results_dir,
                        subject_id=subject_id,
                        session_id=session_id,
                        desc=f"{namespace.lower()}_{key}",
                        overwrite=overwrite,
                    )

                # ScalarMetric or other serializable -> JSON
                elif export_scalars:
                    if isinstance(value, ScalarMetric):
                        data_to_save = value.get_data()
                    else:
                        data_to_save = value

                    try:
                        results_filename = f"{base_name}_desc-{namespace.lower()}_{key}.json"
                        results_path = results_dir / results_filename

                        if results_path.exists() and not overwrite:
                            continue

                        with open(results_path, "w") as f:
                            json.dump(data_to_save, f, indent=2, default=str)
                    except (TypeError, ValueError):
                        # Skip non-serializable results
                        pass

    # Save provenance
    if export_provenance and mask_data.provenance:
        prov_filename = f"{base_name}_desc-provenance.json"
        prov_path = results_dir / prov_filename

        if prov_path.exists() and not overwrite:
            raise FileExistsError(
                f"Provenance file already exists: {prov_path}. Use overwrite=True to replace."
            )

        # Convert provenance to serializable format
        prov_data = []
        for step in mask_data.provenance:
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


def save_nifti(mask_data: MaskData, output_path: str | Path, save_anatomical: bool = False) -> None:
    """
    Save lesion mask to NIfTI file.

    Parameters
    ----------
    mask_data : MaskData
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
    - Subject directories follow naming conventions
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
    - Subject directories match pattern: sub-<label>[/ses-<label>]
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

            # Check for expected subdirectories
            expected_subdirs = ["anat", "results", "func", "dwi"]
            has_subdirs = any((subj_dir / sd).exists() for sd in expected_subdirs)

            if not has_subdirs:
                warnings_list.append(
                    f"Subject '{subj_name}' has no standard BIDS subdirectories "
                    f"(anat, results, func, dwi)"
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
