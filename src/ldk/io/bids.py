"""
BIDS dataset loading and derivative export functionality.

Provides functions to load lesion data from BIDS-compliant datasets and
export analysis results in BIDS derivatives format.
"""

import json
import warnings
from pathlib import Path

from ..core.exceptions import LdkError
from ..core.lesion_data import LesionData


class BidsError(LdkError):
    """Raised when BIDS dataset operations fail."""

    pass


def load_bids_dataset(
    bids_root: str | Path,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    derivatives: bool = False,
    validate_bids: bool = True,
) -> dict[str, LesionData]:
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
    dict of str -> LesionData
        Dictionary mapping subject IDs to LesionData objects.
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
    lesion_data_dict = {}

    # Get all lesion masks
    lesion_files = layout.get(
        suffix=["mask", "roi"],
        label="lesion",
        extension=[".nii", ".nii.gz"],
        subject=subjects,
        session=sessions,
    )

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

        lesion_data = LesionData.from_nifti(
            lesion_path=lesion_file.path, anatomical_path=anatomical_path, metadata=metadata
        )

        lesion_data_dict[subject_key] = lesion_data

    return lesion_data_dict


def _load_bids_manual(
    bids_root: Path,
    subjects: list[str] | None = None,
    sessions: list[str] | None = None,
    derivatives: bool = False,
) -> dict[str, LesionData]:
    """
    Load BIDS dataset without pybids (manual parsing).

    This is a fallback implementation for when pybids is not installed.
    """
    lesion_data_dict = {}

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

                        # Create LesionData
                        subject_key = f"{subject_id}_{session_id}"
                        metadata = {
                            "subject_id": subject_id,
                            "session_id": session_id,
                            "bids_root": str(bids_root),
                            "lesion_path": str(lesion_path),
                        }

                        lesion_data_dict[subject_key] = LesionData.from_nifti(
                            lesion_path=lesion_path,
                            anatomical_path=anatomical_path,
                            metadata=metadata,
                        )
        else:
            # Single-session dataset
            anat_dir = subject_dir / "anat"
            if anat_dir.exists():
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

                    # Create LesionData
                    metadata = {
                        "subject_id": subject_id,
                        "bids_root": str(bids_root),
                        "lesion_path": str(lesion_path),
                    }

                    lesion_data_dict[subject_id] = LesionData.from_nifti(
                        lesion_path=lesion_path, anatomical_path=anatomical_path, metadata=metadata
                    )

    if not found_any:
        raise BidsError(f"No lesion masks found in BIDS dataset: {bids_root}")

    return lesion_data_dict


def export_bids_derivatives(
    lesion_data: LesionData,
    output_dir: str | Path,
    include_images: bool = True,
    include_results: bool = True,
    include_provenance: bool = True,
    overwrite: bool = False,
) -> Path:
    """
    Export LesionData to BIDS derivatives format.

    Parameters
    ----------
    lesion_data : LesionData
        Processed lesion data with analysis results.
    output_dir : str or Path
        Root directory for derivatives (e.g., 'derivatives/ldk-v0.1.0').
    include_images : bool, default=True
        Save normalized lesion masks as NIfTI files.
    include_results : bool, default=True
        Save analysis results as TSV/JSON files.
    include_provenance : bool, default=True
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
        If lesion_data has no subject_id in metadata.

    Examples
    --------
    >>> output_path = export_bids_derivatives(
    ...     lesion_data,
    ...     'derivatives/ldk-v0.1.0',
    ...     include_provenance=True
    ... )
    >>> print(f"Derivatives saved to: {output_path}")
    """
    import nibabel as nib

    output_dir = Path(output_dir)

    # Validate metadata
    if "subject_id" not in lesion_data.metadata:
        raise ValueError("LesionData metadata must contain 'subject_id' for BIDS export")

    subject_id = lesion_data.metadata["subject_id"]
    session_id = lesion_data.metadata.get("session_id")

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
            "Name": "Lesion Decoding Toolkit Derivatives",
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

    # Create subject directories
    if include_images:
        anat_dir = subject_dir / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

    if include_results or include_provenance:
        results_dir = subject_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

    # Save lesion image
    if include_images:
        # Determine output filename
        if session_id:
            base_name = f"{subject_id}_{session_id}"
        else:
            base_name = subject_id

        # Get coordinate space from metadata
        coord_space = lesion_data.get_coordinate_space()
        lesion_filename = f"{base_name}_space-{coord_space}_desc-lesion_mask.nii.gz"
        lesion_path = anat_dir / lesion_filename

        if lesion_path.exists() and not overwrite:
            raise FileExistsError(
                f"Lesion mask already exists: {lesion_path}. Use overwrite=True to replace."
            )

        # Save NIfTI
        nib.save(lesion_data.lesion_img, lesion_path)

    # Save results
    if include_results and lesion_data.results:
        for namespace, results_data in lesion_data.results.items():
            results_filename = f"{base_name}_desc-{namespace.lower()}_results.json"
            results_path = results_dir / results_filename

            if results_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Results file already exists: {results_path}. Use overwrite=True to replace."
                )

            with open(results_path, "w") as f:
                json.dump(results_data, f, indent=2)

    # Save provenance
    if include_provenance and lesion_data.provenance:
        prov_filename = f"{base_name}_desc-provenance.json"
        prov_path = results_dir / prov_filename

        if prov_path.exists() and not overwrite:
            raise FileExistsError(
                f"Provenance file already exists: {prov_path}. Use overwrite=True to replace."
            )

        with open(prov_path, "w") as f:
            json.dump(lesion_data.provenance, f, indent=2)

    return subject_dir


def save_nifti(
    lesion_data: LesionData, output_path: str | Path, save_anatomical: bool = False
) -> None:
    """
    Save lesion mask to NIfTI file.

    Parameters
    ----------
    lesion_data : LesionData
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
    >>> save_nifti(lesion_data, 'output/lesion.nii.gz')
    >>> save_nifti(lesion_data, 'output/lesion.nii.gz', save_anatomical=True)
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
    nib.save(lesion_data.lesion_img, output_path)

    # Optionally save anatomical
    if save_anatomical and lesion_data.anatomical_img is not None:
        # Create anatomical filename
        anat_path = output_path.parent / output_path.name.replace("lesion", "anat")
        nib.save(lesion_data.anatomical_img, anat_path)
