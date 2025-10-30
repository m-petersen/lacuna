"""
Core LesionData class - the central API contract for the toolkit.

This class encapsulates a single subject's lesion data with metadata, provenance
tracking, and analysis results. It serves as the stable interface between all
pipeline modules.
"""

import copy
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from .exceptions import NiftiLoadError
from .validation import check_spatial_match, validate_affine, validate_nifti_image


class LesionData:
    """
    Central data container for a single subject's lesion analysis.

    This class encapsulates lesion image data, optional anatomical scan, spatial
    metadata, subject identifiers, processing provenance, and analysis results.
    It enforces immutability-by-convention: transformations should return new
    instances rather than modifying in place.

    Parameters
    ----------
    lesion_img : nibabel.Nifti1Image
        Binary or continuous lesion mask (3D only).
    anatomical_img : nibabel.Nifti1Image, optional
        Subject's anatomical scan (must match lesion coordinate space).
    metadata : dict, optional
        Subject metadata (must contain 'subject_id' key if provided).
        Defaults to {"subject_id": "sub-unknown"}.
    provenance : list of dict, optional
        Processing history (for deserialization only).
    results : dict, optional
        Analysis results (for deserialization only).

    Raises
    ------
    ValidationError
        If lesion_img is not 3D, affines don't match, or metadata missing 'subject_id'.
    SpatialMismatchError
        If anatomical_img coordinate space doesn't match lesion_img.

    Attributes
    ----------
    lesion_img : nibabel.Nifti1Image
        The lesion mask image (read-only).
    anatomical_img : nibabel.Nifti1Image or None
        Optional anatomical scan (read-only).
    affine : np.ndarray
        4x4 affine transformation matrix (read-only).
    metadata : dict
        Subject and session metadata (read-only view).
    provenance : list
        Processing history (read-only view).
    results : dict
        Analysis results (read-only view).

    Examples
    --------
    >>> import nibabel as nib
    >>> lesion_img = nib.load("lesion.nii.gz")
    >>> lesion = LesionData(lesion_img, metadata={"subject_id": "sub-001"})
    >>> print(f"Volume: {lesion.get_volume_mm3()} mm³")
    >>> print(f"Space: {lesion.get_coordinate_space()}")
    """

    def __init__(
        self,
        lesion_img: nib.Nifti1Image,
        anatomical_img: nib.Nifti1Image | None = None,
        metadata: dict[str, Any] | None = None,
        provenance: list[dict[str, Any]] | None = None,
        results: dict[str, Any] | None = None,
    ):
        # Validate lesion image
        validate_nifti_image(lesion_img, require_3d=True, check_affine=True)

        # Store images
        self._lesion_img = lesion_img
        self._anatomical_img = anatomical_img

        # Extract and validate affine
        self._affine = lesion_img.affine.copy()
        validate_affine(self._affine)

        # Validate anatomical image if provided
        if anatomical_img is not None:
            validate_nifti_image(anatomical_img, require_3d=True)
            check_spatial_match(lesion_img, anatomical_img, check_shape=False, check_affine=True)

        # Setup metadata
        if metadata is None:
            metadata = {}
        if "subject_id" not in metadata:
            metadata["subject_id"] = "sub-unknown"
        self._metadata = metadata.copy()

        # Setup provenance (empty list for new objects)
        self._provenance = list(provenance) if provenance is not None else []

        # Setup results (empty dict for new objects)
        self._results = dict(results) if results is not None else {}

        # Track coordinate space (extracted from provenance or default to native)
        self._coordinate_space = self._infer_coordinate_space()

    @classmethod
    def from_nifti(
        cls,
        lesion_path: str | Path,
        anatomical_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "LesionData":
        """
        Load lesion data from NIfTI file(s).

        Parameters
        ----------
        lesion_path : str or Path
            Path to lesion mask NIfTI file.
        anatomical_path : str or Path, optional
            Path to anatomical NIfTI file.
        metadata : dict, optional
            Subject metadata. Auto-generated 'subject_id' if not provided.

        Returns
        -------
        LesionData
            Loaded lesion data object.

        Raises
        ------
        FileNotFoundError
            If file paths don't exist.
        NiftiLoadError
            If images fail to load or validate.

        Examples
        --------
        >>> lesion = LesionData.from_nifti("lesion.nii.gz")
        >>> lesion = LesionData.from_nifti(
        ...     "lesion.nii.gz",
        ...     anatomical_path="T1w.nii.gz",
        ...     metadata={"subject_id": "sub-001"}
        ... )
        """
        lesion_path = Path(lesion_path)

        # Load lesion image
        try:
            lesion_img = nib.load(lesion_path)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise NiftiLoadError(f"Failed to load lesion from {lesion_path}: {e}") from e

        # Load anatomical if provided
        anatomical_img = None
        if anatomical_path is not None:
            anatomical_path = Path(anatomical_path)
            try:
                anatomical_img = nib.load(anatomical_path)
            except FileNotFoundError:
                raise
            except Exception as e:
                raise NiftiLoadError(
                    f"Failed to load anatomical from {anatomical_path}: {e}"
                ) from e

        # Auto-generate subject_id from filename if not provided
        if metadata is None:
            metadata = {}
        if "subject_id" not in metadata:
            # Try to extract BIDS-like subject ID from filename
            filename = lesion_path.stem.replace(".nii", "")
            if "sub-" in filename:
                # Extract sub-XXX pattern
                parts = filename.split("_")
                for part in parts:
                    if part.startswith("sub-"):
                        metadata["subject_id"] = part
                        break

        return cls(lesion_img, anatomical_img=anatomical_img, metadata=metadata)

    def validate(self) -> bool:
        """
        Validate data integrity.

        Checks that affine is invertible, images are 3D, and spatial properties
        are consistent.

        Returns
        -------
        bool
            True if all checks pass.

        Warns
        -----
        UserWarning
            If lesion mask is empty or has suspicious properties.

        Raises
        ------
        ValidationError
            If critical invariants violated.

        Examples
        --------
        >>> lesion.validate()
        True
        """
        # Validate images
        validate_nifti_image(self._lesion_img, require_3d=True)
        if self._anatomical_img is not None:
            validate_nifti_image(self._anatomical_img, require_3d=True)

        # Validate affine
        validate_affine(self._affine)

        # Check lesion is not empty
        lesion_data = self._lesion_img.get_fdata()
        if not np.any(lesion_data > 0):
            import warnings

            warnings.warn("Lesion mask is empty (no non-zero voxels)", UserWarning, stacklevel=2)

        return True

    def copy(self) -> "LesionData":
        """
        Create a deep copy of this LesionData instance.

        Returns
        -------
        LesionData
            Independent copy with same data.

        Examples
        --------
        >>> lesion_copy = lesion.copy()
        >>> lesion_copy is lesion
        False
        """
        return LesionData(
            lesion_img=self._lesion_img,
            anatomical_img=self._anatomical_img,
            metadata=copy.deepcopy(self._metadata),
            provenance=copy.deepcopy(self._provenance),
            results=copy.deepcopy(self._results),
        )

    def get_coordinate_space(self) -> str:
        """
        Infer current coordinate space from affine and provenance.

        Returns
        -------
        str
            Coordinate space identifier (e.g., 'native', 'MNI152_2mm').

        Examples
        --------
        >>> lesion.get_coordinate_space()
        'native'
        """
        return self._coordinate_space

    def get_volume_mm3(self) -> float:
        """
        Calculate lesion volume in cubic millimeters.

        Returns
        -------
        float
            Total lesion volume (sum of non-zero voxels * voxel volume).

        Examples
        --------
        >>> volume = lesion.get_volume_mm3()
        >>> print(f"Lesion volume: {volume:.2f} mm³")
        """
        lesion_data = self._lesion_img.get_fdata()
        num_voxels = np.sum(lesion_data > 0)

        # Calculate voxel volume from affine
        voxel_dims = np.abs(np.diag(self._affine[:3, :3]))
        voxel_volume_mm3 = np.prod(voxel_dims)

        return float(num_voxels * voxel_volume_mm3)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to JSON-compatible dictionary (excludes image data).

        Returns
        -------
        dict
            Metadata, provenance, and results (no NIfTI arrays).

        Examples
        --------
        >>> data = lesion.to_dict()
        >>> import json
        >>> json.dumps(data)  # Should succeed
        """
        return {
            "metadata": copy.deepcopy(self._metadata),
            "provenance": copy.deepcopy(self._provenance),
            "results": copy.deepcopy(self._results),
            "coordinate_space": self._coordinate_space,
            "affine": self._affine.tolist(),  # Convert numpy to list for JSON
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], lesion_img: nib.Nifti1Image) -> "LesionData":
        """
        Deserialize from dictionary + NIfTI image.

        Parameters
        ----------
        data : dict
            Output from to_dict().
        lesion_img : nibabel.Nifti1Image
            Lesion image (loaded separately).

        Returns
        -------
        LesionData
            Reconstructed object.

        Examples
        --------
        >>> data = lesion.to_dict()
        >>> lesion_img = nib.load("lesion.nii.gz")
        >>> lesion_restored = LesionData.from_dict(data, lesion_img)
        """
        return cls(
            lesion_img=lesion_img,
            metadata=data.get("metadata"),
            provenance=data.get("provenance"),
            results=data.get("results"),
        )

    def add_result(self, namespace: str, results: dict[str, Any]) -> "LesionData":
        """
        Create new LesionData with additional analysis results.

        This method follows immutability-by-convention: it returns a new instance
        with the updated results rather than modifying the current instance.

        Parameters
        ----------
        namespace : str
            Result namespace (e.g., 'LesionNetworkMapping', 'VolumeAnalysis').
            Should match the analysis module name for clarity.
        results : dict
            Analysis results to store (must be JSON-serializable).

        Returns
        -------
        LesionData
            New instance with added results.

        Raises
        ------
        ValueError
            If namespace already exists in results.

        Examples
        --------
        >>> results = {"volume_mm3": 1234.5, "n_voxels": 150}
        >>> lesion_with_results = lesion.add_result("VolumeAnalysis", results)
        >>> "VolumeAnalysis" in lesion_with_results.results
        True
        >>> "VolumeAnalysis" in lesion.results  # Original unchanged
        False
        """
        if namespace in self._results:
            raise ValueError(
                f"Result namespace '{namespace}' already exists. "
                f"Use a different namespace or create a new LesionData instance."
            )

        # Create new results dict with added namespace
        new_results = copy.deepcopy(self._results)
        new_results[namespace] = copy.deepcopy(results)

        # Return new instance
        return LesionData(
            lesion_img=self._lesion_img,
            anatomical_img=self._anatomical_img,
            metadata=copy.deepcopy(self._metadata),
            provenance=copy.deepcopy(self._provenance),
            results=new_results,
        )

    def add_provenance(self, record: dict[str, Any]) -> "LesionData":
        """
        Create new LesionData with additional provenance record.

        This method follows immutability-by-convention: it returns a new instance
        with the updated provenance history rather than modifying the current instance.

        Parameters
        ----------
        record : dict
            Provenance record (from create_provenance_record() or compatible dict).
            Must contain 'function', 'parameters', 'timestamp', and 'version' keys.

        Returns
        -------
        LesionData
            New instance with appended provenance.

        Raises
        ------
        ValueError
            If record is missing required fields.

        Examples
        --------
        >>> from ldk.core.provenance import create_provenance_record
        >>> prov = create_provenance_record(
        ...     function="ldk.preprocess.normalize_to_mni",
        ...     parameters={"template": "MNI152_2mm"},
        ...     version="0.1.0"
        ... )
        >>> lesion_normalized = lesion.add_provenance(prov)
        >>> len(lesion_normalized.provenance) == len(lesion.provenance) + 1
        True
        """
        # Validate record has required fields
        required_fields = ["function", "parameters", "timestamp", "version"]
        missing_fields = [f for f in required_fields if f not in record]
        if missing_fields:
            raise ValueError(
                f"Provenance record missing required fields: {missing_fields}. "
                f"Use create_provenance_record() to create valid records."
            )

        # Create new provenance list with appended record
        new_provenance = copy.deepcopy(self._provenance)
        new_provenance.append(copy.deepcopy(record))

        # Return new instance
        return LesionData(
            lesion_img=self._lesion_img,
            anatomical_img=self._anatomical_img,
            metadata=copy.deepcopy(self._metadata),
            provenance=new_provenance,
            results=copy.deepcopy(self._results),
        )

    def _infer_coordinate_space(self) -> str:
        """Infer coordinate space from provenance or default to 'native'."""
        if self._provenance:
            # Check most recent transformation
            last_transform = self._provenance[-1]
            if "output_space" in last_transform:
                return last_transform["output_space"]
        return "native"

    # Read-only properties

    @property
    def lesion_img(self) -> nib.Nifti1Image:
        """Binary or continuous lesion mask."""
        return self._lesion_img

    @property
    def anatomical_img(self) -> nib.Nifti1Image | None:
        """Subject's anatomical scan (if provided)."""
        return self._anatomical_img

    @property
    def affine(self) -> np.ndarray:
        """4x4 affine transformation matrix (voxel to world)."""
        return self._affine.copy()  # Return copy to prevent modification

    @property
    def metadata(self) -> dict[str, Any]:
        """Subject and session metadata (immutable view)."""
        return self._metadata.copy()  # Return copy to prevent modification

    @property
    def provenance(self) -> list[dict[str, Any]]:
        """Processing history (immutable view)."""
        return copy.deepcopy(self._provenance)  # Deep copy for nested dicts

    @property
    def results(self) -> dict[str, Any]:
        """Analysis results (immutable view)."""
        return copy.deepcopy(self._results)  # Deep copy for nested structures
