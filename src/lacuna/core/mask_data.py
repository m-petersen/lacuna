"""
Core MaskData class - the central API contract for the toolkit.

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
from .validation import validate_affine, validate_nifti_image


class ImmutableDict(dict):
    """
    Immutable dictionary wrapper with custom error messages.

    This class wraps a dictionary and prevents modifications by raising
    clear, informative errors when modification is attempted.
    """

    def __init__(self, data: dict, attribute_name: str):
        super().__init__(data)
        self._attribute_name = attribute_name

    def __setitem__(self, key, value):
        raise TypeError(
            f"Cannot modify MaskData.{self._attribute_name} - it is immutable.\n"
            f"To update {self._attribute_name}, create a new MaskData instance instead."
        )

    def __delitem__(self, key):
        raise TypeError(
            f"Cannot delete from MaskData.{self._attribute_name} - it is immutable.\n"
            f"To modify {self._attribute_name}, create a new MaskData instance instead."
        )

    def update(self, *args, **kwargs):
        raise TypeError(
            f"Cannot update MaskData.{self._attribute_name} - it is immutable.\n"
            f"To update {self._attribute_name}, create a new MaskData instance instead."
        )

    def pop(self, *args, **kwargs):
        raise TypeError(f"Cannot pop from MaskData.{self._attribute_name} - it is immutable.")

    def popitem(self):
        raise TypeError(f"Cannot popitem from MaskData.{self._attribute_name} - it is immutable.")

    def clear(self):
        raise TypeError(f"Cannot clear MaskData.{self._attribute_name} - it is immutable.")

    def setdefault(self, *args, **kwargs):
        raise TypeError(f"Cannot setdefault on MaskData.{self._attribute_name} - it is immutable.")


class MaskData:
    """
    Central data container for a single subject's mask-based analysis.

    This class encapsulates binary mask image data, spatial metadata, subject
    identifiers, processing provenance, and analysis results. It enforces
    immutability-by-convention: transformations should return new instances
    rather than modifying in place.

    Parameters
    ----------
    mask_img : nibabel.Nifti1Image
        Binary mask (3D only, values must be 0 or 1).
    metadata : dict, optional
        Subject metadata. Must contain 'space' and 'resolution' keys.
        'subject_id' defaults to "sub-unknown" if not provided.
    provenance : list of dict, optional
        Processing history (for deserialization only).
    results : dict, optional
        Analysis results (for deserialization only).

    Raises
    ------
    ValueError
        If metadata is missing 'space' or 'resolution' keys,
        if mask_img is not 3D, or if mask_img is not binary (0/1 values only).

    Attributes
    ----------
    mask_img : nibabel.Nifti1Image
        The binary mask image (read-only).
    affine : np.ndarray
        4x4 affine transformation matrix (read-only).
    space : str
        Coordinate space identifier (e.g., 'MNI152NLin6Asym').
    resolution : float
        Spatial resolution in millimeters.
    metadata : ImmutableDict
        Subject and session metadata (read-only view).
    provenance : list
        Processing history (read-only view).
    results : dict
        Analysis results (read-only view, nested structure).

    Examples
    --------
    >>> import nibabel as nib
    >>> mask_img = nib.load("mask.nii.gz")
    >>> mask_data = MaskData(
    ...     mask_img,
    ...     metadata={"subject_id": "sub-001", "space": "MNI152NLin6Asym", "resolution": 2}
    ... )
    >>> print(f"Volume: {mask_data.get_volume_mm3()} mm³")
    >>> print(f"Space: {mask_data.space}")  # New property!
    >>> print(f"Resolution: {mask_data.resolution}mm")  # New property!
    """

    def __init__(
        self,
        mask_img: nib.Nifti1Image,
        metadata: dict[str, Any] | None = None,
        provenance: list[dict[str, Any]] | None = None,
        results: dict[str, Any] | None = None,
    ):
        # Validate lesion image
        validate_nifti_image(mask_img, require_3d=True, check_affine=True)

        # Validate binary mask
        mask_data = mask_img.get_fdata()
        unique_values = np.unique(mask_data)
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(
                "mask_img must be a binary mask with only 0 and 1 values.\n"
                f"Found unique values: {unique_values}\n"
                "Please binarize your lesion mask before creating MaskData."
            )

        # Store image
        self._mask_img = mask_img

        # Extract and validate affine
        self._affine = mask_img.affine.copy()
        validate_affine(self._affine)

        # Setup metadata
        if metadata is None:
            metadata = {}
        if "subject_id" not in metadata:
            metadata["subject_id"] = "sub-unknown"

        # Define supported template spaces (MNI152 variants only)
        SUPPORTED_TEMPLATE_SPACES = [
            "MNI152NLin6Asym",
            "MNI152NLin2009aAsym",
            "MNI152NLin2009cAsym",
        ]

        # Require explicit coordinate space and resolution specification
        if "space" not in metadata:
            raise ValueError(
                "metadata must contain 'space' key to specify coordinate space.\n"
                "This is required for spatial validation in analysis modules.\n"
                f"Supported spaces: {', '.join(SUPPORTED_TEMPLATE_SPACES)}\n"
                "Example: MaskData(img, metadata={{'subject_id': 'sub-001', 'space': 'MNI152NLin6Asym', 'resolution': 2}})"
            )

        # Validate space is in supported list
        if metadata["space"] not in SUPPORTED_TEMPLATE_SPACES:
            raise ValueError(
                f"Invalid space '{metadata['space']}'. "
                f"Supported spaces: {', '.join(SUPPORTED_TEMPLATE_SPACES)}\n"
                "Note: 'native' space is not supported. Use the actual template space instead.\n"
                "Example: MaskData(img, metadata={{'space': 'MNI152NLin6Asym', 'resolution': 2}})"
            )

        if "resolution" not in metadata:
            raise ValueError(
                "metadata must contain 'resolution' key (in mm).\n"
                "This is required for spatial validation and template matching.\n"
                "Common values: 1, 2 (for 1mm or 2mm resolution)\n"
                "Example: MaskData(img, metadata={{'subject_id': 'sub-001', 'space': 'MNI152NLin6Asym', 'resolution': 2}})"
            )

        self._metadata = metadata.copy()

        # Setup provenance (empty list for new objects)
        self._provenance = list(provenance) if provenance is not None else []

        # Setup results (nested dict: analysis -> result_name -> result_object)
        # Handle format migration: dict[str, list] -> dict[str, dict[str, Any]]
        if results is not None:
            self._results = self._normalize_results_format(dict(results))
        else:
            self._results = {}

        # Track coordinate space (extracted from metadata or provenance)
        self._coordinate_space = self._infer_coordinate_space()

    @staticmethod
    def _normalize_results_format(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Convert old results format to new nested dict format.

        Old format: dict[str, list] or dict[str, Any] with non-dict values
        New format: dict[str, dict[str, Any]]

        Parameters
        ----------
        results : dict
            Results in old or new format

        Returns
        -------
        dict[str, dict[str, Any]]
            Results in new nested dict format
        """
        normalized = {}
        for namespace, value in results.items():
            if isinstance(value, dict):
                # Already new format
                normalized[namespace] = value
            elif isinstance(value, list):
                # Old format: list of results -> dict with index keys
                normalized[namespace] = {f"result_{i}": v for i, v in enumerate(value)}
            else:
                # Single result object -> wrap in dict
                normalized[namespace] = {"default": value}
        return normalized

    @classmethod
    def from_nifti(
        cls,
        lesion_path: str | Path,
        space: str | None = None,
        resolution: float | None = None,
        anatomical_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "MaskData":
        """
        Load mask data from NIfTI file.

        Parameters
        ----------
        lesion_path : str or Path
            Path to mask NIfTI file.
        space : str, optional
            Coordinate space identifier (e.g., 'MNI152NLin6Asym').
            If not provided, will attempt auto-detection from image header/filename.
        resolution : float, optional
            Spatial resolution in millimeters (e.g., 1.0, 2.0).
            If not provided, will attempt auto-detection from image header/filename.
        anatomical_path : str or Path, optional
            Deprecated. No longer used.
        metadata : dict, optional
            Additional subject metadata (e.g., session info).
            'subject_id' auto-generated from filename if not provided.

        Returns
        -------
        MaskData
            Loaded mask data object.

        Raises
        ------
        FileNotFoundError
            If file path doesn't exist.
        NiftiLoadError
            If image fails to load or validate.
        ValueError
            If 'space' or 'resolution' cannot be determined.

        Examples
        --------
        >>> mask_data = MaskData.from_nifti(
        ...     "mask.nii.gz",
        ...     space="MNI152NLin6Asym",
        ...     resolution=2.0
        ... )
        >>> mask_data = MaskData.from_nifti(
        ...     "mask.nii.gz",
        ...     space="MNI152NLin6Asym",
        ...     resolution=2.0,
        ...     metadata={"subject_id": "sub-001", "session": "baseline"}
        ... )
        """
        lesion_path = Path(lesion_path)

        # Load lesion image
        try:
            mask_img = nib.load(lesion_path)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise NiftiLoadError(f"Failed to load mask from {lesion_path}: {e}") from e

        # Initialize metadata dict
        if metadata is None:
            metadata = {}
        else:
            metadata = metadata.copy()  # Don't modify caller's dict

        # Auto-generate subject_id from filename if not provided
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

        # Handle space and resolution (priority: kwargs > metadata > auto-detection)
        if space is not None:
            metadata["space"] = space
        if resolution is not None:
            metadata["resolution"] = resolution

        # If coordinate space information is still missing, attempt auto-detection
        if "space" not in metadata or "resolution" not in metadata:
            try:
                # Import lazily to avoid circular imports at module load time
                from .spaces import get_image_space

                detected = get_image_space(mask_img, filepath=lesion_path)
                if detected is not None:
                    # Populate metadata entries if not already present
                    if "space" not in metadata:
                        metadata["space"] = detected.identifier
                    if "resolution" not in metadata:
                        metadata["resolution"] = detected.resolution
            except Exception:
                # Detection is best-effort; leave metadata untouched and allow
                # __init__ to raise a helpful error if necessary.
                pass

        return cls(mask_img=mask_img, metadata=metadata)

    def validate(self) -> bool:
        """
        Validate data integrity.

        Checks that affine is invertible, image is 3D, and spatial properties
        are consistent.

        Returns
        -------
        bool
            True if all checks pass.

        Warns
        -----
        UserWarning
            If mask is empty or has suspicious properties.

        Raises
        ------
        ValidationError
            If critical invariants violated.

        Examples
        --------
        >>> mask_data.validate()
        True
        """
        # Validate images
        validate_nifti_image(self._mask_img, require_3d=True)

        # Validate affine
        validate_affine(self._affine)

        # Check lesion is not empty
        mask_data = self._mask_img.get_fdata()
        if not np.any(mask_data > 0):
            import warnings

            warnings.warn("Mask is empty (no non-zero voxels)", UserWarning, stacklevel=2)

        return True

    def copy(self) -> "MaskData":
        """
        Create a deep copy of this MaskData instance.

        Returns
        -------
        MaskData
            Independent copy with same data.

        Examples
        --------
        >>> mask_copy = mask_data.copy()
        >>> mask_copy is mask_data
        False
        """
        return MaskData(
            mask_img=self._mask_img,
            metadata=copy.deepcopy(self._metadata),
            provenance=copy.deepcopy(self._provenance),
            results=copy.deepcopy(self._results),
        )

    def get_coordinate_space(self) -> str:
        """
        Get current coordinate space from metadata.

        Returns
        -------
        str
            Coordinate space identifier (e.g., 'MNI152NLin6Asym').

        Examples
        --------
        >>> mask_data.get_coordinate_space()
        'MNI152NLin6Asym'
        """
        return self._coordinate_space

    def get_volume_mm3(self) -> float:
        """
        Calculate mask volume in cubic millimeters.

        Returns
        -------
        float
            Total mask volume (sum of non-zero voxels * voxel volume).

        Examples
        --------
        >>> volume = mask_data.get_volume_mm3()
        >>> print(f"Mask volume: {volume:.2f} mm³")
        """
        mask_data = self._mask_img.get_fdata()
        num_voxels = np.sum(mask_data > 0)

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
        >>> data = mask_data.to_dict()
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
    def from_dict(cls, data: dict[str, Any], mask_img: nib.Nifti1Image) -> "MaskData":
        """
        Deserialize from dictionary + NIfTI image.

        Parameters
        ----------
        data : dict
            Output from to_dict().
        mask_img : nibabel.Nifti1Image
            Mask image (loaded separately).

        Returns
        -------
        MaskData
            Reconstructed object.

        Examples
        --------
        >>> data = mask_data.to_dict()
        >>> mask_img = nib.load("mask.nii.gz")
        >>> mask_restored = MaskData.from_dict(data, mask_img)
        """
        return cls(
            mask_img=mask_img,
            metadata=data.get("metadata"),
            provenance=data.get("provenance"),
            results=data.get("results"),
        )

    def add_result(self, namespace: str, results: dict[str, Any]) -> "MaskData":
        """
        Create new MaskData with additional analysis results.

        This method follows immutability-by-convention: it returns a new instance
        with the updated results rather than modifying the current instance.

        Parameters
        ----------
        namespace : str
            Result namespace (e.g., 'LesionNetworkMapping', 'ParcelAggregation').
            Should match the analysis module name for clarity.
        results : dict[str, Any]
            Analysis results as a dict mapping result names to result objects.
            For single result: {"result_name": result_object}
            For multiple results (e.g., multi-atlas): {"Schaefer100": roi_result1, "Tian": roi_result2}

        Returns
        -------
        MaskData
            New instance with added results.

        Raises
        ------
        ValueError
            If namespace already exists in results.

        Examples
        --------
        >>> # Single result
        >>> results = {"default": VoxelMapResult(...)}
        >>> lesion_with_results = lesion.add_result("VolumeAnalysis", results)
        >>> "VolumeAnalysis" in lesion_with_results.results
        True
        >>>
        >>> # Multi-atlas results
        >>> results = {"Schaefer100": roi_result1, "Tian": roi_result2}
        >>> lesion_with_results = lesion.add_result("ParcelAggregation", results)
        >>> lesion_with_results.results["ParcelAggregation"]["Schaefer100"]
        ParcelData(...)
        """
        if namespace in self._results:
            raise ValueError(
                f"Result namespace '{namespace}' already exists. "
                f"Use a different namespace or create a new MaskData instance."
            )

        # Create new results dict with added namespace
        new_results = copy.deepcopy(self._results)
        new_results[namespace] = copy.deepcopy(results)

        # Return new instance
        return MaskData(
            mask_img=self._mask_img,
            metadata=copy.deepcopy(self._metadata),
            provenance=copy.deepcopy(self._provenance),
            results=new_results,
        )

    def add_provenance(self, record: dict[str, Any]) -> "MaskData":
        """
        Create new MaskData with additional provenance record.

        This method follows immutability-by-convention: it returns a new instance
        with the updated provenance history rather than modifying the current instance.

        Parameters
        ----------
        record : dict
            Provenance record (from create_provenance_record() or compatible dict).
            Must contain 'function', 'parameters', 'timestamp', and 'version' keys.

        Returns
        -------
        MaskData
            New instance with appended provenance.

        Raises
        ------
        ValueError
            If record is missing required fields.

        Examples
        --------
        >>> from lacuna.core.provenance import create_provenance_record
        >>> prov = create_provenance_record(
        ...     function="lacuna.analysis.RegionalDamage",
        ...     atlas_names=["Schaefer2018_100Parcels7Networks"],
        ...     version="0.1.0"
        ... )
        >>> result = mask_data.add_provenance(prov)
        >>> len(result.provenance) == len(mask_data.provenance) + 1
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
        return MaskData(
            mask_img=self._mask_img,
            metadata=copy.deepcopy(self._metadata),
            provenance=new_provenance,
            results=copy.deepcopy(self._results),
        )

    def _infer_coordinate_space(self) -> str:
        """
        Get coordinate space from metadata.

        Returns the coordinate space identifier (e.g., 'MNI152NLin6Asym').
        This is always present in metadata (validated in __init__).

        """
        return self._metadata["space"]

    # Read-only properties

    @property
    def mask_img(self) -> nib.Nifti1Image:
        """Binary mask image."""
        return self._mask_img

    @property
    def affine(self) -> np.ndarray:
        """4x4 affine transformation matrix (voxel to world)."""
        return self._affine.copy()  # Return copy to prevent modification

    @property
    def metadata(self) -> ImmutableDict:
        """
        Subject and session metadata (read-only view).

        Returns an immutable dictionary that prevents modifications with clear
        error messages. To update metadata, create a new MaskData instance
        with the desired metadata.

        Returns
        -------
        ImmutableDict
            Read-only view of metadata. Raises TypeError on modification attempts.

        Examples
        --------
        >>> mask_data.metadata["subject_id"]  # OK - reading
        'sub-001'
        >>> mask_data.metadata["new_key"] = "value"  # Raises TypeError
        Traceback (most recent call last):
            ...
        TypeError: Cannot modify MaskData.metadata - it is immutable.
        To update metadata, create a new MaskData instance instead.
        """
        return ImmutableDict(self._metadata, "metadata")

    @property
    def provenance(self) -> list[dict[str, Any]]:
        """Processing history (immutable view)."""
        return copy.deepcopy(self._provenance)  # Deep copy for nested dicts

    @property
    def results(self) -> dict[str, dict[str, Any]]:
        """Analysis results (immutable view).

        Returns dict mapping analysis namespace to result dict.
        Result dict maps result names to result objects.

        Access pattern: results['AnalysisName']['result_name']
        """
        return copy.deepcopy(self._results)  # Deep copy for nested structures

    @property
    def space(self) -> str:
        """
        Coordinate space identifier (e.g., 'MNI152NLin6Asym').

        Returns
        -------
        str
            The coordinate space from metadata.

        Examples
        --------
        >>> mask_data.space
        'MNI152NLin6Asym'
        """
        return self._metadata["space"]

    @property
    def resolution(self) -> float:
        """
        Spatial resolution in millimeters.

        Returns
        -------
        float
            The spatial resolution from metadata.

        Examples
        --------
        >>> mask_data.resolution
        2.0
        """
        return self._metadata["resolution"]

    def __getattr__(self, name: str) -> dict[str, Any]:
        """Enable attribute-based access to analysis results.

        Allows accessing results via `mask_data.AnalysisName` instead of
        `mask_data.results['AnalysisName']`.

        Parameters
        ----------
        name : str
            Analysis namespace (e.g., "ParcelAggregation", "RegionalDamage")

        Returns
        -------
        dict[str, Any]
            Result dictionary for the requested analysis

        Raises
        ------
        AttributeError
            If the attribute doesn't exist in results

        Examples
        --------
        >>> # After running ParcelAggregation:
        >>> mask_data.ParcelAggregation["Schaefer100"]
        ParcelData(...)
        >>> # Equivalent to:
        >>> mask_data.results["ParcelAggregation"]["Schaefer100"]
        ParcelData(...)
        """
        # Only intercept result namespace lookups, not internal attributes
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if name exists in results
        if name in self._results:
            return self._results[name]  # Return reference, not copy

        # Not found - raise AttributeError with helpful message
        available = ", ".join(self._results.keys()) if self._results else "none"
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'.\n"
            f"Available analysis results: {available}"
        )
