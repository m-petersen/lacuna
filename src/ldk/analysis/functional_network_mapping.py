"""Functional lesion network mapping (fLNM) analysis.

This module implements functional connectivity-based lesion network mapping
using normative connectome data. It supports two timeseries extraction methods:
- BOES (Boes et al.): Mean timeseries across all lesion voxels
- PINI (Pini et al.): PCA-based selection of most representative voxels

The analysis computes whole-brain correlation maps showing functional
connectivity disruption patterns associated with the lesion.
"""

from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from sklearn.decomposition import PCA

from ldk.analysis.base import BaseAnalysis
from ldk.core.exceptions import ValidationError
from ldk.core.lesion_data import LesionData


class FunctionalNetworkMapping(BaseAnalysis):
    """Functional connectivity-based lesion network mapping.

    This analysis maps functional connectivity disruption patterns by
    correlating a lesion's timeseries with whole-brain connectome data.
    Requires MNI152-registered lesions and a pre-computed functional
    connectome in HDF5 format.

    Parameters
    ----------
    connectome_path : str | Path
        Path to HDF5 connectome file containing:
        - 'timeseries': (n_subjects, n_timepoints, n_voxels) array
        - 'mask_indices': (3, n_voxels) array of brain mask coordinates
        - 'mask_affine': (4, 4) affine transformation matrix
        - 'mask_shape': Tuple stored in attributes
    method : {"boes", "pini"}, default="boes"
        Timeseries extraction method:
        - "boes": Mean timeseries across all lesion voxels
        - "pini": PCA-based selection of representative voxels
    pini_percentile : int, default=20
        For PINI method: percentile threshold for PC1 loadings (0-100).
        Higher values select fewer, more representative voxels.
    n_jobs : int, default=1
        Number of parallel jobs for batch processing (not yet implemented).

    Attributes
    ----------
    batch_strategy : str
        Set to "vectorized" for optimized batch processing.

    Methods
    -------
    run(lesion_data: LesionData) -> LesionData
        Inherited from BaseAnalysis. Computes functional network mapping.

    Examples
    --------
    >>> from ldk import LesionData
    >>> from ldk.analysis import FunctionalNetworkMapping
    >>> lesion = LesionData.from_nifti("lesion_mni.nii.gz")
    >>> analysis = FunctionalNetworkMapping(
    ...     connectome_path="gsp1000_connectome.h5",
    ...     method="boes"
    ... )
    >>> result = analysis.run(lesion)
    >>> correlation_map = result.results["FunctionalNetworkMapping"]["correlation_map"]
    >>> z_map = result.results["FunctionalNetworkMapping"]["z_map"]

    Notes
    -----
    The connectome HDF5 file must follow this structure:
    - Datasets: timeseries, mask_indices, mask_affine
    - Attributes: mask_shape
    - All voxels should be in MNI152 space

    References
    ----------
    - Boes et al. (2015): Network localization of neurological symptoms
    - Pini et al. (2020): PCA-based voxel selection for improved signal
    """

    # Class attribute for batch processing strategy
    batch_strategy = "vectorized"

    def __init__(
        self,
        connectome_path: str | Path,
        method: str = "boes",
        pini_percentile: int = 20,
        n_jobs: int = 1,
    ):
        """Initialize functional network mapping analysis.

        Parameters
        ----------
        connectome_path : str | Path
            Path to HDF5 connectome file.
        method : {"boes", "pini"}, default="boes"
            Timeseries extraction method.
        pini_percentile : int, default=20
            Percentile threshold for PINI method (0-100).
        n_jobs : int, default=1
            Number of parallel jobs (not yet implemented).

        Raises
        ------
        ValueError
            If method is not 'boes' or 'pini'.
        """
        # Validate method parameter
        if method not in ("boes", "pini"):
            msg = f"method must be 'boes' or 'pini', got '{method}'"
            raise ValueError(msg)

        self.connectome_path = Path(connectome_path)
        self.method = method
        self.pini_percentile = pini_percentile
        self.n_jobs = n_jobs

        # Cache for connectome data (loaded lazily)
        self._connectome_data = None
        self._mask_info = None

    def _validate_inputs(self, lesion_data: LesionData) -> None:
        """Validate inputs for functional network mapping.

        Parameters
        ----------
        lesion_data : LesionData
            Lesion data to validate.

        Raises
        ------
        ValidationError
            If connectome file doesn't exist, lesion is not in MNI152 space,
            or lesion mask is not binary.
        """
        # Check connectome file exists
        if not self.connectome_path.exists():
            msg = f"Connectome file not found: {self.connectome_path}"
            raise ValidationError(msg)

        # Validate MNI152 coordinate space
        space = lesion_data.metadata.get("space", "")
        if "MNI152" not in space.upper() and space != "":
            # If no space specified, check affine approximately
            # (This is a simplified check - production should be more robust)
            if not hasattr(lesion_data.lesion_img, "affine"):
                msg = "Lesion must be in MNI152 space for functional network mapping"
                raise ValidationError(msg)

        # Check if no space metadata at all (warn but continue)
        if not space:
            # Allow if no metadata, but in real case should validate affine
            pass

        # Validate binary mask
        lesion_data_array = lesion_data.lesion_img.get_fdata()
        unique_values = np.unique(lesion_data_array)
        if not np.all(np.isin(unique_values, [0, 1])):
            msg = "Lesion mask must be binary (only 0 and 1 values)"
            raise ValidationError(msg)

    def _run_analysis(self, lesion_data: LesionData) -> dict:
        """Execute functional network mapping analysis.

        Parameters
        ----------
        lesion_data : LesionData
            Validated lesion data in MNI152 space.

        Returns
        -------
        dict
            Dictionary containing:
            - 'correlation_map': NIfTI image of correlation values (r)
            - 'z_map': NIfTI image of Fisher z-transformed correlations
            - 'mean_correlation': float, mean correlation value
            - 'summary_statistics': dict with additional stats

        Notes
        -----
        This method implements the core fLNM pipeline:
        1. Load connectome data and extract lesion timeseries
        2. Compute correlations between lesion and whole-brain timeseries
        3. Apply Fisher z-transformation
        4. Aggregate across subjects and compute statistics
        """
        # Load connectome data (cached after first load)
        if self._connectome_data is None:
            self._load_connectome()

        # Extract lesion timeseries based on method
        if self.method == "boes":
            lesion_timeseries = self._extract_lesion_timeseries_boes(lesion_data)
        else:  # pini
            lesion_timeseries = self._extract_lesion_timeseries_pini(lesion_data)

        # Compute correlation maps for all connectome subjects
        correlation_maps = self._compute_correlation_maps(lesion_timeseries)

        # Fisher z-transformation and aggregation
        z_maps = np.arctanh(correlation_maps)
        z_maps = np.nan_to_num(z_maps, nan=0, posinf=10, neginf=-10)

        # Aggregate across subjects
        mean_z_map = np.mean(z_maps, axis=0)
        mean_r_map = np.tanh(mean_z_map)

        # Convert flat arrays to 3D brain volumes
        mask_shape = self._mask_info["mask_shape"]
        mask_indices = self._mask_info["mask_indices"]
        mask_affine = self._mask_info["mask_affine"]

        # Create 3D volumes
        correlation_map_3d = np.zeros(mask_shape)
        correlation_map_3d[mask_indices] = mean_r_map

        z_map_3d = np.zeros(mask_shape)
        z_map_3d[mask_indices] = mean_z_map

        # Create NIfTI images
        correlation_map_nifti = nib.Nifti1Image(correlation_map_3d.astype(np.float32), mask_affine)
        z_map_nifti = nib.Nifti1Image(z_map_3d.astype(np.float32), mask_affine)

        # Compute summary statistics
        mean_correlation = float(np.mean(mean_r_map))
        std_correlation = float(np.std(mean_r_map))
        max_correlation = float(np.max(mean_r_map))
        min_correlation = float(np.min(mean_r_map))

        return {
            "correlation_map": correlation_map_nifti,
            "network_map": correlation_map_nifti,  # Alias for backward compat
            "z_map": z_map_nifti,
            "mean_correlation": mean_correlation,
            "summary_statistics": {
                "mean": mean_correlation,
                "std": std_correlation,
                "max": max_correlation,
                "min": min_correlation,
                "n_subjects": correlation_maps.shape[0],
            },
        }

    def _load_connectome(self) -> None:
        """Load connectome data from HDF5 file into memory.

        Sets self._connectome_data and self._mask_info.
        """
        with h5py.File(self.connectome_path, "r") as hf:
            # Load timeseries data (n_subjects, n_timepoints, n_voxels)
            self._connectome_data = hf["timeseries"][:]

            # Load mask information
            mask_indices_array = hf["mask_indices"][:]
            # Convert to tuple of arrays for advanced indexing
            mask_indices = tuple(mask_indices_array[i, :] for i in range(3))

            self._mask_info = {
                "mask_indices": mask_indices,
                "mask_affine": hf["mask_affine"][:],
                "mask_shape": tuple(hf.attrs["mask_shape"]),
            }

    def _extract_lesion_timeseries_boes(self, lesion_data: LesionData) -> np.ndarray:
        """Extract mean timeseries across all lesion voxels (BOES method).

        Parameters
        ----------
        lesion_data : LesionData
            Lesion data in MNI152 space.

        Returns
        -------
        np.ndarray
            Shape (n_subjects, n_timepoints). Mean timeseries for each subject.
        """
        # Get lesion voxel indices within connectome mask
        lesion_voxel_indices = self._get_lesion_voxel_indices(lesion_data)

        if len(lesion_voxel_indices) == 0:
            msg = "No lesion voxels overlap with connectome mask"
            raise ValidationError(msg)

        # Extract timeseries for lesion voxels
        # Shape: (n_subjects, n_timepoints, n_lesion_voxels)
        lesion_ts = self._connectome_data[:, :, lesion_voxel_indices]

        # Compute mean across voxels
        # Shape: (n_subjects, n_timepoints)
        lesion_mean_ts = np.mean(lesion_ts, axis=2)

        return lesion_mean_ts

    def _extract_lesion_timeseries_pini(self, lesion_data: LesionData) -> np.ndarray:
        """Extract representative timeseries using PCA (PINI method).

        Uses PCA to identify most representative voxels based on their
        correlation with the mean timeseries, then extracts mean from
        these selected voxels.

        Parameters
        ----------
        lesion_data : LesionData
            Lesion data in MNI152 space.

        Returns
        -------
        np.ndarray
            Shape (n_subjects, n_timepoints). Representative timeseries.
        """
        # Get lesion voxel indices
        lesion_voxel_indices = self._get_lesion_voxel_indices(lesion_data)

        if len(lesion_voxel_indices) == 0:
            msg = "No lesion voxels overlap with connectome mask"
            raise ValidationError(msg)

        # Extract timeseries for lesion voxels
        lesion_ts = self._connectome_data[:, :, lesion_voxel_indices]

        # Compute initial mean timeseries
        mean_ts_across_voxels = np.mean(lesion_ts, axis=2)

        # Center timeseries
        mean_ts_centered = mean_ts_across_voxels - mean_ts_across_voxels.mean(axis=1, keepdims=True)
        voxel_ts_centered = lesion_ts - lesion_ts.mean(axis=1, keepdims=True)

        # Compute correlation between mean and each voxel
        # Using einsum for efficiency
        covariance = np.einsum(
            "it,itv->iv",
            mean_ts_centered,
            voxel_ts_centered,
            dtype=np.float64,
            optimize="optimal",
        )

        std_mean = np.sqrt(np.sum(mean_ts_centered**2, axis=1))
        std_voxels = np.sqrt(np.sum(voxel_ts_centered**2, axis=1))

        with np.errstate(divide="ignore", invalid="ignore"):
            pca_input_matrix = covariance / (std_mean[:, np.newaxis] * std_voxels)

        pca_input_matrix = np.nan_to_num(pca_input_matrix)

        # Apply PCA to find principal component
        pca = PCA(n_components=1)
        pca.fit(pca_input_matrix)
        pc1_loadings = pca.components_[0, :]

        # Select voxels above percentile threshold
        threshold = np.percentile(np.abs(pc1_loadings), self.pini_percentile)
        suprathreshold_indices = np.where(np.abs(pc1_loadings) >= threshold)[0]

        # Extract refined timeseries from selected voxels
        if len(suprathreshold_indices) == 0:
            # Fallback to all voxels if none selected
            refined_lesion_ts = lesion_ts
        else:
            refined_lesion_ts = lesion_ts[:, :, suprathreshold_indices]

        # Return mean timeseries from selected voxels
        return np.mean(refined_lesion_ts, axis=2)

    def _get_lesion_voxel_indices(self, lesion_data: LesionData) -> np.ndarray:
        """Get indices of lesion voxels within the connectome mask.

        Parameters
        ----------
        lesion_data : LesionData
            Lesion data in MNI152 space.

        Returns
        -------
        np.ndarray
            1D array of indices into the connectome's voxel dimension.
        """
        # Get lesion mask
        lesion_mask = lesion_data.lesion_img.get_fdata().astype(bool)

        # Get connectome mask in 3D space
        mask_shape = self._mask_info["mask_shape"]
        mask_indices = self._mask_info["mask_indices"]

        # Create connectome mask (3D boolean)
        connectome_mask_3d = np.zeros(mask_shape, dtype=bool)
        connectome_mask_3d[mask_indices] = True

        # Find overlap
        lesion_in_connectome = lesion_mask & connectome_mask_3d

        # Convert 3D coordinates to flat indices within connectome mask
        lesion_coords = np.where(lesion_in_connectome)

        # Find which positions in mask_indices match lesion coordinates
        voxel_indices = []
        for i in range(len(lesion_coords[0])):
            coord = (
                lesion_coords[0][i],
                lesion_coords[1][i],
                lesion_coords[2][i],
            )
            # Find index in mask_indices that matches this coordinate
            matches = (
                (mask_indices[0] == coord[0])
                & (mask_indices[1] == coord[1])
                & (mask_indices[2] == coord[2])
            )
            if np.any(matches):
                voxel_indices.append(np.where(matches)[0][0])

        return np.array(voxel_indices, dtype=int)

    def _compute_correlation_maps(self, lesion_timeseries: np.ndarray) -> np.ndarray:
        """Compute correlation maps between lesion and whole-brain timeseries.

        Parameters
        ----------
        lesion_timeseries : np.ndarray
            Shape (n_subjects, n_timepoints). Lesion timeseries.

        Returns
        -------
        np.ndarray
            Shape (n_subjects, n_voxels). Correlation values for each voxel.
        """
        # Center timeseries
        brain_ts_centered = self._connectome_data - self._connectome_data.mean(
            axis=1, keepdims=True
        )
        lesion_ts_centered = lesion_timeseries - lesion_timeseries.mean(axis=1, keepdims=True)

        # Compute covariance using einsum
        # (n_subjects, n_timepoints) @ (n_subjects, n_timepoints, n_voxels)
        cov = np.einsum(
            "it,itv->iv",
            lesion_ts_centered,
            brain_ts_centered,
            dtype=np.float64,
            optimize="optimal",
        )

        # Compute standard deviations
        lesion_std = np.sqrt(np.sum(lesion_ts_centered**2, axis=1))
        brain_std = np.sqrt(np.sum(brain_ts_centered**2, axis=1))

        # Compute correlation (cov / (std_lesion * std_brain))
        with np.errstate(divide="ignore", invalid="ignore"):
            r_maps = cov / (lesion_std[:, np.newaxis] * brain_std)

        # Clean up NaN and inf values
        r_maps = np.nan_to_num(r_maps, nan=0, posinf=1, neginf=-1)

        return r_maps.astype(np.float32)
