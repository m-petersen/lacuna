"""
Atlas aggregation module.

Provides flexible ROI-level aggregation of voxel-level maps across multiple
atlases. This is the core tool for extracting region-of-interest statistics
from lesion masks, connectivity maps, or any other spatial maps.

Examples
--------
>>> from ldk import LesionData
>>> from ldk.analysis import AtlasAggregation
>>>
>>> # Compute regional damage (overlap percentages)
>>> lesion = LesionData.from_nifti("lesion.nii.gz")
>>> analysis = AtlasAggregation(
...     atlas_dir="/data/atlases",
...     source="lesion_img",
...     aggregation="percent"
... )
>>> result = analysis.run(lesion)
>>> print(result.results["AtlasAggregation"])
>>>
>>> # Aggregate functional connectivity map by atlas regions
>>> from ldk.analysis import FunctionalNetworkMapping
>>> fnm = FunctionalNetworkMapping(connectome_path="gsp1000.h5")
>>> result = fnm.run(lesion)
>>>
>>> # Now aggregate the network map to atlas ROIs
>>> agg = AtlasAggregation(
...     atlas_dir="/data/atlases",
...     source="FunctionalNetworkMapping.network_map",
...     aggregation="mean"
... )
>>> final = agg.run(result)
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiLabelsMasker

from ldk.analysis.base import BaseAnalysis
from ldk.core.lesion_data import LesionData


class AtlasAggregation(BaseAnalysis):
    """
    Aggregate voxel-level maps to ROI-level statistics using atlases.

    This is a composable analysis that can:
    1. Compute regional damage from lesion masks (percent overlap, volume)
    2. Aggregate connectivity maps from network analyses (mean, sum, etc.)
    3. Extract any voxel-level map to atlas ROI statistics

    The analysis discovers all atlases in the specified directory and computes
    the specified aggregation method for each region in each atlas.

    Parameters
    ----------
    atlas_dir : str or Path
        Directory containing atlas files. Each atlas should have:
        - NIfTI file (.nii or .nii.gz)
        - Labels file with same base name + "_labels.txt" or ".txt"
    source : str, default="lesion_img"
        Source of data to aggregate. Options:
        - "lesion_img": Use the lesion mask directly
        - "anatomical_img": Use the anatomical image
        - "{AnalysisName}.{result_key}": Use result from previous analysis
          Example: "FunctionalNetworkMapping.network_map"
    aggregation : str, default="mean"
        Aggregation method to use. Options:
        - "mean": Mean value across ROI voxels
        - "sum": Sum of values across ROI voxels
        - "percent": Percentage of ROI voxels that are non-zero (for binary masks)
        - "volume": Volume (in mm³) of non-zero voxels in ROI
        - "median": Median value across ROI voxels
        - "std": Standard deviation across ROI voxels
    threshold : float, default=0.5
        For probabilistic atlases: minimum probability to consider a voxel
        as belonging to a region (0.0-1.0).

    Raises
    ------
    ValueError
        If atlas_dir doesn't exist, is empty, or aggregation method is invalid.
    FileNotFoundError
        If specified atlas directory doesn't exist.

    Notes
    -----
    - Both 3D and 4D atlases support automatic resampling to match source data
      spatial resolution via nilearn
    - 3D atlases: integer labels, use NiftiLabelsMasker with nearest-neighbor
      interpolation to preserve labels
    - 4D atlases: automatically detect binary (0/1) vs probabilistic (0.0-1.0)
      * Binary: use nearest-neighbor interpolation to preserve binary masks
      * Probabilistic: use continuous interpolation for probability values
    - For 3D atlases: regions defined by integer labels (automatically rounded)
    - For 4D atlases: each volume is a binary or probability map for one region
    - 4D probabilistic maps are thresholded at `threshold` parameter (default 0.5)
    - Results stored in LesionData.results["AtlasAggregation"] as dict
      mapping atlas_name_region_name -> aggregated_value

    Examples
    --------
    >>> # Regional damage analysis
    >>> analysis = AtlasAggregation(
    ...     atlas_dir="/data/atlases",
    ...     source="lesion_img",
    ...     aggregation="percent"
    ... )
    >>>
    >>> # Average functional connectivity per ROI
    >>> analysis = AtlasAggregation(
    ...     atlas_dir="/data/atlases",
    ...     source="FunctionalNetworkMapping.network_map",
    ...     aggregation="mean"
    ... )

    See Also
    --------
    RegionalDamage : Convenience wrapper for lesion overlap analysis
    BaseAnalysis : Parent class defining analysis interface
    """

    VALID_AGGREGATIONS = ["mean", "sum", "percent", "volume", "median", "std"]
    VALID_SOURCES = ["lesion_img", "anatomical_img"]

    def __init__(
        self,
        atlas_dir: str | Path,
        source: str = "lesion_img",
        aggregation: str = "mean",
        threshold: float = 0.5,
    ):
        """Initialize AtlasAggregation analysis."""
        super().__init__()

        self.atlas_dir = atlas_dir  # Store as provided (str or Path)
        self.source = source
        self.aggregation = aggregation
        self.threshold = threshold

        # Validate aggregation method
        if aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(
                f"Invalid aggregation method: '{aggregation}'\n"
                f"Valid options: {', '.join(self.VALID_AGGREGATIONS)}"
            )

        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

        # Will be populated in _validate_inputs
        self.atlases = []

    def _validate_inputs(self, lesion_data: LesionData) -> None:
        """
        Validate lesion data and discover atlases.

        Parameters
        ----------
        lesion_data : LesionData
            Lesion data to validate

        Raises
        ------
        ValueError
            If lesion data is invalid or source data not found
        FileNotFoundError
            If atlas directory doesn't exist
        """
        # Convert to Path for internal use
        atlas_dir = Path(self.atlas_dir)

        # Validate atlas directory exists
        if not atlas_dir.exists():
            raise FileNotFoundError(
                f"Atlas directory not found: {atlas_dir}\n"
                "Create the directory and add atlas files, or check the path."
            )

        if not atlas_dir.is_dir():
            raise ValueError(f"atlas_dir must be a directory, got file: {atlas_dir}")

        # Check that source data exists
        source_img = self._get_source_image(lesion_data)

        if source_img is None:
            raise ValueError(
                f"Source data not found: {self.source}\n"
                "Check that the source exists in LesionData.\n"
                f"Available sources: lesion_img, anatomical_img, "
                f"or results from previous analyses."
            )

        # Discover atlases in directory
        self.atlases = self._discover_atlases()

        if not self.atlases:
            raise ValueError(
                f"No valid atlases found in {self.atlas_dir}\n"
                "Expected atlas files: <name>.nii.gz and <name>_labels.txt"
            )

    def _run_analysis(self, lesion_data: LesionData) -> dict:
        """
        Compute ROI-level aggregation for all atlases.

        Parameters
        ----------
        lesion_data : LesionData
            Validated lesion data

        Returns
        -------
        dict
            Results dictionary with aggregated values per atlas region
        """
        # Get source data
        source_img = self._get_source_image(lesion_data)
        source_data = source_img.get_fdata()

        # Get voxel sizes for volume calculations
        voxel_volume_mm3 = np.abs(np.linalg.det(source_img.affine[:3, :3]))

        results = {}

        # Process each atlas
        for atlas_info in self.atlases:
            atlas_name = atlas_info["name"]
            atlas_img = nib.load(atlas_info["atlas_path"])
            labels = atlas_info["labels"]

            atlas_data = atlas_img.get_fdata()

            # Warn if resampling will occur
            atlas_shape = atlas_data.shape[:3]  # Handle 4D atlases
            if source_data.shape != atlas_shape:
                import warnings

                warnings.warn(
                    f"Atlas '{atlas_name}' will be resampled to match source data.\n"
                    f"Source shape: {source_data.shape}, Atlas shape: {atlas_shape}",
                    UserWarning,
                    stacklevel=2,
                )

            if atlas_data.ndim == 3:
                # 3D integer-labeled atlas - use nilearn NiftiLabelsMasker
                atlas_results = self._aggregate_3d_atlas(
                    source_img, atlas_img, labels, voxel_volume_mm3
                )
            elif atlas_data.ndim == 4:
                # 4D probabilistic atlas - use nilearn resampling
                atlas_results = self._aggregate_4d_atlas(
                    source_img, atlas_img, labels, voxel_volume_mm3
                )
            else:
                import warnings

                warnings.warn(
                    f"Skipping atlas '{atlas_name}': unexpected dimensions {atlas_data.ndim}D",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Add results with atlas name prefix
            for region_name, value in atlas_results.items():
                key = f"{atlas_name}_{region_name}"
                results[key] = value

        return results

    def _aggregate_3d_atlas(
        self,
        source_img: nib.Nifti1Image,
        atlas_img: nib.Nifti1Image,
        labels: dict[int, str],
        voxel_volume_mm3: float,
    ) -> dict[str, float]:
        """
        Aggregate source data for 3D integer-labeled atlas using nilearn.

        Uses nilearn's NiftiLabelsMasker for robust extraction with automatic
        resampling, masking, and efficient computation.

        Parameters
        ----------
        source_img : nib.Nifti1Image
            Source image to aggregate
        atlas_img : nib.Nifti1Image
            3D atlas with integer labels
        labels : dict[int, str]
            Mapping from region ID to region name
        voxel_volume_mm3 : float
            Volume of one voxel in mm³ (for volume aggregation)

        Returns
        -------
        dict[str, float]
            Mapping from region name to aggregated value
        """
        # Map our aggregation methods to nilearn strategies
        strategy_map = {
            "mean": "mean",
            "sum": "sum",
            "median": "median",
            "std": "standard_deviation",
            "percent": "mean",  # Will multiply by 100
            "volume": "sum",  # Will multiply by voxel_volume_mm3
        }

        if self.aggregation not in strategy_map:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        strategy = strategy_map[self.aggregation]

        # Create label names list (NiftiLabelsMasker expects ordered list)
        # Background (0) should not be included
        atlas_data = atlas_img.get_fdata()

        # Round atlas values to ensure integer labels
        # This handles edge cases where resampling or data type conversion
        # might introduce small floating point values
        atlas_data_rounded = np.round(atlas_data).astype(int)

        region_ids = np.unique(atlas_data_rounded)
        region_ids = region_ids[
            region_ids > 0
        ]  # Exclude background        # Build ordered list of label names
        label_names = [labels.get(int(rid), f"Region{int(rid)}") for rid in sorted(region_ids)]

        # Initialize NiftiLabelsMasker with appropriate settings
        masker = NiftiLabelsMasker(
            labels_img=atlas_img,
            labels=label_names,
            background_label=0,
            strategy=strategy,
            resampling_target="data",  # Resample atlas to match source data
            standardize=False,  # Don't normalize for static maps
            detrend=False,  # No detrending for static maps
            memory=None,  # No caching for now
            verbose=0,
        )

        # Extract values - nilearn expects 4D input (add time dimension if needed)
        if source_img.ndim == 3:
            # Add a dummy 4th dimension for time
            source_data_4d = source_img.get_fdata()[..., np.newaxis]
            source_img_4d = nib.Nifti1Image(source_data_4d, source_img.affine)
        else:
            source_img_4d = source_img

        # Transform: returns (n_timepoints, n_regions) array
        region_values = masker.fit_transform(source_img_4d)

        # Squeeze to get (n_regions,) for single timepoint
        if region_values.shape[0] == 1:
            region_values = region_values.squeeze(axis=0)

        # Apply post-processing based on aggregation type
        if self.aggregation == "percent":
            # Convert mean (0-1) to percentage (0-100)
            region_values = region_values * 100
        elif self.aggregation == "volume":
            # Convert count to volume (mm³)
            region_values = region_values * voxel_volume_mm3

        # Build results dict
        results = {}
        for label_name, value in zip(label_names, region_values, strict=True):
            results[label_name] = float(value)

        return results

    def _aggregate_4d_atlas(
        self,
        source_img: nib.Nifti1Image,
        atlas_img: nib.Nifti1Image,
        labels: dict[int, str],
        voxel_volume_mm3: float,
    ) -> dict[str, float]:
        """
        Aggregate source data for 4D atlas with automatic resampling.

        Uses nilearn's resample_to_img for automatic spatial alignment of atlas
        to source data. Detects binary vs probabilistic atlases and uses appropriate
        interpolation method ('nearest' for binary, 'continuous' for probabilistic).

        Parameters
        ----------
        source_img : nib.Nifti1Image
            Source image to aggregate
        atlas_img : nib.Nifti1Image
            4D atlas (x, y, z, n_regions) with binary or probability maps
        labels : dict[int, str]
            Mapping from region ID to region name
        voxel_volume_mm3 : float
            Volume of one voxel in mm³ (for volume aggregation)

        Returns
        -------
        dict[str, float]
            Mapping from region name to aggregated value
        """
        # Detect if atlas is binary (only 0s and 1s) or probabilistic
        atlas_data_orig = atlas_img.get_fdata()
        unique_values = np.unique(atlas_data_orig)
        is_binary = np.all(np.isin(unique_values, [0.0, 1.0]))

        # Use appropriate interpolation based on atlas type
        # 'nearest' for binary to preserve 0/1 values
        # 'continuous' for probabilistic to interpolate between probability values
        interpolation = "nearest" if is_binary else "continuous"

        # Resample atlas to match source data spatial resolution
        atlas_resampled = resample_to_img(
            atlas_img,
            source_img,
            interpolation=interpolation,
            copy=True,
            force_resample=True,
            copy_header=True,
        )

        source_data = source_img.get_fdata()
        atlas_data = atlas_resampled.get_fdata()

        results = {}
        n_regions = atlas_data.shape[3]

        for region_idx in range(n_regions):
            # Get probability map for this region
            prob_map = atlas_data[:, :, :, region_idx]

            # Threshold to create binary mask
            region_mask = prob_map >= self.threshold

            # Get values in this region
            region_values = source_data[region_mask]

            # Compute aggregation
            value = self._compute_aggregation(region_values, region_mask, voxel_volume_mm3)

            # Region ID for 4D atlases is the volume index + 1
            region_id = region_idx + 1
            region_name = labels.get(region_id, f"Region{region_id}")

            results[region_name] = value

        return results

    def _compute_aggregation(
        self,
        region_values: np.ndarray,
        region_mask: np.ndarray,
        voxel_volume_mm3: float,
    ) -> float:
        """
        Compute specified aggregation method on region values.

        Parameters
        ----------
        region_values : np.ndarray
            Values within the region
        region_mask : np.ndarray
            Boolean mask for the region
        voxel_volume_mm3 : float
            Volume of one voxel in mm³

        Returns
        -------
        float
            Aggregated value
        """
        if len(region_values) == 0:
            return 0.0

        if self.aggregation == "mean":
            return float(np.mean(region_values))

        elif self.aggregation == "sum":
            return float(np.sum(region_values))

        elif self.aggregation == "median":
            return float(np.median(region_values))

        elif self.aggregation == "std":
            return float(np.std(region_values))

        elif self.aggregation == "percent":
            # Percentage of ROI voxels that are non-zero
            n_total = np.sum(region_mask)
            n_nonzero = np.sum(region_values > 0)
            return (n_nonzero / n_total * 100) if n_total > 0 else 0.0

        elif self.aggregation == "volume":
            # Volume of non-zero voxels in mm³
            n_nonzero = np.sum(region_values > 0)
            return n_nonzero * voxel_volume_mm3

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

    def _get_source_image(self, lesion_data: LesionData) -> nib.Nifti1Image | None:
        """
        Get source image from LesionData based on source parameter.

        Parameters
        ----------
        lesion_data : LesionData
            Lesion data containing source

        Returns
        -------
        nib.Nifti1Image or None
            Source image, or None if not found
        """
        # Direct image sources
        if self.source == "lesion_img":
            return lesion_data.lesion_img
        elif self.source == "anatomical_img":
            return lesion_data.anatomical_img

        # Result from previous analysis: "AnalysisName.result_key"
        if "." in self.source:
            analysis_name, result_key = self.source.split(".", 1)

            if analysis_name in lesion_data.results:
                analysis_results = lesion_data.results[analysis_name]

                if result_key in analysis_results:
                    result = analysis_results[result_key]

                    # If it's a NIfTI image, return it
                    if isinstance(result, nib.Nifti1Image):
                        return result

                    # If it's a path, load it
                    if isinstance(result, (str, Path)):
                        result_path = Path(result)
                        if result_path.exists():
                            return nib.load(result_path)

        return None

    def _discover_atlases(self) -> list[dict]:
        """
        Discover atlas files in atlas directory.

        Looks for pairs of:
        - Atlas NIfTI: <name>.nii or <name>.nii.gz
        - Labels file: <name>_labels.txt or <name>.txt

        Returns
        -------
        list[dict]
            List of atlas info dicts with keys:
            - name: Atlas name
            - atlas_path: Path to NIfTI file
            - labels_path: Path to labels file
            - labels: Loaded labels dict
        """
        atlases = []
        atlas_dir = Path(self.atlas_dir)

        # Find all NIfTI files
        nifti_files = list(atlas_dir.glob("*.nii.gz")) + list(atlas_dir.glob("*.nii"))

        for nifti_path in nifti_files:
            # Determine base name (remove .nii.gz or .nii)
            if nifti_path.name.endswith(".nii.gz"):
                base_name = nifti_path.name[:-7]
            else:
                base_name = nifti_path.name[:-4]

            # Look for corresponding labels file
            labels_path = atlas_dir / f"{base_name}_labels.txt"
            if not labels_path.exists():
                labels_path = atlas_dir / f"{base_name}.txt"

            if not labels_path.exists():
                # Skip atlas without labels
                continue

            # Load labels
            try:
                labels = self._load_labels_file(labels_path)
            except Exception:
                # Skip atlas with invalid labels file
                continue

            atlases.append(
                {
                    "name": base_name,
                    "atlas_path": nifti_path,
                    "labels_path": labels_path,
                    "labels": labels,
                }
            )

        return atlases

    def _load_labels_file(self, labels_path: Path) -> dict[int, str]:
        """
        Load atlas labels from text file.

        Expected format: each line contains "region_id region_name"
        Lines starting with # are treated as comments.

        Parameters
        ----------
        labels_path : Path
            Path to labels text file

        Returns
        -------
        dict[int, str]
            Mapping from region ID to region name
        """
        labels = {}

        with open(labels_path) as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Parse "region_id region_name" format
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    try:
                        region_id = int(parts[0])
                        region_name = parts[1]
                        labels[region_id] = region_name
                    except ValueError:
                        # Skip lines that don't start with integer
                        continue
                elif len(parts) == 1:
                    # Handle case where there's just a region name (use line number as ID)
                    # This is for 4D atlases where regions are indexed by volume
                    try:
                        # Try to parse as just a name, assign sequential ID
                        region_name = parts[0]
                        region_id = len(labels) + 1
                        labels[region_id] = region_name
                    except Exception:
                        continue

        if not labels:
            raise ValueError(
                f"No valid labels found in {labels_path}\n"
                "Expected format: 'region_id region_name' per line"
            )

        return labels
