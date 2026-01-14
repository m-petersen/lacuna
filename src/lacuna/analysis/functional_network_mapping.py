"""Functional lesion network mapping (fLNM) analysis.
from __future__ import annotations


This module implements functional connectivity-based lesion network mapping
using normative connectome data. It supports two timeseries extraction methods:
- BOES (Boes et al.): Mean timeseries across all lesion voxels
- PINI (Pini et al.): PCA-based selection of most representative voxels

The analysis computes whole-brain correlation maps showing functional
connectivity disruption patterns associated with the lesion.

Memory-efficient processing:
- Supports single HDF5 file or directory with multiple batched HDF5 files
- Processes connectome batches sequentially to minimize memory footprint
- Accumulates statistics across batches for final aggregation
"""

from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import nibabel as nib
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from lacuna.analysis.base import BaseAnalysis
from lacuna.assets.connectomes import (
    list_functional_connectomes,
    load_functional_connectome,
)
from lacuna.core.data_types import ScalarMetric, VoxelMap
from lacuna.core.exceptions import ValidationError
from lacuna.core.subject_data import SubjectData
from lacuna.utils.logging import ConsoleLogger

if TYPE_CHECKING:
    from lacuna.core.data_types import AnalysisResult


class FunctionalNetworkMapping(BaseAnalysis):
    """Functional connectivity-based lesion network mapping.

    This analysis maps functional connectivity disruption patterns by
    correlating a lesion's timeseries with whole-brain connectome data.
    Requires a pre-computed functional connectome in HDF5 format.

    Computation space is determined dynamically from the registered connectome's
    metadata (space and resolution). Input masks are automatically transformed to
    match the connectome space before analysis.

    Memory-efficient batch processing: Connectomes can be stored as single
    HDF5 files or directories with multiple batched files. All batches are
    processed sequentially to minimize memory usage.

    Parameters
    ----------
    connectome_name : str
        Name of registered functional connectome (e.g., "GSP1000").
        Use list_functional_connectomes() to see available connectomes.
        The connectome must be pre-registered via register_functional_connectome().
        Each HDF5 file must contain:
        - 'timeseries': (n_subjects, n_timepoints, n_voxels) array
        - 'mask_indices': (3, n_voxels) or (n_voxels, 3) brain mask coordinates
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
    run(mask_data: SubjectData) -> SubjectData
        Inherited from BaseAnalysis. Computes functional network mapping.

    Examples
    --------
    >>> from lacuna import SubjectData
    >>> from lacuna.analysis import FunctionalNetworkMapping
    >>> from lacuna.assets.connectomes import (
    ...     list_functional_connectomes,
    ...     register_functional_connectome,
    ... )
    >>>
    >>> # Register a connectome (do this once)
    >>> register_functional_connectome(
    ...     name="GSP1000",
    ...     space="MNI152NLin6Asym",
    ...     resolution=2.0,
    ...     data_path="/data/gsp1000_connectome.h5",
    ...     n_subjects=1000,
    ...     description="GSP1000 voxel-wise connectome"
    ... )
    >>>
    >>> # List available connectomes
    >>> list_functional_connectomes()
    >>>
    >>> # Use registered connectome
    >>> lesion = SubjectData.from_nifti("lesion_mni.nii.gz")
    >>> analysis = FunctionalNetworkMapping(
    ...     connectome_name="GSP1000",
    ...     method="boes"
    ... )
    >>> result = analysis.run(lesion)
    >>> correlation_map = result.results["FunctionalNetworkMapping"]["rmap"]
    >>> z_map = result.results["FunctionalNetworkMapping"]["zmap"]

    Notes
    -----
    The connectome HDF5 file(s) must follow this structure:
    - Datasets: timeseries, mask_indices, mask_affine
    - Attributes: mask_shape
    - All voxels should be in MNI152 space

    When using multiple batch files, all files must have the same mask
    structure (mask_indices, mask_affine, mask_shape).

    References
    ----------
    - Boes et al. (2015): https://doi.org/10.1093/brain/awv228
    - Pini et al. (2020): https://doi.org/10.1093/braincomms/fcab259
    """

    # Class attribute for batch processing strategy
    batch_strategy = "vectorized"

    def __init__(
        self,
        connectome_name: str,
        method: str = "boes",
        pini_percentile: int = 20,
        n_jobs: int = 1,
        verbose: bool = False,
        compute_p_map: bool = True,
        fdr_alpha: float | None = 0.05,
        t_threshold: float | None = None,
        return_in_input_space: bool = True,
        keep_intermediate: bool = False,
    ):
        """Initialize functional network mapping analysis.

        Parameters
        ----------
        connectome_name : str
            Name of registered functional connectome (e.g., "GSP1000").
            Use list_functional_connectomes() to see available options.
        method : {"boes", "pini"}, default="boes"
            Timeseries extraction method.
        pini_percentile : int, default=20
            Percentile threshold for PINI method (0-100).
        n_jobs : int, default=1
            Number of parallel jobs (not yet implemented).
        verbose : bool, default=False
            If True, print progress messages. If False, run silently.
        compute_p_map : bool, default=True
            If True, compute p-value map (two-tailed) from t-statistics.
        fdr_alpha : float, optional, default=0.05
            If provided, compute FDR-corrected p-value map using Benjamini-Hochberg
            procedure at the specified alpha level. Set to None to disable FDR correction.
            Requires compute_p_map=True.
        t_threshold : float, optional
            If provided, create binary mask of voxels with |t| > threshold.
        return_in_input_space : bool, default=True
            If True, transform VoxelMap outputs back to the original input mask space.
            If False, outputs remain in the connectome space (e.g., MNI152NLin6Asym @ 2mm).
            Requires input SubjectData to have valid space/resolution metadata.
        keep_intermediate : bool, default=False
            If True, include intermediate results (e.g., warped mask images)
            in the output. Useful for debugging and quality control.

        Raises
        ------
        ValueError
            If method is not 'boes' or 'pini'.
        KeyError
            If connectome_name not found in registry.
        """
        super().__init__(verbose=verbose, keep_intermediate=keep_intermediate)

        # Validate method parameter
        if method not in ("boes", "pini"):
            msg = f"method must be 'boes' or 'pini', got '{method}'"
            raise ValueError(msg)

        # Load connectome from registry
        try:
            connectome = load_functional_connectome(connectome_name)
        except KeyError as e:
            available = [c.name for c in list_functional_connectomes()]
            raise KeyError(
                f"Connectome '{connectome_name}' not found in registry. "
                f"Available connectomes: {', '.join(available)}. "
                f"Use register_functional_connectome() to add new connectomes."
            ) from e

        # Store connectome information
        self.connectome_name = connectome_name
        self.connectome_path = connectome.data_path
        self.output_space = connectome.metadata.space
        self.output_resolution = connectome.metadata.resolution
        self._is_batch_dir = connectome.is_batched

        # Set TARGET_SPACE dynamically from connectome (used by BaseAnalysis._ensure_target_space)
        self.TARGET_SPACE = connectome.metadata.space
        self.TARGET_RESOLUTION = connectome.metadata.resolution

        # Analysis parameters
        self.method = method
        self.pini_percentile = pini_percentile
        self.n_jobs = n_jobs
        self.compute_p_map = compute_p_map
        self.fdr_alpha = fdr_alpha
        self.t_threshold = t_threshold
        self.return_in_input_space = return_in_input_space

        # Initialize logger
        self.logger = ConsoleLogger(verbose=verbose, width=70)

        # Internal state
        self._batch_files = None
        self._mask_info = None

    def _get_connectome_files(self) -> list[Path]:
        """Get list of HDF5 connectome files to process.

        Returns
        -------
        list[Path]
            List of HDF5 file paths, sorted alphabetically.

        Raises
        ------
        ValidationError
            If no HDF5 files found.
        """
        if self._batch_files is not None:
            return self._batch_files

        if self._is_batch_dir:
            # Find all .h5 files in directory
            h5_files = sorted(self.connectome_path.glob("*.h5"))
            if not h5_files:
                msg = f"No HDF5 files found in directory: {self.connectome_path}"
                raise ValidationError(msg)
            self._batch_files = h5_files
        else:
            # Single file
            self._batch_files = [self.connectome_path]

        return self._batch_files

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """Validate inputs for functional network mapping.

        This method validates that the mask data is ready for FNM analysis.
        By the time this is called, BaseAnalysis.run() has already transformed
        the mask to TARGET_SPACE (the connectome space) via _ensure_target_space().

        Parameters
        ----------
        mask_data : SubjectData
            Mask data to validate (already transformed to connectome space).

        Raises
        ------
        ValidationError
            If connectome file(s) don't exist or mask space doesn't match
            the expected connectome space.

        Notes
        -----
        Binary mask validation is handled by SubjectData.__init__, so we don't
        need to duplicate that check here.

        Space transformation is handled by BaseAnalysis._ensure_target_space(),
        so by the time we get here, mask_data.space should equal self.TARGET_SPACE.
        """
        # Check connectome path exists
        if not self.connectome_path.exists():
            msg = f"Connectome path not found: {self.connectome_path}"
            raise ValidationError(msg)

        # Validate that we have connectome files
        _ = self._get_connectome_files()  # Raises ValidationError if no files found

        # Validate coordinate space matches connectome space
        # (should already be transformed by _ensure_target_space)
        if mask_data.space != self.TARGET_SPACE:
            msg = (
                f"Mask space '{mask_data.space}' does not match connectome space "
                f"'{self.TARGET_SPACE}'. This is unexpected - space transformation "
                f"should have been handled by BaseAnalysis._ensure_target_space()."
            )
            raise ValidationError(msg)

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, "AnalysisResult"]:
        """Execute functional network mapping analysis.

        Processes connectome batches sequentially to minimize memory usage.
        Accumulates z-transformed correlation maps across all batches and
        performs final aggregation.

        Parameters
        ----------
        mask_data : SubjectData
            Validated lesion data in MNI152 space.

        Returns
        -------
        dict[str, AnalysisResult]
            Dictionary containing:
            - 'correlation_map': VoxelMapResult for correlation (r values)
            - 'z_map': VoxelMapResult for Fisher z-transformed
            - 't_map': VoxelMapResult for t-statistics
            - 't_threshold_map': VoxelMapResult (if t_threshold provided)
            - 'summary_statistics': MiscResult for summary statistics

        Notes
        -----
        This method implements a memory-efficient FNM pipeline:
        1. Load mask info once (shared across batches)
        2. For each connectome batch:
           a. Load timeseries data
           b. Extract lesion timeseries
           c. Compute correlation maps
           d. Fisher z-transform and accumulate
           e. Free memory
        3. Aggregate statistics across all batches
        4. Convert to 3D volumes and create NIfTI images
        """
        # Load mask information once
        if self._mask_info is None:
            self.logger.info("Loading mask information from connectome...")
            self._load_mask_info()
            mask_shape = self._mask_info["mask_shape"]
            n_voxels = len(self._mask_info["mask_indices"][0])
            self.logger.success(
                "Mask loaded", details={"shape": str(mask_shape), "n_voxels": n_voxels}
            )

        # Get mask voxel indices and resampled mask (computed once, reused for all batches)
        self.logger.info("Computing mask-connectome overlap...")
        mask_voxel_indices, resampled_mask_img = self._get_mask_voxel_indices(mask_data)

        if len(mask_voxel_indices) == 0:
            msg = "No mask voxels overlap with connectome mask"
            raise ValidationError(msg)

        self.logger.success(f"Found {len(mask_voxel_indices):,} overlapping mask voxels")

        # Store resampled mask as analysis_mask if keep_intermediate=True
        # This is stored as an instance variable to be added to results at the end
        self._resampled_mask_img = resampled_mask_img

        # Initialize accumulators for batch processing
        all_z_maps = []
        total_subjects = 0

        # Process each connectome batch sequentially
        connectome_files = self._get_connectome_files()
        n_batches = len(connectome_files)

        if n_batches == 1:
            self.logger.info("Processing connectome...")
        else:
            self.logger.info(f"Processing {n_batches} connectome batches...")

        for batch_idx, batch_file in enumerate(connectome_files, 1):
            self.logger.progress(f"Loading {batch_file.name}", current=batch_idx, total=n_batches)

            # Load this batch's timeseries
            with h5py.File(batch_file, "r") as hf:
                batch_timeseries = hf["timeseries"][:]
                batch_n_subjects = batch_timeseries.shape[0]

            self.logger.info(
                f"Extracting mask timeseries ({batch_n_subjects} subjects)", indent_level=1
            )

            # Extract mask timeseries for this batch
            if self.method == "boes":
                mask_ts = self._extract_lesion_timeseries_boes_batch(
                    batch_timeseries, mask_voxel_indices
                )
            else:  # pini
                mask_ts = self._extract_lesion_timeseries_pini_batch(
                    batch_timeseries, mask_voxel_indices
                )

            self.logger.info("Computing correlation maps", indent_level=1)

            # Compute correlation maps for this batch
            batch_r_maps = self._compute_correlation_maps_batch(mask_ts, batch_timeseries)

            self.logger.info("Applying Fisher z-transform", indent_level=1)

            # Fisher z-transform
            batch_z_maps = np.arctanh(batch_r_maps)
            batch_z_maps = np.nan_to_num(batch_z_maps, nan=0, posinf=10, neginf=-10)

            # Accumulate
            all_z_maps.append(batch_z_maps)
            total_subjects += batch_timeseries.shape[0]

            # Explicitly free memory
            del batch_timeseries, mask_ts, batch_r_maps, batch_z_maps

        self.logger.info(f"Aggregating results across {total_subjects} subjects...")

        # Concatenate all z-maps
        all_z_maps_array = np.vstack(all_z_maps)

        # Aggregate across all subjects
        mean_z_map = np.mean(all_z_maps_array, axis=0)
        mean_r_map = np.tanh(mean_z_map)

        # Compute t-statistics (always computed)
        self.logger.info("Computing t-statistics...")
        std_z_map = np.std(all_z_maps_array, axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            std_error_map_flat = std_z_map / np.sqrt(total_subjects)
            t_map_flat = np.divide(
                mean_z_map,
                std_error_map_flat,
                where=(std_error_map_flat != 0),
            )
            t_map_flat = np.nan_to_num(t_map_flat)

        # Compute p-values from t-statistics (two-tailed)
        p_map_flat = None
        p_fdr_map_flat = None
        if self.compute_p_map:
            self.logger.info("Computing p-values...")
            df = total_subjects - 1  # degrees of freedom
            # Two-tailed p-value: 2 * (1 - CDF(|t|))
            p_map_flat = 2 * stats.t.sf(np.abs(t_map_flat), df)
            p_map_flat = np.nan_to_num(p_map_flat, nan=1.0)

            # Compute FDR-corrected p-values if requested
            if self.fdr_alpha is not None:
                self.logger.info(f"Computing FDR-corrected p-values (alpha={self.fdr_alpha})...")
                # Benjamini-Hochberg FDR correction
                n_voxels = len(p_map_flat)
                sorted_indices = np.argsort(p_map_flat)
                sorted_pvals = p_map_flat[sorted_indices]

                # Compute FDR-adjusted p-values
                # p_adj[i] = min(p[i] * n / (rank[i]), 1.0)
                ranks = np.arange(1, n_voxels + 1)
                adjusted = sorted_pvals * n_voxels / ranks

                # Enforce monotonicity (cumulative minimum from the right)
                adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
                adjusted = np.clip(adjusted, 0, 1)

                # Map back to original order
                p_fdr_map_flat = np.empty_like(p_map_flat)
                p_fdr_map_flat[sorted_indices] = adjusted

                # Count significant voxels
                n_significant_fdr = np.sum(p_fdr_map_flat < self.fdr_alpha)
                pct_significant_fdr = (n_significant_fdr / n_voxels) * 100
                self.logger.info(
                    f"Found {n_significant_fdr:,} voxels ({pct_significant_fdr:.2f}%) "
                    f"significant at FDR q < {self.fdr_alpha}",
                    indent_level=1,
                )

        # Free memory
        del all_z_maps, all_z_maps_array

        self.logger.info("Creating 3D output volumes...")

        # Convert flat arrays to 3D brain volumes
        mask_shape = self._mask_info["mask_shape"]
        mask_indices = self._mask_info["mask_indices"]
        mask_affine = self._mask_info["mask_affine"]

        # Create 3D volumes
        # mask_indices is tuple of (x_coords, y_coords, z_coords)
        correlation_map_3d = np.zeros(mask_shape, dtype=np.float32)
        correlation_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = mean_r_map

        z_map_3d = np.zeros(mask_shape, dtype=np.float32)
        z_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = mean_z_map

        # Create NIfTI images
        correlation_map_nifti = nib.Nifti1Image(correlation_map_3d, mask_affine)
        z_map_nifti = nib.Nifti1Image(z_map_3d, mask_affine)

        # Create t-map (always computed)
        t_map_3d = np.zeros(mask_shape, dtype=np.float32)
        t_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = t_map_flat
        t_map_nifti = nib.Nifti1Image(t_map_3d, mask_affine)

        # Create thresholded binary map if threshold provided
        t_threshold_map_nifti = None
        if self.t_threshold is not None:
            self.logger.info(f"Creating thresholded map (|t| > {self.t_threshold})")
            t_threshold_mask = np.abs(t_map_flat) > self.t_threshold
            n_significant = np.sum(t_threshold_mask)
            pct_significant = (n_significant / len(t_map_flat)) * 100

            threshold_map_3d = np.zeros(mask_shape, dtype=np.uint8)
            threshold_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = (
                t_threshold_mask.astype(np.uint8)
            )
            t_threshold_map_nifti = nib.Nifti1Image(threshold_map_3d, mask_affine)

            self.logger.info(
                f"Found {n_significant:,} voxels ({pct_significant:.2f}%) above threshold",
                indent_level=1,
            )

        # Create p-value maps if computed
        p_map_nifti = None
        p_fdr_map_nifti = None
        if self.compute_p_map and p_map_flat is not None:
            p_map_3d = np.zeros(mask_shape, dtype=np.float32)
            p_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = p_map_flat
            # Set background to 1.0 (not significant) for clarity
            p_map_nifti = nib.Nifti1Image(p_map_3d, mask_affine)

            # Create FDR-corrected p-value map if computed
            if p_fdr_map_flat is not None:
                p_fdr_map_3d = np.zeros(mask_shape, dtype=np.float32)
                p_fdr_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = p_fdr_map_flat
                p_fdr_map_nifti = nib.Nifti1Image(p_fdr_map_3d, mask_affine)

        # Compute summary statistics
        mean_correlation = float(np.mean(mean_r_map))
        std_correlation = float(np.std(mean_r_map))
        max_correlation = float(np.max(mean_r_map))
        min_correlation = float(np.min(mean_r_map))

        # Success summary
        summary_details = {
            "mean_correlation": mean_correlation,
            "std_correlation": std_correlation,
            "correlation_range": f"[{min_correlation:.4f}, {max_correlation:.4f}]",
            "n_subjects": total_subjects,
        }

        if t_map_nifti is not None:
            t_min = float(np.min(t_map_flat))
            t_max = float(np.max(t_map_flat))
            summary_details["t_range"] = f"[{t_min:.2f}, {t_max:.2f}]"

        self.logger.success("Analysis complete", details=summary_details)

        # Create result objects as dict with descriptive keys
        results = {}

        # Correlation map (r values)
        correlation_result = VoxelMap(
            name="rmap",
            data=correlation_map_nifti,
            space=self.output_space,
            resolution=self.output_resolution,
            metadata={
                "method": self.method,
                "n_subjects": total_subjects,
                "n_batches": len(connectome_files),
                "statistic": "pearson_correlation_coefficient",
            },
        )
        results["rmap"] = correlation_result

        # Z-map (Fisher z-transformed correlations)
        z_result = VoxelMap(
            name="zmap",
            data=z_map_nifti,
            space=self.output_space,
            resolution=self.output_resolution,
            metadata={
                "method": self.method,
                "n_subjects": total_subjects,
                "n_batches": len(connectome_files),
                "statistic": "fisher_z",
            },
        )
        results["zmap"] = z_result

        # Summary statistics
        summary_dict = {
            "mean": mean_correlation,
            "std": std_correlation,
            "max": max_correlation,
            "min": min_correlation,
            "n_subjects": total_subjects,
            "n_batches": len(connectome_files),
        }

        # Add t-map results if computed
        if t_map_nifti is not None:
            t_result = VoxelMap(
                name="tmap",
                data=t_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "n_subjects": total_subjects,
                    "statistic": "t_statistic",
                },
            )
            results["tmap"] = t_result
            summary_dict["t_min"] = float(np.min(t_map_flat))
            summary_dict["t_max"] = float(np.max(t_map_flat))

        if t_threshold_map_nifti is not None:
            threshold_result = VoxelMap(
                name="tthresholdmap",
                data=t_threshold_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "threshold": self.t_threshold,
                    "statistic": "thresholded_t",
                },
            )
            results["tthresholdmap"] = threshold_result
            summary_dict["n_significant_voxels"] = int(n_significant)
            summary_dict["pct_significant_voxels"] = float(pct_significant)

        # Add p-value map if computed
        if p_map_nifti is not None:
            p_result = VoxelMap(
                name="pmap",
                data=p_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "n_subjects": total_subjects,
                    "statistic": "p_value_two_tailed",
                    "degrees_of_freedom": total_subjects - 1,
                },
            )
            results["pmap"] = p_result
            summary_dict["p_min"] = float(np.min(p_map_flat))
            summary_dict["p_max"] = float(np.max(p_map_flat))

        # Add FDR-corrected p-value map if computed
        if p_fdr_map_nifti is not None:
            p_fdr_result = VoxelMap(
                name="pfdrmap",
                data=p_fdr_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "n_subjects": total_subjects,
                    "statistic": "p_value_fdr_corrected",
                    "fdr_alpha": self.fdr_alpha,
                    "correction_method": "benjamini_hochberg",
                },
            )
            results["pfdrmap"] = p_fdr_result
            summary_dict["n_significant_fdr"] = int(n_significant_fdr)
            summary_dict["pct_significant_fdr"] = float(pct_significant_fdr)
            summary_dict["fdr_alpha"] = self.fdr_alpha

            # Create binary significance mask at FDR threshold
            fdr_sig_mask = (p_fdr_map_flat < self.fdr_alpha).astype(np.uint8)
            fdr_sig_map_3d = np.zeros(mask_shape, dtype=np.uint8)
            fdr_sig_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = fdr_sig_mask
            fdr_sig_map_nifti = nib.Nifti1Image(fdr_sig_map_3d, mask_affine)

            fdr_threshold_result = VoxelMap(
                name="pfdrthresholdmap",
                data=fdr_sig_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "n_subjects": total_subjects,
                    "statistic": "fdr_significant_binary",
                    "fdr_alpha": self.fdr_alpha,
                    "n_significant": n_significant_fdr,
                },
            )
            results["pfdrthresholdmap"] = fdr_threshold_result

        # Add summary statistics as ScalarMetric
        summary_result = ScalarMetric(
            name="summarystatistics",
            data=summary_dict,
            metadata={
                "method": self.method,
                "n_subjects": total_subjects,
            },
        )
        results["summarystatistics"] = summary_result

        # Add analysis_mask if keep_intermediate=True
        # This is the mask that was actually used for computation (resampled to connectome grid)
        if self.keep_intermediate and hasattr(self, "_resampled_mask_img"):
            analysis_mask = VoxelMap(
                name="analysis_mask",
                data=self._resampled_mask_img,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "description": "Mask resampled to connectome space for analysis",
                    "original_space": mask_data.metadata.get(
                        "_original_input_space", mask_data.space
                    ),
                    "original_resolution": mask_data.metadata.get(
                        "_original_input_resolution", mask_data.resolution
                    ),
                    "analysis_space": self.output_space,
                    "analysis_resolution": self.output_resolution,
                },
            )
            results["analysis_mask"] = analysis_mask

        # Transform VoxelMap results back to input space if requested
        if self.return_in_input_space:
            results = self._transform_results_to_input_space(results, mask_data)

        self.logger.success(f"Analysis complete ({len(results)} results)")
        return results

    def _load_mask_info(self) -> tuple:
        """Load mask information from first connectome file.

        Mask info is shared across all batch files and only needs to be
        loaded once. Sets self._mask_info.

        Returns
        -------
        tuple
            (mask_indices, mask_affine, mask_shape) tuple

        Raises
        ------
        ValidationError
            If HDF5 file structure is invalid.
        """
        connectome_files = self._get_connectome_files()
        first_file = connectome_files[0]

        with h5py.File(first_file, "r") as hf:
            # Load mask information
            mask_indices_array = hf["mask_indices"][:]

            # Handle both (3, n_voxels) and (n_voxels, 3) formats
            # Store as tuple of 1D arrays for indexing
            if mask_indices_array.shape[0] == 3:
                # Shape is (3, n_voxels) - correct format
                mask_indices = tuple(mask_indices_array[i, :].astype(int) for i in range(3))
            else:
                # Shape is (n_voxels, 3) - transpose it
                mask_indices = tuple(mask_indices_array[:, i].astype(int) for i in range(3))

            mask_affine = hf["mask_affine"][:]
            mask_shape = tuple(hf.attrs["mask_shape"])

            self._mask_info = {
                "mask_indices": mask_indices,
                "mask_affine": mask_affine,
                "mask_shape": mask_shape,
            }

            return mask_indices, mask_affine, mask_shape

    def _extract_lesion_timeseries_boes_batch(
        self, batch_timeseries: np.ndarray, lesion_voxel_indices: np.ndarray
    ) -> np.ndarray:
        """Extract mean timeseries across all lesion voxels (BOES method).

        Memory-efficient version that works on a single batch.

        Parameters
        ----------
        batch_timeseries : np.ndarray
            Shape (n_subjects, n_timepoints, n_voxels). Connectome batch data.
        lesion_voxel_indices : np.ndarray
            1D array of voxel indices within the connectome mask.

        Returns
        -------
        np.ndarray
            Shape (n_subjects, n_timepoints). Mean timeseries for each subject.
        """
        # Extract timeseries for lesion voxels
        # Shape: (n_subjects, n_timepoints, n_lesion_voxels)
        lesion_ts = batch_timeseries[:, :, lesion_voxel_indices]

        # Compute mean across voxels
        # Shape: (n_subjects, n_timepoints)
        lesion_mean_ts = np.mean(lesion_ts, axis=2)

        return lesion_mean_ts

    def _extract_lesion_timeseries_pini_batch(
        self, batch_timeseries: np.ndarray, lesion_voxel_indices: np.ndarray
    ) -> np.ndarray:
        """Extract representative timeseries using PCA (PINI method).

        Memory-efficient version that works on a single batch.

        Uses PCA to identify most representative voxels based on their
        correlation with the mean timeseries, then extracts mean from
        these selected voxels.

        Parameters
        ----------
        batch_timeseries : np.ndarray
            Shape (n_subjects, n_timepoints, n_voxels). Connectome batch data.
        lesion_voxel_indices : np.ndarray
            1D array of voxel indices within the connectome mask.

        Returns
        -------
        np.ndarray
            Shape (n_subjects, n_timepoints). Representative timeseries.
        """
        # Extract timeseries for lesion voxels
        lesion_ts = batch_timeseries[:, :, lesion_voxel_indices]

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

    def _get_mask_voxel_indices(self, mask_data: SubjectData) -> tuple[np.ndarray, nib.Nifti1Image]:
        """Get indices of mask voxels within connectome mask (vectorized O(N) version).

        This uses a lookup array for O(N) complexity instead of O(N×M) nested loops,
        providing massive speedup for large masks.

        **Performance**:
        - Speedup: 15-2000x vs. legacy implementation (increases with mask size)

        **Benchmark Results** (MNI152 @ 2mm, ~335K brain voxels):
        - 100 voxels: 113ms → 7ms (15.7x speedup)
        - 1,000 voxels: 1,078ms → 4.7ms (228x speedup)
        - 10,000 voxels: 9,965ms → 4.9ms (2,025x speedup)

        **Implementation**:
        Uses a 3D lookup array (shape=mask_shape, dtype=int32) that maps
        spatial coordinates directly to flat indices in the connectome mask.
        This eliminates the need for searching through mask_indices for each
        mask voxel.

        Automatically resamples mask to connectome space if dimensions don't match.

        Parameters
        ----------
        mask_data : SubjectData
            Mask data in target space.

        Returns
        -------
        tuple[np.ndarray, nib.Nifti1Image]
            Tuple containing:
            - 1D array of indices into the connectome's voxel dimension
            - The mask image resampled to connectome space (for analysis_mask storage)

        Notes
        -----
        For batch processing scenarios, consider caching the lookup array to avoid
        rebuilding it for every subject
        """
        # Get connectome mask info
        mask_shape = self._mask_info["mask_shape"]
        mask_indices = self._mask_info["mask_indices"]
        mask_affine = self._mask_info["mask_affine"]

        # Get mask image
        mask_img = mask_data.mask_img
        input_shape = mask_img.shape

        # Check if resampling is needed
        if input_shape != mask_shape:
            self.logger.warning(
                f"Resampling mask from {input_shape} to {mask_shape}", indent_level=1
            )

            # Resample mask to connectome space
            from nilearn.image import resample_to_img

            # Create template image in connectome space
            template_img = nib.Nifti1Image(np.zeros(mask_shape), mask_affine)

            # Resample mask to match connectome
            mask_img_resampled = resample_to_img(
                mask_img,
                template_img,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )
            resampled_mask = mask_img_resampled.get_fdata().astype(bool)

            self.logger.success("Resampling complete", indent_level=2)
        else:
            mask_img_resampled = mask_img
            resampled_mask = mask_img.get_fdata().astype(bool)

        # Get mask coordinates
        mask_coords = np.where(resampled_mask)

        # Build lookup array: 3D array mapping coordinates to flat indices
        lookup = np.full(mask_shape, -1, dtype=np.int32)
        lookup[mask_indices] = np.arange(len(mask_indices[0]), dtype=np.int32)

        # Direct O(N) indexing to get flat indices
        flat_indices = lookup[mask_coords]

        # Filter out voxels not in connectome mask (value = -1)
        valid_indices = flat_indices[flat_indices >= 0]

        return valid_indices.astype(int), mask_img_resampled

    def _compute_correlation_maps_batch(
        self, lesion_timeseries: np.ndarray, batch_timeseries: np.ndarray
    ) -> np.ndarray:
        """Compute correlation maps between lesion and whole-brain timeseries.

        Memory-efficient version that works on a single batch.

        Parameters
        ----------
        lesion_timeseries : np.ndarray
            Shape (n_subjects, n_timepoints). Lesion timeseries.
        batch_timeseries : np.ndarray
            Shape (n_subjects, n_timepoints, n_voxels). Connectome batch data.

        Returns
        -------
        np.ndarray
            Shape (n_subjects, n_voxels). Correlation values for each voxel.
        """
        # Center timeseries
        brain_ts_centered = batch_timeseries - batch_timeseries.mean(axis=1, keepdims=True)
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

    def run_batch(self, mask_data_list: list[SubjectData]) -> list[SubjectData]:
        """Process multiple lesions together using vectorized operations.

        This is 10-50x faster than sequential processing because it:
        1. Processes all lesions through each connectome batch together
        2. Uses vectorized einsum: "lit,itv->liv" for batch correlations
        3. Minimizes loop overhead and leverages optimized BLAS operations

        This method is automatically called when using VectorizedStrategy
        for batch processing.

        Parameters
        ----------
        mask_data_list : list[SubjectData]
            Batch of lesions to process together

        Returns
        -------
        list[SubjectData]
            Processed lesions with results added

        Examples
        --------
        >>> from lacuna.batch import VectorizedStrategy
        >>> from lacuna.analysis import FunctionalNetworkMapping
        >>>
        >>> analysis = FunctionalNetworkMapping(...)
        >>> strategy = VectorizedStrategy()
        >>> results = strategy.execute(mask_data_list, analysis)
        """
        self.logger.info(f"Vectorized batch processing: {len(mask_data_list)} lesions together")

        # Validate all lesions first
        for mask_data in mask_data_list:
            self._validate_inputs(mask_data)

        # Load mask info once (shared across all batches)
        mask_indices, mask_affine, mask_shape = self._load_mask_info()
        connectome_files = self._get_connectome_files()

        self.logger.success(
            "Loaded connectome metadata",
            details={
                "connectome_batches": len(connectome_files),
                "mask_shape": mask_shape,
                "n_lesions": len(mask_data_list),
            },
        )

        # Prepare all masks with resampling if needed
        mask_batch = []
        for i, mask_data in enumerate(mask_data_list):
            subject_id = mask_data.metadata.get("subject_id", f"mask_{i}")
            self.logger.info(
                f"Preparing mask {i + 1}/{len(mask_data_list)}: {subject_id}", indent_level=1
            )

            voxel_indices, _ = self._get_mask_voxel_indices(mask_data)

            if len(voxel_indices) == 0:
                raise ValidationError(
                    f"Mask {i} has no overlap with connectome mask after resampling"
                )

            mask_batch.append(
                {
                    "mask_data": mask_data,
                    "voxel_indices": voxel_indices,
                    "index": i,
                }
            )

        # Process through all connectome batches (VECTORIZED)
        self.logger.info("Processing connectome batches...")

        # Get number of voxels from first connectome batch
        with h5py.File(connectome_files[0], "r") as hf:
            n_voxels = hf["timeseries"].shape[2]

        # Initialize streaming aggregators for each mask (MEMORY OPTIMIZED)
        # Instead of storing all correlation maps, we accumulate statistics
        aggregators = []
        for _ in range(len(mask_batch)):
            aggregators.append(
                {
                    "sum_z": np.zeros(n_voxels, dtype=np.float64),  # Need higher precision for sums
                    "sum_z2": np.zeros(n_voxels, dtype=np.float64),
                    "n": 0,
                }
            )

        total_subjects = 0
        batch_times = []  # Track timing for each batch

        for batch_idx, connectome_path in enumerate(connectome_files):
            import time

            batch_start_time = time.time()

            with h5py.File(connectome_path, "r") as hf:
                timeseries_data = hf["timeseries"][:]  # (n_subj, n_time, n_vox)
                n_subjects = timeseries_data.shape[0]
                total_subjects += n_subjects

            # Vectorized processing for ALL masks at once
            batch_r_maps = self._compute_batch_correlations_vectorized(
                mask_batch, timeseries_data
            )  # (n_masks, n_subjects, n_voxels)

            # Convert to Fisher z-scores and update running statistics
            # This is the KEY optimization: we don't store full maps!
            with np.errstate(divide="ignore", invalid="ignore"):
                batch_z_maps = np.arctanh(batch_r_maps)
                batch_z_maps = np.nan_to_num(batch_z_maps, nan=0.0, posinf=3.0, neginf=-3.0)

            # Update aggregators with streaming statistics
            for i in range(len(mask_batch)):
                # Sum across subjects in this batch
                aggregators[i]["sum_z"] += np.sum(batch_z_maps[i], axis=0)
                aggregators[i]["sum_z2"] += np.sum(batch_z_maps[i] ** 2, axis=0)
                aggregators[i]["n"] += n_subjects

            # Memory cleanup - immediately free large arrays
            del timeseries_data, batch_r_maps, batch_z_maps

            # Display timing information
            batch_elapsed = time.time() - batch_start_time
            batch_times.append(batch_elapsed)

            # Show progress with time estimates
            if len(batch_times) > 2:
                avg_time = sum(batch_times) / len(batch_times)
                remaining_batches = len(connectome_files) - (batch_idx + 1)
                est_remaining = avg_time * remaining_batches
                self.logger.progress(
                    f"Batch completed in {batch_elapsed:.2f}s (est. {est_remaining:.1f}s remaining)",
                    current=batch_idx + 1,
                    total=len(connectome_files),
                )
            else:
                self.logger.progress(
                    f"Batch completed in {batch_elapsed:.2f}s",
                    current=batch_idx + 1,
                    total=len(connectome_files),
                )

        total_batch_time = sum(batch_times)
        avg_batch_time = total_batch_time / len(batch_times) if batch_times else 0
        self.logger.success(
            "Batch processing complete",
            details={
                "n_batches": len(connectome_files),
                "total_time": f"{total_batch_time:.2f}s",
                "avg_time_per_batch": f"{avg_batch_time:.2f}s",
            },
        )

        # Compute final statistics from aggregated values
        self.logger.info("Aggregating results...")
        results = []
        for i, mask_info in enumerate(mask_batch):
            subject_id = mask_info["mask_data"].metadata.get("subject_id", f"mask_{i}")
            self.logger.info(f"Aggregating results for: {subject_id}", indent_level=1)

            # Compute statistics from streaming aggregators
            n = aggregators[i]["n"]
            mean_z = aggregators[i]["sum_z"] / n
            mean_r = np.tanh(mean_z).astype(np.float32)

            # Compute sample standard deviation for t-statistics (always computed)
            # Using Bessel's correction (N-1 denominator) to match _run_analysis
            # Population variance: Var_pop = E[X²] - E[X]² = sum_z2/n - mean_z²
            # Sample variance: Var_sample = (n / (n-1)) * Var_pop
            var_z_population = (aggregators[i]["sum_z2"] / n) - (mean_z**2)
            var_z_sample = (n / (n - 1)) * var_z_population
            std_z = np.sqrt(np.maximum(var_z_sample, 0))  # Avoid negative from numerical errors

            # Create a copy of mask_data to avoid modifying input
            mask_copy = mask_info["mask_data"].copy()

            # Aggregate using optimized method that accepts pre-computed statistics
            result = self._aggregate_results_from_statistics(
                mask_copy,
                mean_r,
                mean_z,
                std_z,
                mask_indices,
                mask_affine,
                mask_shape,
                total_subjects,
            )

            results.append(result)

        self.logger.success(
            "Batch processing complete", details={"n_masks_processed": len(results)}
        )

        return results

    def _compute_batch_correlations_vectorized(
        self,
        mask_batch: list[dict],
        timeseries_data: np.ndarray,
    ) -> np.ndarray:
        """Compute correlations for ALL masks at once (vectorized).

        Uses einsum "lit,itv->liv" to compute correlations for all masks simultaneously, 
        reducing overhead and enabling optimized BLAS operations.

        Parameters
        ----------
        mask_batch : list[dict]
            List of mask dictionaries with 'voxel_indices' and 'mask_data'
        timeseries_data : np.ndarray
            Shape (n_subjects, n_timepoints, n_voxels). Connectome batch data.

        Returns
        -------
        np.ndarray
            Shape (n_masks, n_subjects, n_voxels). Correlation maps for all masks.
        """
        # Extract and process timeseries for all masks
        mask_mean_ts_list = []
        for mask_info in mask_batch:
            voxel_indices = mask_info["voxel_indices"]

            # Extract mask timeseries: (n_subjects, n_timepoints, n_mask_voxels)
            mask_ts = timeseries_data[:, :, voxel_indices]

            if self.method == "boes":
                # Simple mean across voxels
                mask_mean_ts = np.mean(mask_ts, axis=2)

            elif self.method == "pini":
                # PINI: PCA-based selection
                mask_mean_ts = self._compute_pini_timeseries_batch(mask_ts)

            mask_mean_ts_list.append(mask_mean_ts)

        # Stack into (n_masks, n_subjects, n_timepoints)
        mask_mean_ts_batch = np.stack(mask_mean_ts_list, axis=0)

        # Center data
        brain_ts_centered = timeseries_data - timeseries_data.mean(axis=1, keepdims=True)
        mask_ts_centered = mask_mean_ts_batch - mask_mean_ts_batch.mean(axis=2, keepdims=True)

        # VECTORIZED CORRELATION: Process all masks at once!
        # einsum: "lit,itv->liv"
        #   l = masks, i = subjects, t = timepoints, v = voxels
        # Use float32 for memory efficiency (sufficient precision for correlations)
        cov = np.einsum(
            "lit,itv->liv",
            mask_ts_centered.astype(np.float32),
            brain_ts_centered.astype(np.float32),
            dtype=np.float32,
            optimize="optimal",
        )

        # Compute standard deviations
        mask_std = np.sqrt(np.sum(mask_ts_centered**2, axis=2))  # (n_masks, n_subjects)
        brain_std = np.sqrt(np.sum(brain_ts_centered**2, axis=1))  # (n_subjects, n_voxels)

        # Compute correlations: cov / (mask_std * brain_std)
        with np.errstate(divide="ignore", invalid="ignore"):
            all_r_maps = cov / (mask_std[:, :, np.newaxis] * brain_std[np.newaxis, :, :])

        # Clean up NaN and inf values
        all_r_maps = np.nan_to_num(all_r_maps, nan=0, posinf=1, neginf=-1)

        return all_r_maps.astype(np.float32)

    def _compute_pini_timeseries_batch(self, lesion_ts: np.ndarray) -> np.ndarray:
        """Compute PINI timeseries for a batch of subjects.

        Parameters
        ----------
        lesion_ts : np.ndarray
            Shape (n_subjects, n_timepoints, n_lesion_voxels)

        Returns
        -------
        np.ndarray
            Shape (n_subjects, n_timepoints). PINI-refined timeseries.
        """
        n_subjects, n_timepoints, n_voxels = lesion_ts.shape

        # Compute mean timeseries first
        mean_ts_across_voxels = np.mean(lesion_ts, axis=2)  # (n_subjects, n_timepoints)

        # Center timeseries
        mean_ts_centered = mean_ts_across_voxels - mean_ts_across_voxels.mean(axis=1, keepdims=True)
        voxel_ts_centered = lesion_ts - lesion_ts.mean(axis=1, keepdims=True)

        # Compute covariance between mean and each voxel
        covariance = np.einsum(
            "it,itv->iv",
            mean_ts_centered,
            voxel_ts_centered,
            dtype=np.float64,
            optimize="optimal",
        )

        # Compute standard deviations
        std_mean = np.sqrt(np.sum(mean_ts_centered**2, axis=1))
        std_voxels = np.sqrt(np.sum(voxel_ts_centered**2, axis=1))

        # Compute correlation matrix for PCA
        with np.errstate(divide="ignore", invalid="ignore"):
            pca_input_matrix = covariance / (std_mean[:, np.newaxis] * std_voxels)

        pca_input_matrix = np.nan_to_num(pca_input_matrix)

        # Apply PCA to find principal component
        pca = PCA(n_components=1)
        pca.fit(pca_input_matrix)
        pc1_loadings = pca.components_[0, :]

        # Select voxels based on percentile threshold
        threshold = np.percentile(np.abs(pc1_loadings), self.pini_percentile)
        suprathreshold_indices = np.where(np.abs(pc1_loadings) >= threshold)[0]

        # Use refined voxel set or fall back to all voxels
        if len(suprathreshold_indices) == 0:
            refined_lesion_ts = lesion_ts
        else:
            refined_lesion_ts = lesion_ts[:, :, suprathreshold_indices]

        # Return mean of refined voxels
        return np.mean(refined_lesion_ts, axis=2)

    def _aggregate_results_from_statistics(
        self,
        mask_data: SubjectData,
        mean_r_map: np.ndarray,
        mean_z_map: np.ndarray,
        std_z_map: np.ndarray,
        mask_indices: tuple,
        mask_affine: np.ndarray,
        mask_shape: tuple,
        total_subjects: int,
    ) -> SubjectData:
        """Aggregate results from pre-computed statistics (memory-optimized).

        This method is used by vectorized batch processing with streaming
        aggregation. Instead of storing all individual correlation maps,
        it accepts pre-computed mean and standard deviation, reducing memory usage.

        Parameters
        ----------
        mask_data : SubjectData
            Original lesion data to add results to
        mean_r_map : np.ndarray
            Shape (n_voxels,). Mean correlation map (already Fisher z-averaged).
        mean_z_map : np.ndarray
            Shape (n_voxels,). Mean Fisher z-transformed map.
        std_z_map : np.ndarray
            Shape (n_voxels,). Standard deviation of z-maps (for t-statistics).
        mask_indices : tuple
            Brain mask voxel indices
        mask_affine : np.ndarray
            Affine transformation matrix
        mask_shape : tuple
            3D mask shape
        total_subjects : int
            Total number of subjects processed

        Returns
        -------
        SubjectData
            Lesion data with analysis results added
        """
        # Compute t-statistics (always computed)
        with np.errstate(divide="ignore", invalid="ignore"):
            std_error_map = std_z_map / np.sqrt(total_subjects)
            t_map_flat = np.divide(
                mean_z_map,
                std_error_map,
                where=(std_error_map != 0),
            )
            t_map_flat = np.nan_to_num(t_map_flat)

        # Compute p-values from t-statistics (two-tailed)
        p_map_flat = None
        p_fdr_map_flat = None
        n_significant_fdr = 0
        pct_significant_fdr = 0.0

        if self.compute_p_map:
            df = total_subjects - 1  # degrees of freedom
            # Two-tailed p-value: 2 * (1 - CDF(|t|))
            p_map_flat = 2 * stats.t.sf(np.abs(t_map_flat), df)
            p_map_flat = np.nan_to_num(p_map_flat, nan=1.0)

            # Compute FDR-corrected p-values if requested
            if self.fdr_alpha is not None:
                # Benjamini-Hochberg FDR correction
                n_voxels = len(p_map_flat)
                sorted_indices = np.argsort(p_map_flat)
                sorted_pvals = p_map_flat[sorted_indices]

                # Compute FDR-adjusted p-values
                # p_adj[i] = min(p[i] * n / (rank[i]), 1.0)
                ranks = np.arange(1, n_voxels + 1)
                adjusted = sorted_pvals * n_voxels / ranks

                # Enforce monotonicity (cumulative minimum from the right)
                adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
                adjusted = np.clip(adjusted, 0, 1)

                # Map back to original order
                p_fdr_map_flat = np.empty_like(p_map_flat)
                p_fdr_map_flat[sorted_indices] = adjusted

                # Count significant voxels
                n_significant_fdr = int(np.sum(p_fdr_map_flat < self.fdr_alpha))
                pct_significant_fdr = (n_significant_fdr / n_voxels) * 100

        # Create 3D volumes
        correlation_map_3d = np.zeros(mask_shape, dtype=np.float32)
        correlation_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = mean_r_map
        correlation_map_nifti = nib.Nifti1Image(correlation_map_3d, mask_affine)

        z_map_3d = np.zeros(mask_shape, dtype=np.float32)
        z_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = mean_z_map.astype(np.float32)
        z_map_nifti = nib.Nifti1Image(z_map_3d, mask_affine)

        # Build results dictionary with snake_case keys (matching _run_analysis)
        # Wrap NIfTI images in VoxelMap for consistent unwrap behavior
        results = {
            "rmap": VoxelMap(
                name="rmap",
                data=correlation_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "n_subjects": total_subjects,
                    "statistic": "pearson_correlation_coefficient",
                },
            ),
            "zmap": VoxelMap(
                name="zmap",
                data=z_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "n_subjects": total_subjects,
                    "statistic": "fisher_z",
                },
            ),
            "summarystatistics": ScalarMetric(
                name="summarystatistics",
                data={
                    "mean": float(np.mean(mean_r_map)),
                    "std": float(np.std(mean_r_map)),
                    "max": float(np.max(mean_r_map)),
                    "min": float(np.min(mean_r_map)),
                    "n_subjects": total_subjects,
                    "n_batches": len(self._get_connectome_files()),
                },
                metadata={"method": self.method},
            ),
        }

        # Add t-map results if computed
        if t_map_flat is not None:
            t_map_3d = np.zeros(mask_shape, dtype=np.float32)
            t_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = t_map_flat
            t_map_nifti = nib.Nifti1Image(t_map_3d, mask_affine)

            results["tmap"] = VoxelMap(
                name="tmap",
                data=t_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "n_subjects": total_subjects,
                    "statistic": "t_statistic",
                },
            )
            # Update summary statistics with t-map info
            results["summarystatistics"].data["t_min"] = float(np.min(t_map_flat))
            results["summarystatistics"].data["t_max"] = float(np.max(t_map_flat))

            # Create thresholded t-map if threshold provided
            if self.t_threshold is not None:
                t_threshold_mask = np.abs(t_map_flat) > self.t_threshold
                threshold_map_3d = np.zeros(mask_shape, dtype=np.uint8)
                threshold_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = (
                    t_threshold_mask.astype(np.uint8)
                )
                t_threshold_map_nifti = nib.Nifti1Image(threshold_map_3d, mask_affine)
                results["tthresholdmap"] = VoxelMap(
                    name="tthresholdmap",
                    data=t_threshold_map_nifti,
                    space=self.output_space,
                    resolution=self.output_resolution,
                    metadata={
                        "method": self.method,
                        "threshold": self.t_threshold,
                        "statistic": "thresholded_t",
                    },
                )
                results["summarystatistics"].data["t_threshold"] = self.t_threshold
                results["summarystatistics"].data["n_significant_voxels"] = int(
                    np.sum(t_threshold_mask)
                )

        # Add p-value map if computed
        if p_map_flat is not None:
            p_map_3d = np.zeros(mask_shape, dtype=np.float32)
            p_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = p_map_flat
            p_map_nifti = nib.Nifti1Image(p_map_3d, mask_affine)

            results["pmap"] = VoxelMap(
                name="pmap",
                data=p_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "n_subjects": total_subjects,
                    "statistic": "p_value_two_tailed",
                    "degrees_of_freedom": total_subjects - 1,
                },
            )
            results["summarystatistics"].data["p_min"] = float(np.min(p_map_flat))
            results["summarystatistics"].data["p_max"] = float(np.max(p_map_flat))

        # Add FDR-corrected p-value map if computed
        if p_fdr_map_flat is not None:
            p_fdr_map_3d = np.zeros(mask_shape, dtype=np.float32)
            p_fdr_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = p_fdr_map_flat
            p_fdr_map_nifti = nib.Nifti1Image(p_fdr_map_3d, mask_affine)

            results["pfdrmap"] = VoxelMap(
                name="pfdrmap",
                data=p_fdr_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "n_subjects": total_subjects,
                    "statistic": "p_value_fdr_corrected",
                    "fdr_alpha": self.fdr_alpha,
                    "correction_method": "benjamini_hochberg",
                },
            )
            results["summarystatistics"].data["n_significant_fdr"] = n_significant_fdr
            results["summarystatistics"].data["pct_significant_fdr"] = pct_significant_fdr
            results["summarystatistics"].data["fdr_alpha"] = self.fdr_alpha

            # Create binary significance mask at FDR threshold
            fdr_sig_mask = (p_fdr_map_flat < self.fdr_alpha).astype(np.uint8)
            fdr_sig_map_3d = np.zeros(mask_shape, dtype=np.uint8)
            fdr_sig_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = fdr_sig_mask
            fdr_sig_map_nifti = nib.Nifti1Image(fdr_sig_map_3d, mask_affine)

            results["pfdrthresholdmap"] = VoxelMap(
                name="pfdrthresholdmap",
                data=fdr_sig_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "n_subjects": total_subjects,
                    "statistic": "fdr_significant_binary",
                    "fdr_alpha": self.fdr_alpha,
                    "n_significant": n_significant_fdr,
                },
            )

        # Transform VoxelMap results back to input space if requested
        if self.return_in_input_space:
            results = self._transform_results_to_input_space(results, mask_data)

        # Add results to mask data (returns new instance with results)
        batch_results = {
            "rmap": results["rmap"],
            "zmap": results["zmap"],
            "summarystatistics": results["summarystatistics"],
        }
        # Add optional results if present
        if "tmap" in results:
            batch_results["tmap"] = results["tmap"]
        if "tthresholdmap" in results:
            batch_results["tthresholdmap"] = results["tthresholdmap"]
        if "pmap" in results:
            batch_results["pmap"] = results["pmap"]
        if "pfdrmap" in results:
            batch_results["pfdrmap"] = results["pfdrmap"]
        if "pfdrthresholdmap" in results:
            batch_results["pfdrthresholdmap"] = results["pfdrthresholdmap"]

        mask_data_with_results = mask_data.add_result(self.__class__.__name__, batch_results)

        return mask_data_with_results

    def _aggregate_results(
        self,
        mask_data: SubjectData,
        all_r_maps: np.ndarray,
        mask_indices: tuple,
        mask_affine: np.ndarray,
        mask_shape: tuple,
        total_subjects: int,
    ) -> SubjectData:
        """Aggregate correlation maps across all subjects into final results.

        This method is reused by both single and batch processing.

        Parameters
        ----------
        mask_data : SubjectData
            Original lesion data to add results to
        all_r_maps : np.ndarray
            Shape (n_subjects, n_voxels). All correlation maps.
        mask_indices : tuple
            Brain mask voxel indices
        mask_affine : np.ndarray
            Affine transformation matrix
        mask_shape : tuple
            3D mask shape
        total_subjects : int
            Total number of subjects processed

        Returns
        -------
        SubjectData
            Lesion data with analysis results added
        """
        # Fisher z-transform
        all_z_maps = np.arctanh(all_r_maps)
        all_z_maps = np.nan_to_num(all_z_maps, nan=0, posinf=10, neginf=-10)

        # Compute statistics
        mean_z_map = np.mean(all_z_maps, axis=0)
        mean_r_map = np.tanh(mean_z_map)

        # Compute t-statistics (always computed)
        std_z_map = np.std(all_z_maps, axis=0, ddof=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            std_error_map = std_z_map / np.sqrt(total_subjects)
            t_map_flat = np.divide(
                mean_z_map,
                std_error_map,
                where=(std_error_map != 0),
            )
            t_map_flat = np.nan_to_num(t_map_flat)

        # Create 3D volumes
        correlation_map_3d = np.zeros(mask_shape, dtype=np.float32)
        correlation_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = mean_r_map
        correlation_map_nifti = nib.Nifti1Image(correlation_map_3d, mask_affine)

        z_map_3d = np.zeros(mask_shape, dtype=np.float32)
        z_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = mean_z_map
        z_map_nifti = nib.Nifti1Image(z_map_3d, mask_affine)

        # Build results dictionary
        # Wrap NIfTI images in VoxelMap for consistent unwrap behavior
        results = {
            "rmap": VoxelMap(
                name="rmap",
                data=correlation_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "n_subjects": total_subjects,
                    "statistic": "pearson_correlation_coefficient",
                },
            ),
            "network_map": correlation_map_nifti,  # Alias for backward compat (raw nifti)
            "zmap": VoxelMap(
                name="zmap",
                data=z_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "n_subjects": total_subjects,
                    "statistic": "fisher_z",
                },
            ),
            "mean_correlation": float(np.mean(mean_r_map)),
            "summarystatistics": ScalarMetric(
                name="summarystatistics",
                data={
                    "mean": float(np.mean(mean_r_map)),
                    "std": float(np.std(mean_r_map)),
                    "max": float(np.max(mean_r_map)),
                    "min": float(np.min(mean_r_map)),
                    "n_subjects": total_subjects,
                    "n_batches": len(self._get_connectome_files()),
                },
                metadata={"method": self.method},
            ),
        }

        # Add t-map results if computed
        if t_map_flat is not None:
            t_map_3d = np.zeros(mask_shape, dtype=np.float32)
            t_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = t_map_flat
            t_map_nifti = nib.Nifti1Image(t_map_3d, mask_affine)

            results["tmap"] = VoxelMap(
                name="tmap",
                data=t_map_nifti,
                space=self.output_space,
                resolution=self.output_resolution,
                metadata={
                    "method": self.method,
                    "n_subjects": total_subjects,
                    "statistic": "t_statistic",
                },
            )
            results["summarystatistics"].data["t_min"] = float(np.min(t_map_flat))
            results["summarystatistics"].data["t_max"] = float(np.max(t_map_flat))

            # Create thresholded t-map if threshold provided
            if self.t_threshold is not None:
                t_threshold_mask = np.abs(t_map_flat) > self.t_threshold
                threshold_map_3d = np.zeros(mask_shape, dtype=np.uint8)
                threshold_map_3d[mask_indices[0], mask_indices[1], mask_indices[2]] = (
                    t_threshold_mask.astype(np.uint8)
                )
                t_threshold_map_nifti = nib.Nifti1Image(threshold_map_3d, mask_affine)
                results["tthresholdmap"] = VoxelMap(
                    name="tthresholdmap",
                    data=t_threshold_map_nifti,
                    space=self.output_space,
                    resolution=self.output_resolution,
                    metadata={
                        "method": self.method,
                        "threshold": self.t_threshold,
                        "statistic": "thresholded_t",
                    },
                )
                results["summarystatistics"].data["t_threshold"] = self.t_threshold
                results["summarystatistics"].data["n_significant_voxels"] = int(
                    np.sum(t_threshold_mask)
                )

        # Add results to lesion data (returns new instance with results)
        # Note: Using individual keys to match _run_analysis() structure
        batch_results = {
            "rmap": results["rmap"],
            "zmap": results["zmap"],
            "summarystatistics": results["summarystatistics"],
        }
        # Add optional results if present
        if "tmap" in results:
            batch_results["tmap"] = results["tmap"]
        if "tthresholdmap" in results:
            batch_results["tthresholdmap"] = results["tthresholdmap"]

        mask_data_with_results = mask_data.add_result(self.__class__.__name__, batch_results)

        return mask_data_with_results

    def _get_parameters(self) -> dict:
        """Get analysis parameters for provenance and display.

        Returns
        -------
        dict
            Dictionary of parameter names and values.
        """
        return {
            "connectome_name": self.connectome_name,
            "method": self.method,
            "pini_percentile": self.pini_percentile,
            "n_jobs": self.n_jobs,
            "t_threshold": self.t_threshold,
            "return_in_input_space": self.return_in_input_space,
            "verbose": self.verbose,
        }

    def _transform_results_to_input_space(self, results: dict, mask_data: SubjectData) -> dict:
        """Transform VoxelMap results back to original input space.

        Parameters
        ----------
        results : dict
            Dictionary of result objects
        mask_data : SubjectData
            Input mask data with space/resolution metadata. If the mask was
            transformed by BaseAnalysis, it will have _original_input_space
            and _original_input_resolution in metadata.

        Returns
        -------
        dict
            Results with transformed VoxelMap objects

        Raises
        ------
        ValueError
            If mask_data lacks space or resolution metadata
        """
        from lacuna.core.spaces import REFERENCE_AFFINES, CoordinateSpace
        from lacuna.spatial.transform import transform_image

        # Get original input space from metadata (if available) or current space
        target_space_id = mask_data.metadata.get("_original_input_space", mask_data.space)
        target_resolution = mask_data.metadata.get(
            "_original_input_resolution", mask_data.resolution
        )

        # If we're already in the input space, no transformation needed
        if target_space_id == self.output_space and target_resolution == self.output_resolution:
            return results

        # Get reference affine for target space
        target_key = (target_space_id, target_resolution)
        if target_key not in REFERENCE_AFFINES:
            raise ValueError(
                f"No reference affine available for {target_space_id}@{target_resolution}mm. "
                f"Available spaces: {list(REFERENCE_AFFINES.keys())}"
            )

        target_space = CoordinateSpace(
            identifier=target_space_id,
            resolution=target_resolution,
            reference_affine=REFERENCE_AFFINES[target_key],
        )

        self.logger.info(
            f"Transforming VoxelMap outputs from {self.output_space}@{self.output_resolution}mm "
            f"to {target_space.identifier}@{target_space.resolution}mm"
        )

        transformed_results = {}
        for key, result in results.items():
            # Only transform VoxelMap results
            if isinstance(result, VoxelMap):
                # Auto-detect interpolation method based on data type
                # Use nearest for binary maps (thresholdmaps), linear for continuous
                data = result.data.get_fdata()
                unique_vals = np.unique(data[~np.isnan(data)])
                is_binary = len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1})
                interpolation = "nearest" if is_binary else "linear"

                # Transform the image
                transformed_img = transform_image(
                    img=result.data,
                    source_space=self.output_space,
                    target_space=target_space,
                    source_resolution=int(self.output_resolution),
                    interpolation=interpolation,
                    verbose=self.verbose,
                )

                # Create new VoxelMap with updated space
                transformed_result = VoxelMap(
                    name=result.name,
                    data=transformed_img,
                    space=target_space.identifier,
                    resolution=target_space.resolution,
                    metadata={
                        **result.metadata,
                        "transformed_from": f"{self.output_space}@{self.output_resolution}mm",
                        "transformed_to": f"{target_space.identifier}@{target_space.resolution}mm",
                    },
                )
                transformed_results[key] = transformed_result
            else:
                # Keep non-VoxelMap results as-is
                transformed_results[key] = result

        return transformed_results
