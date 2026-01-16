"""
Structural lesion network mapping (sLNM) analysis using tractography.

Computes white matter disconnection maps by filtering a whole-brain tractogram
through a lesion mask and comparing the resulting track density to the intact
white matter connectivity.
"""

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import nibabel as nib
import numpy as np

from lacuna.analysis.base import BaseAnalysis
from lacuna.assets import load_template
from lacuna.assets.connectomes import (
    list_structural_connectomes,
    load_structural_connectome,
)
from lacuna.assets.parcellations import list_parcellations, load_parcellation
from lacuna.core.data_types import (
    ConnectivityMatrix,
    ParcelData,
    ScalarMetric,
    Tractogram,
    VoxelMap,
)
from lacuna.core.keys import build_result_key
from lacuna.core.subject_data import SubjectData
from lacuna.utils.cache import get_tdi_cache_dir, get_temp_dir
from lacuna.utils.logging import ConsoleLogger
from lacuna.utils.mrtrix import (
    MRtrixError,
    _get_nthreads_args,
    check_mrtrix_available,
    compute_disconnection_map,
    compute_tdi_map,
    filter_tractogram_by_mask,
    run_mrtrix_command,
)

if TYPE_CHECKING:
    from lacuna.core.data_types import AnalysisResult


class StructuralNetworkMapping(BaseAnalysis):
    """
    Structural lesion network mapping using tractography-based disconnection.

    This analysis quantifies white matter disconnection caused by a lesion by:
    1. Filtering a whole-brain tractogram to streamlines passing through the lesion
    2. Computing track density images (TDI) for both lesion and whole-brain tracts
    3. Calculating disconnection probability as the ratio of lesion TDI to whole-brain TDI
    4. Optionally computing parcellated connectivity matrices (if atlas provided)

    **Outputs:**
    - **Voxel-wise disconnection map** (always): 3D NIfTI showing disconnection probability per voxel
    - **Connectivity matrices** (optional): Parcellated edge-wise disconnection when atlas is provided

    **Computation Space:**
    All computations are performed in the tractogram's native space (typically
    MNI152NLin2009cAsym @ 1mm for high-resolution structural connectivity).
    Lesions are transformed to this high-resolution template space for accurate
    white matter fiber filtering.

    The analysis requires MRtrix3 to be installed and available in the system PATH.

    Attributes
    ----------
    batch_strategy : str
        Batch processing strategy. Set to "parallel" as each subject is independent
        and tractogram filtering is compute-intensive.

    Parameters
    ----------
    connectome_name : str
        Name of registered structural connectome (e.g., "dTOR985").
        Use list_structural_connectomes() to see available connectomes.
        The connectome must be pre-registered via register_structural_connectome().
    parcellation_name : str, optional
        Name of registered atlas for parcellated connectivity matrices.
        Use list_parcellations() to see available atlases.
    compute_disconnectivity_matrix : bool, default=False
        If True and parcellation_name provided, compute disconnectivity matrices
        including per-ROI disconnection percentages.
    output_resolution : {1, 2}, default=2
        Output resolution in mm (must match connectome resolution).
    cache_tdi : bool, default=True
        If True, cache computed TDI maps for reuse.
    n_jobs : int, default=1
        Number of threads for MRtrix3 processing.
    keep_intermediate : bool, default=False
        If True, include intermediate results in output (lesion tractogram, lesion TDI,
        warped atlas if transformed). Useful for debugging and quality control.
    cleanup_temp_files : bool, default=True
        If True, delete temporary working directory after analysis completes.
        Set to False to preserve temp files on disk for inspection.
        Note: This is independent of keep_intermediate.
    check_dependencies : bool, default=True
        If True, checks for MRtrix3 availability.
    verbose : bool, default=True
        If True, print progress messages. If False, run silently.
    show_mrtrix_output : bool, default=False
        If True, display MRtrix3 command outputs. If False, suppress verbose
        MRtrix3 messages for cleaner output.

    Raises
    ------
    MRtrixError
        If MRtrix3 is not installed or not available in PATH
    FileNotFoundError
        If tractogram_path or whole_brain_tdi files don't exist

    Examples
    --------
    **Register and use structural connectome:**

    >>> from lacuna import SubjectData
    >>> from lacuna.analysis import StructuralNetworkMapping
    >>> from lacuna.assets.connectomes import (
    ...     list_structural_connectomes,
    ...     register_structural_connectome,
    ... )
    >>>
    >>> # Register connectome (do this once)
    >>> register_structural_connectome(
    ...     name="dTOR985",
    ...     space="MNI152NLin2009cAsym",
    ...     tractogram_path="/data/dtor/dtor985_tractogram.tck",
    ...     description="HCP dTOR tractogram - TDI computed on-demand"
    ... )
    >>>
    >>> # List available connectomes
    >>> list_structural_connectomes()
    >>>
    >>> # Load lesion data
    >>> lesion = SubjectData.from_nifti("lesion.nii.gz")
    >>>
    >>> # Interactive analysis
    >>> analysis = StructuralNetworkMapping(
    ...     connectome_name="dTOR985",
    ...     n_jobs=8,
    ... )
    >>> result = analysis.run(lesion)
    >>> disconn_map = result.results["StructuralNetworkMapping"]["disconnection_map"]
    >>> disconn_map.orthoview()

    **Batch processing:**

    >>> from lacuna import batch_process
    >>>
    >>> analysis = StructuralNetworkMapping(
    ...     connectome_name="dTOR985",
    ...     n_jobs=8,
    ... )
    >>> results = batch_process(lesions, analysis, n_jobs=2)
    >>>
    >>> # Save results
    >>> for result in results:
    ...     disconn_map = result.results["StructuralNetworkMapping"]["disconnection_map"]
    ...     nib.save(disconn_map, f"output/{subject_id}_disconn.nii.gz")

    Notes
    -----
    - Requires MRtrix3: https://www.mrtrix.org/download/
    - Processing time scales with lesion size and tractogram density
    - For large tractograms, processing can take several minutes per subject

    See Also
    --------
    FunctionalNetworkMapping : Functional connectivity-based lesion network mapping
    RegionalDamage : Atlas-based regional overlap quantification
    """

    #: Preferred batch processing strategy - sequential because MRtrix3's tckedit
    #: uses internal parallelization (-nthreads) and running multiple instances
    #: in parallel causes resource contention and memory-mapping conflicts
    batch_strategy: str = "sequential"

    def __init__(
        self,
        connectome_name: str,
        parcellation_name: str | None = None,
        compute_disconnectivity_matrix: bool = False,
        output_resolution: Literal[1, 2] = 2,
        cache_tdi: bool = True,
        n_jobs: int = 1,
        keep_intermediate: bool = False,
        cleanup_temp_files: bool = True,
        check_dependencies: bool = True,
        verbose: bool = False,
        show_mrtrix_output: bool = False,
        return_in_input_space: bool = True,
    ):
        """Initialize StructuralNetworkMapping analysis.

        Parameters
        ----------
        connectome_name : str
            Name of registered structural connectome (e.g., "HCP842_dTOR").
            Use list_structural_connectomes() to see available options.
        parcellation_name : str, optional
            Name of registered atlas for parcellated connectivity matrices.
        compute_disconnectivity_matrix : bool, default=False
            If True and parcellation_name provided, compute disconnectivity matrices
            including per-ROI disconnection percentages.
        output_resolution : {1, 2}, default=2
            Output resolution in mm (must match connectome resolution).
        cache_tdi : bool, default=True
            If True, cache computed TDI maps.
        n_jobs : int, default=1
            Number of threads for MRtrix3.
        keep_intermediate : bool, default=False
            If True, include intermediate results in output (lesion tractogram,
            lesion TDI, warped atlas). Useful for debugging and QC.
        cleanup_temp_files : bool, default=True
            If True, delete temporary working directory after analysis.
            Set to False to preserve temp files on disk for inspection.
        check_dependencies : bool, default=True
            If True, checks for MRtrix3 availability.
        verbose : bool, default=True
            If True, print progress messages. If False, run silently.
        show_mrtrix_output : bool, default=False
            If True, display MRtrix3 command outputs. If False, suppress verbose
            MRtrix3 messages for cleaner output.
        return_in_input_space : bool, default=True
            If True, transform VoxelMap outputs back to the original input mask space.
            If False, outputs remain in the connectome space.
            Requires input SubjectData to have valid space/resolution metadata.

        Raises
        ------
        MRtrixError
            If MRtrix3 is not available and check_dependencies=True.
        KeyError
            If connectome_name not found in registry.
        ValueError
            If output_resolution is not 1 or 2.
        """
        super().__init__(verbose=verbose, keep_intermediate=keep_intermediate)

        # Validate output_resolution
        if output_resolution not in (1, 2):
            raise ValueError(f"output_resolution must be 1 or 2, got: {output_resolution}")

        # Load connectome from registry
        try:
            connectome = load_structural_connectome(connectome_name)
        except KeyError as e:
            available = [c.name for c in list_structural_connectomes()]
            raise KeyError(
                f"Connectome '{connectome_name}' not found in registry. "
                f"Available connectomes: {', '.join(available)}. "
                f"Use register_structural_connectome() to add new connectomes."
            ) from e

        # Store connectome information
        self.connectome_name = connectome_name
        self.tractogram_path = connectome.tractogram_path
        self.tractogram_space = connectome.metadata.space
        self.template = connectome.template_path  # May be None

        # Store analysis parameters
        self.parcellation_name = parcellation_name
        self.compute_disconnectivity_matrix = compute_disconnectivity_matrix
        self.output_resolution = output_resolution
        self.cache_tdi = cache_tdi
        self.n_jobs = n_jobs
        self.keep_intermediate = keep_intermediate
        self.cleanup_temp_files = cleanup_temp_files
        self.show_mrtrix_output = show_mrtrix_output
        self.return_in_input_space = return_in_input_space

        # Target space matches tractogram space, resolution from output_resolution
        # (used by BaseAnalysis._ensure_target_space)
        self.TARGET_SPACE = self.tractogram_space
        self.TARGET_RESOLUTION = self.output_resolution

        # Initialize logger
        self.logger = ConsoleLogger(verbose=verbose, width=70)

        # Internal state
        self.whole_brain_tdi = None  # Will be set during validation
        self._atlas_image = None
        self._atlas_labels = None
        self._parcellation_resolved = None
        self._full_connectivity_matrix = None
        self._cached_tdi_path = None

        # Check MRtrix3 availability if requested
        if check_dependencies:
            try:
                check_mrtrix_available()
            except MRtrixError as e:
                raise MRtrixError(
                    f"MRtrix3 is required for StructuralNetworkMapping but is not available.\n{e}"
                ) from e

    def _get_tdi_cache_path(self) -> Path:
        """Get deterministic cache path for whole-brain TDI.

        Returns
        -------
        Path
            Path to cached TDI file, unique to tractogram and resolution
        """
        # Create deterministic hash from tractogram path and resolution
        tractogram_str = str(self.tractogram_path.resolve())
        hash_input = f"{tractogram_str}_{self.output_resolution}mm"
        file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]

        # Use unified cache directory
        cache_dir = get_tdi_cache_dir()

        cache_filename = f"tdi_{file_hash}_{self.output_resolution}mm.nii.gz"
        return cache_dir / cache_filename

    def _compute_tdi_to_path(self, output_path: Path) -> None:
        """Compute whole-brain TDI and save to specified path.

        Parameters
        ----------
        output_path : Path
            Where to save the computed TDI
        """
        compute_tdi_map(
            tractogram_path=self.tractogram_path,
            template=self.template,
            output_path=output_path,
            n_jobs=self.n_jobs,
            verbose=self.show_mrtrix_output,
        )

    def _ensure_tdi_cached(self, cache_path: Path) -> None:
        """Ensure TDI is computed and cached, with file locking for parallel safety.

        Uses file locking to prevent race conditions when multiple parallel
        workers try to compute the same TDI simultaneously. The first worker
        to acquire the lock computes the TDI; others wait and use the cached file.

        Parameters
        ----------
        cache_path : Path
            Cache file path from _get_tdi_cache_path()
        """
        import fcntl

        # Check if already cached (fast path, no lock needed)
        if cache_path.exists():
            self.logger.info(f"Using cached TDI: {cache_path}")
            return

        # Use a lock file to coordinate parallel workers
        lock_path = cache_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lock_path, "w") as lock_file:
            self.logger.debug(f"Acquiring TDI cache lock: {lock_path}")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                # Re-check after acquiring lock (another worker may have computed it)
                if cache_path.exists():
                    self.logger.info(
                        f"Using cached TDI (computed by another process): {cache_path}"
                    )
                    return

                # We have the lock and TDI doesn't exist - compute it
                self.logger.info(
                    f"Computing whole-brain TDI at {self.output_resolution}mm resolution..."
                )
                self._compute_tdi_to_path(cache_path)
                self.logger.info(f"Cached TDI to: {cache_path}")
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        # Clean up lock file (best effort, ignore errors)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _compute_and_cache_tdi(self, cache_path: Path) -> None:
        """Compute whole-brain TDI and save to cache.

        Parameters
        ----------
        cache_path : Path
            Cache file path from _get_tdi_cache_path()
        """
        self._compute_tdi_to_path(cache_path)
        self.logger.info(f"Cached TDI to: {cache_path}")

    def run(self, mask_data: SubjectData) -> SubjectData:
        """Run structural network mapping analysis.

        Automatically transforms lesion to tractogram space if needed.

        Parameters
        ----------
        mask_data : SubjectData
            Lesion data to analyze (can be in any MNI152 space)

        Returns
        -------
        SubjectData
            Analysis results

        Raises
        ------
        ValueError
            If lesion is in native space (cannot transform)
        """
        # Check for native space (cannot transform)
        space = mask_data.get_coordinate_space()

        if space.lower() == "native":
            raise ValueError(
                "Native space lesions are not supported for structural network mapping. "
                f"Lesions must be in a standard space. Tractogram is in {self.tractogram_space}."
            )

        # Transform to tractogram space (handled by base class)
        # The base class will handle space equivalence and transformations
        return super().run(mask_data)

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """
        Validate that lesion data meets requirements for structural network mapping.

        This method validates that the mask data is ready for SNM analysis and
        performs essential setup (template loading, TDI computation, atlas loading).
        By the time this is called, BaseAnalysis.run() has already transformed
        the mask to TARGET_SPACE (the tractogram space) via _ensure_target_space().

        Parameters
        ----------
        mask_data : SubjectData
            Lesion data to validate (already transformed to tractogram space).

        Raises
        ------
        ValidationError
            If mask space doesn't match the expected tractogram space.
        FileNotFoundError
            If required input files don't exist (tractogram, template).

        Notes
        -----
        Binary mask validation is handled by SubjectData.__init__, so we don't
        need to duplicate that check here.

        Space transformation is handled by BaseAnalysis._ensure_target_space(),
        so by the time we get here, mask_data.space should equal self.TARGET_SPACE.
        """
        # Validate that required files exist
        if not self.tractogram_path.exists():
            raise FileNotFoundError(f"Tractogram file not found: {self.tractogram_path}")

        # Validate coordinate space matches tractogram space
        # (should already be transformed by _ensure_target_space)
        if mask_data.space != self.TARGET_SPACE:
            raise ValueError(
                f"Mask space '{mask_data.space}' does not match tractogram space "
                f"'{self.TARGET_SPACE}'. This is unexpected - space transformation "
                f"should have been handled by BaseAnalysis._ensure_target_space()."
            )

        # Load template from asset management if not provided (MUST BE DONE BEFORE TDI COMPUTATION)
        if self.template is None:
            # Use output_resolution for template (not lesion resolution)
            # This ensures TDI and template match
            template_name = f"{self.TARGET_SPACE}_res-{self.output_resolution}"

            try:
                template_path = load_template(template_name)
                self.template = template_path
            except (KeyError, FileNotFoundError) as e:
                raise FileNotFoundError(
                    f"Could not load template for {self.TARGET_SPACE} at {self.output_resolution}mm resolution. "
                    f"Template '{template_name}' not found in registry."
                ) from e
        else:
            self.template = Path(self.template)

        # Compute or load TDI with caching (template must be set first!)
        if self.cache_tdi:
            tdi_cache_path = self._get_tdi_cache_path()
            # Use file locking to prevent race conditions in parallel processing
            # Multiple workers may try to compute the same TDI simultaneously
            self._ensure_tdi_cached(tdi_cache_path)
            self.whole_brain_tdi = tdi_cache_path
        else:
            # Compute TDI without caching (temporary file)
            temp_tdi = get_temp_dir(prefix="tdi_") / "whole_brain_tdi.nii.gz"
            self.logger.info(
                f"Computing whole-brain TDI at {self.output_resolution}mm resolution..."
            )
            self._compute_tdi_to_path(temp_tdi)
            self.whole_brain_tdi = temp_tdi

        # Verify template exists
        if not self.template.exists():
            raise FileNotFoundError(f"Template not found: {self.template}")

        # Load atlas from registry
        if self.parcellation_name is not None:
            try:
                atlas = load_parcellation(self.parcellation_name)
                # Store the atlas image for use in analysis
                self._atlas_image = atlas.image
                self._atlas_labels = atlas.labels

                # Check if atlas space matches tractogram space
                atlas_space = atlas.metadata.space
                atlas_resolution = atlas.metadata.resolution

                if atlas_space != self.tractogram_space:
                    self.logger.info(
                        f"Atlas space ({atlas_space}) differs from tractogram space "
                        f"({self.tractogram_space}). Transforming atlas..."
                    )

                    # Transform atlas to tractogram space
                    from lacuna.core.spaces import CoordinateSpace
                    from lacuna.spatial.transform import transform_image
                    from lacuna.utils.cache import get_cache_dir

                    # Define target space matching tractogram
                    template_img = (
                        nib.load(self.template)
                        if isinstance(self.template, (str, Path))
                        else self.template
                    )
                    target_space = CoordinateSpace(
                        identifier=self.tractogram_space,
                        resolution=self.output_resolution,
                        reference_affine=template_img.affine,
                    )

                    # Transform atlas (nearest neighbor interpolation for label preservation)
                    # Logging is handled by transform_image
                    transformed_atlas_img = transform_image(
                        img=self._atlas_image,
                        source_space=atlas_space,
                        target_space=target_space,
                        source_resolution=atlas_resolution,
                        interpolation="nearest",  # Preserve integer labels
                        image_name=f"atlas '{self.parcellation_name}'",
                        verbose=self.verbose,
                    )

                    # Save transformed atlas to cache
                    atlas_cache_dir = get_cache_dir() / "atlases"
                    atlas_cache_dir.mkdir(exist_ok=True, parents=True)

                    # Create deterministic filename based on atlas name and target space
                    atlas_hash = hashlib.md5(
                        f"{self.parcellation_name}_{self.tractogram_space}_{self.output_resolution}".encode()
                    ).hexdigest()[:12]
                    transformed_atlas_path = atlas_cache_dir / f"atlas_{atlas_hash}.nii.gz"

                    # Save and update references
                    nib.save(transformed_atlas_img, transformed_atlas_path)
                    self._parcellation_resolved = transformed_atlas_path
                    self._atlas_image = transformed_atlas_img
                    # Store metadata for intermediate output
                    self._atlas_was_transformed = True
                    self._original_atlas_space = atlas_space
                    self._original_atlas_resolution = atlas_resolution

                    self.logger.info(f"Atlas transformed and cached to: {transformed_atlas_path}")
                else:
                    # No transformation needed - use original atlas file
                    self._atlas_was_transformed = False
                    from lacuna.assets.parcellations.loader import BUNDLED_PARCELLATIONS_DIR

                    atlas_filename_path = Path(atlas.metadata.parcellation_filename)
                    if atlas_filename_path.is_absolute():
                        self._parcellation_resolved = atlas_filename_path
                    else:
                        self._parcellation_resolved = (
                            BUNDLED_PARCELLATIONS_DIR / atlas.metadata.parcellation_filename
                        )

                    if not self._parcellation_resolved.exists():
                        raise FileNotFoundError(
                            f"Atlas file not found: {self._parcellation_resolved}"
                        )

            except KeyError as e:
                available = [a.name for a in list_parcellations()]
                raise ValueError(
                    f"Atlas '{self.parcellation_name}' not found in registry. "
                    f"Available parcellations: {', '.join(available[:5])}... "
                    f"Use list_parcellations() to see all options."
                ) from e

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, "AnalysisResult"]:
        """
        Execute structural network mapping analysis.

        Parameters
        ----------
        mask_data : SubjectData
            Input lesion data

        Returns
        -------
        dict[str, AnalysisResult]
            Dictionary mapping result names to results:
            - 'disconnection_map': VoxelMapResult for disconnection map
            - 'summary_statistics': MiscResult for summary statistics
            - 'mask_tractogram': TractogramResult (if keep_intermediate=True)
            - 'mask_tdi': VoxelMapResult (if keep_intermediate=True)
            - Connectivity results (if atlas provided): see _compute_connectivity_matrices

        Notes
        -----
        Processing steps:
        1. Filter tractogram to streamlines passing through mask (tckedit)
        2. Compute TDI from filtered tractogram (tckmap)
        3. Compute disconnection as ratio of mask TDI to whole-brain TDI (mrcalc)
        """
        # Get subject ID for informative output
        subject_id = mask_data.metadata.get("subject_id", "unknown")

        # Log analysis start
        self.logger.info("Filtering tractogram by mask...")

        # Create temporary directory for intermediate files
        temp_dir_path = get_temp_dir(prefix=f"slnm_{subject_id}_")

        if self.keep_intermediate:
            self.logger.info(f"Intermediate files will be saved to: {temp_dir_path}")

        try:
            # Step 1: Filter tractogram by mask
            mask_tck_path = temp_dir_path / "mask_streamlines.tck"
            filter_tractogram_by_mask(
                tractogram_path=self.tractogram_path,
                mask=mask_data.mask_img,
                output_path=mask_tck_path,
                n_jobs=self.n_jobs,
                force=True,
                verbose=self.show_mrtrix_output,
            )
            self.logger.success("Tractogram filtered")

            # Step 2: Compute TDI from mask-filtered tractogram
            self.logger.info("Computing track-density image (TDI)...")
            # Use anatomical template to define output grid
            mask_tdi_path = temp_dir_path / "mask_tdi.nii.gz"
            compute_tdi_map(
                tractogram_path=mask_tck_path,
                template=self.template,  # Use anatomical template
                output_path=mask_tdi_path,
                n_jobs=self.n_jobs,
                force=True,
                verbose=self.show_mrtrix_output,
            )

            # Step 3: Compute disconnection map
            self.logger.info("Computing disconnection map...")
            disconn_map_path = temp_dir_path / "disconnection_map.nii.gz"
            compute_disconnection_map(
                mask_tdi=mask_tdi_path,
                whole_brain_tdi=self.whole_brain_tdi,
                output_path=disconn_map_path,
                force=True,
                verbose=self.show_mrtrix_output,
            )

            # Load results
            # Use memory-mapped loading for computing statistics efficiently
            disconn_map = nib.load(disconn_map_path, mmap=True)

            # Compute summary statistics (this will load data temporarily but release it)
            disconn_array = disconn_map.get_fdata()
            mean_disconnection = float(np.mean(disconn_array[disconn_array > 0]))

            # Free memory immediately after computing statistics
            del disconn_array

            # Count streamlines in mask tractogram (from TDI sum)
            mask_tdi = nib.load(mask_tdi_path, mmap=True)
            mask_streamline_count = int(np.sum(mask_tdi.get_fdata()))

            # Load disconnection map into memory
            # This ensures results are independent of temp directory lifecycle
            disconn_data = nib.load(disconn_map_path).get_fdata()
            final_disconn_map = nib.Nifti1Image(
                disconn_data, disconn_map.affine, disconn_map.header
            )

            # Build results dict
            results = {}

            # VoxelMapResult for disconnection map
            disconnection_result = VoxelMap(
                name="disconnection_map",
                data=final_disconn_map,
                space=self.tractogram_space,
                resolution=float(self.output_resolution),
                metadata={
                    "tractogram": str(self.tractogram_path),
                    "whole_brain_tdi": str(self.whole_brain_tdi),
                    "template": str(self.template),
                    "n_jobs": self.n_jobs,
                    "keep_intermediate": self.keep_intermediate,
                },
            )
            results["disconnection_map"] = disconnection_result

            # MiscResult for summary statistics
            summary_result = ScalarMetric(
                name="summarystatistics",
                data={
                    "mean_disconnection": mean_disconnection,
                    "mask_streamline_count": mask_streamline_count,
                },
                metadata={
                    "tractogram": str(self.tractogram_path),
                },
            )
            results["summarystatistics"] = summary_result

            # Add intermediate results if keep_intermediate=True
            if self.keep_intermediate:
                # Add mask tractogram as TractogramResult
                mask_tractogram_result = Tractogram(
                    name="mask_tractogram",
                    streamlines=None,  # Not loading into memory
                    tractogram_path=mask_tck_path,
                    metadata={
                        "description": "Tractogram filtered by mask",
                        "temp_directory": str(temp_dir_path),
                    },
                )
                results["mask_tractogram"] = mask_tractogram_result

                # Add mask TDI as VoxelMapResult
                mask_tdi_path = temp_dir_path / "mask_tdi.nii.gz"
                if mask_tdi_path.exists():
                    mask_tdi_img = nib.load(mask_tdi_path)
                    mask_tdi_result = VoxelMap(
                        name="mask_tdi",
                        data=mask_tdi_img,
                        space=self.tractogram_space,
                        resolution=self.output_resolution,
                        metadata={
                            "description": "Track density image for mask-filtered tractogram",
                            "temp_directory": str(temp_dir_path),
                        },
                    )
                    results["mask_tdi"] = mask_tdi_result

                # Add warped atlas if atlas was transformed
                if self._parcellation_resolved is not None and getattr(
                    self, "_atlas_was_transformed", False
                ):
                    warped_atlas_result = VoxelMap(
                        name=f"warped_atlas_{self.parcellation_name}",
                        data=self._atlas_image,
                        space=self.tractogram_space,
                        resolution=self.output_resolution,
                        metadata={
                            "description": (
                                f"Atlas '{self.parcellation_name}' transformed from "
                                f"{self._original_atlas_space}@{self._original_atlas_resolution}mm "
                                f"to {self.tractogram_space}@{self.output_resolution}mm"
                            ),
                            "original_space": self._original_atlas_space,
                            "original_resolution": self._original_atlas_resolution,
                            "parcellation_name": self.parcellation_name,
                            "cached_path": str(self._parcellation_resolved),
                        },
                    )
                    results["warped_atlas"] = warped_atlas_result

            # Optional: Compute parcellated connectivity matrices if atlas provided
            if self._parcellation_resolved is not None:
                self.logger.info("Computing connectivity matrices...")
                connectivity_results = self._compute_connectivity_matrices(
                    mask_data=mask_data,
                    mask_tck_path=mask_tck_path,
                    temp_dir_path=temp_dir_path,
                    subject_id=subject_id,
                )
                # Merge connectivity matrix results into results dict
                results.update(connectivity_results)

            # Transform VoxelMap results back to input space if requested
            if self.return_in_input_space:
                results = self._transform_results_to_input_space(results, mask_data)

            self.logger.success(f"Analysis complete ({len(results)} results)")
            return results

        finally:
            # Clean up temp files based on cleanup_temp_files flag (independent of keep_intermediate)
            if self.cleanup_temp_files:
                import shutil

                shutil.rmtree(temp_dir_path, ignore_errors=True)
            else:
                self.logger.success(f"Temp files preserved in: {temp_dir_path}")
                self.logger.info("Files saved:", indent_level=1)
                self.logger.info("- mask_streamlines.tck", indent_level=2)
                self.logger.info("- mask_tdi.nii.gz", indent_level=2)
                self.logger.info("- disconnection_map.nii.gz", indent_level=2)

    def _compute_connectivity_matrices(
        self,
        mask_data: SubjectData,
        mask_tck_path: Path,
        temp_dir_path: Path,
        subject_id: str,
    ) -> dict[str, "AnalysisResult"]:
        """Compute parcellated connectivity matrices.

        Parameters
        ----------
        mask_data : SubjectData
            Subject data with mask image
        mask_tck_path : Path
            Path to mask-filtered tractogram
        temp_dir_path : Path
            Temporary directory for intermediate files
        subject_id : str
            SubjectData identifier for file naming

        Returns
        -------
        dict[str, AnalysisResult]
            Dictionary containing:
            - 'mask_connectivity_matrix': ConnectivityMatrixResult
            - 'disconnectivity_percent': ConnectivityMatrixResult
            - 'full_connectivity_matrix': ConnectivityMatrixResult
            - 'intact_connectivity_matrix': ConnectivityMatrixResult (if compute_disconnectivity_matrix=True)
            - 'roi_disconnection': ParcelData (if compute_disconnectivity_matrix=True)
            - 'matrix_statistics': MiscResult
        """

        # Step 1: Compute full-brain connectivity matrix (cached)
        if self._full_connectivity_matrix is None:
            self.logger.info(
                "Computing full-brain connectivity matrix (will be cached)", indent_level=1
            )
            self._full_connectivity_matrix = self._compute_connectivity_matrix(
                tractogram_path=self.tractogram_path,
                matrix_name="full_connectivity",
            )
        else:
            self.logger.info("Using cached full-brain connectivity matrix", indent_level=1)

        full_matrix = self._full_connectivity_matrix

        # Step 2: Compute mask connectivity matrix
        self.logger.info("Computing mask connectivity matrix", indent_level=1)
        mask_matrix = self._compute_connectivity_matrix(
            tractogram_path=mask_tck_path,
            matrix_name=f"{subject_id}_mask_connectivity",
        )

        # Step 3: Compute disconnectivity percentage
        self.logger.info("Computing disconnectivity percentage", indent_level=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            disconn_pct = (mask_matrix / full_matrix) * 100

        # Handle division by zero
        disconn_pct = np.nan_to_num(disconn_pct, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 4: Optional - compute intact (post-disconnection) connectivity
        intact_matrix = None
        if self.compute_disconnectivity_matrix:
            self.logger.info("Computing intact (post-disconnection) connectivity matrix", indent_level=1)

            # Save mask temporarily for tckedit -exclude
            exclude_mask_path = temp_dir_path / f"{subject_id}_exclude_mask.nii.gz"
            nib.save(mask_data.mask_img, exclude_mask_path)

            # Filter tractogram to EXCLUDE streamlines through mask
            intact_tck_path = temp_dir_path / f"{subject_id}_intact.tck"
            command = [
                "tckedit",
                str(self.tractogram_path),
                str(intact_tck_path),
                "-exclude",
                str(exclude_mask_path),
                "-force",
            ]
            command.extend(_get_nthreads_args(self.n_jobs))
            run_mrtrix_command(command, verbose=self.show_mrtrix_output)

            # Compute intact connectivity matrix
            intact_matrix = self._compute_connectivity_matrix(
                tractogram_path=intact_tck_path,
                matrix_name=f"{subject_id}_intact_connectivity",
            )

        # Step 5: Compute summary statistics
        matrix_stats = self._compute_matrix_statistics(
            full_matrix=full_matrix,
            mask_matrix=mask_matrix,
            disconn_pct=disconn_pct,
            intact_matrix=intact_matrix,
        )

        # Build results dict
        results = {}

        # Get atlas labels for ConnectivityMatrixResult
        # Convert dict[int, str] to list[str] ordered by region ID
        atlas_labels = [f"region_{i}" for i in range(mask_matrix.shape[0])]
        if hasattr(self, "_atlas_labels") and self._atlas_labels is not None:
            # Sort by region ID and extract names
            sorted_regions = sorted(self._atlas_labels.items())
            atlas_labels = [name for region_id, name in sorted_regions]

        # ConnectivityMatrixResult for mask connectivity
        mask_connectivity_result = ConnectivityMatrix(
            name="mask_connectivity_matrix",
            matrix=mask_matrix,
            region_labels=atlas_labels,
            matrix_type="structural",
            metadata={
                "atlas": self.parcellation_name,
                "tractogram": str(self.tractogram_path),
            },
        )
        mask_conn_key = build_result_key(
            atlas=self.parcellation_name,
            source="StructuralNetworkMapping",
            desc="mask_connectivity_matrix",
        )
        results[mask_conn_key] = mask_connectivity_result

        # ConnectivityMatrixResult for disconnectivity percentage
        disconn_result = ConnectivityMatrix(
            name="disconnectivity_percent",
            matrix=disconn_pct,
            region_labels=atlas_labels,
            matrix_type="structural",
            metadata={
                "atlas": self.parcellation_name,
                "description": "Percentage of streamlines disconnected by mask",
            },
        )
        disconn_pct_key = build_result_key(
            atlas=self.parcellation_name,
            source="StructuralNetworkMapping",
            desc="disconnectivity_percent",
        )
        results[disconn_pct_key] = disconn_result

        # ConnectivityMatrixResult for full connectivity (reference)
        full_connectivity_result = ConnectivityMatrix(
            name="full_connectivity_matrix",
            matrix=full_matrix,
            region_labels=atlas_labels,
            matrix_type="structural",
            metadata={
                "atlas": self.parcellation_name,
                "description": "Full brain connectivity matrix (reference)",
            },
        )
        full_conn_key = build_result_key(
            atlas=self.parcellation_name,
            source="StructuralNetworkMapping",
            desc="full_connectivity_matrix",
        )
        results[full_conn_key] = full_connectivity_result

        # Optional: intact (post-disconnection) connectivity matrix
        if intact_matrix is not None:
            intact_result = ConnectivityMatrix(
                name="intact_connectivity_matrix",
                matrix=intact_matrix,
                region_labels=atlas_labels,
                matrix_type="structural",
                metadata={
                    "atlas": self.parcellation_name,
                    "description": "Intact connectivity excluding disconnected streamlines",
                },
            )
            intact_conn_key = build_result_key(
                atlas=self.parcellation_name,
                source="StructuralNetworkMapping",
                desc="intact_connectivity_matrix",
            )
            results[intact_conn_key] = intact_result

            # Compute per-ROI disconnection percentage
            # For each ROI: (streamlines through mask connecting ROI) / (all streamlines connecting ROI)
            # Sum rows to get total streamlines per ROI (row sum = degree weighted by streamline count)
            full_roi_streamlines = np.sum(full_matrix, axis=1)
            mask_roi_streamlines = np.sum(mask_matrix, axis=1)

            with np.errstate(divide="ignore", invalid="ignore"):
                roi_disconnection_pct = (mask_roi_streamlines / full_roi_streamlines) * 100

            # Handle division by zero
            roi_disconnection_pct = np.nan_to_num(
                roi_disconnection_pct, nan=0.0, posinf=0.0, neginf=0.0
            )

            # Create ParcelData with per-ROI disconnection percentages
            # Convert array to dict mapping region labels to values
            roi_disconnection_data = {
                label: float(roi_disconnection_pct[i]) for i, label in enumerate(atlas_labels)
            }
            roi_disconnection_result = ParcelData(
                name="roi_disconnection",
                data=roi_disconnection_data,
                region_labels=atlas_labels,
                aggregation_method="percent",
                metadata={
                    "atlas": self.parcellation_name,
                    "description": "Percentage of streamlines disconnected per ROI",
                    "unit": "percent",
                },
            )
            # Build BIDS-style result key with atlas prefix
            roi_disconnection_key = build_result_key(
                atlas=self.parcellation_name,
                source="StructuralNetworkMapping",
                desc="roi_disconnection",
            )
            results[roi_disconnection_key] = roi_disconnection_result

        # MiscResult for matrix statistics
        stats_result = ScalarMetric(
            name="matrix_statistics",
            data=matrix_stats,
            metadata={
                "atlas": self.parcellation_name,
            },
        )
        matrix_stats_key = build_result_key(
            atlas=self.parcellation_name,
            source="StructuralNetworkMapping",
            desc="matrix_statistics",
        )
        results[matrix_stats_key] = stats_result

        return results

    def _compute_connectivity_matrix(
        self,
        tractogram_path: Path,
        matrix_name: str,
    ) -> np.ndarray:
        """Compute connectivity matrix from tractogram using tck2connectome.

        Parameters
        ----------
        tractogram_path : Path
            Path to tractogram file
        matrix_name : str
            Name for temporary CSV file

        Returns
        -------
        np.ndarray
            Connectivity matrix (n_parcels x n_parcels)
        """
        from lacuna.utils.cache import make_temp_file

        with make_temp_file(suffix=".csv", delete=True, mode="w") as tmp_csv:
            output_csv = Path(tmp_csv.name)

            command = [
                "tck2connectome",
                str(tractogram_path),
                str(self._parcellation_resolved),
                str(output_csv),
                "-symmetric",
                "-zero_diagonal",
                "-force",
            ]
            command.extend(_get_nthreads_args(self.n_jobs))

            run_mrtrix_command(command, verbose=self.show_mrtrix_output)

            # Load matrix
            matrix = np.loadtxt(output_csv, delimiter=",")

        return matrix

    def _compute_matrix_statistics(
        self,
        full_matrix: np.ndarray,
        mask_matrix: np.ndarray,
        disconn_pct: np.ndarray,
        intact_matrix: np.ndarray | None,
    ) -> dict:
        """Compute summary statistics for connectivity matrices.

        Parameters
        ----------
        full_matrix : np.ndarray
            Full connectivity matrix
        mask_matrix : np.ndarray
            Connectivity matrix from streamlines passing through mask
        disconn_pct : np.ndarray
            Disconnectivity percentage matrix
        intact_matrix : np.ndarray | None
            Intact connectivity matrix (excluding disconnected streamlines)

        Returns
        -------
        dict
            Summary statistics
        """
        # Edge-wise statistics
        n_edges = int(np.sum(full_matrix > 0))
        n_affected_edges = int(np.sum(mask_matrix > 0))
        mean_disconnection_pct = (
            float(np.mean(disconn_pct[full_matrix > 0])) if n_edges > 0 else 0.0
        )

        # Node-wise statistics (degree)
        full_degree = np.sum(full_matrix > 0, axis=1)
        mask_degree = np.sum(mask_matrix > 0, axis=1)
        degree_reduction = full_degree - mask_degree

        stats = {
            "n_parcels": full_matrix.shape[0],
            "n_edges_total": n_edges,
            "n_edges_affected": n_affected_edges,
            "percent_edges_affected": (
                float(n_affected_edges / n_edges * 100) if n_edges > 0 else 0.0
            ),
            "mean_disconnection_percent": mean_disconnection_pct,
            "max_disconnection_percent": float(np.max(disconn_pct)),
            "mean_degree_reduction": float(np.mean(degree_reduction)),
            "max_degree_reduction": int(np.max(degree_reduction)),
            "most_affected_parcel": int(np.argmax(degree_reduction)),
        }

        # Add intact matrix statistics if computed
        if intact_matrix is not None:
            intact_degree = np.sum(intact_matrix > 0, axis=1)
            stats["intact_mean_degree"] = float(np.mean(intact_degree))

            # Quality control: mask + intact should approximately equal full
            combined = mask_matrix + intact_matrix
            preservation = np.sum(combined > 0) / n_edges if n_edges > 0 else 0
            stats["connectivity_preservation_ratio"] = float(preservation)

        return stats

    def _get_version(self) -> str:
        """Get analysis version for provenance tracking."""
        from .. import __version__

        return __version__

    def _get_parameters(self) -> dict:
        """Get analysis parameters for provenance and display.

        Returns
        -------
        dict
            Dictionary of parameter names and values.
        """
        return {
            "connectome_name": self.connectome_name,
            "parcellation_name": str(self.parcellation_name) if self.parcellation_name else None,
            "compute_disconnectivity_matrix": self.compute_disconnectivity_matrix,
            "output_resolution": self.output_resolution,
            "n_jobs": self.n_jobs,
            "keep_intermediate": self.keep_intermediate,
            "cleanup_temp_files": self.cleanup_temp_files,
            "show_mrtrix_output": self.show_mrtrix_output,
            "return_in_input_space": self.return_in_input_space,
            "verbose": self.verbose,
        }

    def _transform_results_to_input_space(self, results: dict, mask_data: SubjectData) -> dict:
        """Transform VoxelMap results back to original input mask space.

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

        Notes
        -----
        Transforms to original input space but keeps output_resolution for
        consistent output resolution. The output_resolution parameter controls
        both TDI generation and final output resolution.
        """
        from lacuna.core.spaces import REFERENCE_AFFINES, CoordinateSpace
        from lacuna.spatial.transform import transform_image

        # Get original input space from metadata (set by BaseAnalysis before transformation)
        # Fall back to current mask_data.space if not available (already in input space)
        original_space = mask_data.metadata.get("_original_input_space", mask_data.space)

        # Use output_resolution for final output (not input resolution)
        # This ensures consistent output resolution as specified by user
        target_resolution = self.output_resolution

        # Get reference affine for target space
        target_key = (original_space, target_resolution)
        if target_key not in REFERENCE_AFFINES:
            raise ValueError(
                f"No reference affine available for {original_space}@{target_resolution}mm. "
                f"Available spaces: {list(REFERENCE_AFFINES.keys())}"
            )

        target_space = CoordinateSpace(
            identifier=original_space,
            resolution=target_resolution,
            reference_affine=REFERENCE_AFFINES[target_key],
        )

        # Check if transformation is actually needed
        if original_space == self.TARGET_SPACE and target_resolution == self.output_resolution:
            return results

        self.logger.info(
            f"Transforming VoxelMap outputs from {self.TARGET_SPACE}@{self.output_resolution}mm "
            f"to {target_space.identifier}@{target_space.resolution}mm"
        )

        transformed_results = {}
        for key, result in results.items():
            # Only transform VoxelMap results
            from lacuna.core.data_types import VoxelMap

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
                    source_space=self.TARGET_SPACE,
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
                        "transformed_from": f"{self.TARGET_SPACE}@{self.output_resolution}mm",
                        "transformed_to": f"{target_space.identifier}@{target_space.resolution}mm",
                    },
                )
                transformed_results[key] = transformed_result
            else:
                # Keep non-VoxelMap results as-is
                transformed_results[key] = result

        return transformed_results
