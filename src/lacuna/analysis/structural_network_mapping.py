"""
Structural lesion network mapping (sLNM) analysis using tractography.

Computes white matter disconnection maps by filtering a whole-brain tractogram
through a lesion mask and comparing the resulting track density to the intact
white matter connectivity.
"""

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import nibabel as nib
import numpy as np

from lacuna.analysis.base import BaseAnalysis
from lacuna.assets import load_template
from lacuna.assets.atlases import list_atlases, load_atlas
from lacuna.core.data_types import (
    ConnectivityMatrix,
    ScalarMetric,
    Tractogram,
    VoxelMap,
)
from lacuna.core.mask_data import MaskData
from lacuna.utils.cache import get_tdi_cache_dir
from lacuna.utils.logging import ConsoleLogger
from lacuna.utils.mrtrix import (
    MRtrixError,
    check_mrtrix_available,
    compute_disconnection_map,
    compute_tdi_map,
    filter_tractogram_by_lesion,
    run_mrtrix_command,
)

if TYPE_CHECKING:
    from lacuna.core.data_types import AnalysisResult

logger = logging.getLogger(__name__)


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
    tractogram_path : str or Path
        Path to whole-brain tractogram file (.tck format from MRtrix3)
    whole_brain_tdi : str or Path
        Path to pre-computed whole-brain TDI map (reference connectivity)
    template : str or Path, optional
        Path to template image defining output grid. If not provided, automatically
        loads bundled MNI152 template matching the lesion resolution (1mm or 2mm).
    n_jobs : int, default=1
        Number of threads for MRtrix3 processing
    keep_intermediate : bool, default=False
        If True, keeps intermediate tractogram files (.tck) for debugging/inspection
    load_to_memory : bool, default=True
        If True, loads disconnection maps into memory (convenient for visualization).
        If False, uses memory-mapped files (memory-efficient for batch processing).
        For large batch jobs, set to False to minimize RAM usage.
    check_dependencies : bool, default=True
        If True, checks for MRtrix3 availability during initialization

    Raises
    ------
    MRtrixError
        If MRtrix3 is not installed or not available in PATH
    FileNotFoundError
        If tractogram_path or whole_brain_tdi files don't exist

    Examples
    --------
    **Interactive analysis (default - results loaded into memory):**

    >>> from lacuna import MaskData
    >>> from lacuna.analysis import StructuralNetworkMapping
    >>>
    >>> # Load lesion data
    >>> lesion = MaskData.from_nifti("lesion.nii.gz")
    >>>
    >>> # Create analysis (template auto-detected from lesion resolution)
    >>> analysis = StructuralNetworkMapping(
    ...     tractogram_path="/data/dTOR_full_tractogram.tck",
    ...     whole_brain_tdi="/data/dTOR_tdi_2mm.nii.gz",
    ...     n_jobs=8,
    ...     load_to_memory=True  # default - convenient for visualization
    ... )
    >>>
    >>> # Run analysis
    >>> result = analysis.run(lesion)
    >>>
    >>> # Visualize immediately (data in memory)
    >>> disconn_map = result.results["StructuralNetworkMapping"]["disconnection_map"]
    >>> disconn_map.orthoview()

    **Memory-efficient batch processing:**

    >>> from lacuna import batch_process
    >>>
    >>> # For large batch jobs, use memory-mapped files
    >>> analysis = StructuralNetworkMapping(
    ...     tractogram_path="/data/dTOR_full_tractogram.tck",
    ...     whole_brain_tdi="/data/dTOR_tdi_2mm.nii.gz",
    ...     n_jobs=8,
    ...     keep_intermediate=True,  # required to keep temp files
    ...     load_to_memory=False     # memory-efficient mode
    ... )
    >>>
    >>> results = batch_process(lesions, analysis, n_jobs=2)
    >>>
    >>> # Save results immediately (temp files will be cleaned up eventually)
    >>> for result in results:
    ...     disconn_map = result.results["StructuralNetworkMapping"]["disconnection_map"]
    ...     nib.save(disconn_map, f"output/{subject_id}_disconn.nii.gz")

    Notes
    -----
    - Requires MRtrix3: https://www.mrtrix.org/download/
    - Lesion must be in MNI152 space (1mm or 2mm resolution supported)
    - Set lesion.metadata['space'] = 'MNI152NLin6Asym' or 'MNI152NLin2009cAsym' before analysis
    - Lesion must be binary (0/1 values only)
    - Template and TDI resolution should match your lesion resolution
    - Processing time scales with lesion size and tractogram density
    - For large tractograms, processing can take several minutes per subject
    - **Memory management**: Use load_to_memory=False for large batch jobs to minimize RAM usage

    See Also
    --------
    FunctionalNetworkMapping : Functional connectivity-based lesion network mapping
    RegionalDamage : Atlas-based regional overlap quantification
    """

    #: Preferred batch processing strategy
    batch_strategy: str = "parallel"

    def __init__(
        self,
        tractogram_path: str | Path,
        tractogram_space: str = "MNI152NLin2009cAsym",
        template: str | Path | None = None,
        atlas_name: str | None = None,
        compute_lesioned: bool = False,
        output_resolution: Literal[1, 2] = 2,
        cache_tdi: bool = True,
        n_jobs: int = 1,
        keep_intermediate: bool = False,
        load_to_memory: bool = True,
        check_dependencies: bool = True,
        verbose: bool = False,
    ):
        """Initialize StructuralNetworkMapping analysis.

        Parameters
        ----------
        tractogram_path : str | Path
            Path to whole-brain tractogram (.tck file)
        tractogram_space : str, default="MNI152NLin2009cAsym"
            Coordinate space of the tractogram. This determines how lesions
            in different spaces are transformed before analysis.
            Supported: "MNI152NLin6Asym", "MNI152NLin2009cAsym"
        template : str | Path | None, optional
            Template image for output grid. Auto-detected if None.
        atlas_name : str | None, optional
            Name of atlas from registry for computing connectivity matrices.
            Options:
            - None: Skip matrix computation (default, voxel-wise map only)
            - Atlas name from registry (e.g., "Schaefer2018_100Parcels7Networks")
            Use list_atlases() to see available atlases.
            When provided, computes lesion connectivity matrix and
            disconnectivity percentage per edge.
        compute_lesioned : bool, default=False
            If True and atlas_name provided, also compute the "lesioned"
            connectivity matrix (streamlines NOT passing through lesion),
            representing intact structural connectivity.
        output_resolution : {1, 2}, default=2
            Output resolution in millimeters. Determines the template resolution
            and TDI computation grid.
            - 1: High resolution (182×218×182 voxels)
            - 2: Standard resolution (91×109×91 voxels, faster)
        cache_tdi : bool, default=True
            Cache computed whole-brain TDI for reuse in batch processing.
            When True, TDI is computed once and reused across subjects.
            When False, TDI is recomputed for each subject.
        n_jobs : int, default=1
            Number of threads for MRtrix3 operations
        keep_intermediate : bool, default=False
            Keep intermediate tractogram files for inspection
        load_to_memory : bool, default=True
            Load maps into memory (True) or use memory-mapped files (False)
        check_dependencies : bool, default=True
            Check MRtrix3 availability at initialization

        Raises
        ------
        ValueError
            If output_resolution is not 1 or 2
        """
        super().__init__()

        # Validate output_resolution
        if output_resolution not in (1, 2):
            raise ValueError(f"output_resolution must be 1 or 2, got: {output_resolution}")

        self.tractogram_path = Path(tractogram_path)
        self.tractogram_space = tractogram_space
        self.output_resolution = output_resolution
        self.cache_tdi = cache_tdi
        self.whole_brain_tdi = None  # Will be set during validation

        self.template = Path(template) if template is not None else None
        self.atlas_name = atlas_name  # Atlas name from registry
        self.compute_lesioned = compute_lesioned
        self.n_jobs = n_jobs
        self.keep_intermediate = keep_intermediate
        self.load_to_memory = load_to_memory
        self._check_dependencies = check_dependencies
        self.verbose = verbose

        # Target space matches tractogram space
        self.TARGET_SPACE = self.tractogram_space

        # Initialize logger
        self.logger = ConsoleLogger(verbose=verbose, width=70)

        # Will be resolved to Path during validation
        self._atlas_resolved = None
        self._atlas_labels = None
        # Cache for full connectivity matrix (computed once if atlas provided)
        self._full_connectivity_matrix = None
        # Cache for computed whole-brain TDI
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
            verbose=self.verbose,
        )

    def _compute_and_cache_tdi(self, cache_path: Path) -> None:
        """Compute whole-brain TDI and save to cache.

        Parameters
        ----------
        cache_path : Path
            Cache file path from _get_tdi_cache_path()
        """
        self._compute_tdi_to_path(cache_path)
        logger.info(f"Cached TDI to: {cache_path}")

    def run(self, mask_data: MaskData) -> MaskData:
        """Run structural network mapping analysis.

        Automatically transforms lesion to tractogram space if needed.

        Parameters
        ----------
        mask_data : MaskData
            Lesion data to analyze (can be in any MNI152 space)

        Returns
        -------
        MaskData
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

    def _validate_inputs(self, mask_data: MaskData) -> None:
        """
        Validate that lesion data meets requirements for structural network mapping.

        Parameters
        ----------
        mask_data : MaskData
            Lesion data to validate

        Raises
        ------
        ValueError
            If lesion is not in MNI152 space or is not binary
        FileNotFoundError
            If required input files don't exist
        """
        # Validate that required files exist
        if not self.tractogram_path.exists():
            raise FileNotFoundError(f"Tractogram file not found: {self.tractogram_path}")

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
            if tdi_cache_path.exists():
                self.whole_brain_tdi = tdi_cache_path
                logger.info(f"Using cached TDI: {tdi_cache_path}")
            else:
                # Compute TDI and cache it
                logger.info(
                    f"Computing whole-brain TDI at {self.output_resolution}mm resolution..."
                )
                self._compute_and_cache_tdi(tdi_cache_path)
                self.whole_brain_tdi = tdi_cache_path
        else:
            # Compute TDI without caching (temporary file)
            temp_tdi = Path(tempfile.mkdtemp()) / "whole_brain_tdi.nii.gz"
            logger.info(f"Computing whole-brain TDI at {self.output_resolution}mm resolution...")
            self._compute_tdi_to_path(temp_tdi)
            self.whole_brain_tdi = temp_tdi

        # Space validation is handled in run() method before transformation
        # At this point, mask_data should already be in TARGET_SPACE

        # Verify template exists
        if not self.template.exists():
            raise FileNotFoundError(f"Template not found: {self.template}")

        # Load atlas from registry
        if self.atlas_name is not None:
            try:
                atlas = load_atlas(self.atlas_name)
                # Store the atlas image for use in analysis
                self._atlas_image = atlas.image
                self._atlas_labels = atlas.labels

                # Check if atlas space matches tractogram space
                atlas_space = atlas.metadata.space
                atlas_resolution = atlas.metadata.resolution

                if atlas_space != self.tractogram_space:
                    logger.info(
                        f"Atlas space ({atlas_space}) differs from tractogram space ({self.tractogram_space}). "
                        f"Transforming atlas to {self.tractogram_space}..."
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
                        image_name=f"atlas '{self.atlas_name}'",
                        log_level=self.log_level,
                    )

                    # Save transformed atlas to cache
                    atlas_cache_dir = get_cache_dir() / "atlases"
                    atlas_cache_dir.mkdir(exist_ok=True, parents=True)

                    # Create deterministic filename based on atlas name and target space
                    atlas_hash = hashlib.md5(
                        f"{self.atlas_name}_{self.tractogram_space}_{self.output_resolution}".encode()
                    ).hexdigest()[:12]
                    transformed_atlas_path = atlas_cache_dir / f"atlas_{atlas_hash}.nii.gz"

                    # Save and update references
                    nib.save(transformed_atlas_img, transformed_atlas_path)
                    self._atlas_resolved = transformed_atlas_path
                    self._atlas_image = transformed_atlas_img

                    logger.info(f"Atlas transformed and cached to: {transformed_atlas_path}")
                else:
                    # No transformation needed - use original atlas file
                    from lacuna.assets.atlases.loader import BUNDLED_ATLASES_DIR

                    atlas_filename_path = Path(atlas.metadata.atlas_filename)
                    if atlas_filename_path.is_absolute():
                        self._atlas_resolved = atlas_filename_path
                    else:
                        self._atlas_resolved = BUNDLED_ATLASES_DIR / atlas.metadata.atlas_filename

                    if not self._atlas_resolved.exists():
                        raise FileNotFoundError(f"Atlas file not found: {self._atlas_resolved}")

            except KeyError as e:
                available = [a.name for a in list_atlases()]
                raise ValueError(
                    f"Atlas '{self.atlas_name}' not found in registry. "
                    f"Available atlases: {', '.join(available[:5])}... "
                    f"Use list_atlases() to see all options."
                ) from e

        # Check that lesion is binary
        lesion_array = mask_data.mask_img.get_fdata()
        unique_vals = np.unique(lesion_array)

        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(
                f"Structural network mapping requires binary lesion mask (0 and 1 only).\n"
                f"Found values: {unique_vals}\n"
                f"Use thresholding or binarization to convert continuous maps."
            )

    def _run_analysis(self, mask_data: MaskData) -> dict[str, "AnalysisResult"]:
        """
        Execute structural network mapping analysis.

        Parameters
        ----------
        mask_data : MaskData
            Input lesion data

        Returns
        -------
        dict[str, AnalysisResult]
            Dictionary mapping result names to results:
            - 'disconnection_map': VoxelMap for disconnection map
            - 'summary_statistics': ScalarMetric for summary statistics
            - 'lesion_tractogram': Tractogram (if keep_intermediate=True)
            - 'lesion_tdi': VoxelMap (if keep_intermediate=True)
            - Connectivity results (if atlas provided): see _compute_connectivity_matrices

        Notes
        -----
        Processing steps:
        1. Filter tractogram to streamlines passing through lesion (tckedit)
        2. Compute TDI from filtered tractogram (tckmap)
        3. Compute disconnection as ratio of lesion TDI to whole-brain TDI (mrcalc)
        """
        # Get subject ID for informative output
        subject_id = mask_data.metadata.get("subject_id", "unknown")

        # Subject header
        self.logger.section(f"PROCESSING: {subject_id}")

        # Create temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp(prefix=f"slnm_{subject_id}_")
        temp_dir_path = Path(temp_dir)

        if self.keep_intermediate:
            self.logger.info(f"Intermediate files will be saved to: {temp_dir}")

        try:
            # Step 1: Filter tractogram by lesion
            lesion_tck_path = temp_dir_path / "lesion_streamlines.tck"
            filter_tractogram_by_lesion(
                tractogram_path=self.tractogram_path,
                lesion_mask=mask_data.mask_img,
                output_path=lesion_tck_path,
                n_jobs=self.n_jobs,
                force=True,
                verbose=self.verbose,
            )

            # Step 2: Compute TDI from lesion-filtered tractogram
            # Use anatomical template to define output grid
            lesion_tdi_path = temp_dir_path / "lesion_tdi.nii.gz"
            compute_tdi_map(
                tractogram_path=lesion_tck_path,
                template=self.template,  # Use anatomical template
                output_path=lesion_tdi_path,
                n_jobs=self.n_jobs,
                force=True,
                verbose=self.verbose,
            )

            # Step 3: Compute disconnection map
            disconn_map_path = temp_dir_path / "disconnection_map.nii.gz"
            compute_disconnection_map(
                lesion_tdi=lesion_tdi_path,
                whole_brain_tdi=self.whole_brain_tdi,
                output_path=disconn_map_path,
                force=True,
                verbose=self.verbose,
            )

            # Load results
            # Use memory-mapped loading for computing statistics efficiently
            disconn_map = nib.load(disconn_map_path, mmap=True)

            # Compute summary statistics (this will load data temporarily but release it)
            disconn_array = disconn_map.get_fdata()
            mean_disconnection = float(np.mean(disconn_array[disconn_array > 0]))

            # Free memory immediately after computing statistics
            del disconn_array

            # Count streamlines in lesion tractogram (from TDI sum)
            lesion_tdi = nib.load(lesion_tdi_path, mmap=True)
            lesion_streamline_count = int(np.sum(lesion_tdi.get_fdata()))

            # Determine how to store the disconnection map based on load_to_memory setting
            if self.load_to_memory:
                # Load into memory for convenient visualization/access
                # Suitable for interactive analysis and small-to-medium batch jobs
                disconn_data = nib.load(disconn_map_path).get_fdata()
                final_disconn_map = nib.Nifti1Image(
                    disconn_data, disconn_map.affine, disconn_map.header
                )
            else:
                # Keep as memory-mapped file for minimal memory footprint
                # Requires keep_intermediate=True to prevent file deletion
                # Suitable for large batch processing with limited RAM
                if not self.keep_intermediate:
                    raise ValueError(
                        "load_to_memory=False requires keep_intermediate=True "
                        "to prevent temp file cleanup before results can be accessed."
                    )
                final_disconn_map = disconn_map

            # Build results dict
            results = {}

            # VoxelMap for disconnection map
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
                    "load_to_memory": self.load_to_memory,
                },
            )
            results["disconnection_map"] = disconnection_result

            # ScalarMetric for summary statistics
            summary_result = ScalarMetric(
                name="summary_statistics",
                data={
                    "mean_disconnection": mean_disconnection,
                    "lesion_streamline_count": lesion_streamline_count,
                },
                metadata={
                    "tractogram": str(self.tractogram_path),
                },
            )
            results["summary_statistics"] = summary_result

            # Add intermediate results if keep_intermediate=True
            if self.keep_intermediate:
                # Add lesion tractogram as Tractogram
                lesion_tractogram_result = Tractogram(
                    name="lesion_tractogram",
                    streamlines=None,  # Not loading into memory
                    tractogram_path=lesion_tck_path,
                    metadata={
                        "description": "Tractogram filtered by lesion mask",
                        "temp_directory": str(temp_dir_path),
                    },
                )
                results["lesion_tractogram"] = lesion_tractogram_result

                # Add lesion TDI as VoxelMap
                lesion_tdi_path = temp_dir_path / "lesion_tdi.nii.gz"
                if lesion_tdi_path.exists():
                    lesion_tdi_img = nib.load(lesion_tdi_path)
                    lesion_tdi_result = VoxelMap(
                        name="lesion_tdi",
                        data=lesion_tdi_img,
                        space=self.tractogram_space,
                        resolution=self.output_resolution,
                        metadata={
                            "description": "Track density image for lesion-filtered tractogram",
                            "temp_directory": str(temp_dir_path),
                        },
                    )
                    results["lesion_tdi"] = lesion_tdi_result

            # Optional: Compute parcellated connectivity matrices if atlas provided
            if self._atlas_resolved is not None:
                self.logger.subsection("Computing Connectivity Matrices")
                connectivity_results = self._compute_connectivity_matrices(
                    mask_data=mask_data,
                    lesion_tck_path=lesion_tck_path,
                    temp_dir_path=temp_dir_path,
                    subject_id=subject_id,
                )
                # Merge connectivity matrix results into results dict
                results.update(connectivity_results)

            return results

        finally:
            # Clean up intermediate files unless keep_intermediate=True
            if not self.keep_intermediate:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                self.logger.success(f"Intermediate files preserved in: {temp_dir}")
                self.logger.info("Files saved:", indent_level=1)
                self.logger.info("- lesion_streamlines.tck", indent_level=2)
                self.logger.info("- lesion_tdi.nii.gz", indent_level=2)
                self.logger.info("- disconnection_map.nii.gz", indent_level=2)

    def _compute_connectivity_matrices(
        self,
        mask_data: MaskData,
        lesion_tck_path: Path,
        temp_dir_path: Path,
        subject_id: str,
    ) -> dict[str, "AnalysisResult"]:
        """Compute parcellated connectivity matrices.

        Parameters
        ----------
        mask_data : MaskData
            Lesion data with mask image
        lesion_tck_path : Path
            Path to lesion-filtered tractogram
        temp_dir_path : Path
            Temporary directory for intermediate files
        subject_id : str
            Subject identifier for file naming

        Returns
        -------
        dict[str, AnalysisResult]
            Dictionary containing:
            - 'lesion_connectivity_matrix': ConnectivityMatrix
            - 'disconnectivity_percent': ConnectivityMatrix
            - 'full_connectivity_matrix': ConnectivityMatrix
            - 'lesioned_connectivity_matrix': ConnectivityMatrix (if compute_lesioned=True)
            - 'matrix_statistics': ScalarMetric
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

        # Step 2: Compute lesion connectivity matrix
        self.logger.info("Computing lesion connectivity matrix", indent_level=1)
        lesion_matrix = self._compute_connectivity_matrix(
            tractogram_path=lesion_tck_path,
            matrix_name=f"{subject_id}_lesion_connectivity",
        )

        # Step 3: Compute disconnectivity percentage
        self.logger.info("Computing disconnectivity percentage", indent_level=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            disconn_pct = (lesion_matrix / full_matrix) * 100

        # Handle division by zero
        disconn_pct = np.nan_to_num(disconn_pct, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 4: Optional - compute lesioned (intact) connectivity
        lesioned_matrix = None
        if self.compute_lesioned:
            self.logger.info("Computing lesioned (intact) connectivity matrix", indent_level=1)

            # Save lesion mask temporarily for tckedit -exclude
            lesion_mask_path = temp_dir_path / f"{subject_id}_lesion_mask.nii.gz"
            nib.save(mask_data.mask_img, lesion_mask_path)

            # Filter tractogram to EXCLUDE streamlines through lesion
            lesioned_tck_path = temp_dir_path / f"{subject_id}_lesioned.tck"
            command = [
                "tckedit",
                str(self.tractogram_path),
                str(lesioned_tck_path),
                "-exclude",
                str(lesion_mask_path),
                "-nthreads",
                str(self.n_jobs),
                "-force",
            ]
            run_mrtrix_command(command, verbose=self.verbose)

            # Compute lesioned connectivity matrix
            lesioned_matrix = self._compute_connectivity_matrix(
                tractogram_path=lesioned_tck_path,
                matrix_name=f"{subject_id}_lesioned_connectivity",
            )

        # Step 5: Compute summary statistics
        matrix_stats = self._compute_matrix_statistics(
            full_matrix=full_matrix,
            lesion_matrix=lesion_matrix,
            disconn_pct=disconn_pct,
            lesioned_matrix=lesioned_matrix,
        )

        # Build results dict
        results = {}

        # Get atlas labels for ConnectivityMatrix
        # Convert dict[int, str] to list[str] ordered by region ID
        atlas_labels = [f"region_{i}" for i in range(lesion_matrix.shape[0])]
        if hasattr(self, "_atlas_labels") and self._atlas_labels is not None:
            # Sort by region ID and extract names
            sorted_regions = sorted(self._atlas_labels.items())
            atlas_labels = [name for region_id, name in sorted_regions]

        # ConnectivityMatrix for lesion connectivity
        lesion_connectivity_result = ConnectivityMatrix(
            name="lesion_connectivity_matrix",
            matrix=lesion_matrix,
            region_labels=atlas_labels,
            matrix_type="structural",
            metadata={
                "atlas": self.atlas_name,
                "tractogram": str(self.tractogram_path),
            },
        )
        results["lesion_connectivity_matrix"] = lesion_connectivity_result

        # ConnectivityMatrix for disconnectivity percentage
        disconn_result = ConnectivityMatrix(
            name="disconnectivity_percent",
            matrix=disconn_pct,
            region_labels=atlas_labels,
            matrix_type="structural",
            metadata={
                "atlas": self.atlas_name,
                "description": "Percentage of streamlines disconnected by lesion",
            },
        )
        results["disconnectivity_percent"] = disconn_result

        # ConnectivityMatrix for full connectivity (reference)
        full_connectivity_result = ConnectivityMatrix(
            name="full_connectivity_matrix",
            matrix=full_matrix,
            region_labels=atlas_labels,
            matrix_type="structural",
            metadata={
                "atlas": self.atlas_name,
                "description": "Full brain connectivity matrix (reference)",
            },
        )
        results["full_connectivity_matrix"] = full_connectivity_result

        # Optional: lesioned (intact) connectivity matrix
        if lesioned_matrix is not None:
            lesioned_result = ConnectivityMatrix(
                name="lesioned_connectivity_matrix",
                matrix=lesioned_matrix,
                region_labels=atlas_labels,
                matrix_type="structural",
                metadata={
                    "atlas": self.atlas_name,
                    "description": "Intact connectivity excluding lesion streamlines",
                },
            )
            results["lesioned_connectivity_matrix"] = lesioned_result

        # ScalarMetric for matrix statistics
        stats_result = ScalarMetric(
            name="matrix_statistics",
            data=matrix_stats,
            metadata={
                "atlas": self.atlas_name,
            },
        )
        results["matrix_statistics"] = stats_result

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
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=True, mode="w") as tmp_csv:
            output_csv = Path(tmp_csv.name)

            command = [
                "tck2connectome",
                str(tractogram_path),
                str(self._atlas_resolved),
                str(output_csv),
                "-symmetric",
                "-zero_diagonal",
                "-nthreads",
                str(self.n_jobs),
                "-force",
            ]

            run_mrtrix_command(command, verbose=self.verbose)

            # Load matrix
            matrix = np.loadtxt(output_csv, delimiter=",")

        return matrix

    def _compute_matrix_statistics(
        self,
        full_matrix: np.ndarray,
        lesion_matrix: np.ndarray,
        disconn_pct: np.ndarray,
        lesioned_matrix: np.ndarray | None,
    ) -> dict:
        """Compute summary statistics for connectivity matrices.

        Parameters
        ----------
        full_matrix : np.ndarray
            Full connectivity matrix
        lesion_matrix : np.ndarray
            Lesion connectivity matrix
        disconn_pct : np.ndarray
            Disconnectivity percentage matrix
        lesioned_matrix : np.ndarray | None
            Lesioned (intact) connectivity matrix

        Returns
        -------
        dict
            Summary statistics
        """
        # Edge-wise statistics
        n_edges = int(np.sum(full_matrix > 0))
        n_affected_edges = int(np.sum(lesion_matrix > 0))
        mean_disconnection_pct = (
            float(np.mean(disconn_pct[full_matrix > 0])) if n_edges > 0 else 0.0
        )

        # Node-wise statistics (degree)
        full_degree = np.sum(full_matrix > 0, axis=1)
        lesion_degree = np.sum(lesion_matrix > 0, axis=1)
        degree_reduction = full_degree - lesion_degree

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

        # Add lesioned matrix statistics if computed
        if lesioned_matrix is not None:
            lesioned_degree = np.sum(lesioned_matrix > 0, axis=1)
            stats["lesioned_mean_degree"] = float(np.mean(lesioned_degree))

            # Quality control: lesion + lesioned should approximately equal full
            combined = lesion_matrix + lesioned_matrix
            preservation = np.sum(combined > 0) / n_edges if n_edges > 0 else 0
            stats["connectivity_preservation_ratio"] = float(preservation)

        return stats

    def _get_parameters(self) -> dict:
        """Get analysis parameters for provenance tracking."""
        return {
            "tractogram_path": str(self.tractogram_path),
            "whole_brain_tdi": str(self.whole_brain_tdi),
            "template": str(self.template),
            "n_jobs": self.n_jobs,
            "keep_intermediate": self.keep_intermediate,
            "load_to_memory": self.load_to_memory,
            "verbose": self.verbose,
        }

    def _get_version(self) -> str:
        """Get analysis version."""
        return "0.1.0"

    def _get_parameters(self) -> dict:
        """Get analysis parameters for provenance and display.

        Returns
        -------
        dict
            Dictionary of parameter names and values.
        """
        return {
            "tractogram_path": str(self.tractogram_path),
            "whole_brain_tdi": str(self.whole_brain_tdi),
            "template": str(self.template) if self.template else None,
            "atlas_name": str(self.atlas_name) if self.atlas_name else None,
            "compute_lesioned": self.compute_lesioned,
            "n_jobs": self.n_jobs,
            "keep_intermediate": self.keep_intermediate,
            "load_to_memory": self.load_to_memory,
            "verbose": self.verbose,
        }
