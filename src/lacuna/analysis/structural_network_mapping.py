"""
Structural lesion network mapping (sLNM) analysis using tractography.

Computes white matter disconnection maps by filtering a whole-brain tractogram
through a lesion mask and comparing the resulting track density to the intact
white matter connectivity.
"""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from lacuna.analysis.base import BaseAnalysis
from lacuna.core.lesion_data import LesionData
from lacuna.data import get_bundled_atlas, get_template_path
from lacuna.utils.logging import ConsoleLogger
from lacuna.utils.mrtrix import (
    MRtrixError,
    check_mrtrix_available,
    compute_disconnection_map,
    compute_tdi_map,
    filter_tractogram_by_lesion,
    run_mrtrix_command,
)


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

    >>> from lacuna import LesionData
    >>> from lacuna.analysis import StructuralNetworkMapping
    >>>
    >>> # Load lesion data
    >>> lesion = LesionData.from_nifti("lesion.nii.gz")
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
    - Set lesion.metadata['space'] = 'MNI152_2mm' or 'MNI152_1mm' before analysis
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
        whole_brain_tdi: str | Path,
        template: str | Path | None = None,
        atlas_path: str | Path | None = None,
        compute_lesioned: bool = False,
        n_jobs: int = 1,
        keep_intermediate: bool = False,
        load_to_memory: bool = True,
        check_dependencies: bool = True,
        verbose: bool = True,
    ):
        """Initialize StructuralNetworkMapping analysis.

        Parameters
        ----------
        tractogram_path : str | Path
            Path to whole-brain tractogram (.tck file)
        whole_brain_tdi : str | Path
            Path to pre-computed whole-brain TDI map
        template : str | Path | None, optional
            Template image for output grid. Auto-detected if None.
        atlas_path : str | Path | None, optional
            Parcellation atlas for computing connectivity matrices.
            Options:
            - None: Skip matrix computation (default, voxel-wise map only)
            - "schaefer100", "schaefer200", etc.: Use bundled atlas
            - Path to custom atlas NIfTI file
            When provided, computes lesion connectivity matrix and
            disconnectivity percentage per edge.
        compute_lesioned : bool, default=False
            If True and atlas_path provided, also compute the "lesioned"
            connectivity matrix (streamlines NOT passing through lesion),
            representing intact structural connectivity.
        n_jobs : int, default=1
            Number of threads for MRtrix3 operations
        keep_intermediate : bool, default=False
            Keep intermediate tractogram files for inspection
        load_to_memory : bool, default=True
            Load maps into memory (True) or use memory-mapped files (False)
        check_dependencies : bool, default=True
            Check MRtrix3 availability at initialization
        """
        super().__init__()

        self.tractogram_path = Path(tractogram_path)
        self.whole_brain_tdi = Path(whole_brain_tdi)
        self.template = Path(template) if template is not None else None
        self.atlas_path = atlas_path  # Can be str (bundled) or Path (custom)
        self.compute_lesioned = compute_lesioned
        self.n_jobs = n_jobs
        self.keep_intermediate = keep_intermediate
        self.load_to_memory = load_to_memory
        self._check_dependencies = check_dependencies
        self.verbose = verbose

        # Initialize logger
        self.logger = ConsoleLogger(verbose=verbose, width=70)

        # Will be resolved to Path during validation
        self._atlas_resolved = None
        # Cache for full connectivity matrix (computed once if atlas provided)
        self._full_connectivity_matrix = None

        # Check MRtrix3 availability if requested
        if check_dependencies:
            try:
                check_mrtrix_available()
            except MRtrixError as e:
                raise MRtrixError(
                    f"MRtrix3 is required for StructuralNetworkMapping but is not available.\n{e}"
                ) from e

    def _validate_inputs(self, lesion_data: LesionData) -> None:
        """
        Validate that lesion data meets requirements for structural network mapping.

        Parameters
        ----------
        lesion_data : LesionData
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
        if not self.whole_brain_tdi.exists():
            raise FileNotFoundError(f"Whole-brain TDI not found: {self.whole_brain_tdi}")

        # Check coordinate space - strict validation (case-sensitive)
        space = lesion_data.get_coordinate_space()

        # Validate exact format: must be exactly 'MNI152_1mm' or 'MNI152_2mm'
        if space not in ("MNI152_1mm", "MNI152_2mm"):
            raise ValueError(
                f"Invalid coordinate space: '{space}'. "
                f"Structural network mapping requires exactly 'MNI152_1mm' or 'MNI152_2mm' (case-sensitive). "
                f"Got: '{space}'"
            )

        # Load template from bundled data if not provided
        if self.template is None:
            # Determine resolution from validated space
            resolution_mm = 1 if space == "MNI152_1mm" else 2

            # Get path to appropriate MNI152 template from bundled package data
            self.template = get_template_path(resolution=resolution_mm)
        else:
            # Template path was provided - verify it exists
            if not self.template.exists():
                raise FileNotFoundError(f"Template not found: {self.template}")

        # Load atlas from bundled data or validate custom atlas path
        if self.atlas_path is not None:
            if isinstance(self.atlas_path, str):
                # Bundled atlas requested (e.g., "schaefer100", "aal3")
                try:
                    atlas_file, labels_file = get_bundled_atlas(self.atlas_path)
                    self._atlas_resolved = atlas_file
                except (FileNotFoundError, ValueError) as e:
                    raise ValueError(
                        f"Bundled atlas '{self.atlas_path}' not found. "
                        f"Use list_bundled_atlases() to see available options."
                    ) from e
            else:
                # Custom atlas path provided
                self._atlas_resolved = Path(self.atlas_path)
                if not self._atlas_resolved.exists():
                    raise FileNotFoundError(f"Atlas file not found: {self._atlas_resolved}")

        # Check that lesion is binary
        lesion_array = lesion_data.lesion_img.get_fdata()
        unique_vals = np.unique(lesion_array)

        if not np.all(np.isin(unique_vals, [0, 1])):
            raise ValueError(
                f"Structural network mapping requires binary lesion mask (0 and 1 only).\n"
                f"Found values: {unique_vals}\n"
                f"Use thresholding or binarization to convert continuous maps."
            )

    def _run_analysis(self, lesion_data: LesionData) -> dict:
        """
        Execute structural network mapping analysis.

        Parameters
        ----------
        lesion_data : LesionData
            Input lesion data

        Returns
        -------
        dict
            Analysis results containing:
            - disconnection_map: nibabel.Nifti1Image of voxel-wise disconnection
            - mean_disconnection: float, mean disconnection across brain
            - lesion_streamline_count: int, number of streamlines through lesion
            - metadata: dict with processing parameters

        Notes
        -----
        Processing steps:
        1. Filter tractogram to streamlines passing through lesion (tckedit)
        2. Compute TDI from filtered tractogram (tckmap)
        3. Compute disconnection as ratio of lesion TDI to whole-brain TDI (mrcalc)
        """
        # Get subject ID for informative output
        subject_id = lesion_data.metadata.get("subject_id", "unknown")

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
                lesion_mask=lesion_data.lesion_img,
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

            # Build results dictionary
            results = {
                "disconnection_map": final_disconn_map,
                "mean_disconnection": mean_disconnection,
                "lesion_streamline_count": lesion_streamline_count,
                "metadata": {
                    "tractogram": str(self.tractogram_path),
                    "whole_brain_tdi": str(self.whole_brain_tdi),
                    "template": str(self.template),
                    "n_jobs": self.n_jobs,
                    "keep_intermediate": self.keep_intermediate,
                    "load_to_memory": self.load_to_memory,
                },
            }

            # Optional: Compute parcellated connectivity matrices if atlas provided
            if self._atlas_resolved is not None:
                self.logger.subsection("Computing Connectivity Matrices")
                connectivity_results = self._compute_connectivity_matrices(
                    lesion_data=lesion_data,
                    lesion_tck_path=lesion_tck_path,
                    temp_dir_path=temp_dir_path,
                    subject_id=subject_id,
                )
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
        lesion_data: LesionData,
        lesion_tck_path: Path,
        temp_dir_path: Path,
        subject_id: str,
    ) -> dict:
        """Compute parcellated connectivity matrices.

        Parameters
        ----------
        lesion_data : LesionData
            Lesion data with mask image
        lesion_tck_path : Path
            Path to lesion-filtered tractogram
        temp_dir_path : Path
            Temporary directory for intermediate files
        subject_id : str
            Subject identifier for file naming

        Returns
        -------
        dict
            Dictionary with connectivity matrix results:
            - lesion_connectivity_matrix: np.ndarray
            - disconnectivity_percent: np.ndarray
            - lesioned_connectivity_matrix: np.ndarray (optional)
            - matrix_statistics: dict
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
            nib.save(lesion_data.lesion_img, lesion_mask_path)

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

        return {
            "lesion_connectivity_matrix": lesion_matrix,
            "disconnectivity_percent": disconn_pct,
            "full_connectivity_matrix": full_matrix,
            "lesioned_connectivity_matrix": lesioned_matrix,
            "matrix_statistics": matrix_stats,
        }

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
            "atlas_path": str(self.atlas_path) if self.atlas_path else None,
            "compute_lesioned": self.compute_lesioned,
            "n_jobs": self.n_jobs,
            "keep_intermediate": self.keep_intermediate,
            "load_to_memory": self.load_to_memory,
            "verbose": self.verbose,
        }
