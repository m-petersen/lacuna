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

from ldk.analysis.base import BaseAnalysis
from ldk.core.lesion_data import LesionData
from ldk.data import get_template_path
from ldk.utils.mrtrix import (
    MRtrixError,
    check_mrtrix_available,
    compute_disconnection_map,
    compute_tdi_map,
    filter_tractogram_by_lesion,
)


class StructuralNetworkMapping(BaseAnalysis):
    """
    Structural lesion network mapping using tractography-based disconnection.

    This analysis quantifies white matter disconnection caused by a lesion by:
    1. Filtering a whole-brain tractogram to streamlines passing through the lesion
    2. Computing track density images (TDI) for both lesion and whole-brain tracts
    3. Calculating disconnection probability as the ratio of lesion TDI to whole-brain TDI

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

    >>> from ldk import LesionData
    >>> from ldk.analysis import StructuralNetworkMapping
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

    >>> from ldk import batch_process
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
        n_jobs: int = 1,
        keep_intermediate: bool = False,
        load_to_memory: bool = True,
        check_dependencies: bool = True,
    ):
        """Initialize StructuralNetworkMapping analysis."""
        super().__init__()

        self.tractogram_path = Path(tractogram_path)
        self.whole_brain_tdi = Path(whole_brain_tdi)
        self.template = Path(template) if template is not None else None
        self.n_jobs = n_jobs
        self.keep_intermediate = keep_intermediate
        self.load_to_memory = load_to_memory
        self._check_dependencies = check_dependencies

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
        # Create temporary directory for intermediate files
        temp_dir = tempfile.mkdtemp(prefix="slnm_")
        temp_dir_path = Path(temp_dir)

        if self.keep_intermediate:
            print(f"ℹ️  Intermediate files will be saved to: {temp_dir}")

        try:
            # Step 1: Filter tractogram by lesion
            lesion_tck_path = temp_dir_path / "lesion_streamlines.tck"
            filter_tractogram_by_lesion(
                tractogram_path=self.tractogram_path,
                lesion_mask=lesion_data.lesion_img,
                output_path=lesion_tck_path,
                n_jobs=self.n_jobs,
                force=True,
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
            )

            # Step 3: Compute disconnection map
            disconn_map_path = temp_dir_path / "disconnection_map.nii.gz"
            compute_disconnection_map(
                lesion_tdi=lesion_tdi_path,
                whole_brain_tdi=self.whole_brain_tdi,
                output_path=disconn_map_path,
                force=True,
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

            return results

        finally:
            # Clean up intermediate files unless keep_intermediate=True
            if not self.keep_intermediate:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                print(f"✓ Intermediate files preserved in: {temp_dir}")
                print("   - lesion_streamlines.tck")
                print("   - lesion_tdi.nii.gz")
                print("   - disconnection_map.nii.gz")

    def _get_parameters(self) -> dict:
        """Get analysis parameters for provenance tracking."""
        return {
            "tractogram_path": str(self.tractogram_path),
            "whole_brain_tdi": str(self.whole_brain_tdi),
            "template": str(self.template),
            "n_jobs": self.n_jobs,
            "keep_intermediate": self.keep_intermediate,
            "load_to_memory": self.load_to_memory,
        }

    def _get_version(self) -> str:
        """Get analysis version."""
        return "0.1.0"
