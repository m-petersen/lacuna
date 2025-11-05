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
    template : str or Path
        Path to template image defining output grid (e.g., MNI152_T1_2mm.nii.gz)
    n_jobs : int, default=1
        Number of threads for MRtrix3 processing
    keep_intermediate : bool, default=False
        If True, keeps intermediate tractogram files (for debugging)
    check_dependencies : bool, default=True
        If True, checks for MRtrix3 availability during initialization

    Raises
    ------
    MRtrixError
        If MRtrix3 is not installed or not available in PATH
    FileNotFoundError
        If tractogram_path, whole_brain_tdi, or template files don't exist

    Examples
    --------
    >>> from ldk import LesionData
    >>> from ldk.analysis import StructuralNetworkMapping
    >>>
    >>> # Load lesion data
    >>> lesion = LesionData.from_nifti("lesion.nii.gz")
    >>>
    >>> # Create analysis
    >>> analysis = StructuralNetworkMapping(
    ...     tractogram_path="/data/dTOR_full_tractogram.tck",
    ...     whole_brain_tdi="/data/dTOR_tdi_2mm.nii.gz",
    ...     template="/data/MNI152_T1_2mm.nii.gz",
    ...     n_jobs=8
    ... )
    >>>
    >>> # Run analysis
    >>> result = analysis.run(lesion)
    >>>
    >>> # Access results
    >>> disconn_map = result.results["StructuralNetworkMapping"]["disconnection_map"]
    >>> mean_disconn = result.results["StructuralNetworkMapping"]["mean_disconnection"]
    >>> print(f"Mean disconnection: {mean_disconn:.2%}")

    Notes
    -----
    - Requires MRtrix3: https://www.mrtrix.org/download/
    - Lesion must be in same coordinate space as tractogram (typically MNI152)
    - Lesion must be binary (0/1 values only)
    - Processing time scales with lesion size and tractogram density
    - For large tractograms, processing can take several minutes per subject

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
        template: str | Path,
        n_jobs: int = 1,
        keep_intermediate: bool = False,
        check_dependencies: bool = True,
    ):
        """Initialize StructuralNetworkMapping analysis."""
        super().__init__()

        self.tractogram_path = Path(tractogram_path)
        self.whole_brain_tdi = Path(whole_brain_tdi)
        self.template = Path(template)
        self.n_jobs = n_jobs
        self.keep_intermediate = keep_intermediate
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
        if not self.template.exists():
            raise FileNotFoundError(f"Template not found: {self.template}")

        # Check coordinate space
        space = lesion_data.get_coordinate_space()
        if "MNI152" not in space:
            raise ValueError(
                f"Structural network mapping requires lesion in MNI152 space.\n"
                f"Got space: {space}\n"
                f"Use lesion_data.to_space('MNI152') to transform to standard space."
            )

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
            lesion_tdi_path = temp_dir_path / "lesion_tdi.nii.gz"
            compute_tdi_map(
                tractogram_path=lesion_tck_path,
                template=self.template,
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
            disconn_map = nib.load(disconn_map_path)
            disconn_array = disconn_map.get_fdata()

            # Compute summary statistics
            mean_disconnection = float(np.mean(disconn_array[disconn_array > 0]))

            # Count streamlines in lesion tractogram (from TDI sum)
            lesion_tdi = nib.load(lesion_tdi_path)
            lesion_streamline_count = int(np.sum(lesion_tdi.get_fdata()))

            # Build results dictionary
            results = {
                "disconnection_map": disconn_map,
                "mean_disconnection": mean_disconnection,
                "lesion_streamline_count": lesion_streamline_count,
                "metadata": {
                    "tractogram": str(self.tractogram_path),
                    "whole_brain_tdi": str(self.whole_brain_tdi),
                    "template": str(self.template),
                    "n_jobs": self.n_jobs,
                },
            }

            return results

        finally:
            # Clean up intermediate files unless keep_intermediate=True
            if not self.keep_intermediate:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)

    def _get_parameters(self) -> dict:
        """Get analysis parameters for provenance tracking."""
        return {
            "tractogram_path": str(self.tractogram_path),
            "whole_brain_tdi": str(self.whole_brain_tdi),
            "template": str(self.template),
            "n_jobs": self.n_jobs,
            "keep_intermediate": self.keep_intermediate,
        }

    def _get_version(self) -> str:
        """Get analysis version."""
        return "0.1.0"
