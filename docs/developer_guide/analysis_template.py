"""
Analysis Template for Lacuna

This file provides a ready-to-use template for creating new analysis modules.
Copy this file to `src/lacuna/analysis/` and customize it for your analysis.

Instructions:
1. Copy this file to `src/lacuna/analysis/my_analysis.py`
2. Rename the class to something descriptive (e.g., `MyCustomAnalysis`)
3. Implement `_validate_inputs()` to check input requirements
4. Implement `_run_analysis()` to perform your computation
5. Update the docstrings with your analysis description
6. The analysis will be auto-discovered and available via `list_analyses()`

Auto-Discovery:
- Your analysis class will automatically appear in `list_analyses()`
- Users can instantiate it via `get_analysis("MyCustomAnalysis")()`
- No registration or configuration needed - just create the file!
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lacuna.analysis.base import BaseAnalysis

if TYPE_CHECKING:
    from lacuna.core.subject_data import SubjectData


class AnalysisTemplate(BaseAnalysis):
    """
    Template for creating a custom analysis module.

    This analysis demonstrates the structure and conventions for creating
    new analysis modules in Lacuna. Replace this docstring with a description
    of what your analysis does.

    Parameters
    ----------
    my_parameter : str, optional
        Description of what this parameter controls.
        Default is "default_value".
    threshold : float, optional
        Example numeric parameter with validation.
        Default is 0.5.
    log_level : int, optional
        Logging verbosity (0=silent, 1=standard, 2=verbose).
        Default is 1.

    Attributes
    ----------
    batch_strategy : str
        Set to "parallel" for analyses that can run independently on each
        subject, or "vectorized" for analyses that process all subjects at once.

    Examples
    --------
    >>> from lacuna.analysis import get_analysis
    >>> MyAnalysis = get_analysis("AnalysisTemplate")
    >>> analysis = MyAnalysis(my_parameter="custom", threshold=0.7)
    >>> result = analysis.run(subject_data)
    >>> print(result.results["AnalysisTemplate"])

    Notes
    -----
    Describe any important implementation details, assumptions, or references
    to relevant papers/methods here.

    See Also
    --------
    lacuna.analysis.regional_damage.RegionalDamage : Example of a simple analysis
    lacuna.analysis.functional_network_mapping.FunctionalNetworkMapping : Example with connectome
    """

    # Batch processing strategy: "parallel" or "vectorized"
    batch_strategy: str = "parallel"

    def __init__(
        self,
        my_parameter: str = "default_value",
        threshold: float = 0.5,
        log_level: int = 1,
    ):
        """
        Initialize the analysis with configuration parameters.

        All parameters should be validated here or in _validate_inputs().
        Store parameters as instance attributes for use in _run_analysis().
        """
        super().__init__(log_level=log_level)

        # Validate parameters
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0 and 1, got {threshold}")

        # Store parameters
        self.my_parameter = my_parameter
        self.threshold = threshold

    def _validate_inputs(self, data: SubjectData) -> None:
        """
        Validate that the input data meets requirements for this analysis.

        This method is called automatically before _run_analysis().
        Raise ValueError or TypeError if validation fails.

        Parameters
        ----------
        data : SubjectData
            The input data to validate.

        Raises
        ------
        ValueError
            If the data doesn't meet requirements.
        TypeError
            If the data is the wrong type.
        """
        # Example: Check that mask_img exists
        if data.mask_img is None:
            raise ValueError("SubjectData must have a mask_img")

        # Example: Check coordinate space
        if data.space is None:
            raise ValueError("SubjectData must have a defined coordinate space")

        # Example: Check for specific prior results (if needed)
        # if "RegionalDamage" not in data.results:
        #     raise ValueError("This analysis requires RegionalDamage results first")

    def _run_analysis(self, data: SubjectData) -> dict:
        """
        Perform the analysis computation.

        This method contains your analysis logic. It should:
        1. Extract data from the SubjectData object
        2. Perform computations
        3. Return a dictionary of results

        The returned dictionary will be stored in data.results[ClassName].
        All values should be JSON-serializable (numbers, strings, lists, dicts).

        Parameters
        ----------
        data : SubjectData
            The validated input data.

        Returns
        -------
        dict
            Dictionary of analysis results. Keys become result attributes.
            Values should be JSON-serializable.
        """
        # Extract data
        mask_array = data.mask_img.get_fdata()

        # Perform computation (replace with your analysis logic)
        volume_voxels = (mask_array > self.threshold).sum()
        voxel_volume_mm3 = abs(data.mask_img.header.get_zooms()[0] ** 3)
        volume_mm3 = volume_voxels * voxel_volume_mm3

        # Log progress if verbose
        if self.log_level >= 2:
            print(f"  Processing with parameter: {self.my_parameter}")
            print(f"  Found {volume_voxels} voxels above threshold")

        # Return results dictionary
        return {
            "volume_voxels": int(volume_voxels),
            "volume_mm3": float(volume_mm3),
            "threshold_used": self.threshold,
            "parameter_used": self.my_parameter,
        }

    def __repr__(self) -> str:
        """Return a string representation of the analysis configuration."""
        return (
            f"{self.__class__.__name__}("
            f"my_parameter={self.my_parameter!r}, "
            f"threshold={self.threshold}, "
            f"log_level={self.log_level})"
        )


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
#
# After copying this file to src/lacuna/analysis/:
#
# 1. Test auto-discovery:
#    >>> from lacuna.analysis import list_analyses
#    >>> for name, cls in list_analyses():
#    ...     print(name)
#    AnalysisTemplate  # Your new analysis should appear!
#
# 2. Use in analyze() API:
#    >>> from lacuna import analyze
#    >>> result = analyze(subject_data)  # Won't include your analysis by default
#
# 3. Use directly:
#    >>> from lacuna.analysis import get_analysis
#    >>> MyAnalysis = get_analysis("AnalysisTemplate")
#    >>> analysis = MyAnalysis(my_parameter="custom")
#    >>> result = analysis.run(subject_data)
#
# 4. Use in Pipeline:
#    >>> from lacuna import Pipeline
#    >>> from lacuna.analysis import get_analysis
#    >>> MyAnalysis = get_analysis("AnalysisTemplate")
#    >>> pipeline = Pipeline().add(MyAnalysis(threshold=0.3))
#    >>> result = pipeline.run(subject_data)
#
# ============================================================================
