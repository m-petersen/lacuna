"""
Base class for all analysis modules.

Provides the abstract interface and workflow orchestration for plug-and-play
analysis extensibility.
"""

from abc import ABC, abstractmethod
from typing import Any, final

from ldk.core.lesion_data import LesionData
from ldk.core.provenance import create_provenance_record


class BaseAnalysis(ABC):
    """
    Abstract base class for all analysis modules.

    This class defines the contract that all analysis implementations must follow,
    enabling plug-and-play extensibility. Subclasses must implement two abstract
    methods:
    - `_validate_inputs`: Check that input data meets analysis requirements
    - `_run_analysis`: Perform the actual analysis computation

    The public `run()` method orchestrates the workflow and cannot be overridden.

    Examples
    --------
    >>> class MyAnalysis(BaseAnalysis):
    ...     def __init__(self, threshold=0.5):
    ...         super().__init__()
    ...         self.threshold = threshold
    ...
    ...     def _validate_inputs(self, lesion_data):
    ...         if lesion_data.get_coordinate_space() != "MNI152":
    ...             raise ValueError("Must be in MNI152 space")
    ...
    ...     def _run_analysis(self, lesion_data):
    ...         volume = lesion_data.get_volume_mm3()
    ...         return {"volume": volume, "above_threshold": volume > self.threshold}
    ...
    >>> analysis = MyAnalysis(threshold=100.0)
    >>> result = analysis.run(lesion_data)
    >>> print(result.results["MyAnalysis"])
    {"volume": 523.5, "above_threshold": True}
    """

    def __init__(self) -> None:
        """
        Initialize the analysis module.

        Subclasses should override this to accept analysis-specific parameters
        and store them as instance attributes for provenance tracking.
        """
        pass

    @final
    def run(self, lesion_data: LesionData) -> LesionData:
        """
        Execute the analysis on a LesionData object.

        This is the ONLY public method users should call. It orchestrates
        the complete analysis workflow:
        1. Validate inputs via _validate_inputs()
        2. Run analysis via _run_analysis()
        3. Namespace results under the analysis class name
        4. Create new LesionData with updated results
        5. Record provenance
        6. Return new LesionData instance

        The input LesionData is never modified (immutability principle).

        Parameters
        ----------
        lesion_data : LesionData
            Input data containing lesion mask, metadata, and any prior results.

        Returns
        -------
        LesionData
            A NEW LesionData instance with analysis results added to the
            .results dictionary under a namespace key (the analysis class name).

        Raises
        ------
        ValueError
            If input validation fails (via _validate_inputs).
        RuntimeError
            If analysis computation fails (via _run_analysis).

        Notes
        -----
        This method is marked @final to prevent subclasses from overriding it.
        All customization must happen in the protected abstract methods.

        Examples
        --------
        >>> analysis = LesionNetworkMapping(connectome='HCP1200')
        >>> result = analysis.run(lesion_data)
        >>> print(result.results['LesionNetworkMapping']['network_scores'])
        """
        # Step 1: Validate inputs
        self._validate_inputs(lesion_data)

        # Step 2: Run analysis computation
        analysis_results = self._run_analysis(lesion_data)

        # Step 3: Namespace results under class name
        namespace_key = self.__class__.__name__
        updated_results = lesion_data.results.copy()
        updated_results[namespace_key] = analysis_results

        # Step 4: Create new LesionData with updated results
        # Create a new instance with updated results (manual approach for namespace overwriting)
        result_lesion_data = LesionData(
            lesion_img=lesion_data.lesion_img,
            anatomical_img=lesion_data.anatomical_img,
            metadata=lesion_data.metadata,
            provenance=lesion_data.provenance,
            results=updated_results,
        )

        # Step 5: Record provenance
        provenance_record = create_provenance_record(
            function=f"{self.__class__.__module__}.{self.__class__.__name__}",
            parameters=self._get_parameters(),
            version=self._get_version(),
        )
        result_lesion_data = result_lesion_data.add_provenance(provenance_record)

        return result_lesion_data

    @abstractmethod
    def _validate_inputs(self, lesion_data: LesionData) -> None:
        """
        Validate that lesion_data meets the requirements for this analysis.

        Parameters
        ----------
        lesion_data : LesionData
            Input data to validate.

        Raises
        ------
        ValueError
            If validation fails. Error message should clearly explain what
            requirement was not met and how to fix it.

        Notes
        -----
        Common validations include:
        - Checking coordinate space (e.g., must be in MNI152)
        - Verifying lesion mask is binary
        - Ensuring required metadata fields are present
        - Checking for prerequisite results from other analyses

        This method is called automatically by run() before _run_analysis().

        Examples
        --------
        >>> def _validate_inputs(self, lesion_data: LesionData) -> None:
        ...     if lesion_data.get_coordinate_space() != "MNI152_2mm":
        ...         raise ValueError(
        ...             "LesionNetworkMapping requires data in MNI152 space. "
        ...             "Use ldk.preprocess.normalize_to_mni() first."
        ...         )
        ...
        ...     data = lesion_data.lesion_img.get_fdata()
        ...     if not np.all(np.isin(data, [0, 1])):
        ...         raise ValueError("Lesion mask must be binary (0s and 1s).")
        """
        pass

    @abstractmethod
    def _run_analysis(self, lesion_data: LesionData) -> dict[str, Any]:
        """
        Perform the core analysis computation.

        Parameters
        ----------
        lesion_data : LesionData
            Validated input data.

        Returns
        -------
        Dict[str, Any]
            Analysis results as a dictionary. Structure is module-specific,
            but must be JSON-serializable for provenance tracking.

        Raises
        ------
        RuntimeError
            If analysis computation fails.

        Notes
        -----
        This method contains the scientific logic of your analysis.
        It is called automatically by run() after validation succeeds.

        The returned dictionary will be automatically namespaced under
        self.__class__.__name__ in the output LesionData.results attribute.

        Do NOT modify the input lesion_data object. Extract what you need,
        perform computations, and return results as a new dictionary.

        Examples
        --------
        >>> def _run_analysis(self, lesion_data: LesionData) -> Dict[str, Any]:
        ...     lesion_array = lesion_data.lesion_img.get_fdata()
        ...
        ...     network_scores = self._compute_network_scores(lesion_array)
        ...     affected_regions = self._identify_affected_regions(lesion_array)
        ...
        ...     return {
        ...         'network_scores': network_scores.tolist(),
        ...         'affected_regions': affected_regions,
        ...         'summary_statistics': {
        ...             'mean_disruption': float(np.mean(network_scores)),
        ...             'max_disruption': float(np.max(network_scores))
        ...         }
        ...     }
        """
        pass

    def _get_parameters(self) -> dict[str, Any]:
        """
        Get analysis parameters for provenance tracking.

        Returns
        -------
        Dict[str, Any]
            Dictionary of parameter names and values.

        Notes
        -----
        Override this method if your analysis has parameters that should
        be recorded in provenance. Default implementation returns an empty dict.

        Examples
        --------
        >>> def _get_parameters(self) -> Dict[str, Any]:
        ...     return {
        ...         'threshold': self.threshold,
        ...         'method': self.method,
        ...         'connectome': self.connectome
        ...     }
        """
        return {}

    def _get_version(self) -> str:
        """
        Get analysis version for provenance tracking.

        Returns
        -------
        str
            Version string (e.g., "0.1.0").

        Notes
        -----
        Override this method to provide version information for your analysis.
        Default implementation returns "0.1.0".

        Examples
        --------
        >>> def _get_version(self) -> str:
        ...     return "1.2.3"
        """
        return "0.1.0"
