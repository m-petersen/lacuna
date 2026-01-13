"""
Base class for all analysis modules.

Provides the abstract interface and workflow orchestration for plug-and-play
analysis extensibility.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, final

from lacuna.core.provenance import create_provenance_record
from lacuna.core.subject_data import SubjectData

if TYPE_CHECKING:
    from lacuna.core.data_types import DataContainer


class BaseAnalysis(ABC):
    """
    Abstract base class for all analysis modules.

    This class defines the contract that all analysis implementations must follow,
    enabling plug-and-play extensibility. Subclasses must implement two abstract
    methods:
    - `_validate_inputs`: Check that input data meets analysis requirements
    - `_run_analysis`: Perform the actual analysis computation

    The public `run()` method orchestrates the workflow and cannot be overridden.

    Attributes
    ----------
    TARGET_SPACE : str or None
        Coordinate space where computations are performed (e.g., "MNI152NLin6Asym").
        Can be defined as a class attribute (static) or instance attribute (dynamic).
        For connectome-based analyses, this is typically set in __init__ based on
        the registered connectome's space. Set to None or "atlas" for analyses
        that adapt to input data spaces (e.g., ParcelAggregation).
    TARGET_RESOLUTION : float or None
        Resolution in mm where computations are performed (e.g., 1 or 2).
        Can be defined as a class attribute or instance attribute.
        Set to None for analyses that adapt to input data resolution.
    batch_strategy : str
        Preferred batch processing strategy ("parallel", or "vectorized").
        Default is "parallel". Subclasses should override this if they benefit from
        a different strategy (e.g., vectorized for network mapping analyses).

    Examples
    --------
    >>> class LesionVolume(BaseAnalysis):
    ...     # Declare computation space - masks will be transformed to this space
    ...     TARGET_SPACE = "MNI152NLin6Asym"
    ...     TARGET_RESOLUTION = 2
    ...     batch_strategy = "parallel"  # Process subjects in parallel
    ...
    ...     def __init__(self):
    ...         super().__init__()
    ...
    ...     def _validate_inputs(self, mask_data):
    ...         # Validation happens AFTER automatic transformation to TARGET_SPACE
    ...         space = mask_data.get_coordinate_space()
    ...         if space != self.TARGET_SPACE:
    ...             raise ValueError(f"Expected {self.TARGET_SPACE}, got {space}")
    ...
    ...     def _run_analysis(self, mask_data):
    ...         # Lesion is guaranteed to be in TARGET_SPACE @ TARGET_RESOLUTION
    ...         volume = mask_data.get_volume_mm3()
    ...         return {"volume": volume}
    ...
    >>> analysis = LesionVolume()
    >>> result = analysis.run(mask_data)
    >>> print(result.results["LesionVolume"])
    {"volume": 523.5}
    """

    #: Preferred batch processing strategy (default: parallel)
    batch_strategy: str = "parallel"

    def __init__(self, verbose: bool = False, keep_intermediate: bool = False) -> None:
        """
        Initialize the analysis module.

        Parameters
        ----------
        verbose : bool, default=False
            If True, print progress messages. If False, run silently.
        keep_intermediate : bool, default=False
            If True, include intermediate results (e.g., warped mask images)
            in the output. Useful for debugging and quality control.

        Notes
        -----
        Subclasses should override this to accept analysis-specific parameters
        and store them as instance attributes for provenance tracking.
        Always call super().__init__(verbose=verbose, keep_intermediate=keep_intermediate)
        when overriding.
        """
        self.verbose = verbose
        self.keep_intermediate = keep_intermediate

    def __repr__(self) -> str:
        """
        Return detailed string representation of the analysis object.

        Returns
        -------
        str
            String in format "ClassName(param1=value1, param2=value2, ...)"

        Examples
        --------
        >>> analysis = FunctionalNetworkMapping(connectome_name='GSP1000')
        >>> repr(analysis)
        "FunctionalNetworkMapping(connectome_name='GSP1000', ...)"
        """
        params = self._get_parameters()
        class_name = self.__class__.__name__

        if not params:
            return f"{class_name}()"

        # Format parameters - truncate long values
        param_strs = []
        for key, value in params.items():
            if isinstance(value, str) and len(value) > 50:
                value_str = f"'{value[:47]}...'"
            elif isinstance(value, str):
                value_str = f"'{value}'"
            else:
                value_str = str(value)
            param_strs.append(f"{key}={value_str}")

        params_formatted = ", ".join(param_strs)
        return f"{class_name}({params_formatted})"

    def __str__(self) -> str:
        """
        Return user-friendly string representation of the analysis.

        Returns
        -------
        str
            Human-readable description of the analysis configuration.

        Examples
        --------
        >>> analysis = FunctionalNetworkMapping(method='pearson')
        >>> print(analysis)
        FunctionalNetworkMapping Analysis
        Configuration:
          - method: pearson
          - connectome_path: /path/to/connectome.h5
          - t_threshold: 9.0
        """
        class_name = self.__class__.__name__
        params = self._get_parameters()

        if not params:
            return f"{class_name} Analysis (no parameters)"

        lines = [f"{class_name} Analysis", "Configuration:"]
        for key, value in params.items():
            # Truncate long strings for readability
            if isinstance(value, str) and len(value) > 60:
                value_str = f"{value[:57]}..."
            else:
                value_str = str(value)
            lines.append(f"  - {key}: {value_str}")

        return "\n".join(lines)

    @final
    def run(self, mask_data: SubjectData) -> SubjectData:
        """
        Execute the analysis on a SubjectData object.

        This is the ONLY public method users should call. It orchestrates
        the complete analysis workflow:
        1. Transform to TARGET_SPACE if defined
        2. Validate inputs via _validate_inputs()
        3. Run analysis via _run_analysis()
        4. Namespace results under the analysis class name
        5. Create new SubjectData with updated results
        6. Record provenance
        7. Return new SubjectData instance

        The input SubjectData is never modified (immutability principle).

        Parameters
        ----------
        mask_data : SubjectData
            Input data containing lesion mask, metadata, and any prior results.

        Returns
        -------
        SubjectData
            A NEW SubjectData instance with analysis results added to the
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

        When keep_intermediate=True, the mask used for analysis is included
        in results as 'analysis_mask'. This VoxelMap includes metadata
        indicating whether transformation occurred (was_transformed=True/False).

        Examples
        --------
        >>> analysis = FunctionalNetworkMapping(connectome='GSP1000')
        >>> result = analysis.run(mask_data)
        >>> print(result.results['FunctionalNetworkMapping']['rmap'])
        """
        # Track original input space info for analyses that need to transform back
        original_space = mask_data.space
        original_resolution = mask_data.resolution

        # Step 1: Transform to target space if TARGET_SPACE is defined
        transformed_data = self._ensure_target_space(mask_data)

        # Store original input info in metadata for _run_analysis to access
        # This allows analyses to transform results back to input space if requested
        if transformed_data is not mask_data:
            updated_metadata = transformed_data.metadata.copy()
            updated_metadata["_original_input_space"] = original_space
            updated_metadata["_original_input_resolution"] = original_resolution
            transformed_data = SubjectData(
                mask_img=transformed_data.mask_img,
                space=transformed_data.space,
                resolution=transformed_data.resolution,
                metadata=updated_metadata,
                provenance=transformed_data.provenance,
                results=transformed_data.results,
            )

        # Step 2: Validate inputs
        self._validate_inputs(transformed_data)

        # Step 3: Run analysis computation
        analysis_results = self._run_analysis(transformed_data)

        # Step 4: Namespace results under class name
        results_dict = analysis_results

        # Add analysis mask to intermediates if keep_intermediate=True
        # Note: This is a fallback that stores the space-transformed mask.
        # Analyses that resample further (e.g., FNM to connectome grid) should
        # store their own 'analysis_mask' in _run_analysis with the actual
        # mask used for computation. That will override this default.
        if self.keep_intermediate and "analysis_mask" not in results_dict:
            from lacuna.core.data_types import VoxelMap

            was_transformed = (
                original_space != transformed_data.space
                or original_resolution != transformed_data.resolution
            )
            analysis_mask = VoxelMap(
                name="analysis_mask",
                data=transformed_data.mask_img,
                space=transformed_data.space,
                resolution=transformed_data.resolution,
                metadata={
                    "description": (
                        "Mask transformed to analysis target space"
                        if was_transformed
                        else "Mask used for analysis (no transformation needed)"
                    ),
                    "was_transformed": was_transformed,
                    "original_space": original_space,
                    "original_resolution": original_resolution,
                    "analysis_space": transformed_data.space,
                    "analysis_resolution": transformed_data.resolution,
                },
            )
            results_dict["analysis_mask"] = analysis_mask

        namespace_key = self.__class__.__name__
        updated_results = transformed_data.results.copy()
        updated_results[namespace_key] = results_dict

        # Step 4: Create new SubjectData with updated results
        # Create a new instance with updated results (manual approach for namespace overwriting)
        result_mask_data = SubjectData(
            mask_img=transformed_data.mask_img,
            space=transformed_data.space,
            resolution=transformed_data.resolution,
            metadata=transformed_data.metadata,
            provenance=transformed_data.provenance,
            results=updated_results,
        )

        # Step 5: Record provenance
        provenance_record = create_provenance_record(
            function=f"{self.__class__.__module__}.{self.__class__.__name__}",
            parameters=self._get_parameters(),
            version=self._get_version(),
        )
        result_mask_data = result_mask_data.add_provenance(provenance_record)

        return result_mask_data

    @abstractmethod
    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """
        Validate that mask_data meets the requirements for this analysis.

        Parameters
        ----------
        mask_data : SubjectData
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
        >>> def _validate_inputs(self, mask_data: SubjectData) -> None:
        ...     if mask_data.get_coordinate_space() != "MNI152Nlin6Asym":
        ...         raise ValueError(
        ...             "ExampleAnalysis requires data in MNI152Nlin6Asym space. "
        ...         )
        ...
        ...     data = mask_data.mask_img.get_fdata()
        ...     if not np.all(np.isin(data, [0, 1])):
        ...         raise ValueError("Mask data must be binary (0s and 1s).")
        """
        pass

    @abstractmethod
    def _run_analysis(self, mask_data: SubjectData) -> list["DataContainer"]:
        """
        Perform the core analysis computation.

        Parameters
        ----------
        mask_data : SubjectData
            Validated input data.

        Returns
        -------
        list[DataContainer]
            Analysis results as a list of DataContainer objects. Each result
            represents a distinct output (voxel map, ROI data, matrix, etc.).

        Raises
        ------
        RuntimeError
            If analysis computation fails.

        Notes
        -----
        This method contains the scientific logic of your analysis.
        It is called automatically by run() after validation succeeds.

        The returned list will be automatically namespaced under
        self.__class__.__name__ in the output SubjectData.results attribute.

        Do NOT modify the input mask_data object. Extract what you need,
        perform computations, and return results as a list of DataContainer objects.

        Examples
        --------
        >>> from lacuna.core.data_types import VoxelMap, ScalarMetric
        >>> def _run_analysis(self, mask_data: SubjectData) -> list[DataContainer]:
        ...     mask_array = mask_data.mask_img.get_fdata()
        ...
        ...     # Create voxel map result
        ...     correlation_img = self._compute_correlation_map(mask_array)
        ...     voxel_result = VoxelMap(
        ...         name="rmap",
        ...         data=correlation_img,
        ...         output_space=self.computation_space,
        ...         lesion_space=mask_data.coordinate_space
        ...     )
        ...
        ...     # Create summary statistics result
        ...     summary_result = ScalarMetric(
        ...         name="summarystatistics",
        ...         data={"sum": float(np.sum(mask_array))}
        ...     )
        ...
        ...     return [voxel_result, summary_result]
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
        be recorded in provenance. The base implementation returns verbose.
        Subclasses should call super()._get_parameters() and merge with their
        own parameters.

        Examples
        --------
        >>> def _get_parameters(self) -> Dict[str, Any]:
        ...     params = super()._get_parameters()  # Get verbose
        ...     params.update({
        ...         'threshold': self.threshold,
        ...         'method': self.method,
        ...         'connectome': self.connectome
        ...     })
        ...     return params
        """
        return {"verbose": self.verbose}

    def _ensure_target_space(self, mask_data: SubjectData) -> SubjectData:
        """
        Automatically transform lesion data to TARGET_SPACE if defined.

        This method is called automatically by run() before validation and analysis.
        If TARGET_SPACE and TARGET_RESOLUTION are defined (as class or instance
        attributes), the lesion data will be transformed to that space.

        Special cases:
        - If TARGET_SPACE is None or "atlas", no transformation is performed
          (analysis adapts to input space)
        - If TARGET_RESOLUTION is None, current resolution is preserved

        Note: Instance attributes take precedence over class attributes, allowing
        analyses like FunctionalNetworkMapping and StructuralNetworkMapping to
        dynamically set TARGET_SPACE based on connectome metadata.

        Parameters
        ----------
        mask_data : SubjectData
            Input lesion data

        Returns
        -------
        SubjectData
            Transformed lesion data (or original if no transformation needed)
        """
        # Check if this analysis defines a target space
        # Use getattr(self, ...) to pick up instance attributes (e.g., from connectome)
        # as well as class attributes
        target_space = getattr(self, "TARGET_SPACE", None)
        target_resolution = getattr(self, "TARGET_RESOLUTION", None)

        # Skip transformation if no target space defined or if set to "atlas" (adaptive)
        if target_space is None or target_space == "atlas":
            return mask_data

        # Get current space
        current_space = mask_data.space
        current_resolution = mask_data.resolution

        if current_space is None:
            raise ValueError(
                f"{self.__class__.__name__} requires lesion data with 'space' metadata. "
                f"Expected space: {target_space}"
            )

        # Validate that resolution is present when space is specified
        # This prevents silently ignoring resolution mismatches
        if current_resolution is None:
            raise ValueError(
                f"{self.__class__.__name__} requires lesion data with 'resolution' metadata. "
                f"Resolution is required when space is specified. "
                f"Got space='{current_space}' but resolution=None"
            )

        # Check if transformation needed
        needs_space_transform = current_space != target_space
        needs_resolution_change = (
            target_resolution is not None
            and current_resolution is not None
            and current_resolution != target_resolution
        )

        if not needs_space_transform and not needs_resolution_change:
            # Already in target space
            return mask_data

        # Import here to avoid circular imports
        from lacuna.core.spaces import REFERENCE_AFFINES, CoordinateSpace
        from lacuna.spatial.transform import transform_mask_data

        # Determine target resolution (use current if not specified)
        final_resolution = (
            target_resolution if target_resolution is not None else current_resolution
        )

        # Create target space object
        target_space_obj = CoordinateSpace(
            identifier=target_space,
            resolution=final_resolution,
            reference_affine=REFERENCE_AFFINES.get(
                (target_space, final_resolution), mask_data.affine
            ),
        )

        # Transform (logging handled by transform_mask_data)
        return transform_mask_data(
            mask_data, target_space_obj, image_name="mask", verbose=self.verbose
        )

    def _get_version(self) -> str:
        """
        Get analysis version for provenance tracking.

        Returns
        -------
        str
            Version string (e.g., "0.1.0").

        Notes
        -----
        Returns the lacuna package version for consistent provenance tracking.
        Override this method if you need custom version information.

        Examples
        --------
        >>> def _get_version(self) -> str:
        ...     from .. import __version__
        ...     return __version__
        """
        from .. import __version__

        return __version__

    def _validate_and_transform_space(
        self,
        mask_data: SubjectData,
        required_space: str,
        required_resolution: float | None = None,
    ) -> SubjectData:
        """Validate coordinate space and auto-transform if needed.

        This helper method provides a standard pattern for analysis modules
        to validate spatial requirements and automatically transform data
        to the required space if needed.

        Parameters
        ----------
        mask_data : SubjectData
            Input lesion data
        required_space : str
            Required coordinate space identifier (e.g., 'MNI152NLin2009cAsym')
        required_resolution : float, optional
            Required resolution in mm. If None, any resolution accepted.

        Returns
        -------
        SubjectData
            Original data (if already in required space) or transformed data

        Raises
        ------
        ValueError
            If space cannot be determined or transformation not available

        Examples
        --------
        >>> def _validate_inputs(self, mask_data: SubjectData) -> None:
        ...     # Ensure data is in MNI152NLin2009cAsym space at 2mm
        ...     mask_data = self._validate_and_transform_space(
        ...         mask_data,
        ...         required_space='MNI152NLin2009cAsym',
        ...         required_resolution=2
        ...     )
        ...     return mask_data
        """
        # Get current space from metadata
        current_space = mask_data.space
        current_resolution = mask_data.resolution

        if current_space is None:
            raise ValueError(
                "Cannot determine coordinate space from lesion data. "
                "Ensure metadata contains 'space' key."
            )

        # Check if transformation needed
        needs_space_transform = current_space != required_space
        needs_resolution_change = (
            required_resolution is not None and current_resolution != required_resolution
        )

        if not needs_space_transform and not needs_resolution_change:
            # Already in required space - no transformation needed
            return mask_data

        # Import transformation utilities
        from lacuna.core.spaces import REFERENCE_AFFINES, CoordinateSpace
        from lacuna.spatial.transform import transform_mask_data

        # Create target space
        target_resolution = (
            required_resolution if required_resolution is not None else current_resolution
        )
        target_space = CoordinateSpace(
            identifier=required_space,
            resolution=target_resolution,
            reference_affine=REFERENCE_AFFINES.get(
                (required_space, target_resolution), mask_data.affine
            ),
        )

        # Transform data (logging handled by transform_mask_data)
        return transform_mask_data(mask_data, target_space, image_name="mask", verbose=self.verbose)
