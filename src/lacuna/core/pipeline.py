"""
High-level pipeline API for declarative analysis workflows.

This module provides a Pipeline class that allows defining and running
complete analysis workflows in a declarative manner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from lacuna.core.data_types import ParcelData
from lacuna.core.subject_data import SubjectData
from lacuna.utils.logging import ConsoleLogger

if TYPE_CHECKING:
    from lacuna.analysis.base import BaseAnalysis


@dataclass
class PipelineStep:
    """
    A single step in an analysis pipeline.

    Attributes
    ----------
    analysis : BaseAnalysis
        The analysis module to run
    name : str, optional
        Human-readable name for the step (defaults to class name)
    """

    analysis: BaseAnalysis
    name: str | None = None

    def __post_init__(self):
        if self.name is None:
            self.name = self.analysis.__class__.__name__


class Pipeline:
    """
    Declarative analysis workflow definition.

    Pipeline allows defining a sequence of analyses that will be run
    in order on each subject. It supports batch processing with
    configurable parallelization.

    Parameters
    ----------
    name : str, optional
        Human-readable name for the pipeline
    description : str, optional
        Description of what the pipeline does

    Examples
    --------
    >>> from lacuna.analysis import RegionalDamage, FunctionalNetworkMapping, ParcelAggregation
    >>> from lacuna import Pipeline

    >>> # Define pipeline
    >>> pipeline = Pipeline(name="Standard Lesion Analysis")
    >>> pipeline.add(RegionalDamage())
    >>> pipeline.add(FunctionalNetworkMapping())
    >>> pipeline.add(ParcelAggregation(parc_names=["Schaefer100"]))

    >>> # Run on single subject
    >>> result = pipeline.run(mask_data)

    >>> # Run on multiple subjects in parallel
    >>> results = pipeline.run_batch(subjects, n_jobs=-1)

    >>> # Get workflow description
    >>> print(pipeline.describe())
    Pipeline: Standard Lesion Analysis
    Steps:
      1. RegionalDamage
      2. FunctionalNetworkMapping (atlas=schaefer100)
      3. ParcelAggregation (parc_names=['Schaefer100'])
    """

    def __init__(
        self,
        name: str | None = None,
        description: str | None = None,
    ):
        self.name = name or "Unnamed Pipeline"
        self.description = description
        self._steps: list[PipelineStep] = []

    def add(
        self,
        analysis: BaseAnalysis,
        name: str | None = None,
    ) -> Pipeline:
        """
        Add an analysis step to the pipeline.

        Parameters
        ----------
        analysis : BaseAnalysis
            The analysis module to add
        name : str, optional
            Human-readable name for this step

        Returns
        -------
        Pipeline
            Self for method chaining
        """
        step = PipelineStep(analysis=analysis, name=name)
        self._steps.append(step)
        return self

    def run(self, data: SubjectData, verbose: bool = False) -> SubjectData:
        """
        Run the pipeline on a single subject.

        Parameters
        ----------
        data : SubjectData
            Input data to process
        verbose : bool, default=False
            If True, print progress messages. If False, run silently.

        Returns
        -------
        SubjectData
            Processed data with all analysis results

        Raises
        ------
        TypeError
            If data is not a SubjectData instance
        """
        # Validate input type
        if not isinstance(data, SubjectData):
            raise TypeError(
                f"Unsupported input type: {type(data).__name__}\n" "Supported types: SubjectData"
            )

        result = data

        # Create logger for analysis section headers
        logger = ConsoleLogger(verbose=verbose, width=70)

        for step in self._steps:
            # Run the analysis
            if verbose:
                logger.section(f"Running {step.name}")

            result = step.analysis.run(result)

        return result

    def run_batch(
        self,
        data_list: list[SubjectData],
        n_jobs: int = -1,
        show_progress: bool = True,
        parallel: bool = True,
    ) -> list[SubjectData | ParcelData]:
        """
        Run the pipeline on multiple subjects.

        Parameters
        ----------
        data_list : list of SubjectData
            List of subjects to process
        n_jobs : int, default=-1
            Number of parallel jobs (-1 uses all CPUs)
        show_progress : bool, default=True
            Show progress bar
        parallel : bool, default=True
            Whether to process subjects in parallel

        Returns
        -------
        list of SubjectData or ParcelData
            Processed data for each subject
        """
        if not parallel or n_jobs == 1:
            # Sequential processing
            results: list[SubjectData | ParcelData] = []
            iterator = data_list
            if show_progress:
                from tqdm import tqdm

                iterator = tqdm(data_list, desc=self.name)

            for data in iterator:
                results.append(self.run(data, verbose=False))
            return results

        # Parallel processing - run each step as a batch
        step_results: list[SubjectData | ParcelData] = list(data_list)

        for step in self._steps:
            # Use batch_process for this step
            from lacuna.batch.api import batch_process

            step_results = batch_process(
                inputs=step_results,  # type: ignore[arg-type]
                analysis=step.analysis,
                n_jobs=n_jobs,
                show_progress=show_progress,
            )

        return step_results

    def describe(self) -> str:
        """
        Get a human-readable description of the pipeline.

        Returns
        -------
        str
            Multi-line description of the pipeline
        """
        lines = [f"Pipeline: {self.name}"]

        if self.description:
            lines.append(f"  {self.description}")

        lines.append("")
        lines.append("Steps:")

        for i, step in enumerate(self._steps, 1):
            # Get analysis parameters for display
            params = self._get_analysis_params(step.analysis)
            if params:
                lines.append(f"  {i}. {step.name} ({params})")
            else:
                lines.append(f"  {i}. {step.name}")

        return "\n".join(lines)

    def _get_analysis_params(self, analysis: BaseAnalysis) -> str:
        """Extract key parameters from analysis for display."""
        params = []

        # Try common parameter names
        for attr in ["parc_names", "atlas", "threshold", "source"]:
            if hasattr(analysis, attr):
                val = getattr(analysis, attr)
                if val is not None:
                    params.append(f"{attr}={val!r}")

        return ", ".join(params)

    def __len__(self) -> int:
        """Return number of steps in pipeline."""
        return len(self._steps)

    def __repr__(self) -> str:
        return f"Pipeline(name={self.name!r}, steps={len(self._steps)})"


def analyze(
    data: SubjectData | list[SubjectData],
    *,
    steps: dict[str, dict | None],
    n_jobs: int = 1,
    show_progress: bool = True,
    verbose: bool = False,
) -> SubjectData | list[SubjectData]:
    """
    Run an analysis pipeline defined by a steps dictionary.

    This function provides a flexible interface for running analysis workflows.
    The `steps` dictionary defines which analyses to run and their parameters.
    Analyses are executed in the order they appear in the dictionary.

    Parameters
    ----------
    data : SubjectData or list of SubjectData
        Input data to analyze. Single subject or batch of subjects.
    steps : dict[str, dict | None]
        Analysis steps to run. Keys are analysis class names (must match
        exactly), values are dicts of kwargs for that analysis, or None
        for defaults.

        Available analyses (use `list_analyses()` to see all):
        - "RegionalDamage": Parcel-based lesion quantification
        - "FunctionalNetworkMapping": Functional lesion network mapping
        - "StructuralNetworkMapping": Structural lesion network mapping
        - "ParcelAggregation": Aggregate voxel maps to parcels

        Required parameters vary by analysis:
        - FunctionalNetworkMapping requires "connectome_name"
        - StructuralNetworkMapping requires "connectome_name"
        - Others have sensible defaults
    n_jobs : int, default=1
        Number of parallel jobs for batch processing. Use -1 for all CPUs.
    show_progress : bool, default=True
        Show tqdm progress bar during batch processing.
    verbose : bool, default=True
        If True, print progress messages. If False, run silently.

    Returns
    -------
    SubjectData or list of SubjectData
        Analyzed data with results. If input was a list, returns a list.
        Results are stored in `subject.results` dict keyed by analysis name.

    Raises
    ------
    TypeError
        If data is not SubjectData or list of SubjectData.
    KeyError
        If an analysis name in steps is not recognized.
    ValueError
        If required parameters are missing for an analysis.

    Examples
    --------
    Basic usage with RegionalDamage defaults:

    >>> from lacuna import analyze, SubjectData
    >>> result = analyze(mask_data, steps={"RegionalDamage": None})

    With functional network mapping (connectome_name is required):

    >>> result = analyze(
    ...     mask_data,
    ...     steps={
    ...         "RegionalDamage": None,
    ...         "FunctionalNetworkMapping": {"connectome_name": "GSP1000"},
    ...     }
    ... )

    With custom parameters:

    >>> result = analyze(
    ...     mask_data,
    ...     steps={
    ...         "RegionalDamage": {"parcel_names": ["Schaefer2018_100Parcels7Networks"]},
    ...         "FunctionalNetworkMapping": {
    ...             "connectome_name": "GSP1000",
    ...             "method": "boes",
    ...         },
    ...     }
    ... )

    Batch processing with parallelization:

    >>> results = analyze(
    ...     [subject1, subject2, subject3],
    ...     steps={"FunctionalNetworkMapping": {"connectome_name": "GSP1000"}},
    ...     n_jobs=-1,
    ...     show_progress=True,
    ... )
    """
    from lacuna.analysis import get_analysis, list_analyses

    # Validate steps is not empty
    if not steps:
        raise ValueError(
            "steps cannot be empty. Provide at least one analysis, e.g., "
            '{"RegionalDamage": None}'
        )

    # Get available analysis names for validation
    available_analyses = dict(list_analyses())

    # Build list of analysis instances
    analyses: list = []
    for analysis_name, kwargs in steps.items():
        # Strict validation: analysis must exist
        if analysis_name not in available_analyses:
            available_names = sorted(available_analyses.keys())
            raise KeyError(
                f"Unknown analysis: {analysis_name!r}. " f"Available analyses: {available_names}"
            )

        # Get the analysis class
        analysis_cls = get_analysis(analysis_name)

        # Handle None kwargs (use defaults)
        if kwargs is None:
            kwargs = {}
        else:
            # Make a copy to avoid mutating the input
            kwargs = kwargs.copy()

        # Add verbose if not specified
        if "verbose" not in kwargs:
            kwargs["verbose"] = verbose

        # Instantiate the analysis
        try:
            analysis = analysis_cls(**kwargs)
        except TypeError as e:
            raise ValueError(
                f"Invalid parameters for {analysis_name}: {e}. "
                f"Check required parameters for this analysis."
            ) from e

        analyses.append(analysis)

    # Build pipeline
    pipeline = Pipeline(name="analyze")
    for analysis in analyses:
        pipeline.add(analysis)

    # Helper function to run on single subject
    def run_single(subject: SubjectData) -> SubjectData:
        return pipeline.run(subject, verbose=verbose)

    # Handle batch vs single input
    if isinstance(data, list):
        if n_jobs == 1:
            # Sequential processing
            if show_progress:
                try:
                    from tqdm import tqdm

                    return [run_single(d) for d in tqdm(data, desc="Analyzing")]
                except ImportError:
                    return [run_single(d) for d in data]
            else:
                return [run_single(d) for d in data]
        else:
            # Parallel processing with joblib
            try:
                from joblib import Parallel, delayed

                if show_progress:
                    try:
                        from tqdm import tqdm

                        results = Parallel(n_jobs=n_jobs)(
                            delayed(run_single)(d) for d in tqdm(data, desc="Analyzing")
                        )
                    except ImportError:
                        results = Parallel(n_jobs=n_jobs)(delayed(run_single)(d) for d in data)
                else:
                    results = Parallel(n_jobs=n_jobs)(delayed(run_single)(d) for d in data)
                return list(results)
            except ImportError:
                # Fallback to sequential if joblib not available
                return [run_single(d) for d in data]

    return run_single(data)
