"""
High-level pipeline API for declarative analysis workflows.

This module provides a Pipeline class that allows defining and running
complete analysis workflows in a declarative manner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lacuna.analysis.base import BaseAnalysis
    from lacuna.core.mask_data import MaskData


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

    def run(self, data: MaskData, log_level: int = 1) -> MaskData:
        """
        Run the pipeline on a single subject.

        Parameters
        ----------
        data : MaskData
            Input data to process
        log_level : int, default=1
            Logging verbosity (0=silent, 1=standard, 2=verbose)

        Returns
        -------
        MaskData
            Processed data with all analysis results
        """
        result = data

        for step in self._steps:
            # Run the analysis
            if log_level >= 2:
                print(f"Running {step.name}...")

            result = step.analysis.run(result)

        return result

    def run_batch(
        self,
        data_list: list[MaskData],
        n_jobs: int = -1,
        show_progress: bool = True,
        parallel: bool = True,
    ) -> list[MaskData]:
        """
        Run the pipeline on multiple subjects.

        Parameters
        ----------
        data_list : list of MaskData
            List of subjects to process
        n_jobs : int, default=-1
            Number of parallel jobs (-1 uses all CPUs)
        show_progress : bool, default=True
            Show progress bar
        parallel : bool, default=True
            Whether to process subjects in parallel

        Returns
        -------
        list of MaskData
            Processed data for each subject
        """
        if not parallel or n_jobs == 1:
            # Sequential processing
            results = []
            iterator = data_list
            if show_progress:
                from tqdm import tqdm

                iterator = tqdm(data_list, desc=self.name)

            for data in iterator:
                results.append(self.run(data, log_level=0))
            return results

        # Parallel processing - run each step as a batch
        results = data_list

        for step in self._steps:
            # Use batch_process for this step
            from lacuna.batch.api import batch_process

            results = batch_process(
                inputs=results,
                analysis=step.analysis,
                n_jobs=n_jobs,
                show_progress=show_progress,
            )

        return results

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
    data: MaskData | list[MaskData],
    *,
    functional_connectome: str | None = None,
    structural_connectome: str | None = None,
) -> MaskData | list[MaskData]:
    """
    Convenience function for common analysis workflows.

    This is a simple entry point that runs the standard lesion analysis
    pipeline. By default, it runs RegionalDamage (parcel-based lesion
    quantification). Optionally, you can enable functional or structural
    lesion network mapping by providing registered connectome names.

    Parameters
    ----------
    data : MaskData or list of MaskData
        Input data to analyze
    functional_connectome : str, optional
        Name of a registered functional connectome to enable functional
        lesion network mapping (fLNM). Use `list_functional_connectomes()`
        to see available connectomes.
    structural_connectome : str, optional
        Name of a registered structural connectome to enable structural
        lesion network mapping (sLNM). Use `list_structural_connectomes()`
        to see available connectomes.

    Returns
    -------
    MaskData or list of MaskData
        Analyzed data with results

    Examples
    --------
    Basic usage (parcel aggregation only):

    >>> from lacuna import analyze, MaskData
    >>> result = analyze(mask_data)
    >>> print(result.results.keys())

    With functional network mapping:

    >>> result = analyze(mask_data, functional_connectome="GSP1000")

    With structural network mapping:

    >>> result = analyze(mask_data, structural_connectome="dTOR985")

    With both:

    >>> result = analyze(
    ...     mask_data,
    ...     functional_connectome="GSP1000",
    ...     structural_connectome="dTOR985"
    ... )
    """
    # Import here to avoid circular imports
    from lacuna.analysis import (
        FunctionalNetworkMapping,
        RegionalDamage,
        StructuralNetworkMapping,
    )
    from lacuna.batch.api import batch_process

    # Build pipeline of analyses
    analyses: list = [RegionalDamage()]

    if functional_connectome:
        analyses.append(FunctionalNetworkMapping(connectome_name=functional_connectome))

    if structural_connectome:
        analyses.append(StructuralNetworkMapping(connectome_name=structural_connectome))

    # Single analysis: run directly
    if len(analyses) == 1:
        if isinstance(data, list):
            return batch_process(inputs=data, analysis=analyses[0])
        return analyses[0].run(data)

    # Multiple analyses: use pipeline
    pipeline = Pipeline(name="analyze")
    for analysis in analyses:
        pipeline.add(analysis)

    if isinstance(data, list):
        return [pipeline.run(d) for d in data]

    return pipeline.run(data)
