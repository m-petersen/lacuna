"""
Export utilities for analysis results.

Provides convenient functions for exporting MaskData analysis results
to various formats (CSV, TSV, JSON) for downstream analysis and visualization.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.mask_data import MaskData


def export_results_to_csv(
    mask_data: MaskData,
    output_path: str | Path,
    analysis_name: str | None = None,
    include_metadata: bool = True,
) -> Path:
    """
    Export analysis results to CSV format.

    Converts nested results dictionary to a flat CSV structure suitable
    for statistical analysis or visualization in external tools.

    Parameters
    ----------
    mask_data : MaskData
        MaskData object with analysis results
    output_path : str or Path
        Output CSV file path
    analysis_name : str, optional
        Specific analysis to export. If None, exports all results.
        Example: "RegionalDamage", "AtlasAggregation"
    include_metadata : bool, default=True
        Include subject metadata (subject_id, session_id, etc.) as columns

    Returns
    -------
    Path
        Path to created CSV file

    Raises
    ------
    ValueError
        If mask_data has no results or specified analysis not found

    Examples
    --------
    >>> from lacuna import MaskData
    >>> from lacuna.analysis import RegionalDamage
    >>> from lacuna.io import export_results_to_csv
    >>>
    >>> lesion = MaskData.from_nifti("lesion.nii.gz")
    >>> analysis = RegionalDamage()
    >>> result = analysis.run(lesion)
    >>>
    >>> # Export all results
    >>> export_results_to_csv(result, "results.csv")
    >>>
    >>> # Export specific analysis
    >>> export_results_to_csv(result, "damage.csv", analysis_name="RegionalDamage")

    Notes
    -----
    - Results are flattened: nested dicts become columns with dot notation
    - Example: {"AtlasAggregation": {"region1": 0.5}} becomes columns
      "AtlasAggregation.region1" with value 0.5
    - Multiple analyses create multiple columns
    - Metadata columns (if included): subject_id, session_id, coordinate_space
    """
    output_path = Path(output_path)

    if not mask_data.results:
        raise ValueError("MaskData has no results to export")

    # Filter by analysis name if specified
    if analysis_name:
        if analysis_name not in mask_data.results:
            available = list(mask_data.results.keys())
            raise ValueError(
                f"Analysis '{analysis_name}' not found in results.\nAvailable analyses: {available}"
            )
        results_to_export = {analysis_name: mask_data.results[analysis_name]}
    else:
        results_to_export = mask_data.results

    # Flatten results to single row
    row_data = {}

    # Add metadata if requested
    if include_metadata:
        row_data["subject_id"] = mask_data.metadata.get("subject_id", "unknown")
        row_data["session_id"] = mask_data.metadata.get("session_id", "")
        row_data["coordinate_space"] = mask_data.get_coordinate_space()

    # Flatten nested results
    for analysis, results_dict in results_to_export.items():
        if isinstance(results_dict, dict):
            for key, value in results_dict.items():
                # Create column name: Analysis.key
                col_name = f"{analysis}.{key}"
                # Convert to scalar if possible
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    row_data[col_name] = value[0]
                else:
                    row_data[col_name] = value
        else:
            # Non-dict result, store as-is
            row_data[analysis] = results_dict

    # Create DataFrame and save
    df = pd.DataFrame([row_data])
    df.to_csv(output_path, index=False)

    return output_path


def export_results_to_tsv(
    mask_data: MaskData,
    output_path: str | Path,
    analysis_name: str | None = None,
    include_metadata: bool = True,
) -> Path:
    """
    Export analysis results to TSV (tab-separated values) format.

    Identical to export_results_to_csv but uses tab delimiter.
    TSV is preferred in neuroimaging for BIDS compatibility.

    Parameters
    ----------
    mask_data : MaskData
        MaskData object with analysis results
    output_path : str or Path
        Output TSV file path
    analysis_name : str, optional
        Specific analysis to export. If None, exports all results.
    include_metadata : bool, default=True
        Include subject metadata as columns

    Returns
    -------
    Path
        Path to created TSV file

    Raises
    ------
    ValueError
        If mask_data has no results or specified analysis not found

    Examples
    --------
    >>> from lacuna.io import export_results_to_tsv
    >>>
    >>> # Export to TSV (BIDS-compatible format)
    >>> export_results_to_tsv(result, "results.tsv")
    >>>
    >>> # Export specific analysis without metadata
    >>> export_results_to_tsv(
    ...     result,
    ...     "atlas_only.tsv",
    ...     analysis_name="AtlasAggregation",
    ...     include_metadata=False
    ... )

    See Also
    --------
    export_results_to_csv : CSV export (identical but comma-delimited)
    """
    output_path = Path(output_path)

    if not mask_data.results:
        raise ValueError("MaskData has no results to export")

    # Filter by analysis name if specified
    if analysis_name:
        if analysis_name not in mask_data.results:
            available = list(mask_data.results.keys())
            raise ValueError(
                f"Analysis '{analysis_name}' not found in results.\nAvailable analyses: {available}"
            )
        results_to_export = {analysis_name: mask_data.results[analysis_name]}
    else:
        results_to_export = mask_data.results

    # Flatten results to single row
    row_data = {}

    # Add metadata if requested
    if include_metadata:
        row_data["subject_id"] = mask_data.metadata.get("subject_id", "unknown")
        row_data["session_id"] = mask_data.metadata.get("session_id", "")
        row_data["coordinate_space"] = mask_data.get_coordinate_space()

    # Flatten nested results
    for analysis, results_dict in results_to_export.items():
        if isinstance(results_dict, dict):
            for key, value in results_dict.items():
                col_name = f"{analysis}.{key}"
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    row_data[col_name] = value[0]
                else:
                    row_data[col_name] = value
        else:
            row_data[analysis] = results_dict

    # Create DataFrame and save with tab delimiter
    df = pd.DataFrame([row_data])
    df.to_csv(output_path, sep="\t", index=False)

    return output_path


def export_provenance_to_json(
    mask_data: MaskData,
    output_path: str | Path,
    indent: int = 2,
) -> Path:
    """
    Export provenance data to JSON format.

    Saves the complete processing history and metadata as a standalone
    JSON file for reproducibility and audit trails.

    Parameters
    ----------
    mask_data : MaskData
        MaskData object with provenance data
    output_path : str or Path
        Output JSON file path
    indent : int, default=2
        JSON indentation for readability (0 for compact)

    Returns
    -------
    Path
        Path to created JSON file

    Raises
    ------
    ValueError
        If mask_data has no provenance data

    Examples
    --------
    >>> from lacuna.io import export_provenance_to_json
    >>>
    >>> # Export provenance history
    >>> export_provenance_to_json(result, "provenance.json")
    >>>
    >>> # Export compact JSON
    >>> export_provenance_to_json(result, "prov.json", indent=0)

    Notes
    -----
    Provenance includes:
    - Source file paths
    - Processing steps (transformations, analyses)
    - Software versions
    - Timestamps
    - Parameters used for each operation
    """
    output_path = Path(output_path)

    if not mask_data.provenance:
        raise ValueError(
            "MaskData has no provenance data to export.\n"
            "Provenance is automatically tracked during analysis operations."
        )

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write provenance as JSON
    with open(output_path, "w") as f:
        json.dump(mask_data.provenance, f, indent=indent if indent > 0 else None)

    return output_path


def export_results_to_json(
    mask_data: MaskData,
    output_path: str | Path,
    analysis_name: str | None = None,
    include_metadata: bool = True,
    include_provenance: bool = False,
    indent: int = 2,
) -> Path:
    """
    Export analysis results to JSON format.

    Creates a JSON file with analysis results, optionally including
    metadata and provenance. Useful for web applications or further
    programmatic processing.

    Parameters
    ----------
    mask_data : MaskData
        MaskData object with analysis results
    output_path : str or Path
        Output JSON file path
    analysis_name : str, optional
        Specific analysis to export. If None, exports all results.
    include_metadata : bool, default=True
        Include subject metadata in JSON
    include_provenance : bool, default=False
        Include provenance data in JSON
    indent : int, default=2
        JSON indentation for readability (0 for compact)

    Returns
    -------
    Path
        Path to created JSON file

    Raises
    ------
    ValueError
        If mask_data has no results or specified analysis not found

    Examples
    --------
    >>> from lacuna.io import export_results_to_json
    >>>
    >>> # Export all results with metadata
    >>> export_results_to_json(result, "results.json")
    >>>
    >>> # Export specific analysis with full provenance
    >>> export_results_to_json(
    ...     result,
    ...     "damage_full.json",
    ...     analysis_name="RegionalDamage",
    ...     include_provenance=True
    ... )
    >>>
    >>> # Compact JSON for web APIs
    >>> export_results_to_json(result, "api_response.json", indent=0)

    Notes
    -----
    JSON structure:
    {
        "metadata": {...},          # If include_metadata=True
        "results": {...},           # Analysis results
        "provenance": {...}         # If include_provenance=True
    }
    """
    output_path = Path(output_path)

    if not mask_data.results:
        raise ValueError("MaskData has no results to export")

    # Build export data structure
    export_data: dict[str, Any] = {}

    # Add metadata if requested
    if include_metadata:
        export_data["metadata"] = dict(mask_data.metadata)
        export_data["metadata"]["coordinate_space"] = mask_data.get_coordinate_space()

    # Add results
    if analysis_name:
        if analysis_name not in mask_data.results:
            available = list(mask_data.results.keys())
            raise ValueError(
                f"Analysis '{analysis_name}' not found in results.\nAvailable analyses: {available}"
            )
        export_data["results"] = {analysis_name: mask_data.results[analysis_name]}
    else:
        export_data["results"] = mask_data.results

    # Add provenance if requested
    if include_provenance and mask_data.provenance:
        export_data["provenance"] = mask_data.provenance

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=indent if indent > 0 else None)

    return output_path


def batch_export_to_csv(
    mask_data_list: list[MaskData],
    output_path: str | Path,
    analysis_name: str | None = None,
    include_metadata: bool = True,
) -> Path:
    """
    Export results from multiple MaskData objects to a single CSV.

    Combines results from multiple subjects into one CSV file with each
    row representing one subject. Ideal for group-level statistical analysis.

    Parameters
    ----------
    mask_data_list : list[MaskData]
        List of MaskData objects (typically from batch processing)
    output_path : str or Path
        Output CSV file path
    analysis_name : str, optional
        Specific analysis to export. If None, exports all results.
    include_metadata : bool, default=True
        Include subject metadata as columns

    Returns
    -------
    Path
        Path to created CSV file

    Raises
    ------
    ValueError
        If list is empty or subjects have no results

    Examples
    --------
    >>> from lacuna.io import load_bids_dataset, batch_export_to_csv
    >>> from lacuna.analysis import RegionalDamage
    >>>
    >>> # Load multiple subjects
    >>> dataset = load_bids_dataset("bids_dir")
    >>> analysis = RegionalDamage()
    >>>
    >>> # Run analysis on all subjects
    >>> results = [analysis.run(lesion) for lesion in dataset.values()]
    >>>
    >>> # Export to single CSV for group analysis
    >>> batch_export_to_csv(results, "group_results.csv")

    Notes
    -----
    - All subjects must have the same analysis results structure
    - Missing values are filled with NaN
    - Each row represents one subject
    - Columns are shared across all subjects
    """
    if not mask_data_list:
        raise ValueError("mask_data_list is empty")

    output_path = Path(output_path)

    # Collect all rows
    rows = []
    for mask_data in mask_data_list:
        if not mask_data.results:
            continue  # Skip subjects without results

        row_data = {}

        # Add metadata if requested
        if include_metadata:
            row_data["subject_id"] = mask_data.metadata.get("subject_id", "unknown")
            row_data["session_id"] = mask_data.metadata.get("session_id", "")
            row_data["coordinate_space"] = mask_data.get_coordinate_space()

        # Filter by analysis name
        if analysis_name:
            if analysis_name not in mask_data.results:
                continue  # Skip subjects without this analysis
            results_to_export = {analysis_name: mask_data.results[analysis_name]}
        else:
            results_to_export = mask_data.results

        # Flatten results
        for analysis, results_dict in results_to_export.items():
            if isinstance(results_dict, dict):
                for key, value in results_dict.items():
                    col_name = f"{analysis}.{key}"
                    if isinstance(value, (list, tuple)) and len(value) == 1:
                        row_data[col_name] = value[0]
                    else:
                        row_data[col_name] = value
            else:
                row_data[analysis] = results_dict

        rows.append(row_data)

    if not rows:
        raise ValueError("No results to export. Ensure subjects have analysis results.")

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return output_path


def batch_export_to_tsv(
    mask_data_list: list[MaskData],
    output_path: str | Path,
    analysis_name: str | None = None,
    include_metadata: bool = True,
) -> Path:
    """
    Export results from multiple MaskData objects to a single TSV.

    Identical to batch_export_to_csv but uses tab delimiter.
    TSV is preferred in neuroimaging for BIDS compatibility.

    Parameters
    ----------
    mask_data_list : list[MaskData]
        List of MaskData objects
    output_path : str or Path
        Output TSV file path
    analysis_name : str, optional
        Specific analysis to export
    include_metadata : bool, default=True
        Include subject metadata as columns

    Returns
    -------
    Path
        Path to created TSV file

    Raises
    ------
    ValueError
        If list is empty or subjects have no results

    Examples
    --------
    >>> from lacuna.io import batch_export_to_tsv
    >>>
    >>> # Export group results to BIDS-compatible TSV
    >>> batch_export_to_tsv(results, "group_results.tsv")

    See Also
    --------
    batch_export_to_csv : CSV batch export
    """
    if not mask_data_list:
        raise ValueError("mask_data_list is empty")

    output_path = Path(output_path)

    # Collect all rows (same as CSV version)
    rows = []
    for mask_data in mask_data_list:
        if not mask_data.results:
            continue

        row_data = {}

        if include_metadata:
            row_data["subject_id"] = mask_data.metadata.get("subject_id", "unknown")
            row_data["session_id"] = mask_data.metadata.get("session_id", "")
            row_data["coordinate_space"] = mask_data.get_coordinate_space()

        if analysis_name:
            if analysis_name not in mask_data.results:
                continue
            results_to_export = {analysis_name: mask_data.results[analysis_name]}
        else:
            results_to_export = mask_data.results

        for analysis, results_dict in results_to_export.items():
            if isinstance(results_dict, dict):
                for key, value in results_dict.items():
                    col_name = f"{analysis}.{key}"
                    if isinstance(value, (list, tuple)) and len(value) == 1:
                        row_data[col_name] = value[0]
                    else:
                        row_data[col_name] = value
            else:
                row_data[analysis] = results_dict

        rows.append(row_data)

    if not rows:
        raise ValueError("No results to export. Ensure subjects have analysis results.")

    # Create DataFrame and save with tab delimiter
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)

    return output_path
