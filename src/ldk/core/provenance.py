"""
Provenance tracking utilities for reproducibility.

Functions for recording and managing transformation history.
"""

from datetime import datetime, timezone
from typing import Any

from .exceptions import ProvenanceError


def create_provenance_record(
    function: str,
    parameters: dict[str, Any],
    version: str,
    output_space: str | None = None,
) -> dict[str, Any]:
    """
    Create a provenance record for a transformation.

    Parameters
    ----------
    function : str
        Fully qualified function name (e.g., 'ldk.preprocess.normalize_to_mni').
    parameters : dict
        Function parameters (must be JSON-serializable).
    version : str
        Package version at time of execution.
    output_space : str, optional
        Resulting coordinate space (if spatial operation).

    Returns
    -------
    dict
        Provenance record with function, parameters, timestamp, version.

    Raises
    ------
    ProvenanceError
        If parameters are not JSON-serializable.

    Examples
    --------
    >>> record = create_provenance_record(
    ...     function="ldk.preprocess.normalize_to_mni",
    ...     parameters={"template": "MNI152_2mm"},
    ...     version="0.1.0",
    ...     output_space="MNI152_2mm"
    ... )
    >>> record['function']
    'ldk.preprocess.normalize_to_mni'
    """
    # Validate parameters are serializable
    try:
        import json

        json.dumps(parameters)
    except (TypeError, ValueError) as e:
        raise ProvenanceError(f"Parameters must be JSON-serializable, got error: {e}") from e

    # Create record
    record = {
        "function": function,
        "parameters": parameters,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": version,
    }

    if output_space is not None:
        record["output_space"] = output_space

    return record


def validate_provenance_record(record: dict[str, Any]) -> None:
    """
    Validate a provenance record structure.

    Parameters
    ----------
    record : dict
        Provenance record to validate.

    Raises
    ------
    ProvenanceError
        If record structure is invalid.
    """
    required_fields = ["function", "parameters", "timestamp", "version"]

    for field in required_fields:
        if field not in record:
            raise ProvenanceError(f"Provenance record missing required field: {field}")

    # Validate timestamp format
    try:
        datetime.fromisoformat(record["timestamp"])
    except (ValueError, TypeError) as e:
        raise ProvenanceError(f"Invalid timestamp format: {record['timestamp']}") from e

    # Validate parameters are dict
    if not isinstance(record["parameters"], dict):
        raise ProvenanceError(f"Parameters must be dict, got {type(record['parameters'])}")


def merge_provenance(base_provenance: list, new_provenance: list) -> list:
    """
    Merge two provenance lists.

    Parameters
    ----------
    base_provenance : list
        Base provenance history.
    new_provenance : list
        New provenance to append.

    Returns
    -------
    list
        Merged provenance list (ordered chronologically).
    """
    merged = list(base_provenance)  # Copy to avoid mutation
    merged.extend(new_provenance)
    return merged
