"""
Provenance tracking utilities for reproducibility.

Functions for recording and managing transformation history.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .exceptions import ProvenanceError


@dataclass
class TransformationRecord:
    """Record of a spatial transformation operation.

    Attributes:
        source_space: Source coordinate space identifier (e.g., 'MNI152NLin6Asym')
        source_resolution: Source resolution in mm
        target_space: Target coordinate space identifier
        target_resolution: Target resolution in mm
        method: Transformation method (e.g., 'nitransforms')
        interpolation: Interpolation method used (e.g., 'linear', 'nearest')
        timestamp: ISO 8601 timestamp of transformation
        rationale: Optional explanation for transformation strategy choice
        transform_file: Optional path/identifier of transform file used
    """

    source_space: str
    source_resolution: float
    target_space: str
    target_resolution: float
    method: str
    interpolation: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    rationale: str | None = None
    transform_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for provenance tracking."""
        result = {
            "type": "spatial_transformation",
            "source_space": self.source_space,
            "source_resolution": self.source_resolution,
            "target_space": self.target_space,
            "target_resolution": self.target_resolution,
            "method": self.method,
            "interpolation": self.interpolation,
            "timestamp": self.timestamp,
        }
        if self.rationale is not None:
            result["rationale"] = self.rationale
        if self.transform_file is not None:
            result["transform_file"] = self.transform_file
        return result


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
        Fully qualified function name (e.g., 'lacuna.analysis.RegionalDamage').
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
    ...     function="lacuna.analysis.RegionalDamage",
    ...     parameters={"atlas_names": ["Schaefer2018_100Parcels7Networks"]},
    ...     version="0.1.0",
    ... )
    >>> record['function']
    'lacuna.analysis.RegionalDamage'
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

    for field_name in required_fields:
        if field_name not in record:
            raise ProvenanceError(f"Provenance record missing required field: {field_name}")

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
