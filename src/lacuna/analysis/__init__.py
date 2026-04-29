"""
Analysis modules for lesion decoding.

This module provides the base infrastructure and specific analysis implementations
for processing lesion data.

Functions
---------
list_analyses()
    List all available analysis classes discovered via auto-discovery.
get_analysis(name)
    Get an analysis class by name.

Classes
-------
BaseAnalysis
    Abstract base class for all analyses.
FunctionalNetworkMapping
    Functional lesion network mapping analysis.
ParcelAggregation
    Aggregate voxel-wise results to parcels.
LocalDamage
    Regional damage analysis.
StructuralNetworkMapping
    Structural lesion network mapping analysis.

Examples
--------
>>> from lacuna.analysis import list_analyses, get_analysis
>>> for name, cls in list_analyses():
...     print(f"{name}: {cls.batch_strategy}")
FunctionalNetworkMapping: vectorized
ParcelAggregation: parallel
LocalDamage: parallel
StructuralNetworkMapping: parallel

>>> FNM = get_analysis("FunctionalNetworkMapping")
>>> analysis = FNM(connectome_name="GSP1000")
"""

from lacuna.analysis.accelerated_functional_network_mapping import (
    AcceleratedFunctionalNetworkMapping,
)
from lacuna.analysis.base import BaseAnalysis
from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping
from lacuna.analysis.functional_neurotransmitter_fingerprinting import (
    FunctionalNeurotransmitterFingerprinting,
)
from lacuna.analysis.local_neurotransmitter_fingerprinting import LocalNeurotransmitterFingerprinting
from lacuna.analysis.parcel_aggregation import ParcelAggregation
from lacuna.analysis.structural_neurotransmitter_fingerprinting import (
    StructuralNeurotransmitterFingerprinting,
)
from lacuna.analysis.local_damage import LocalDamage
from lacuna.analysis.registry import get_analysis, list_analyses
from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

__all__ = [
    # Functions
    "list_analyses",
    "get_analysis",
    # Classes
    "BaseAnalysis",
    "ParcelAggregation",
    "FunctionalNetworkMapping",
    "LocalDamage",
    "AcceleratedFunctionalNetworkMapping",
    "LocalNeurotransmitterFingerprinting",
    "StructuralNeurotransmitterFingerprinting",
    "FunctionalNeurotransmitterFingerprinting",
    "StructuralNetworkMapping",
]
