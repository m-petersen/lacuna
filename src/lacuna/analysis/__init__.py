"""
Analysis modules for lesion decoding.

This module provides the base infrastructure and specific analysis implementations
for processing lesion data.
"""

from lacuna.analysis.base import BaseAnalysis
from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping
from lacuna.analysis.parcel_aggregation import ParcelAggregation
from lacuna.analysis.regional_damage import RegionalDamage
from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping

__all__ = [
    "BaseAnalysis",
    "ParcelAggregation",
    "FunctionalNetworkMapping",
    "RegionalDamage",
    "StructuralNetworkMapping",
]
