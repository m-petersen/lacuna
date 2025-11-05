"""
Analysis modules for lesion decoding.

This module provides the base infrastructure and specific analysis implementations
for processing lesion data.
"""

from ldk.analysis.atlas_aggregation import AtlasAggregation
from ldk.analysis.base import BaseAnalysis
from ldk.analysis.regional_damage import RegionalDamage
from ldk.analysis.structural_network_mapping import StructuralNetworkMapping

__all__ = ["BaseAnalysis", "AtlasAggregation", "RegionalDamage", "StructuralNetworkMapping"]
