"""
Analysis modules for lesion decoding.

This module provides the base infrastructure and specific analysis implementations
for processing lesion data.
"""

from ldk.analysis.atlas_aggregation import AtlasAggregation
from ldk.analysis.base import BaseAnalysis
from ldk.analysis.regional_damage import RegionalDamage

__all__ = ["BaseAnalysis", "AtlasAggregation", "RegionalDamage"]
