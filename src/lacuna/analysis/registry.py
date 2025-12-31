"""
Lacuna analysis auto-discovery registry module.

This module provides automatic discovery of BaseAnalysis subclasses
using the pkgutil pattern, similar to scikit-learn and nilearn.

Classes:
    AnalysisRegistry: Registry for discovered analysis classes.

Functions:
    list_analyses: List all available analysis classes.
    get_analysis: Get an analysis class by name.
"""

from __future__ import annotations

import inspect
import logging
import pkgutil
import warnings
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lacuna.analysis.base import BaseAnalysis

logger = logging.getLogger(__name__)

_ANALYSIS_ROOT = str(Path(__file__).parent)
_MODULES_TO_IGNORE = {"tests", "conftest", "registry", "base"}


def _is_abstract(cls: type) -> bool:
    """Check if a class is abstract."""
    if not hasattr(cls, "__abstractmethods__"):
        return False
    return len(cls.__abstractmethods__) > 0


def _discover_analysis_classes() -> list[tuple[str, type[BaseAnalysis]]]:
    """
    Discover all concrete BaseAnalysis subclasses.

    Returns
    -------
    list of tuple
        List of (name, class) tuples for all discovered analyses,
        sorted alphabetically by name.
    """
    from lacuna.analysis.base import BaseAnalysis

    discovered: list[tuple[str, type[BaseAnalysis]]] = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        for _finder, module_name, _is_pkg in pkgutil.iter_modules(
            path=[_ANALYSIS_ROOT], prefix="lacuna.analysis."
        ):
            short_name = module_name.split(".")[-1]
            if short_name in _MODULES_TO_IGNORE or short_name.startswith("_"):
                continue

            try:
                module = import_module(module_name)
            except Exception as e:
                logger.warning(f"Failed to import analysis module '{module_name}': {e}")
                continue

            for name, cls in inspect.getmembers(module, inspect.isclass):
                # Skip private classes and classes imported from other modules
                if name.startswith("_") or cls.__module__ != module_name:
                    continue
                # Skip BaseAnalysis itself
                if cls is BaseAnalysis:
                    continue
                # Check if it's a subclass of BaseAnalysis
                try:
                    if not issubclass(cls, BaseAnalysis):
                        continue
                except TypeError:
                    continue
                # Skip abstract classes
                if _is_abstract(cls):
                    continue

                discovered.append((name, cls))

    return sorted(set(discovered), key=lambda x: x[0])


class AnalysisRegistry:
    """
    Registry for discovered analysis classes.

    This class manages auto-discovery and lookup of analysis classes
    using lazy initialization and caching.

    Class Methods
    -------------
    discover() -> dict[str, type[BaseAnalysis]]
        Discover and cache all analysis classes.
    list_analyses() -> list[tuple[str, type[BaseAnalysis]]]
        List all available analyses as (name, class) tuples.
    get(name: str) -> type[BaseAnalysis]
        Get analysis class by name.
    clear_cache() -> None
        Clear the discovery cache (for testing).
    """

    _discovered: dict[str, type[BaseAnalysis]] | None = None

    @classmethod
    def discover(cls) -> dict[str, type[BaseAnalysis]]:
        """
        Discover and cache all analysis classes.

        Returns
        -------
        dict
            Dictionary mapping analysis names to their classes.
        """
        if cls._discovered is None:
            cls._discovered = {}
            for name, analysis_cls in _discover_analysis_classes():
                if name in cls._discovered:
                    logger.warning(f"Duplicate analysis name '{name}' - keeping first occurrence")
                    continue
                cls._discovered[name] = analysis_cls
        return cls._discovered

    @classmethod
    def list_analyses(cls) -> list[tuple[str, type[BaseAnalysis]]]:
        """
        List all available analyses as (name, class) tuples.

        Returns
        -------
        list of tuple
            List of (name, class) tuples, sorted alphabetically.
        """
        return sorted(cls.discover().items())

    @classmethod
    def get(cls, name: str) -> type[BaseAnalysis]:
        """
        Get analysis class by name.

        Parameters
        ----------
        name : str
            Name of the analysis class.

        Returns
        -------
        type[BaseAnalysis]
            The analysis class.

        Raises
        ------
        KeyError
            If the analysis name is not found.
        """
        analyses = cls.discover()
        if name not in analyses:
            available = ", ".join(sorted(analyses.keys()))
            raise KeyError(f"Unknown analysis '{name}'. Available: {available}")
        return analyses[name]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the discovery cache (useful for testing)."""
        cls._discovered = None


def list_analyses() -> list[tuple[str, type[BaseAnalysis]]]:
    """
    List all available analysis classes.

    This function returns all discovered analyses that:
    - Are subclasses of BaseAnalysis
    - Are concrete (not abstract)
    - Are not private (name doesn't start with '_')

    Returns
    -------
    list of tuple
        List of (name, class) tuples for all available analyses,
        sorted alphabetically by name.

    Examples
    --------
    >>> from lacuna.analysis import list_analyses
    >>> for name, cls in list_analyses():
    ...     print(f"{name}: {cls.batch_strategy}")
    FunctionalNetworkMapping: vectorized
    ParcelAggregation: parallel
    RegionalDamage: parallel
    StructuralNetworkMapping: parallel
    """
    return AnalysisRegistry.list_analyses()


def get_analysis(name: str) -> type[BaseAnalysis]:
    """
    Get an analysis class by name.

    Parameters
    ----------
    name : str
        Name of the analysis class (e.g., "FunctionalNetworkMapping").

    Returns
    -------
    type[BaseAnalysis]
        The analysis class.

    Raises
    ------
    KeyError
        If the analysis name is not found.

    Examples
    --------
    >>> from lacuna.analysis import get_analysis
    >>> FNM = get_analysis("FunctionalNetworkMapping")
    >>> analysis = FNM(connectome_name="GSP1000")
    """
    return AnalysisRegistry.get(name)
