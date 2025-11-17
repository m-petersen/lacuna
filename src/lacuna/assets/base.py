"""Base classes for all asset types in Lacuna.

This module provides the foundational classes and patterns used across
all asset management (atlases, templates, transforms, connectomes).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar


@dataclass(frozen=True)
class AssetMetadata(ABC):
    """Base class for all asset metadata.
    
    All asset types must subclass this and implement the abstract methods.
    
    Attributes
    ----------
    name : str
        Unique identifier for the asset
    description : str
        Human-readable description
    """
    
    name: str
    description: str
    
    @abstractmethod
    def validate(self) -> None:
        """Validate metadata consistency.
        
        Raises
        ------
        ValueError
            If metadata is invalid
        """
        pass
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage.
        
        Returns
        -------
        dict
            Dictionary representation of metadata
        """
        return self.__dict__.copy()


@dataclass(frozen=True)
class SpatialAssetMetadata(AssetMetadata):
    """Base class for assets with spatial properties.
    
    Adds coordinate space and resolution tracking.
    
    Attributes
    ----------
    name : str
        Unique identifier for the asset
    description : str
        Human-readable description
    space : str
        Coordinate space identifier (e.g., "MNI152NLin2009cAsym")
    resolution : float
        Voxel resolution in mm (e.g., 1.0, 2.0)
    """
    
    space: str
    resolution: float
    
    def validate(self) -> None:
        """Validate space and resolution.
        
        Raises
        ------
        ValueError
            If space or resolution is invalid
        """
        from lacuna.core.spaces import SUPPORTED_SPACES, SPACE_ALIASES
        
        # Check if space is supported (either directly or as alias)
        if self.space not in SUPPORTED_SPACES and self.space not in SPACE_ALIASES:
            raise ValueError(
                f"Unsupported space: {self.space}. "
                f"Supported: {SUPPORTED_SPACES}"
            )
        
        # Check resolution
        valid_resolutions = [0.5, 1.0, 2.0]
        if self.resolution not in valid_resolutions:
            raise ValueError(
                f"Unsupported resolution: {self.resolution}. "
                f"Supported: {valid_resolutions}"
            )


T = TypeVar('T', bound=AssetMetadata)


class AssetRegistry(Generic[T]):
    """Generic registry for any asset type.
    
    Provides consistent registration, listing, and retrieval
    patterns across all asset types.
    
    Parameters
    ----------
    asset_type_name : str
        Human-readable name of asset type (for error messages)
    
    Examples
    --------
    >>> registry = AssetRegistry[AtlasMetadata]("atlas")
    >>> registry.register(atlas_metadata)
    >>> atlases = registry.list(space="MNI152NLin2009cAsym")
    >>> atlas = registry.get("Schaefer2018_100Parcels7Networks")
    """
    
    def __init__(self, asset_type_name: str = "asset"):
        """Initialize empty registry.
        
        Parameters
        ----------
        asset_type_name : str
            Name of asset type for error messages
        """
        self._registry: dict[str, T] = {}
        self._asset_type_name = asset_type_name
    
    def register(self, metadata: T) -> None:
        """Register an asset.
        
        Parameters
        ----------
        metadata : T
            Asset metadata to register
        
        Raises
        ------
        ValueError
            If asset already registered or metadata invalid
        """
        if metadata.name in self._registry:
            raise ValueError(
                f"{self._asset_type_name.capitalize()} already registered: {metadata.name}"
            )
        
        # Validate before registering
        metadata.validate()
        
        self._registry[metadata.name] = metadata
    
    def unregister(self, name: str) -> None:
        """Unregister an asset.
        
        Parameters
        ----------
        name : str
            Name of asset to unregister
        
        Raises
        ------
        KeyError
            If asset not found
        """
        if name not in self._registry:
            raise KeyError(
                f"{self._asset_type_name.capitalize()} not found: {name}"
            )
        del self._registry[name]
    
    def list(self, **filters) -> list[T]:
        """List assets matching filters.
        
        Parameters
        ----------
        **filters
            Attribute filters (e.g., space="MNI152NLin2009cAsym")
        
        Returns
        -------
        list[T]
            Matching assets
        
        Examples
        --------
        >>> # Get all assets
        >>> all_assets = registry.list()
        >>> 
        >>> # Filter by space
        >>> mni_assets = registry.list(space="MNI152NLin2009cAsym")
        >>> 
        >>> # Filter by multiple criteria
        >>> assets = registry.list(space="MNI152NLin2009cAsym", resolution=1.0)
        """
        assets = list(self._registry.values())
        
        for key, value in filters.items():
            if value is not None:
                assets = [a for a in assets if getattr(a, key, None) == value]
        
        return assets
    
    def get(self, name: str) -> T:
        """Get asset metadata by name.
        
        Parameters
        ----------
        name : str
            Asset name
        
        Returns
        -------
        T
            Asset metadata
        
        Raises
        ------
        KeyError
            If asset not found
        """
        if name not in self._registry:
            raise KeyError(
                f"{self._asset_type_name.capitalize()} not found: {name}. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]
    
    def __contains__(self, name: str) -> bool:
        """Check if asset is registered.
        
        Parameters
        ----------
        name : str
            Asset name
        
        Returns
        -------
        bool
            True if registered
        """
        return name in self._registry
    
    def __len__(self) -> int:
        """Get number of registered assets.
        
        Returns
        -------
        int
            Count of assets
        """
        return len(self._registry)
    
    def keys(self) -> list[str]:
        """Get all registered asset names.
        
        Returns
        -------
        list[str]
            Asset names
        """
        return list(self._registry.keys())


__all__ = [
    "AssetMetadata",
    "SpatialAssetMetadata",
    "AssetRegistry",
]
