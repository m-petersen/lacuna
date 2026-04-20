"""Atlas type definitions for the lacuna atlas engine.

Defines VoxelAtlas, ParcelAtlas, and SurfaceAtlas. VoxelAtlas is the primary
concrete type used in v1; the others are reserved for future releases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import nibabel as nib


@dataclass
class VoxelAtlas:
    """A collection of named 3D brain maps in a common coordinate space.

    Holds one z-scored, averaged NIfTI map per neurotransmitter target.
    All maps are expected to share the same affine and shape.

    Parameters
    ----------
    maps : dict[str, nib.Nifti1Image]
        One map per target name.
    space : str
        Coordinate space identifier, e.g. ``"MNI152NLin6Asym"``.
    resolution : float
        Isotropic voxel size in mm.
    domain : str
        Domain label, e.g. ``"neurotransmitter"``.
    metadata : dict
        Optional free-form metadata.

    Raises
    ------
    ValueError
        If *maps* is empty.
    """

    maps: dict[str, Any]  # dict[str, nib.Nifti1Image]
    space: str
    resolution: float
    domain: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.maps:
            raise ValueError("maps must not be empty; provide at least one target map.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def targets(self) -> list[str]:
        """Sorted list of target names."""
        return sorted(self.maps.keys())

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_map(self, target: str) -> nib.Nifti1Image:
        """Return the NIfTI image for *target*.

        Parameters
        ----------
        target : str
            Target name.

        Returns
        -------
        nib.Nifti1Image

        Raises
        ------
        KeyError
            If *target* is not in this atlas.
        """
        if target not in self.maps:
            raise KeyError(target)
        return self.maps[target]

    def subset(self, targets: list[str]) -> VoxelAtlas:
        """Return a new atlas containing only the specified targets.

        Parameters
        ----------
        targets : list[str]
            Target names to keep.

        Returns
        -------
        VoxelAtlas

        Raises
        ------
        KeyError
            If any requested target is absent from this atlas.
        """
        for t in targets:
            if t not in self.maps:
                raise KeyError(t)

        return VoxelAtlas(
            maps={t: self.maps[t] for t in targets},
            space=self.space,
            resolution=self.resolution,
            domain=self.domain,
            metadata=dict(self.metadata),
        )

    def to_matrix(self, mask: np.ndarray) -> np.ndarray:
        """Extract masked voxels for all targets into a 2-D matrix.

        Parameters
        ----------
        mask : np.ndarray
            Boolean 3-D array of the same spatial shape as the maps.

        Returns
        -------
        np.ndarray
            Shape ``(n_targets, n_masked_voxels)``, rows ordered by
            :attr:`targets` (sorted target names).
        """
        rows = []
        for target in self.targets:
            img = self.maps[target]
            data = np.asarray(img.dataobj)
            rows.append(data[mask])
        return np.vstack(rows)

    def resample_to(
        self,
        target_affine: np.ndarray,
        target_shape: tuple[int, int, int],
    ) -> VoxelAtlas:
        """Resample all maps to a new affine/shape using nilearn.

        Parameters
        ----------
        target_affine : np.ndarray
            4x4 affine for the target grid.
        target_shape : tuple[int, int, int]
            Spatial shape of the target grid.

        Returns
        -------
        VoxelAtlas
            New atlas with resampled maps.
        """
        from nilearn.image import resample_img

        resampled = {
            target: resample_img(
                img,
                target_affine=target_affine,
                target_shape=target_shape,
                interpolation="continuous",
            )
            for target, img in self.maps.items()
        }
        return VoxelAtlas(
            maps=resampled,
            space=self.space,
            resolution=float(np.abs(target_affine[0, 0])),
            domain=self.domain,
            metadata=dict(self.metadata),
        )


class ParcelAtlas:
    """Not implemented in v1."""


class SurfaceAtlas:
    """Not implemented in v1."""
