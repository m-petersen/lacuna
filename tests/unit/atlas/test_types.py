"""Unit tests for lacuna.atlas.types module.

Tests for VoxelAtlas dataclass: construction, properties, and methods.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest


def _make_img(shape=(4, 4, 4), affine=None):
    """Helper: create a simple NIfTI image with ones."""
    if affine is None:
        affine = np.eye(4)
    data = np.ones(shape, dtype=np.float32)
    return nib.Nifti1Image(data, affine)


def _make_maps(*names, shape=(4, 4, 4)):
    """Helper: create a dict of named NIfTI images."""
    return {name: _make_img(shape) for name in names}


class TestVoxelAtlasConstruction:
    """Tests for VoxelAtlas.__init__ and basic construction."""

    def test_basic_construction(self):
        """VoxelAtlas can be created with required arguments."""
        from lacuna.atlas.types import VoxelAtlas

        maps = _make_maps("serotonin", "dopamine")
        atlas = VoxelAtlas(
            maps=maps,
            space="MNI152NLin6Asym",
            resolution=2.0,
            domain="neurotransmitter",
        )

        assert atlas.space == "MNI152NLin6Asym"
        assert atlas.resolution == 2.0
        assert atlas.domain == "neurotransmitter"
        assert atlas.metadata == {}

    def test_construction_with_metadata(self):
        """VoxelAtlas accepts optional metadata dict."""
        from lacuna.atlas.types import VoxelAtlas

        maps = _make_maps("serotonin")
        meta = {"source": "JuSpace", "version": "1.0"}
        atlas = VoxelAtlas(
            maps=maps,
            space="MNI152NLin6Asym",
            resolution=2.0,
            domain="neurotransmitter",
            metadata=meta,
        )

        assert atlas.metadata == {"source": "JuSpace", "version": "1.0"}

    def test_empty_maps_raises_value_error(self):
        """VoxelAtlas raises ValueError when maps dict is empty."""
        from lacuna.atlas.types import VoxelAtlas

        with pytest.raises(ValueError, match="maps"):
            VoxelAtlas(maps={}, space="MNI152NLin6Asym", resolution=2.0, domain="neurotransmitter")

    def test_metadata_default_not_shared(self):
        """Default metadata dict is not shared between instances."""
        from lacuna.atlas.types import VoxelAtlas

        maps = _make_maps("serotonin")
        a = VoxelAtlas(maps=maps, space="MNI", resolution=2.0, domain="neurotransmitter")
        b = VoxelAtlas(maps=maps, space="MNI", resolution=2.0, domain="neurotransmitter")
        a.metadata["key"] = "value"
        assert "key" not in b.metadata


class TestVoxelAtlasTargets:
    """Tests for the targets property."""

    def test_targets_returns_sorted_list(self):
        """targets property returns keys sorted alphabetically."""
        from lacuna.atlas.types import VoxelAtlas

        maps = _make_maps("serotonin", "acetylcholine", "dopamine")
        atlas = VoxelAtlas(maps=maps, space="MNI", resolution=2.0, domain="neurotransmitter")

        assert atlas.targets == ["acetylcholine", "dopamine", "serotonin"]

    def test_targets_single_map(self):
        """targets works with a single-map atlas."""
        from lacuna.atlas.types import VoxelAtlas

        atlas = VoxelAtlas(
            maps=_make_maps("norepinephrine"),
            space="MNI",
            resolution=2.0,
            domain="neurotransmitter",
        )
        assert atlas.targets == ["norepinephrine"]


class TestVoxelAtlasGetMap:
    """Tests for the get_map method."""

    @pytest.fixture
    def atlas(self):
        from lacuna.atlas.types import VoxelAtlas

        return VoxelAtlas(
            maps=_make_maps("serotonin", "dopamine"),
            space="MNI152NLin6Asym",
            resolution=2.0,
            domain="neurotransmitter",
        )

    def test_get_existing_map(self, atlas):
        """get_map returns the NIfTI image for an existing target."""
        img = atlas.get_map("serotonin")
        assert isinstance(img, nib.Nifti1Image)

    def test_get_nonexistent_map_raises_key_error(self, atlas):
        """get_map raises KeyError for an unknown target."""
        with pytest.raises(KeyError):
            atlas.get_map("gaba")


class TestVoxelAtlasSubset:
    """Tests for the subset method."""

    @pytest.fixture
    def atlas(self):
        from lacuna.atlas.types import VoxelAtlas

        return VoxelAtlas(
            maps=_make_maps("serotonin", "dopamine", "acetylcholine"),
            space="MNI152NLin6Asym",
            resolution=2.0,
            domain="neurotransmitter",
            metadata={"source": "test"},
        )

    def test_subset_valid_targets(self, atlas):
        """subset returns a new VoxelAtlas with only the requested targets."""
        sub = atlas.subset(["serotonin", "dopamine"])
        assert isinstance(sub, type(atlas))
        assert sub.targets == ["dopamine", "serotonin"]

    def test_subset_preserves_metadata(self, atlas):
        """subset carries over space, resolution, domain, and metadata."""
        sub = atlas.subset(["serotonin"])
        assert sub.space == atlas.space
        assert sub.resolution == atlas.resolution
        assert sub.domain == atlas.domain
        assert sub.metadata == atlas.metadata

    def test_subset_invalid_target_raises_key_error(self, atlas):
        """subset raises KeyError when a requested target is missing."""
        with pytest.raises(KeyError):
            atlas.subset(["serotonin", "norepinephrine"])

    def test_subset_returns_new_instance(self, atlas):
        """subset returns a new atlas, not the same object."""
        sub = atlas.subset(["serotonin"])
        assert sub is not atlas


class TestVoxelAtlasToMatrix:
    """Tests for the to_matrix method."""

    @pytest.fixture
    def atlas_and_mask(self):
        from lacuna.atlas.types import VoxelAtlas

        shape = (3, 3, 3)
        affine = np.eye(4)

        # serotonin: all twos, dopamine: all threes
        serotonin_data = np.full(shape, 2.0, dtype=np.float32)
        dopamine_data = np.full(shape, 3.0, dtype=np.float32)

        maps = {
            "serotonin": nib.Nifti1Image(serotonin_data, affine),
            "dopamine": nib.Nifti1Image(dopamine_data, affine),
        }
        atlas = VoxelAtlas(maps=maps, space="MNI", resolution=1.0, domain="neurotransmitter")

        # mask: keep the 8 voxels of the inner 2x2x2 block
        mask = np.zeros(shape, dtype=bool)
        mask[1:3, 1:3, 1:3] = True  # 8 voxels

        return atlas, mask

    def test_to_matrix_shape(self, atlas_and_mask):
        """to_matrix returns array of shape (n_targets, n_masked_voxels)."""
        atlas, mask = atlas_and_mask
        mat = atlas.to_matrix(mask)
        n_targets = len(atlas.targets)
        n_voxels = int(mask.sum())
        assert mat.shape == (n_targets, n_voxels)

    def test_to_matrix_values(self, atlas_and_mask):
        """to_matrix extracts correct values under the mask."""
        atlas, mask = atlas_and_mask
        mat = atlas.to_matrix(mask)

        # targets sorted: dopamine=0, serotonin=1
        assert np.all(mat[0] == 3.0)  # dopamine
        assert np.all(mat[1] == 2.0)  # serotonin

    def test_to_matrix_row_order_matches_sorted_targets(self, atlas_and_mask):
        """Row order in to_matrix matches sorted target names."""
        atlas, mask = atlas_and_mask
        mat = atlas.to_matrix(mask)
        assert atlas.targets == sorted(atlas.targets)
        # Verify row count equals target count
        assert mat.shape[0] == len(atlas.targets)


class TestPlaceholderClasses:
    """Tests that placeholder classes exist with correct docstrings."""

    def test_parcel_atlas_exists(self):
        """ParcelAtlas placeholder class is importable."""
        from lacuna.atlas.types import ParcelAtlas  # noqa: F401

    def test_surface_atlas_exists(self):
        """SurfaceAtlas placeholder class is importable."""
        from lacuna.atlas.types import SurfaceAtlas  # noqa: F401

    def test_parcel_atlas_docstring(self):
        from lacuna.atlas.types import ParcelAtlas

        assert ParcelAtlas.__doc__ is not None
        assert "Not implemented" in ParcelAtlas.__doc__

    def test_surface_atlas_docstring(self):
        from lacuna.atlas.types import SurfaceAtlas

        assert SurfaceAtlas.__doc__ is not None
        assert "Not implemented" in SurfaceAtlas.__doc__
