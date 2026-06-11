"""Regression tests for REFERENCE_AFFINES / REFERENCE_SHAPES grid correctness.

Background
----------
The reference grids define the canonical voxel grid for each (space, resolution)
and are used as regrid targets and for provenance. A latent bug had the 2mm
entries reuse the *1mm* origin (e.g. 6Asym -91, 2009c -96) instead of
TemplateFlow's actual 2mm origin (-90, -96.5): a 2mm MNI grid is NOT an
origin-preserving downsample of the 1mm grid. These tests pin the grids to the
canonical TemplateFlow definitions so the bug cannot silently return.

The canonical origins below come from TemplateFlow's own
``tpl-<space>/template_description.json`` "res" blocks.
"""

from __future__ import annotations

import glob

import nibabel as nib
import numpy as np
import pytest

from lacuna.core.spaces import REFERENCE_AFFINES, REFERENCE_SHAPES

# Canonical voxel-grid origins from TemplateFlow template_description.json.
# (space, resolution) -> origin (world coords of voxel [0,0,0] in RAS+).
CANONICAL_ORIGINS = {
    ("MNI152NLin6Asym", 1): (-91.0, -126.0, -72.0),
    ("MNI152NLin6Asym", 2): (-90.0, -126.0, -72.0),
    ("MNI152NLin2009cAsym", 1): (-96.0, -132.0, -78.0),
    ("MNI152NLin2009cAsym", 2): (-96.5, -132.5, -78.5),
}


def _world_bbox(affine: np.ndarray, shape: tuple[int, int, int]):
    """Return (min, max) world coordinates over the voxel-grid corners."""
    corners = [
        affine @ np.array([i, j, k, 1.0])
        for i in (0, shape[0] - 1)
        for j in (0, shape[1] - 1)
        for k in (0, shape[2] - 1)
    ]
    c = np.array(corners)[:, :3]
    return c.min(axis=0), c.max(axis=0)


@pytest.mark.parametrize(("key", "origin"), list(CANONICAL_ORIGINS.items()))
def test_reference_affine_origin_matches_templateflow(key, origin):
    """REFERENCE_AFFINES origins must match TemplateFlow's canonical grids.

    Guards specifically against the 2mm-reuses-1mm-origin regression.
    """
    affine = REFERENCE_AFFINES[key]
    assert tuple(affine[:3, 3]) == pytest.approx(
        origin, abs=1e-6
    ), f"{key} origin {tuple(affine[:3, 3])} != canonical {origin}"


def test_2mm_origin_is_not_the_1mm_origin():
    """Explicit guard: the historical bug set 2mm origin == 1mm origin."""
    for space in ("MNI152NLin6Asym", "MNI152NLin2009cAsym"):
        o1 = REFERENCE_AFFINES[(space, 1)][:3, 3]
        o2 = REFERENCE_AFFINES[(space, 2)][:3, 3]
        assert not np.allclose(
            o1, o2
        ), f"{space}: 2mm origin equals 1mm origin — the resolution-origin bug is back"


def test_reference_affine_matches_bundled_atlas_1mm():
    """The 1mm 6Asym reference grid must match the bundled atlas grid exactly.

    Orientation-independent (compares world bounding boxes), network-free.
    """
    matches = glob.glob("src/lacuna/data/atlases/*MNI152NLin6Asym_res-01*Parcels*.nii.gz")
    if not matches:
        pytest.skip("no bundled 6Asym 1mm atlas found")
    atlas = nib.load(sorted(matches)[0])

    key = ("MNI152NLin6Asym", 1)
    ref_min, ref_max = _world_bbox(REFERENCE_AFFINES[key], REFERENCE_SHAPES[key])
    atlas_min, atlas_max = _world_bbox(atlas.affine, atlas.shape[:3])

    assert atlas.shape[:3] == REFERENCE_SHAPES[key]
    np.testing.assert_allclose(atlas_min, ref_min, atol=1e-3)
    np.testing.assert_allclose(atlas_max, ref_max, atol=1e-3)


def test_reference_shapes_present_for_all_affines():
    """Every reference affine must have a matching reference shape."""
    for key in REFERENCE_AFFINES:
        assert key in REFERENCE_SHAPES, f"missing REFERENCE_SHAPES entry for {key}"
