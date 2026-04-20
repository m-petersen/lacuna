"""Tests for atlas scoring functions."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.atlas.scoring import (
    score_focal,
    score_functional_overlap,
    score_ace_temporal,
    score_structural_endpoints,
)
from lacuna.atlas.types import VoxelAtlas


@pytest.fixture
def simple_atlas():
    """Atlas with known values for predictable scoring."""
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    shape = (10, 10, 10)

    # 5HT1a: value 2.0 everywhere except zeros at edges
    data_5ht = np.full(shape, 2.0, dtype=np.float32)
    data_5ht[0, :, :] = 0.0

    # D1: value 1.0 everywhere, no zeros
    data_d1 = np.full(shape, 1.0, dtype=np.float32)

    maps = {
        "5HT1a": nib.Nifti1Image(data_5ht, affine),
        "D1": nib.Nifti1Image(data_d1, affine),
    }
    return VoxelAtlas(
        maps=maps, space="MNI152NLin6Asym", resolution=2.0, domain="neurotransmitter"
    )


class TestScoreFocal:
    def test_mean_score(self, simple_atlas):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True
        mask[5, 5, 6] = True
        scores = score_focal(simple_atlas, mask)
        assert scores["D1"] == pytest.approx(1.0)
        assert scores["5HT1a"] == pytest.approx(2.0)

    def test_zeros_excluded(self, simple_atlas):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[0, 5, 5] = True  # this is a zero voxel in 5HT1a
        mask[5, 5, 5] = True  # this is 2.0 in 5HT1a
        scores = score_focal(simple_atlas, mask)
        # Zero excluded, so mean of [2.0] = 2.0
        assert scores["5HT1a"] == pytest.approx(2.0)
        # D1 has no zeros, mean of [1.0, 1.0] = 1.0
        assert scores["D1"] == pytest.approx(1.0)

    def test_sum_aggregation(self, simple_atlas):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True
        mask[5, 5, 6] = True
        scores = score_focal(simple_atlas, mask, aggregation="sum")
        assert scores["D1"] == pytest.approx(2.0)
        assert scores["5HT1a"] == pytest.approx(4.0)

    def test_empty_mask_returns_nan(self, simple_atlas):
        mask = np.zeros((10, 10, 10), dtype=bool)
        scores = score_focal(simple_atlas, mask)
        assert np.isnan(scores["D1"])

    def test_parcel_mask_restricts_scoring(self, simple_atlas):
        lesion_mask = np.zeros((10, 10, 10), dtype=bool)
        lesion_mask[5, 5, 5] = True
        lesion_mask[5, 5, 6] = True
        parcel_mask = np.zeros((10, 10, 10), dtype=bool)
        parcel_mask[5, 5, 5] = True  # only one voxel overlaps
        scores = score_focal(simple_atlas, lesion_mask, parcel_mask=parcel_mask)
        assert scores["D1"] == pytest.approx(1.0)


class TestScoreStructuralEndpoints:
    def test_basic_scoring(self, simple_atlas):
        # 3 streamlines, each with 2 endpoints (start, end) in (x,y,z)
        endpoints_start = np.array([[5, 5, 5], [5, 5, 6], [5, 5, 7]], dtype=np.int32)
        endpoints_end = np.array([[5, 5, 8], [5, 5, 9], [5, 5, 5]], dtype=np.int32)
        intersecting_ids = np.array([0, 2])  # streamlines 0 and 2 intersect lesion

        scores, count = score_structural_endpoints(
            simple_atlas, endpoints_start, endpoints_end, intersecting_ids
        )
        assert count == 2
        # D1 is 1.0 everywhere -> each streamline mean = 1.0 -> sum of 2 = 2.0
        assert scores["D1"] == pytest.approx(2.0)

    def test_returns_streamline_count(self, simple_atlas):
        endpoints_start = np.array([[5, 5, 5]], dtype=np.int32)
        endpoints_end = np.array([[5, 5, 6]], dtype=np.int32)
        intersecting_ids = np.array([0])
        _, count = score_structural_endpoints(
            simple_atlas, endpoints_start, endpoints_end, intersecting_ids
        )
        assert count == 1

    def test_empty_intersecting_returns_zero(self, simple_atlas):
        endpoints_start = np.array([[5, 5, 5]], dtype=np.int32)
        endpoints_end = np.array([[5, 5, 6]], dtype=np.int32)
        intersecting_ids = np.array([], dtype=np.int32)
        scores, count = score_structural_endpoints(
            simple_atlas, endpoints_start, endpoints_end, intersecting_ids
        )
        assert count == 0
        assert scores["D1"] == 0.0


class TestScoreFunctionalOverlap:
    def test_positive_connectivity_only(self, simple_atlas):
        # z-map with positive and negative values
        z_data = np.zeros((10, 10, 10), dtype=np.float32)
        z_data[5, 5, 5] = 0.5  # positive
        z_data[5, 5, 6] = -0.3  # negative -- should be excluded
        z_map = nib.Nifti1Image(z_data, np.eye(4) * 2)

        scores = score_functional_overlap(simple_atlas, z_map)
        # Only positive voxel [5,5,5] contributes
        # D1 at [5,5,5] = 1.0, z = 0.5 -> weighted = 0.5 / 0.5 = 1.0 (normalized)
        assert scores["D1"] > 0
        assert scores["D1"] == pytest.approx(1.0)

    def test_all_negative_returns_nan(self, simple_atlas):
        z_data = np.full((10, 10, 10), -0.5, dtype=np.float32)
        z_map = nib.Nifti1Image(z_data, np.eye(4) * 2)
        scores = score_functional_overlap(simple_atlas, z_map)
        assert np.isnan(scores["D1"])


class TestScoreAceTemporal:
    def test_perfect_correlation(self):
        n_timepoints = 100
        nt_timeseries = {
            "D1": np.sin(np.linspace(0, 4 * np.pi, n_timepoints)),
            "5HT1a": np.cos(np.linspace(0, 4 * np.pi, n_timepoints)),
        }
        lesion_ts = nt_timeseries["D1"].copy()  # identical to D1
        scores = score_ace_temporal(nt_timeseries, lesion_ts)
        assert scores["D1"] == pytest.approx(1.0, abs=0.01)
        assert abs(scores["5HT1a"]) < 0.3  # sin/cos are ~uncorrelated

    def test_returns_all_targets(self):
        n_timepoints = 50
        nt_timeseries = {
            "D1": np.random.default_rng(42).standard_normal(n_timepoints),
            "5HT1a": np.random.default_rng(43).standard_normal(n_timepoints),
        }
        lesion_ts = np.random.default_rng(44).standard_normal(n_timepoints)
        scores = score_ace_temporal(nt_timeseries, lesion_ts)
        assert set(scores.keys()) == {"D1", "5HT1a"}
