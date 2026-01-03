"""Unit tests for SubjectData.get_result() method.

Tests the glob pattern-based result key access.
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna.core.keys import build_result_key
from lacuna.core.subject_data import SubjectData


@pytest.fixture
def sample_mask_img():
    """Create a simple binary mask image for testing."""
    data = np.zeros((10, 10, 10), dtype=np.float32)
    data[3:7, 3:7, 3:7] = 1.0  # Binary mask
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    return nib.Nifti1Image(data, affine)


@pytest.fixture
def mask_data_with_results(sample_mask_img):
    """Create SubjectData with some pre-populated results."""
    mask = SubjectData(
        sample_mask_img,
        space="MNI152NLin6Asym",
        resolution=2.0,
    )

    # Add ParcelAggregation results with BIDS-style keys
    # Note: build_result_key("...", "SubjectData", "maskimg") returns
    # "atlas-..._source-InputMask" (no desc for InputMask)
    pa_results = {
        build_result_key("Schaefer100", "SubjectData", "maskimg"): {"parcels": [1, 2, 3]},
        build_result_key("Schaefer100", "FunctionalNetworkMapping", "rmap"): {"parcels": [4, 5, 6]},
        build_result_key("Tian_S4", "SubjectData", "maskimg"): {"parcels": [7, 8, 9]},
    }
    mask = mask.add_result("ParcelAggregation", pa_results)

    # Add FunctionalNetworkMapping results
    fnm_results = {
        "rmap": {"data": [1.0, 2.0]},
        "zscoremap": {"data": [3.0, 4.0]},
    }
    mask = mask.add_result("FunctionalNetworkMapping", fnm_results)

    return mask


class TestGetResultAnalysisLevel:
    """Tests for get_result() at analysis namespace level."""

    def test_get_all_results_for_analysis(self, mask_data_with_results):
        """Get all results for an analysis namespace."""
        results = mask_data_with_results.get_result("ParcelAggregation")
        assert len(results) == 3
        # build_result_key("Schaefer100", "SubjectData", ...) returns atlas-Schaefer100_source-InputMask
        assert build_result_key("Schaefer100", "SubjectData") in results

    def test_get_fnm_results(self, mask_data_with_results):
        """Get FunctionalNetworkMapping results."""
        results = mask_data_with_results.get_result("FunctionalNetworkMapping")
        assert "rmap" in results
        assert "zscoremap" in results

    def test_unknown_analysis_raises_keyerror(self, mask_data_with_results):
        """Unknown analysis namespace raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            mask_data_with_results.get_result("NonExistentAnalysis")

    def test_unknown_analysis_suggests_similar(self, mask_data_with_results):
        """Unknown analysis provides suggestions."""
        with pytest.raises(KeyError, match="Did you mean"):
            mask_data_with_results.get_result("ParcelAggragation")  # typo


class TestGetResultWithPattern:
    """Tests for get_result() with glob pattern filtering."""

    def test_get_specific_result_by_pattern(self, mask_data_with_results):
        """Get specific result using glob pattern."""
        result = mask_data_with_results.get_result(
            "ParcelAggregation",
            pattern="*Schaefer100*FunctionalNetworkMapping*rmap*",
        )
        assert result == {"parcels": [4, 5, 6]}

    def test_get_mask_source_result(self, mask_data_with_results):
        """Get result from SubjectData source using pattern."""
        result = mask_data_with_results.get_result(
            "ParcelAggregation",
            pattern="*Tian_S4*InputMask*",
        )
        assert result == {"parcels": [7, 8, 9]}

    def test_partial_filter_by_parc_pattern(self, mask_data_with_results):
        """Partial filtering by parc pattern returns matching results."""
        results = mask_data_with_results.get_result(
            "ParcelAggregation",
            pattern="*Schaefer100*",
        )
        # Should return dict of all Schaefer100 results (multiple matches)
        assert isinstance(results, dict)
        assert len(results) == 2

    def test_unknown_pattern_raises_keyerror(self, mask_data_with_results):
        """Unknown pattern raises KeyError."""
        with pytest.raises(KeyError, match="No results found"):
            mask_data_with_results.get_result(
                "ParcelAggregation",
                pattern="*NonExistent*",
            )

    def test_unknown_pattern_suggests_similar(self, mask_data_with_results):
        """Unknown pattern provides helpful error message."""
        with pytest.raises(KeyError, match="No results found"):
            mask_data_with_results.get_result(
                "ParcelAggregation",
                pattern="*correltion_map*",  # typo
            )


class TestGetResultEmptyResults:
    """Tests for get_result() with empty results."""

    def test_no_results_raises_keyerror(self, sample_mask_img):
        """Empty results raises KeyError."""
        mask = SubjectData(
            sample_mask_img,
            space="MNI152NLin6Asym",
            resolution=2.0,
        )
        with pytest.raises(KeyError, match="not found"):
            mask.get_result("ParcelAggregation")
