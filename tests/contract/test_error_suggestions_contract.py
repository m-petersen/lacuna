"""Contract tests for error message improvements with fuzzy matching."""

import nibabel as nib
import numpy as np
import pytest

from lacuna import SubjectData


class TestErrorMessageSuggestionsContract:
    """Contract tests for error messages with fuzzy suggestions."""

    def test_invalid_space_suggests_similar(self):
        """Contract: Invalid space error includes fuzzy suggestions."""
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)

        # Typo in space name
        with pytest.raises(ValueError) as exc_info:
            SubjectData(
                mask_img=img,
                space="MNI152Lin6Asym",  # Missing 'N'
                resolution=2,
            )

        error_msg = str(exc_info.value)
        # Should suggest the correct space
        assert "MNI152NLin6Asym" in error_msg
        assert "Did you mean" in error_msg or "MNI152NLin6Asym" in error_msg

    def test_invalid_space_lists_valid_options(self):
        """Contract: Invalid space error lists all valid options."""
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)

        with pytest.raises(ValueError) as exc_info:
            SubjectData(
                mask_img=img,
                space="invalid_space",
                resolution=2,
            )

        error_msg = str(exc_info.value)
        assert "MNI152NLin6Asym" in error_msg
        assert "MNI152NLin2009aAsym" in error_msg


class TestParcelAggregationErrorSuggestionsContract:
    """Contract tests for ParcelAggregation error messages."""

    def test_invalid_aggregation_suggests_similar(self):
        """Contract: Invalid aggregation method suggests similar options."""
        from lacuna.analysis import ParcelAggregation

        with pytest.raises(ValueError) as exc_info:
            ParcelAggregation(aggregation="average")  # Should be 'mean'

        error_msg = str(exc_info.value)
        assert "mean" in error_msg or "Valid options" in error_msg

    def test_invalid_source_suggests_similar(self):
        """Contract: Invalid source name suggests available sources."""
        from lacuna.analysis import ParcelAggregation

        # Create SubjectData with results
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0
        img = nib.Nifti1Image(data, affine)
        mask_data = SubjectData(
            mask_img=img,
            space="MNI152NLin6Asym",
            resolution=2,
        )

        # Try with invalid source name
        agg = ParcelAggregation(
            source=["SubjectData.correltion_map"],  # Typo
            parcel_names=["Schaefer100"],
        )

        with pytest.raises(ValueError) as exc_info:
            agg.run(mask_data)

        error_msg = str(exc_info.value)
        # Should mention available sources or suggest similar
        assert "not found" in error_msg.lower() or "available" in error_msg.lower()


class TestGetResultErrorSuggestionsContract:
    """Contract tests for SubjectData.get_result() error messages."""

    @pytest.fixture
    def mask_data_with_results(self):
        """Create SubjectData with some results for testing."""
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0
        img = nib.Nifti1Image(data, affine)
        mask_data = SubjectData(
            mask_img=img,
            space="MNI152NLin6Asym",
            resolution=2,
        )
        # Add some mock results
        mask_data._results["RegionalDamage"] = {"volume": 100}
        mask_data._results["FunctionalNetworkMapping"] = {"rmap": None}
        return mask_data

    def test_invalid_analysis_suggests_similar(self, mask_data_with_results):
        """Contract: Invalid analysis name suggests similar options."""
        with pytest.raises(KeyError) as exc_info:
            mask_data_with_results.get_result("RegionalDmage")  # Typo

        error_msg = str(exc_info.value)
        assert "RegionalDamage" in error_msg
