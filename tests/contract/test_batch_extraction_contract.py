"""Contract tests for batch result extraction utilities.

These tests define the expected behavior for the unified extract() function.

User Story 3: Provide extraction utilities for batch processing results.

Contract: T030 - Batch Result Extraction
"""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from lacuna.core.data_types import ParcelData
from lacuna.core.keys import build_result_key
from lacuna.core.mask_data import MaskData


@pytest.fixture
def batch_results_with_parcel_data():
    """Create batch results dict with multiple subjects."""
    results: dict[MaskData, dict[str, ParcelData]] = {}

    for i in range(3):
        # Create a new MaskData for each subject
        shape = (10, 10, 10)
        affine = np.eye(4) * 2
        affine[3, 3] = 1

        data = np.zeros(shape, dtype=np.int8)
        data[4:6, 4:6, 4:6] = 1
        img = nib.Nifti1Image(data, affine)

        mask_data = MaskData(
            mask_img=img,
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": f"sub-{i:03d}"},
        )

        # Add ParcelData result with BIDS-style key
        parcel_data = ParcelData(
            name="parcel_means",
            data={"region_A": 0.5 + i * 0.1, "region_B": 0.3 + i * 0.1, "region_C": 0.8 - i * 0.1},
            parcel_names=["AAL116"],
        )
        # build_result_key(parc, source, desc)
        parcel_key = build_result_key("AAL116", "ParcelAggregation", "parcel_means")

        # Create mask_data with results, then add to batch
        mask_data = mask_data.add_result("ParcelAggregation", {parcel_key: parcel_data})
        results[mask_data] = mask_data.results

    return results


class TestExtractContract:
    """Contract tests for the unified extract() function."""

    def test_extract_returns_dict(self, batch_results_with_parcel_data):
        """extract() should return a dict mapping identifiers to results."""
        from lacuna.batch.extract import extract

        extracted = extract(batch_results_with_parcel_data, parc="AAL116")

        assert isinstance(extracted, dict)
        assert len(extracted) == 3

    def test_extract_uses_subject_id(self, batch_results_with_parcel_data):
        """extract() should use subject_id as key."""
        from lacuna.batch.extract import extract

        extracted = extract(batch_results_with_parcel_data, parc="AAL116")

        # Should use subject_id from metadata
        assert "sub-000" in extracted
        assert "sub-001" in extracted
        assert "sub-002" in extracted

    def test_extract_filters_by_parc(self, batch_results_with_parcel_data):
        """extract() should filter results by parcellation."""
        from lacuna.batch.extract import extract

        extracted = extract(batch_results_with_parcel_data, parc="AAL116")

        # All results should have AAL116 in the key
        for subject_results in extracted.values():
            for key in subject_results:
                assert "parc-AAL116" in key

    def test_extract_filters_by_source(self, batch_results_with_parcel_data):
        """extract() should filter results by source."""
        from lacuna.batch.extract import extract

        extracted = extract(batch_results_with_parcel_data, source="ParcelAggregation")

        # All results should have ParcelAggregation in the key
        for subject_results in extracted.values():
            for key in subject_results:
                assert "source-ParcelAggregation" in key

    def test_extract_as_dataframe(self, batch_results_with_parcel_data):
        """extract() with as_dataframe=True should return DataFrame."""
        from lacuna.batch.extract import extract

        df = extract(batch_results_with_parcel_data, parc="AAL116", as_dataframe=True)

        assert isinstance(df, pd.DataFrame)
        assert "subject" in df.columns
        assert len(df) == 3

    def test_extract_unwrap_calls_get_data(self, batch_results_with_parcel_data):
        """extract() with unwrap=True should call get_data() on results."""
        from lacuna.batch.extract import extract

        extracted = extract(batch_results_with_parcel_data, parc="AAL116", unwrap=True)

        # Values should be raw data (dicts), not ParcelData objects
        for subject_results in extracted.values():
            for value in subject_results.values():
                assert isinstance(value, dict)
                assert "region_A" in value

    def test_extract_without_unwrap_returns_wrapper(self, batch_results_with_parcel_data):
        """extract() with unwrap=False should return wrapper objects."""
        from lacuna.batch.extract import extract

        extracted = extract(batch_results_with_parcel_data, parc="AAL116", unwrap=False)

        # Values should be ParcelData objects
        for subject_results in extracted.values():
            for value in subject_results.values():
                assert isinstance(value, ParcelData)


class TestExtractErrorHandling:
    """Contract tests for error handling in extract()."""

    def test_empty_results_raises_error(self):
        """extract() should raise ValueError for empty results."""
        from lacuna.batch.extract import extract

        with pytest.raises(ValueError, match="batch_results is empty"):
            extract({}, parc="AAL116")

    def test_no_matching_results_raises_error(self, batch_results_with_parcel_data):
        """extract() should raise ValueError when no results match filters."""
        from lacuna.batch.extract import extract

        with pytest.raises(ValueError, match="No results found matching filters"):
            extract(batch_results_with_parcel_data, parc="NonExistentAtlas")


class TestExtractModuleExports:
    """Contract tests for module exports."""

    def test_extract_is_exported(self):
        """extract should be exported from lacuna.batch.extract."""
        from lacuna.batch.extract import extract

        assert callable(extract)

    def test_extract_in_all(self):
        """extract should be in __all__."""
        import importlib

        extract_module = importlib.import_module("lacuna.batch.extract")
        assert "extract" in extract_module.__all__

    def test_legacy_functions_removed(self):
        """Legacy functions should not be exported."""
        import importlib

        extract_module = importlib.import_module("lacuna.batch.extract")
        assert not hasattr(extract_module, "extract_voxelmaps")
        assert not hasattr(extract_module, "extract_parcel_table")
        assert not hasattr(extract_module, "extract_scalars")
