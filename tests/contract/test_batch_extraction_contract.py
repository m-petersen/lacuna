"""Contract tests for batch result extraction utilities.

These tests define the expected behavior for extract_voxelmaps(),
extract_parcel_table(), and extract_scalars() utilities.

User Story 3: Provide extraction utilities for batch processing results.

Contract: T030 - Batch Result Extraction
"""

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from lacuna.core.data_types import VoxelMap, ParcelData, ScalarMetric
from lacuna.core.mask_data import MaskData


@pytest.fixture
def sample_mask_data_with_results():
    """Create MaskData with analysis results for extraction testing."""
    shape = (10, 10, 10)
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    
    # Create binary mask
    data = np.zeros(shape, dtype=np.int8)
    data[4:6, 4:6, 4:6] = 1
    img = nib.Nifti1Image(data, affine)
    
    mask_data = MaskData(
        mask_img=img,
        space="MNI152NLin6Asym",
        resolution=2.0,
        metadata={"subject_id": "sub-001"}
    )
    
    # Add VoxelMap result
    voxel_data = np.random.rand(*shape).astype(np.float32)
    voxel_img = nib.Nifti1Image(voxel_data, affine)
    voxel_map = VoxelMap(
        name="correlation_map",
        data=voxel_img,
        space="MNI152NLin6Asym",
        resolution=2.0
    )
    
    # Add ParcelData result
    parcel_data = ParcelData(
        name="parcel_means",
        data={"region_A": 0.5, "region_B": 0.3, "region_C": 0.8},
        parcel_names=["TestAtlas"]
    )
    
    # Add ScalarMetric result  
    scalar_metric = ScalarMetric(
        name="lesion_volume",
        data=125.5
    )
    
    # Simulate results structure
    results = {
        "TestAnalysis": {
            "correlation_map": voxel_map,
            "parcel_means": parcel_data,
            "lesion_volume": scalar_metric
        }
    }
    
    return mask_data.add_result("TestAnalysis", results["TestAnalysis"])


@pytest.fixture
def sample_parcel_data_list():
    """Create list of ParcelData for extraction testing."""
    parcel_list = []
    for i in range(3):
        pd_item = ParcelData(
            name=f"subject_{i:03d}_parcels",
            data={
                "region_A": 0.5 + i * 0.1,
                "region_B": 0.3 + i * 0.1,
                "region_C": 0.8 - i * 0.1
            },
            parcel_names=["TestAtlas"],
            metadata={"subject_id": f"sub-{i:03d}"}
        )
        parcel_list.append(pd_item)
    return parcel_list


@pytest.fixture
def batch_results_with_parcel_data(sample_mask_data_with_results):
    """Create batch results list with multiple subjects."""
    results = []
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
            metadata={"subject_id": f"sub-{i:03d}"}
        )
        
        # Add ParcelData result
        parcel_data = ParcelData(
            name="parcel_means",
            data={
                "region_A": 0.5 + i * 0.1,
                "region_B": 0.3 + i * 0.1,
                "region_C": 0.8 - i * 0.1
            },
            parcel_names=["TestAtlas"]
        )
        
        mask_data = mask_data.add_result(
            "ParcelAggregation",
            {"parcel_means": parcel_data}
        )
        results.append(mask_data)
    
    return results


class TestExtractVoxelmapsContract:
    """Contract tests for extract_voxelmaps function."""

    def test_extract_voxelmaps_returns_dict(self, sample_mask_data_with_results):
        """extract_voxelmaps should return a dict mapping identifiers to VoxelMaps."""
        from lacuna.batch.extract import extract_voxelmaps
        
        results = [sample_mask_data_with_results]
        voxelmaps = extract_voxelmaps(
            results,
            analysis="TestAnalysis",
            key="correlation_map"
        )
        
        assert isinstance(voxelmaps, dict)
        assert len(voxelmaps) == 1

    def test_extract_voxelmaps_uses_subject_id(self, sample_mask_data_with_results):
        """extract_voxelmaps should use subject_id as key."""
        from lacuna.batch.extract import extract_voxelmaps
        
        results = [sample_mask_data_with_results]
        voxelmaps = extract_voxelmaps(
            results,
            analysis="TestAnalysis",
            key="correlation_map"
        )
        
        # Should use subject_id from metadata
        assert "sub-001" in voxelmaps

    def test_extract_voxelmaps_values_are_voxelmaps(self, sample_mask_data_with_results):
        """extract_voxelmaps values should be VoxelMap instances."""
        from lacuna.batch.extract import extract_voxelmaps
        
        results = [sample_mask_data_with_results]
        voxelmaps = extract_voxelmaps(
            results,
            analysis="TestAnalysis",
            key="correlation_map"
        )
        
        for vm in voxelmaps.values():
            assert isinstance(vm, VoxelMap)


class TestExtractParcelTableContract:
    """Contract tests for extract_parcel_table function."""

    def test_extract_parcel_table_returns_dataframe(self, batch_results_with_parcel_data):
        """extract_parcel_table should return a pandas DataFrame."""
        from lacuna.batch.extract import extract_parcel_table
        
        df = extract_parcel_table(
            batch_results_with_parcel_data,
            analysis="ParcelAggregation",
            key="parcel_means"
        )
        
        assert isinstance(df, pd.DataFrame)

    def test_extract_parcel_table_has_subject_index(self, batch_results_with_parcel_data):
        """DataFrame should have subject identifiers as index."""
        from lacuna.batch.extract import extract_parcel_table
        
        df = extract_parcel_table(
            batch_results_with_parcel_data,
            analysis="ParcelAggregation",
            key="parcel_means"
        )
        
        # Index should contain subject IDs
        assert "sub-000" in df.index
        assert "sub-001" in df.index
        assert "sub-002" in df.index

    def test_extract_parcel_table_has_region_columns(self, batch_results_with_parcel_data):
        """DataFrame should have region names as columns."""
        from lacuna.batch.extract import extract_parcel_table
        
        df = extract_parcel_table(
            batch_results_with_parcel_data,
            analysis="ParcelAggregation",
            key="parcel_means"
        )
        
        assert "region_A" in df.columns
        assert "region_B" in df.columns
        assert "region_C" in df.columns

    def test_extract_parcel_table_preserves_values(self, batch_results_with_parcel_data):
        """DataFrame values should match input parcel data."""
        from lacuna.batch.extract import extract_parcel_table
        
        df = extract_parcel_table(
            batch_results_with_parcel_data,
            analysis="ParcelAggregation",
            key="parcel_means"
        )
        
        # Check first subject values
        assert df.loc["sub-000", "region_A"] == pytest.approx(0.5, rel=1e-5)
        assert df.loc["sub-001", "region_A"] == pytest.approx(0.6, rel=1e-5)

    def test_extract_parcel_table_accepts_parcel_data_list(self, sample_parcel_data_list):
        """extract_parcel_table should accept list[ParcelData] directly."""
        from lacuna.batch.extract import extract_parcel_table
        
        df = extract_parcel_table(sample_parcel_data_list)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3


class TestExtractScalarsContract:
    """Contract tests for extract_scalars function."""

    def test_extract_scalars_returns_series(self, sample_mask_data_with_results):
        """extract_scalars should return a pandas Series."""
        from lacuna.batch.extract import extract_scalars
        
        results = [sample_mask_data_with_results]
        scalars = extract_scalars(
            results,
            analysis="TestAnalysis",
            key="lesion_volume"
        )
        
        assert isinstance(scalars, pd.Series)

    def test_extract_scalars_has_subject_index(self, sample_mask_data_with_results):
        """Series should have subject identifiers as index."""
        from lacuna.batch.extract import extract_scalars
        
        results = [sample_mask_data_with_results]
        scalars = extract_scalars(
            results,
            analysis="TestAnalysis",
            key="lesion_volume"
        )
        
        assert "sub-001" in scalars.index

    def test_extract_scalars_preserves_values(self, sample_mask_data_with_results):
        """Series values should match input scalar data."""
        from lacuna.batch.extract import extract_scalars
        
        results = [sample_mask_data_with_results]
        scalars = extract_scalars(
            results,
            analysis="TestAnalysis",
            key="lesion_volume"
        )
        
        assert scalars["sub-001"] == pytest.approx(125.5, rel=1e-5)


class TestExtractionErrorHandling:
    """Contract tests for error handling in extraction utilities."""

    def test_missing_result_returns_nan(self, batch_results_with_parcel_data):
        """Missing results should return NaN, not raise errors."""
        from lacuna.batch.extract import extract_parcel_table
        
        # Try to extract non-existent key
        df = extract_parcel_table(
            batch_results_with_parcel_data,
            analysis="ParcelAggregation",
            key="nonexistent_key"
        )
        
        # Should return empty or NaN DataFrame, not raise
        assert isinstance(df, pd.DataFrame)

    def test_empty_list_returns_empty_result(self):
        """Empty input list should return empty result."""
        from lacuna.batch.extract import extract_parcel_table
        
        df = extract_parcel_table([])
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
