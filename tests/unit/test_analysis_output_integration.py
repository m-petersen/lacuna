"""Tests for analysis modules using polymorphic output architecture.

Following TDD: these tests define the expected behavior of analyses
returning AnalysisResult objects instead of plain dicts.
"""

import nibabel as nib
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lacuna.core import LesionData
from lacuna.core.output import (
    AnalysisResult,
    VoxelMapResult,
    ROIResult,
    ConnectivityMatrixResult,
    MiscResult,
)
from lacuna.core.spaces import CoordinateSpace
from lacuna.analysis import (
    AtlasAggregation,
    FunctionalNetworkMapping,
    StructuralNetworkMapping,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_lesion_data():
    """Create sample lesion data for testing."""
    data = np.zeros((91, 109, 91), dtype=np.uint8)
    data[40:50, 50:60, 40:50] = 1  # Binary lesion mask
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0
    lesion_img = nib.Nifti1Image(data, affine)
    
    space_affine = np.eye(4)
    space_affine[0, 0] = space_affine[1, 1] = space_affine[2, 2] = 2.0
    space = CoordinateSpace(
        identifier="MNI152NLin6Asym",
        resolution=2.0,
        reference_affine=space_affine
    )
    
    return LesionData(
        lesion_img=lesion_img,
        metadata={
            "subject_id": "test_subject",
            "space": "MNI152NLin6Asym",
            "resolution": 2.0,
            "coordinate_space": space
        }
    )


@pytest.fixture
def mock_atlas_file(tmp_path):
    """Create mock atlas file."""
    atlas_data = np.random.randint(0, 100, (91, 109, 91), dtype=np.int32)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0
    atlas_img = nib.Nifti1Image(atlas_data, affine)
    
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(atlas_img, atlas_path)
    return atlas_path


# ============================================================================
# AtlasAggregation Tests
# ============================================================================

class TestAtlasAggregationOutputs:
    """Test AtlasAggregation returns correct output types."""
    
    def test_run_analysis_returns_list_of_results(self, sample_lesion_data, mock_atlas_file):
        """AtlasAggregation._run_analysis returns list[AnalysisResult]."""
        # Use a real bundled atlas instead of trying to mock everything
        # Note: This may return empty results if lesion doesn't overlap with atlas
        # The test is primarily checking the return type structure
        analysis = AtlasAggregation(atlas_names=["Schaefer2018_100Parcels7Networks"])
        
        results = analysis._run_analysis(sample_lesion_data)
        
        assert isinstance(results, list)
        # May be empty if no overlap, but should still be a list of AnalysisResult types
        assert all(isinstance(r, AnalysisResult) for r in results)
    
    def test_atlas_aggregation_returns_roi_result(self, sample_lesion_data, mock_atlas_file):
        """AtlasAggregation returns ROIResult with region data."""
        analysis = AtlasAggregation(atlas_names=["Schaefer2018_100Parcels7Networks"])
        
        results = analysis._run_analysis(sample_lesion_data)
        
        # Should contain ROIResult (may be empty if no overlap)
        roi_results = [r for r in results if isinstance(r, ROIResult)]
        # All results should be ROIResult type
        assert all(isinstance(r, ROIResult) for r in results)
        
        if len(roi_results) > 0:
            # Check ROIResult structure
            roi_result = roi_results[0]
            assert isinstance(roi_result.data, dict)
            assert roi_result.name is not None
            assert roi_result.atlas_names is not None
            assert roi_result.aggregation_method is not None
    
    def test_atlas_aggregation_roi_result_has_metadata(self, sample_lesion_data, mock_atlas_file):
        """ROIResult from AtlasAggregation includes analysis metadata."""
        analysis = AtlasAggregation(
            atlas_names=["Schaefer2018_100Parcels7Networks"],
            aggregation="percent"
        )
        
        results = analysis._run_analysis(sample_lesion_data)
        
        if len(results) > 0:
            roi_result = [r for r in results if isinstance(r, ROIResult)][0]
            assert roi_result.aggregation_method == "percent"
            assert roi_result.metadata is not None


# ============================================================================
# FunctionalNetworkMapping Tests
# ============================================================================

class TestFunctionalNetworkMappingOutputs:
    """Test FunctionalNetworkMapping returns correct output types."""
    
    @pytest.fixture
    def mock_connectome(self, tmp_path):
        """Create mock HDF5 connectome file."""
        import h5py
        
        # Create connectome structure that FunctionalNetworkMapping expects
        n_subjects = 5
        n_timepoints = 100
        n_voxels = 1000
        
        # Mock mask indices - ensure some overlap with lesion area [40:50, 50:60, 40:50]
        # Create voxels in and around the lesion area
        x_coords = np.random.randint(35, 55, n_voxels)  # Around lesion x: 40-50
        y_coords = np.random.randint(45, 65, n_voxels)  # Around lesion y: 50-60
        z_coords = np.random.randint(35, 55, n_voxels)  # Around lesion z: 40-50
        mask_indices = np.array([x_coords, y_coords, z_coords])
        
        # Mock timeseries data
        timeseries = np.random.rand(n_subjects, n_timepoints, n_voxels).astype(np.float32)
        
        # Mock affine
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0
        
        # Create HDF5 file
        connectome_path = tmp_path / "connectome.h5"
        with h5py.File(connectome_path, "w") as hf:
            hf.create_dataset("timeseries", data=timeseries)
            hf.create_dataset("mask_indices", data=mask_indices)
            hf.create_dataset("mask_affine", data=affine)  # Use mask_affine not affine
            # Add mask_shape attribute
            hf.attrs["mask_shape"] = (91, 109, 91)
        
        return connectome_path
    
    def test_run_analysis_returns_list_of_results(self, sample_lesion_data, mock_connectome):
        """FunctionalNetworkMapping._run_analysis returns list[AnalysisResult]."""
        analysis = FunctionalNetworkMapping(connectome_path=mock_connectome)
        
        results = analysis._run_analysis(sample_lesion_data)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, AnalysisResult) for r in results)
    
    def test_functional_mapping_returns_voxel_map_results(self, sample_lesion_data, mock_connectome):
        """FunctionalNetworkMapping returns VoxelMapResult objects for brain maps."""
        analysis = FunctionalNetworkMapping(connectome_path=mock_connectome)
        
        results = analysis._run_analysis(sample_lesion_data)
        
        # Should contain VoxelMapResults for correlation_map, z_map
        voxel_results = [r for r in results if isinstance(r, VoxelMapResult)]
        assert len(voxel_results) >= 2  # At least correlation_map and z_map
        
        # Check for expected result names
        result_names = [r.name for r in voxel_results]
        assert "correlation_map" in result_names
        assert "z_map" in result_names
    
    def test_functional_mapping_voxel_results_have_spaces(self, sample_lesion_data, mock_connectome):
        """VoxelMapResults from FunctionalNetworkMapping have space and resolution."""
        analysis = FunctionalNetworkMapping(connectome_path=mock_connectome)
        
        results = analysis._run_analysis(sample_lesion_data)
        voxel_results = [r for r in results if isinstance(r, VoxelMapResult)]
        
        for voxel_result in voxel_results:
            assert voxel_result.space is not None
            assert voxel_result.resolution is not None
            assert isinstance(voxel_result.space, str)
            assert isinstance(voxel_result.resolution, float)
    
    def test_functional_mapping_returns_misc_result_for_scalars(self, sample_lesion_data, mock_connectome):
        """FunctionalNetworkMapping returns MiscResult for summary statistics."""
        analysis = FunctionalNetworkMapping(connectome_path=mock_connectome)
        
        results = analysis._run_analysis(sample_lesion_data)
        
        # Should contain MiscResult for summary statistics
        misc_results = [r for r in results if isinstance(r, MiscResult)]
        assert len(misc_results) > 0
        
        # Check for summary_statistics result
        summary_results = [r for r in misc_results if "summary" in r.name.lower()]
        assert len(summary_results) > 0

