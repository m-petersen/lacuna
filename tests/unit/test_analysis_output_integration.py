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
    data = np.random.rand(91, 109, 91)
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
        analysis = AtlasAggregation(atlas_names=["atlas"])
        
        # Mock atlas loading
        with patch('lacuna.assets.atlases.load_atlas') as mock_load:
            mock_atlas_data = np.random.randint(0, 100, (91, 109, 91), dtype=np.int32)
            mock_affine = np.eye(4)
            mock_affine[0, 0] = mock_affine[1, 1] = mock_affine[2, 2] = 2.0
            mock_atlas_img = nib.Nifti1Image(mock_atlas_data, mock_affine)
            mock_load.return_value = (mock_atlas_img, {1: "region_1", 2: "region_2"}, Mock())
            
            results = analysis._run_analysis(sample_lesion_data)
            
            assert isinstance(results, list)
            assert len(results) > 0
            assert all(isinstance(r, AnalysisResult) for r in results)
    
    def test_atlas_aggregation_returns_roi_result(self, sample_lesion_data):
        """AtlasAggregation returns ROIResult with region data."""
        analysis = AtlasAggregation(atlas_names=["atlas"])
        
        with patch('lacuna.assets.atlases.load_atlas') as mock_load:
            mock_atlas_data = np.random.randint(0, 100, (91, 109, 91), dtype=np.int32)
            mock_affine = np.eye(4)
            mock_affine[0, 0] = mock_affine[1, 1] = mock_affine[2, 2] = 2.0
            mock_atlas_img = nib.Nifti1Image(mock_atlas_data, mock_affine)
            mock_load.return_value = (mock_atlas_img, {1: "region_1", 2: "region_2"}, Mock())
            
            results = analysis._run_analysis(sample_lesion_data)
            
            # Should contain at least one ROIResult
            roi_results = [r for r in results if isinstance(r, ROIResult)]
            assert len(roi_results) > 0
            
            # Check ROIResult structure
            roi_result = roi_results[0]
            assert isinstance(roi_result.data, dict)
            assert roi_result.name is not None
            assert roi_result.atlas_names is not None
            assert roi_result.aggregation_method is not None
    
    def test_atlas_aggregation_roi_result_has_metadata(self, sample_lesion_data):
        """ROIResult from AtlasAggregation includes analysis metadata."""
        analysis = AtlasAggregation(
            atlas_names=["atlas"],
            aggregation="percent"
        )
        
        with patch('lacuna.assets.atlases.load_atlas') as mock_load:
            mock_atlas_data = np.random.randint(0, 100, (91, 109, 91), dtype=np.int32)
            mock_affine = np.eye(4)
            mock_affine[0, 0] = mock_affine[1, 1] = mock_affine[2, 2] = 2.0
            mock_atlas_img = nib.Nifti1Image(mock_atlas_data, mock_affine)
            mock_load.return_value = (mock_atlas_img, {1: "region_1", 2: "region_2"}, Mock())
            
            results = analysis._run_analysis(sample_lesion_data)
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
        
        # Mock mask indices (voxel coordinates)
        x_coords = np.random.randint(0, 91, n_voxels)
        y_coords = np.random.randint(0, 109, n_voxels)
        z_coords = np.random.randint(0, 91, n_voxels)
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


# ============================================================================
# StructuralNetworkMapping Tests
# ============================================================================

@pytest.mark.skip(reason="Requires complex MRtrix mocking - tested in integration tests")
class TestStructuralNetworkMappingOutputs:
    """Test StructuralNetworkMapping returns correct output types."""
    
    @pytest.fixture
    def mock_tractogram(self, tmp_path):
        """Create mock tractogram file."""
        tractogram_path = tmp_path / "tractogram.tck"
        tractogram_path.write_text("mock tractogram data")
        return tractogram_path
    
    @pytest.fixture
    def mock_tdi(self, tmp_path):
        """Create mock TDI file."""
        tdi_data = np.random.rand(91, 109, 91)
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0
        tdi_img = nib.Nifti1Image(tdi_data, affine)
        
        tdi_path = tmp_path / "tdi.nii.gz"
        nib.save(tdi_img, tdi_path)
        return tdi_path
    
    def test_run_analysis_returns_list_of_results(
        self, sample_lesion_data, mock_tractogram, mock_tdi
    ):
        """StructuralNetworkMapping._run_analysis returns list[AnalysisResult]."""
        analysis = StructuralNetworkMapping(
            tractogram_path=mock_tractogram,
            check_dependencies=False  # Don't check for MRtrix in tests
        )
        # Set whole_brain_tdi directly as it would be after validation
        analysis.whole_brain_tdi = mock_tdi
        
        # Mock the MRtrix functions that would be called
        with patch('lacuna.analysis.structural_network_mapping.filter_tractogram_by_lesion') as mock_filter:
            with patch('lacuna.analysis.structural_network_mapping.compute_tdi_map') as mock_tdi_compute:
                with patch('lacuna.analysis.structural_network_mapping.compute_disconnection_map') as mock_disconn:
                    with patch('lacuna.analysis.structural_network_mapping.nib.load') as mock_nib_load:
                        # Setup mock NIfTI images
                        mock_img = MagicMock()
                        mock_img.get_fdata.return_value = np.random.rand(91, 109, 91)
                        mock_img.affine = np.eye(4)
                        mock_img.header = MagicMock()
                        mock_nib_load.return_value = mock_img
                        
                        results = analysis._run_analysis(sample_lesion_data)
                        
                        assert isinstance(results, list)
                        assert len(results) > 0
                        assert all(isinstance(r, AnalysisResult) for r in results)
    
    def test_structural_mapping_returns_voxel_map_result(
        self, sample_lesion_data, mock_tractogram, mock_tdi
    ):
        """StructuralNetworkMapping returns VoxelMapResult for disconnection map."""
        analysis = StructuralNetworkMapping(
            tractogram_path=mock_tractogram,
            check_dependencies=False
        )
        analysis.whole_brain_tdi = mock_tdi
        
        with patch('lacuna.analysis.structural_network_mapping.filter_tractogram_by_lesion'):
            with patch('lacuna.analysis.structural_network_mapping.compute_tdi_map'):
                with patch('lacuna.analysis.structural_network_mapping.compute_disconnection_map'):
                    with patch('lacuna.analysis.structural_network_mapping.nib.load') as mock_nib_load:
                        mock_img = MagicMock()
                        mock_img.get_fdata.return_value = np.random.rand(91, 109, 91)
                        mock_img.affine = np.eye(4)
                        mock_img.header = MagicMock()
                        mock_nib_load.return_value = mock_img
                        
                        results = analysis._run_analysis(sample_lesion_data)
                        
                        # Should contain VoxelMapResult for disconnection_map
                        voxel_results = [r for r in results if isinstance(r, VoxelMapResult)]
                        assert len(voxel_results) > 0
                        
                        result_names = [r.name for r in voxel_results]
                        assert "disconnection_map" in result_names
    
    def test_structural_mapping_returns_connectivity_matrices(
        self, sample_lesion_data, mock_tractogram, mock_tdi, mock_atlas_file
    ):
        """StructuralNetworkMapping returns ConnectivityMatrixResult when atlas provided."""
        analysis = StructuralNetworkMapping(
            tractogram_path=mock_tractogram,
            atlas_name="test_atlas",
            check_dependencies=False
        )
        analysis.whole_brain_tdi = mock_tdi
        # Mock that atlas was resolved
        analysis._atlas_resolved = MagicMock()
        analysis._atlas_resolved.labels = [f"region_{i}" for i in range(100)]
        
        with patch('lacuna.analysis.structural_network_mapping.filter_tractogram_by_lesion'):
            with patch('lacuna.analysis.structural_network_mapping.compute_tdi_map'):
                with patch('lacuna.analysis.structural_network_mapping.compute_disconnection_map'):
                    with patch('lacuna.analysis.structural_network_mapping.nib.load') as mock_nib_load:
                        mock_img = MagicMock()
                        mock_img.get_fdata.return_value = np.random.rand(91, 109, 91)
                        mock_img.affine = np.eye(4)
                        mock_img.header = MagicMock()
                        mock_nib_load.return_value = mock_img
                        
                        with patch.object(analysis, '_compute_connectivity_matrices') as mock_compute:
                            # Mock connectivity matrix computation
                            # Now _compute_connectivity_matrices returns list[AnalysisResult]
                            intact = np.random.rand(100, 100)
                            labels = [f"region_{i}" for i in range(100)]
                            
                            mock_results = [
                                ConnectivityMatrixResult(
                                    name="lesion_connectivity_matrix",
                                    data=intact,
                                    row_labels=labels,
                                    column_labels=labels,
                                )
                            ]
                            mock_compute.return_value = mock_results
                            
                            results = analysis._run_analysis(sample_lesion_data)
                            
                            # Should contain ConnectivityMatrixResult
                            matrix_results = [r for r in results if isinstance(r, ConnectivityMatrixResult)]
                            assert len(matrix_results) > 0
                            
                            matrix_result = matrix_results[0]
                assert matrix_result.data is not None
                assert matrix_result.row_labels is not None


# ============================================================================
# Base Analysis Integration Tests
# ============================================================================

@pytest.mark.skip(reason="Requires full analysis pipeline - tested in integration tests")
class TestBaseAnalysisIntegration:
    """Test base analysis properly handles result objects."""
    
    def test_run_method_accepts_result_list(self, sample_lesion_data, mock_atlas_file):
        """BaseAnalysis.run() properly handles list[AnalysisResult] from _run_analysis."""
        # AtlasAggregation doesn't need any special setup - it discovers atlases from registry
        analysis = AtlasAggregation()
        
        # Mock the atlas discovery to use our test atlas
        with patch('lacuna.assets.atlases.list_atlases') as mock_list:
            mock_list.return_value = ["test_atlas"]
            with patch('lacuna.assets.atlases.load_atlas') as mock_load:
                mock_load.return_value = nib.load(mock_atlas_file)
            
            # run() should complete without errors
            result_data = analysis.run(sample_lesion_data)
            
            assert isinstance(result_data, LesionData)
            assert "AtlasAggregation" in result_data.results
    
    def test_results_stored_as_list_in_lesion_data(self, sample_lesion_data, mock_atlas_file):
        """Results are stored as list[AnalysisResult] in LesionData."""
        analysis = AtlasAggregation()
        
        with patch('lacuna.assets.atlases.list_atlases') as mock_list:
            mock_list.return_value = ["test_atlas"]
            with patch('lacuna.assets.atlases.load_atlas') as mock_load:
                mock_load.return_value = nib.load(mock_atlas_file)
            
            result_data = analysis.run(sample_lesion_data)
            
            results = result_data.results["AtlasAggregation"]
            assert isinstance(results, list)
            assert all(isinstance(r, AnalysisResult) for r in results)
    
    def test_multiple_analyses_chain_correctly(self, sample_lesion_data, mock_atlas_file):
        """Multiple analyses can be chained and results accumulate."""
        atlas_analysis = AtlasAggregation()
        
        with patch('lacuna.assets.atlases.list_atlases') as mock_list:
            mock_list.return_value = ["test_atlas"]
            with patch('lacuna.assets.atlases.load_atlas') as mock_load:
                mock_load.return_value = nib.load(mock_atlas_file)
            
            # First analysis
            result_data = atlas_analysis.run(sample_lesion_data)
            assert "AtlasAggregation" in result_data.results
            
            # Can run second analysis on result
            # (would need another analysis instance for real chaining)
            assert isinstance(result_data, LesionData)


# ============================================================================
# Backward Compatibility Tests
# ============================================================================

@pytest.mark.skip(reason="Requires full analysis pipeline - tested in integration tests")
class TestBackwardCompatibility:
    """Test that new output format maintains expected behavior."""
    
    def test_can_iterate_over_results(self, sample_lesion_data, mock_atlas_file):
        """Results can be iterated to access individual result objects."""
        analysis = AtlasAggregation()
        
        with patch('lacuna.assets.atlases.list_atlases') as mock_list:
            mock_list.return_value = ["test_atlas"]
            with patch('lacuna.assets.atlases.load_atlas') as mock_load:
                mock_load.return_value = nib.load(mock_atlas_file)
            
            result_data = analysis.run(sample_lesion_data)
            results = result_data.results["AtlasAggregation"]
            
            # Can iterate
            for result in results:
                assert isinstance(result, AnalysisResult)
                assert hasattr(result, 'get_data')
                assert hasattr(result, 'summary')
    
    def test_can_filter_results_by_type(self, sample_lesion_data, mock_atlas_file):
        """Results can be filtered by type for processing."""
        analysis = AtlasAggregation()
        
        with patch('lacuna.assets.atlases.list_atlases') as mock_list:
            mock_list.return_value = ["test_atlas"]
            with patch('lacuna.assets.atlases.load_atlas') as mock_load:
                mock_load.return_value = nib.load(mock_atlas_file)
            
            result_data = analysis.run(sample_lesion_data)
            results = result_data.results["AtlasAggregation"]
            
            # Filter by type
            roi_results = [r for r in results if isinstance(r, ROIResult)]
            assert len(roi_results) > 0
