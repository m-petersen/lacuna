"""
Unit tests for StructuralNetworkMapping resolution control and TDI caching.

Tests the new features:
- User-selectable output resolution (1mm or 2mm)
- Automatic TDI computation from tractogram
- TDI caching for batch processing efficiency
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture
def dummy_tractogram(tmp_path):
    """Create a dummy tractogram file."""
    tck_path = tmp_path / "tractogram.tck"
    tck_path.touch()
    return tck_path


@pytest.fixture
def synthetic_lesion_data():
    """Create synthetic lesion data in MNI152NLin6Asym space."""
    from lacuna import LesionData
    
    lesion_array = np.zeros((91, 109, 91))
    lesion_array[45:50, 54:59, 45:50] = 1
    lesion_img = nib.Nifti1Image(lesion_array, np.eye(4))
    
    return LesionData(
        lesion_img=lesion_img,
        metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )


class TestOutputResolutionParameter:
    """Test user-selectable output resolution."""
    
    def test_accepts_output_resolution_parameter(self, dummy_tractogram):
        """Test that output_resolution parameter is accepted."""
        from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
        
        # Should accept 1mm
        analysis_1mm = StructuralNetworkMapping(
            tractogram_path=dummy_tractogram,
            output_resolution=1,
            check_dependencies=False,
        )
        assert analysis_1mm.output_resolution == 1
        
        # Should accept 2mm
        analysis_2mm = StructuralNetworkMapping(
            tractogram_path=dummy_tractogram,
            output_resolution=2,
            check_dependencies=False,
        )
        assert analysis_2mm.output_resolution == 2
    
    def test_output_resolution_defaults_to_2mm(self, dummy_tractogram):
        """Test that output_resolution defaults to 2mm."""
        from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
        
        analysis = StructuralNetworkMapping(
            tractogram_path=dummy_tractogram,
            check_dependencies=False,
        )
        assert analysis.output_resolution == 2
    
    def test_output_resolution_must_be_1_or_2(self, dummy_tractogram):
        """Test that output_resolution must be 1 or 2."""
        from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
        
        with pytest.raises(ValueError, match="output_resolution must be 1 or 2"):
            StructuralNetworkMapping(
                tractogram_path=dummy_tractogram,
                output_resolution=3,
                check_dependencies=False,
            )
    
    @patch("lacuna.analysis.structural_network_mapping.DataAssetManager")
    def test_loads_template_matching_output_resolution(self, mock_asset_mgr, dummy_tractogram, tmp_path):
        """Test that template is loaded matching the output_resolution."""
        from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
        
        # Create a dummy template
        template_1mm = tmp_path / "template_1mm.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((182, 218, 182)), np.eye(4)), template_1mm)
        
        mock_asset_mgr.return_value.get_template.return_value = template_1mm
        
        analysis = StructuralNetworkMapping(
            tractogram_path=dummy_tractogram,
            output_resolution=1,
            check_dependencies=False,
        )
        
        # Template should be loaded during validation with resolution=1
        # We'll test this in integration tests
        assert analysis.output_resolution == 1


class TestAutomaticTDIComputation:
    """Test automatic TDI computation from tractogram."""
    
    def test_no_longer_requires_whole_brain_tdi_parameter(self, dummy_tractogram):
        """Test that whole_brain_tdi parameter is no longer required."""
        from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
        
        # Should work without whole_brain_tdi
        analysis = StructuralNetworkMapping(
            tractogram_path=dummy_tractogram,
            output_resolution=2,
            check_dependencies=False,
        )
        assert analysis is not None
    
    def test_whole_brain_tdi_parameter_deprecated(self, dummy_tractogram, tmp_path):
        """Test that whole_brain_tdi parameter raises deprecation warning."""
        from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
        
        dummy_tdi = tmp_path / "tdi.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((91, 109, 91)), np.eye(4)), dummy_tdi)
        
        with pytest.warns(DeprecationWarning, match="whole_brain_tdi parameter is deprecated"):
            analysis = StructuralNetworkMapping(
                tractogram_path=dummy_tractogram,
                whole_brain_tdi=dummy_tdi,
                output_resolution=2,
                check_dependencies=False,
            )
    
    @patch("lacuna.analysis.structural_network_mapping.compute_tdi_map")
    def test_computes_tdi_from_tractogram(self, mock_compute_tdi, dummy_tractogram, tmp_path, synthetic_lesion_data):
        """Test that TDI is computed from tractogram during run()."""
        from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
        
        # Mock TDI computation to return a dummy TDI
        tdi_path = tmp_path / "computed_tdi.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((91, 109, 91)), np.eye(4)), tdi_path)
        mock_compute_tdi.return_value = tdi_path
        
        analysis = StructuralNetworkMapping(
            tractogram_path=dummy_tractogram,
            output_resolution=2,
            check_dependencies=False,
        )
        
        # Mock other dependencies
        with patch.object(analysis, '_validate_inputs'), \
             patch.object(analysis, '_compute_disconnection'), \
             patch("lacuna.analysis.structural_network_mapping.DataAssetManager"):
            
            result = analysis.run(synthetic_lesion_data)
            
            # Should have called compute_tdi_map with correct resolution
            mock_compute_tdi.assert_called_once()
            call_kwargs = mock_compute_tdi.call_args[1]
            assert call_kwargs['template_resolution'] == 2


class TestTDICaching:
    """Test TDI caching mechanism for batch processing."""
    
    def test_tdi_cached_by_default(self, dummy_tractogram):
        """Test that TDI caching is enabled by default."""
        from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
        
        analysis = StructuralNetworkMapping(
            tractogram_path=dummy_tractogram,
            output_resolution=2,
            check_dependencies=False,
        )
        assert analysis.cache_tdi is True
    
    def test_can_disable_tdi_caching(self, dummy_tractogram):
        """Test that TDI caching can be disabled."""
        from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
        
        analysis = StructuralNetworkMapping(
            tractogram_path=dummy_tractogram,
            output_resolution=2,
            cache_tdi=False,
            check_dependencies=False,
        )
        assert analysis.cache_tdi is False
    
    @patch("lacuna.analysis.structural_network_mapping.compute_tdi_map")
    def test_tdi_computed_once_and_reused(self, mock_compute_tdi, dummy_tractogram, tmp_path, synthetic_lesion_data):
        """Test that TDI is computed once and reused for multiple subjects."""
        from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
        
        # Mock TDI computation
        tdi_path = tmp_path / "cached_tdi.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((91, 109, 91)), np.eye(4)), tdi_path)
        mock_compute_tdi.return_value = tdi_path
        
        analysis = StructuralNetworkMapping(
            tractogram_path=dummy_tractogram,
            output_resolution=2,
            cache_tdi=True,
            check_dependencies=False,
        )
        
        # Mock other dependencies
        with patch.object(analysis, '_validate_inputs'), \
             patch.object(analysis, '_compute_disconnection') as mock_disconnection, \
             patch("lacuna.analysis.structural_network_mapping.DataAssetManager"):
            
            # Process two subjects
            result1 = analysis.run(synthetic_lesion_data)
            result2 = analysis.run(synthetic_lesion_data)
            
            # TDI should only be computed once
            assert mock_compute_tdi.call_count == 1
    
    @patch("lacuna.analysis.structural_network_mapping.compute_tdi_map")
    def test_tdi_recomputed_when_caching_disabled(self, mock_compute_tdi, dummy_tractogram, tmp_path, synthetic_lesion_data):
        """Test that TDI is recomputed each time when caching is disabled."""
        from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
        
        # Mock TDI computation
        tdi_path = tmp_path / "uncached_tdi.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((91, 109, 91)), np.eye(4)), tdi_path)
        mock_compute_tdi.return_value = tdi_path
        
        analysis = StructuralNetworkMapping(
            tractogram_path=dummy_tractogram,
            output_resolution=2,
            cache_tdi=False,
            check_dependencies=False,
        )
        
        # Mock other dependencies
        with patch.object(analysis, '_validate_inputs'), \
             patch.object(analysis, '_compute_disconnection') as mock_disconnection, \
             patch("lacuna.analysis.structural_network_mapping.DataAssetManager"):
            
            # Process two subjects
            result1 = analysis.run(synthetic_lesion_data)
            result2 = analysis.run(synthetic_lesion_data)
            
            # TDI should be computed twice
            assert mock_compute_tdi.call_count == 2
    
    def test_cached_tdi_path_stored_in_temp_directory(self, dummy_tractogram, tmp_path):
        """Test that cached TDI is stored with deterministic name."""
        from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
        
        analysis = StructuralNetworkMapping(
            tractogram_path=dummy_tractogram,
            output_resolution=2,
            cache_tdi=True,
            check_dependencies=False,
        )
        
        # Should have a method to get cache path
        cache_path = analysis._get_tdi_cache_path()
        
        # Path should be deterministic based on tractogram path and resolution
        assert cache_path is not None
        assert "tractogram" in str(cache_path) or "tdi" in str(cache_path)
        assert "2mm" in str(cache_path) or "_2_" in str(cache_path)
    
    def test_cache_path_differs_by_resolution(self, dummy_tractogram):
        """Test that different resolutions use different cache paths."""
        from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
        
        analysis_1mm = StructuralNetworkMapping(
            tractogram_path=dummy_tractogram,
            output_resolution=1,
            cache_tdi=True,
            check_dependencies=False,
        )
        
        analysis_2mm = StructuralNetworkMapping(
            tractogram_path=dummy_tractogram,
            output_resolution=2,
            cache_tdi=True,
            check_dependencies=False,
        )
        
        cache_1mm = analysis_1mm._get_tdi_cache_path()
        cache_2mm = analysis_2mm._get_tdi_cache_path()
        
        assert cache_1mm != cache_2mm
