"""
Test atlas space transformation in StructuralNetworkMapping.

Ensures that atlases in different spaces are automatically transformed
to match the tractogram space.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.structural_network_mapping import StructuralNetworkMapping
from lacuna.core.lesion_data import LesionData


def test_atlas_transformed_when_space_mismatch():
    """Test that atlas is automatically transformed when space doesn't match tractogram."""
    
    # Create dummy tractogram
    with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as tmp:
        tractogram_path = Path(tmp.name)
        tmp.write(b"dummy tractogram")
    
    try:
        # Create dummy lesion
        lesion_data_array = np.zeros((10, 10, 10), dtype=np.float32)
        lesion_data_array[4:6, 4:6, 4:6] = 1.0
        lesion_img = nib.Nifti1Image(lesion_data_array, np.eye(4))
        
        lesion = LesionData(
            lesion_img=lesion_img,
            metadata={
                "subject_id": "test",
                "space": "MNI152NLin2009cAsym",
                "resolution": 2,
            }
        )
        
        # Mock dependencies
        with patch("lacuna.analysis.structural_network_mapping.check_mrtrix_available"):
            with patch("lacuna.analysis.structural_network_mapping.load_template") as mock_load_template:
                # Mock template
                template_path = Path(tempfile.mkdtemp()) / "template.nii.gz"
                template_data = np.zeros((91, 109, 91), dtype=np.float32)
                template_img = nib.Nifti1Image(template_data, np.eye(4))
                nib.save(template_img, template_path)
                mock_load_template.return_value = template_path
                
                with patch("lacuna.analysis.structural_network_mapping.compute_tdi_map"):
                    with patch("lacuna.analysis.structural_network_mapping.load_atlas") as mock_load_atlas:
                        # Mock atlas in different space (MNI152NLin6Asym)
                        atlas_data = np.random.randint(0, 101, size=(182, 218, 182), dtype=np.int16)
                        atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
                        
                        mock_atlas = Mock()
                        mock_atlas.image = atlas_img
                        mock_atlas.labels = {str(i): f"Region_{i}" for i in range(1, 101)}
                        mock_atlas.metadata = Mock()
                        mock_atlas.metadata.space = "MNI152NLin6Asym"  # Different from tractogram space
                        mock_atlas.metadata.resolution = 1
                        mock_atlas.metadata.atlas_filename = "test_atlas.nii.gz"
                        
                        mock_load_atlas.return_value = mock_atlas

                        with patch("lacuna.spatial.transform.transform_image") as mock_transform:
                            # Mock transformation - returns transformed atlas image
                            transformed_atlas_data = np.random.randint(0, 101, size=(91, 109, 91), dtype=np.int16)
                            transformed_atlas_img = nib.Nifti1Image(transformed_atlas_data, np.eye(4))
                            mock_transform.return_value = transformed_atlas_img
                            
                            # Initialize analysis with atlas
                            analysis = StructuralNetworkMapping(
                                tractogram_path=tractogram_path,
                                tractogram_space="MNI152NLin2009cAsym",  # Different from atlas space
                                output_resolution=2,
                                atlas_name="Schaefer2018_100Parcels7Networks",
                                cache_tdi=False,
                                check_dependencies=False,
                            )
                            
                            # Validate inputs - this should trigger atlas transformation
                            analysis._validate_inputs(lesion)
                            
                            # Verify transformation was called
                            assert mock_transform.called, "Atlas should be transformed when spaces don't match"
                            
                            # Verify transform was called with correct interpolation
                            call_kwargs = mock_transform.call_args[1]
                            assert call_kwargs.get('interpolation') == 'nearest', \
                                "Should use nearest neighbor interpolation for labels"
                            
                            # Verify target space resolution
                            target_space_arg = call_kwargs.get('target_space')
                            assert target_space_arg.resolution == 2, \
                                "Should use output_resolution for target space"
                            
                            # Verify transformed atlas was saved and path set
                            assert analysis._atlas_resolved is not None
                            assert "atlas_" in str(analysis._atlas_resolved)
                            
    finally:
        tractogram_path.unlink()


def test_atlas_not_transformed_when_space_matches():
    """Test that atlas is not transformed when space already matches tractogram."""
    
    # Create dummy tractogram
    with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as tmp:
        tractogram_path = Path(tmp.name)
        tmp.write(b"dummy tractogram")
    
    try:
        # Create dummy lesion
        lesion_data_array = np.zeros((10, 10, 10), dtype=np.float32)
        lesion_data_array[4:6, 4:6, 4:6] = 1.0
        lesion_img = nib.Nifti1Image(lesion_data_array, np.eye(4))
        
        lesion = LesionData(
            lesion_img=lesion_img,
            metadata={
                "subject_id": "test",
                "space": "MNI152NLin6Asym",
                "resolution": 2,
            }
        )
        
        # Mock dependencies
        with patch("lacuna.analysis.structural_network_mapping.check_mrtrix_available"):
            with patch("lacuna.analysis.structural_network_mapping.load_template") as mock_load_template:
                # Mock template
                template_path = Path(tempfile.mkdtemp()) / "template.nii.gz"
                template_data = np.zeros((91, 109, 91), dtype=np.float32)
                template_img = nib.Nifti1Image(template_data, np.eye(4))
                nib.save(template_img, template_path)
                mock_load_template.return_value = template_path
                
                with patch("lacuna.analysis.structural_network_mapping.compute_tdi_map"):
                    with patch("lacuna.analysis.structural_network_mapping.load_atlas") as mock_load_atlas:
                        # Create temp atlas file
                        atlas_dir = Path(tempfile.mkdtemp())
                        atlas_file = atlas_dir / "test_atlas.nii.gz"
                        atlas_data = np.random.randint(0, 101, size=(91, 109, 91), dtype=np.int16)
                        atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
                        nib.save(atlas_img, atlas_file)
                        
                        # Mock atlas in SAME space
                        mock_atlas = Mock()
                        mock_atlas.image = atlas_img
                        mock_atlas.labels = {str(i): f"Region_{i}" for i in range(1, 101)}
                        mock_atlas.metadata = Mock()
                        mock_atlas.metadata.space = "MNI152NLin6Asym"  # Same as tractogram
                        mock_atlas.metadata.resolution = 1
                        mock_atlas.metadata.atlas_filename = str(atlas_file)
                        
                        mock_load_atlas.return_value = mock_atlas
                        
                        with patch("lacuna.spatial.transform.transform_image") as mock_transform:
                            # Mock - should NOT be called when spaces match
                            mock_transform.side_effect = Exception("Should not transform when spaces match")
                            
                            # Initialize analysis with atlas - same space as atlas
                            analysis = StructuralNetworkMapping(
                                tractogram_path=tractogram_path,
                                tractogram_space="MNI152NLin6Asym",  # Same as atlas space
                                output_resolution=2,
                                atlas_name="Schaefer2018_100Parcels7Networks",
                                cache_tdi=False,
                                check_dependencies=False,
                            )
                            
                            # Validate inputs
                            analysis._validate_inputs(lesion)
                            
                            # Verify transformation was NOT called
                            assert not mock_transform.called, \
                                "Atlas should not be transformed when spaces match"
                            
                            # Verify original atlas file is used
                            assert analysis._atlas_resolved == atlas_file
                            
    finally:
        tractogram_path.unlink()


def test_atlas_transformation_uses_correct_resolution():
    """Test that atlas transformation uses the output_resolution parameter."""
    
    # Create dummy tractogram
    with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as tmp:
        tractogram_path = Path(tmp.name)
        tmp.write(b"dummy tractogram")
    
    try:
        # Create dummy lesion
        lesion_data_array = np.zeros((10, 10, 10), dtype=np.float32)
        lesion_data_array[4:6, 4:6, 4:6] = 1.0
        lesion_img = nib.Nifti1Image(lesion_data_array, np.eye(4))
        
        lesion = LesionData(
            lesion_img=lesion_img,
            metadata={
                "subject_id": "test",
                "space": "MNI152NLin2009cAsym",
                "resolution": 2,
            }
        )
        
        # Mock dependencies
        with patch("lacuna.analysis.structural_network_mapping.check_mrtrix_available"):
            with patch("lacuna.analysis.structural_network_mapping.load_template") as mock_load_template:
                # Mock template
                template_path = Path(tempfile.mkdtemp()) / "template.nii.gz"
                template_data = np.zeros((91, 109, 91), dtype=np.float32)
                template_img = nib.Nifti1Image(template_data, np.eye(4))
                nib.save(template_img, template_path)
                mock_load_template.return_value = template_path
                
                with patch("lacuna.analysis.structural_network_mapping.compute_tdi_map"):
                    with patch("lacuna.analysis.structural_network_mapping.load_atlas") as mock_load_atlas:
                        # Mock atlas
                        atlas_data = np.random.randint(0, 101, size=(182, 218, 182), dtype=np.int16)
                        atlas_img = nib.Nifti1Image(atlas_data, np.eye(4))
                        
                        mock_atlas = Mock()
                        mock_atlas.image = atlas_img
                        mock_atlas.labels = {str(i): f"Region_{i}" for i in range(1, 101)}
                        mock_atlas.metadata = Mock()
                        mock_atlas.metadata.space = "MNI152NLin6Asym"
                        mock_atlas.metadata.resolution = 1
                        mock_atlas.metadata.atlas_filename = "test_atlas.nii.gz"
                        
                        mock_load_atlas.return_value = mock_atlas
                        
                        with patch("lacuna.spatial.transform.transform_image") as mock_transform:
                            # Mock transformation - returns transformed atlas image
                            transformed_atlas_data = np.random.randint(0, 101, size=(91, 109, 91), dtype=np.int16)
                            transformed_atlas_img = nib.Nifti1Image(transformed_atlas_data, np.eye(4))
                            mock_transform.return_value = transformed_atlas_img
                            
                            # Initialize with specific output resolution
                            output_res = 2
                            analysis = StructuralNetworkMapping(
                                tractogram_path=tractogram_path,
                                tractogram_space="MNI152NLin2009cAsym",
                                output_resolution=output_res,
                                atlas_name="Schaefer2018_100Parcels7Networks",
                                cache_tdi=False,
                                check_dependencies=False,
                            )
                            
                            # Validate inputs
                            analysis._validate_inputs(lesion)
                            
                            # Verify transformation used correct resolution
                            call_kwargs = mock_transform.call_args[1]
                            target_space = call_kwargs.get('target_space')
                            assert target_space.resolution == output_res, \
                                f"Should use output_resolution ({output_res}mm) for atlas transformation"
                            
    finally:
        tractogram_path.unlink()
