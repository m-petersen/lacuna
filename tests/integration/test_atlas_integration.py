"""Integration tests for atlas registry and analysis modules.

Tests the complete workflow of atlas loading, discovery, and usage
in analysis modules.
"""

import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from lacuna import LesionData
from lacuna.analysis import RegionalDamage, AtlasAggregation
from lacuna.assets.atlases.registry import list_atlases
from lacuna.assets.atlases.loader import load_atlas


class TestMultiAtlasAnalysisWorkflow:
    """Test multi-atlas analysis workflow (T067)."""

    def test_regional_damage_with_single_atlas(self):
        """RegionalDamage can use a single named atlas."""
        # Create a simple binary lesion
        lesion_data = np.zeros((91, 109, 91))
        lesion_data[45:50, 54:59, 45:50] = 1
        lesion_img = nib.Nifti1Image(lesion_data.astype(np.float32), np.eye(4))
        lesion = LesionData(
            lesion_img=lesion_img,
            metadata={"space": "MNI152NLin2009cAsym", "subject_id": "test001"}
        )

        # Use specific atlas by name
        analysis = RegionalDamage(atlas="Schaefer100")
        
        # Validate it was configured correctly
        assert analysis.atlas == ["Schaefer100"]
        assert analysis.aggregation == "percent"

    def test_regional_damage_with_multiple_atlases(self):
        """RegionalDamage can use multiple named atlases."""
        # Create a simple binary lesion
        lesion_data = np.zeros((91, 109, 91))
        lesion_data[45:50, 54:59, 45:50] = 1
        lesion_img = nib.Nifti1Image(lesion_data.astype(np.float32), np.eye(4))
        lesion = LesionData(
            lesion_img=lesion_img,
            metadata={"space": "MNI152NLin2009cAsym", "subject_id": "test001"}
        )

        # Use multiple atlases by name
        analysis = RegionalDamage(atlas=["Schaefer100", "Schaefer200"])
        
        # Validate it was configured correctly
        assert analysis.atlas == ["Schaefer100", "Schaefer200"]
        assert len(analysis.atlas) == 2

    def test_atlas_aggregation_with_named_atlas(self):
        """AtlasAggregation can use named atlas from registry."""
        # Create a simple binary lesion
        lesion_data = np.zeros((91, 109, 91))
        lesion_data[45:50, 54:59, 45:50] = 1
        lesion_img = nib.Nifti1Image(lesion_data.astype(np.float32), np.eye(4))
        lesion = LesionData(
            lesion_img=lesion_img,
            metadata={"space": "MNI152NLin2009cAsym", "subject_id": "test001"}
        )

        # Use atlas by name with different aggregation
        analysis = AtlasAggregation(
            atlas="TianS2",
            source="lesion_img",
            aggregation="mean"
        )
        
        # Validate configuration
        assert analysis.atlas == ["TianS2"]
        assert analysis.aggregation == "mean"

    def test_backward_compatibility_with_atlas_dir(self):
        """Analysis modules still support legacy atlas_dir parameter."""
        # Create a simple binary lesion
        lesion_data = np.zeros((91, 109, 91))
        lesion_data[45:50, 54:59, 45:50] = 1
        lesion_img = nib.Nifti1Image(lesion_data.astype(np.float32), np.eye(4))
        lesion = LesionData(
            lesion_img=lesion_img,
            metadata={"space": "MNI152NLin2009cAsym", "subject_id": "test001"}
        )

        # Create temporary atlas directory
        with tempfile.TemporaryDirectory() as tmpdir:
            atlas_dir = Path(tmpdir)
            
            # Legacy usage should work (atlas_dir is deprecated but still functional)
            analysis = RegionalDamage(atlas_dir=str(atlas_dir))
            
            # Should still store the atlas_dir
            assert analysis.atlas_dir == str(atlas_dir)


class TestAtlasDiscovery:
    """Test atlas discovery with list_atlases() (T068)."""

    def test_list_all_bundled_atlases(self):
        """list_atlases() returns all bundled atlases."""
        atlases = list_atlases()
        
        # Should have atlases
        assert len(atlases) > 0
        
        # Each should be AtlasMetadata with required fields
        for atlas in atlases:
            assert hasattr(atlas, "name")
            assert hasattr(atlas, "space")
            assert hasattr(atlas, "n_regions")
            assert hasattr(atlas, "atlas_type")

    def test_filter_atlases_by_space(self):
        """list_atlases() can filter by space."""
        # Get atlases in specific space
        mni2009c_atlases = list_atlases(space="MNI152NLin2009cAsym")
        
        # All should be in that space
        for atlas in mni2009c_atlases:
            assert atlas.space == "MNI152NLin2009cAsym"

    def test_filter_atlases_by_type(self):
        """list_atlases() can filter by type."""
        # Get network atlases
        network_atlases = list_atlases(atlas_type="network")
        
        # All should be network type
        for atlas in network_atlases:
            assert atlas.atlas_type == "network"

    def test_filter_atlases_by_region_count(self):
        """list_atlases() can filter by region count range."""
        # Get atlases with 100-500 regions
        medium_atlases = list_atlases(min_regions=100, max_regions=500)
        
        # All should be in range
        for atlas in medium_atlases:
            assert 100 <= atlas.n_regions <= 500

    def test_combined_filters(self):
        """list_atlases() can combine multiple filters."""
        # Get network atlases in MNI2009c with 200-400 regions
        filtered = list_atlases(
            space="MNI152NLin2009cAsym",
            atlas_type="network",
            min_regions=200,
            max_regions=400
        )
        
        # All should match all criteria
        for atlas in filtered:
            assert atlas.space == "MNI152NLin2009cAsym"
            assert atlas.atlas_type == "network"
            assert 200 <= atlas.n_regions <= 400

    def test_discover_specific_atlases(self):
        """list_atlases() includes expected bundled atlases."""
        all_atlases = list_atlases()
        atlas_names = [a.name for a in all_atlases]
        
        # Should include Schaefer parcellations
        assert "Schaefer100" in atlas_names
        assert "Schaefer200" in atlas_names
        assert "Schaefer400" in atlas_names
        
        # Should include Tian subcortical atlases
        assert "TianS2" in atlas_names


class TestAtlasLoadingInDifferentSpaces:
    """Test atlas loading in different coordinate spaces (T069)."""

    def test_load_atlas_returns_correct_space(self):
        """load_atlas() returns atlas with correct coordinate space."""
        # Load atlas
        atlas = load_atlas("Schaefer400")
        
        # Should have correct metadata
        assert atlas.metadata.space == "MNI152NLin6Asym"
        assert atlas.metadata.n_regions == 400

    def test_load_atlas_with_labels(self):
        """load_atlas() and get_atlas_labels() work together."""
        # Load atlas
        atlas = load_atlas("TianS2")
        labels = get_atlas_labels("TianS2")
        
        # Labels should match region count
        assert len(labels) == atlas.metadata.n_regions
        
        # Labels should be dict mapping int -> str
        assert isinstance(labels, dict)
        for region_id, region_name in labels.items():
            assert isinstance(region_id, int)
            assert isinstance(region_name, str)

    def test_load_different_schaefer_parcellations(self):
        """Can load different Schaefer parcellation scales."""
        atlases = ["Schaefer100", "Schaefer200", "Schaefer400", "Schaefer1000"]
        expected_regions = [100, 200, 400, 1000]
        
        for atlas_name, expected_n in zip(atlases, expected_regions):
            atlas = load_atlas(atlas_name)
            assert atlas.metadata.n_regions == expected_n
            assert atlas.metadata.space == "MNI152NLin6Asym"

    def test_load_tian_subcortical_atlases(self):
        """Can load different Tian subcortical atlas scales."""
        atlases = ["TianS1", "TianS2", "TianS3"]
        expected_regions = [16, 32, 50]
        
        for atlas_name, expected_n in zip(atlases, expected_regions):
            atlas = load_atlas(atlas_name)
            assert atlas.metadata.n_regions == expected_n
            assert atlas.metadata.space == "MNI152NLin6Asym"

    def test_atlas_metadata_consistency(self):
        """Atlas metadata from registry is consistent."""
        # Get metadata from registry
        all_atlases = list_atlases()
        
        # Check first 3 for consistency
        for atlas_info in all_atlases[:3]:
            # Metadata should have all required fields
            assert hasattr(atlas_info, "name")
            assert hasattr(atlas_info, "space")
            assert hasattr(atlas_info, "n_regions")
            assert hasattr(atlas_info, "atlas_type")
            assert hasattr(atlas_info, "filename")
            
            # Values should be reasonable
            assert isinstance(atlas_info.name, str)
            assert len(atlas_info.name) > 0
            assert atlas_info.n_regions > 0

    def test_invalid_atlas_name_raises_error(self):
        """load_atlas() raises ValueError for invalid atlas name."""
        with pytest.raises(ValueError, match="not found.*registry"):
            load_atlas("NonexistentAtlas12345")

    def test_atlas_registry_contains_expected_atlases(self):
        """Atlas registry contains all expected bundled atlases."""
        all_atlases = list_atlases()
        
        # Check that we have a reasonable number of atlases
        assert len(all_atlases) >= 8, f"Expected at least 8 atlases, got {len(all_atlases)}"
        
        # All should have proper metadata
        for atlas_info in all_atlases:
            assert isinstance(atlas_info.name, str)
            assert isinstance(atlas_info.space, str)
            assert atlas_info.n_regions > 0
            assert isinstance(atlas_info.filename, str)
