"""Integration tests for atlas registry and analysis modules.

Tests the complete workflow of atlas loading, discovery, and usage
in analysis modules.
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna import MaskData
from lacuna.analysis import ParcelAggregation, RegionalDamage
from lacuna.assets.parcellations.loader import load_parcellation
from lacuna.assets.parcellations.registry import list_parcellations


@pytest.fixture
def sample_mask_data():
    """Create a sample MaskData for testing."""
    mask_data = np.zeros((91, 109, 91))
    mask_data[45:50, 54:59, 45:50] = 1
    mask_img = nib.Nifti1Image(mask_data.astype(np.float32), np.eye(4))
    return MaskData(
        mask_img=mask_img,
        space="MNI152NLin6Asym",
        resolution=1.0,
        metadata={"subject_id": "test001"},
    )


class TestMultiAtlasAnalysisWorkflow:
    """Test multi-atlas analysis workflow."""

    def test_regional_damage_with_single_atlas(self, sample_mask_data):
        """RegionalDamage can use a single named atlas."""
        # Use specific parcellation by name
        analysis = RegionalDamage(parcel_names=["Schaefer2018_100Parcels7Networks"])

        # Validate it was configured correctly
        assert analysis.parcel_names == ["Schaefer2018_100Parcels7Networks"]
        assert analysis.aggregation == "percent"

    def test_regional_damage_with_multiple_atlases(self, sample_mask_data):
        """RegionalDamage can use multiple named atlases."""
        # Use multiple parcellations by name
        analysis = RegionalDamage(
            parcel_names=[
                "Schaefer2018_100Parcels7Networks",
                "Schaefer2018_200Parcels7Networks",
            ]
        )

        # Validate it was configured correctly
        assert len(analysis.parcel_names) == 2
        assert "Schaefer2018_100Parcels7Networks" in analysis.parcel_names
        assert "Schaefer2018_200Parcels7Networks" in analysis.parcel_names

    def test_parcel_aggregation_with_named_atlas(self, sample_mask_data):
        """ParcelAggregation can use named parcellation from registry."""
        # Use parcellation by name with different aggregation
        analysis = ParcelAggregation(
            parcel_names=["TianSubcortex_3TS2"],
            source="mask_img",
            aggregation="mean",
        )

        # Validate configuration
        assert analysis.parcel_names == ["TianSubcortex_3TS2"]
        assert analysis.aggregation == "mean"

    def test_regional_damage_default_parcellations(self, sample_mask_data):
        """RegionalDamage with None uses all available parcellations."""
        analysis = RegionalDamage(parcel_names=None)
        assert analysis.parcel_names is None


class TestAtlasDiscovery:
    """Test atlas discovery with list_parcellations()."""

    def test_list_all_bundled_atlases(self):
        """list_parcellations() returns all bundled atlases."""
        atlases = list_parcellations()

        # Should have atlases
        assert len(atlases) > 0

        # Each should be ParcellationMetadata with required fields
        for atlas in atlases:
            assert hasattr(atlas, "name")
            assert hasattr(atlas, "space")
            assert hasattr(atlas, "resolution")
            assert hasattr(atlas, "n_regions")
            assert hasattr(atlas, "parcellation_filename")

    def test_filter_atlases_by_space(self):
        """list_parcellations() can filter by space."""
        # Get atlases in specific space
        mni6_atlases = list_parcellations(space="MNI152NLin6Asym")

        # All should be in that space
        for atlas in mni6_atlases:
            assert atlas.space == "MNI152NLin6Asym"

    def test_filter_atlases_by_resolution(self):
        """list_parcellations() can filter by resolution."""
        # Get 1mm resolution atlases
        res1_atlases = list_parcellations(resolution=1)

        # All should have 1mm resolution
        for atlas in res1_atlases:
            assert atlas.resolution == 1

    def test_combined_filters(self):
        """list_parcellations() can combine space and resolution filters."""
        # Get atlases in MNI6Asym with 1mm resolution
        filtered = list_parcellations(space="MNI152NLin6Asym", resolution=1)

        # All should match all criteria
        for atlas in filtered:
            assert atlas.space == "MNI152NLin6Asym"
            assert atlas.resolution == 1

    def test_discover_specific_atlases(self):
        """list_parcellations() includes expected bundled atlases."""
        all_atlases = list_parcellations()
        parcel_names = [a.name for a in all_atlases]

        # Should include Schaefer parcellations with full names
        assert "Schaefer2018_100Parcels7Networks" in parcel_names
        assert "Schaefer2018_200Parcels7Networks" in parcel_names
        assert "Schaefer2018_400Parcels7Networks" in parcel_names

        # Should include Tian subcortical atlases
        assert any("Tian" in name for name in parcel_names)


class TestAtlasLoadingInDifferentSpaces:
    """Test atlas loading in different coordinate spaces."""

    def test_load_parcellation_returns_correct_metadata(self):
        """load_parcellation() returns atlas with correct metadata."""
        # Load atlas
        atlas = load_parcellation("Schaefer2018_400Parcels7Networks")

        # Should have correct metadata
        assert atlas.metadata.name == "Schaefer2018_400Parcels7Networks"
        assert atlas.metadata.n_regions == 400

    def test_load_parcellation_returns_nifti_image(self):
        """load_parcellation() returns a proper NIfTI image."""
        atlas = load_parcellation("Schaefer2018_100Parcels7Networks")

        # Should return Parcellation with image data
        assert hasattr(atlas, "image")
        assert hasattr(atlas, "metadata")
        assert hasattr(atlas, "labels")
        # The image should be a nibabel image
        assert hasattr(atlas.image, "get_fdata")

    def test_load_different_schaefer_parcellations(self):
        """Can load different Schaefer parcellation scales."""
        atlases = [
            "Schaefer2018_100Parcels7Networks",
            "Schaefer2018_200Parcels7Networks",
            "Schaefer2018_400Parcels7Networks",
        ]
        expected_regions = [100, 200, 400]

        for atlas_name, expected_n in zip(atlases, expected_regions, strict=False):
            atlas = load_parcellation(atlas_name)
            assert atlas.metadata.n_regions == expected_n

    def test_load_tian_subcortical_atlases(self):
        """Can load Tian subcortical atlas scales."""
        # Check which Tian atlases exist
        all_atlases = list_parcellations()
        tian_names = [a.name for a in all_atlases if "Tian" in a.name]

        # Should have at least one Tian atlas
        assert len(tian_names) >= 1

        # Load the first one
        atlas = load_parcellation(tian_names[0])
        assert atlas.metadata is not None
        assert atlas.metadata.n_regions > 0

    def test_atlas_metadata_consistency(self):
        """Atlas metadata from registry is consistent."""
        # Get metadata from registry
        all_atlases = list_parcellations()

        # Check first 3 for consistency
        for atlas_info in all_atlases[:3]:
            # Metadata should have all required fields
            assert hasattr(atlas_info, "name")
            assert hasattr(atlas_info, "space")
            assert hasattr(atlas_info, "resolution")
            assert hasattr(atlas_info, "n_regions")
            assert hasattr(atlas_info, "parcellation_filename")

            # Values should be reasonable
            assert isinstance(atlas_info.name, str)
            assert len(atlas_info.name) > 0
            assert atlas_info.n_regions > 0

    def test_invalid_atlas_name_raises_error(self):
        """load_parcellation() raises KeyError for invalid atlas name."""
        with pytest.raises(KeyError, match="not found"):
            load_parcellation("NonexistentAtlas12345")

    def test_atlas_registry_contains_expected_atlases(self):
        """Atlas registry contains all expected bundled atlases."""
        all_atlases = list_parcellations()

        # Check that we have a reasonable number of atlases
        assert len(all_atlases) >= 4, f"Expected at least 4 atlases, got {len(all_atlases)}"

        # All should have proper metadata
        for atlas_info in all_atlases:
            assert isinstance(atlas_info.name, str)
            assert isinstance(atlas_info.space, str)
            assert atlas_info.n_regions > 0
            assert isinstance(atlas_info.parcellation_filename, str)
