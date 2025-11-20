"""Integration tests for atlas registry with space handling.

Tests the complete workflow of loading atlases from the registry and
handling space transformations automatically.
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna import MaskData
from lacuna.analysis import AtlasAggregation
from lacuna.assets.atlases.loader import load_atlas
from lacuna.assets.atlases.registry import (
    ATLAS_REGISTRY,
    list_atlases,
    register_atlas_from_files,
    register_atlases_from_directory,
    unregister_atlas,
)


class TestAtlasRegistryBasics:
    """Test basic atlas registry functionality."""

    def test_list_bundled_atlases(self):
        """Test listing bundled atlases from registry."""
        atlases = list_atlases()

        # Should have at least the bundled atlases
        assert len(atlases) >= 8

        # Check for specific bundled atlases
        atlas_names = [a.name for a in atlases]
        assert "Schaefer2018_100Parcels7Networks" in atlas_names
        assert "Schaefer2018_200Parcels7Networks" in atlas_names
        assert "TianSubcortex_3TS1" in atlas_names

    def test_filter_atlases_by_space(self):
        """Test filtering atlases by coordinate space."""
        # Get atlases in MNI152NLin6Asym space
        atlases_lin6 = list_atlases(space="MNI152NLin6Asym")
        assert len(atlases_lin6) > 0
        assert all(a.space == "MNI152NLin6Asym" for a in atlases_lin6)

        # Get atlases in MNI152NLin2009aAsym space
        atlases_lin2009a = list_atlases(space="MNI152NLin2009aAsym")
        assert len(atlases_lin2009a) >= 1  # At least HCP1065

    def test_filter_atlases_by_resolution(self):
        """Test filtering atlases by resolution."""
        # Get 1mm atlases
        atlases_1mm = list_atlases(resolution=1)
        assert len(atlases_1mm) > 0
        assert all(a.resolution == 1 for a in atlases_1mm)

    def test_load_bundled_atlas(self):
        """Test loading a bundled atlas."""
        atlas = load_atlas("Schaefer2018_100Parcels7Networks")

        # Check metadata
        assert atlas.metadata.name == "Schaefer2018_100Parcels7Networks"
        assert atlas.metadata.space == "MNI152NLin6Asym"
        assert atlas.metadata.resolution == 1
        assert atlas.metadata.n_regions == 100

        # Check image
        assert isinstance(atlas.image, nib.Nifti1Image)
        # Shape should be roughly 182x218x182 for 1mm MNI space
        assert all(dim > 150 for dim in atlas.image.shape)

        # Check labels
        assert isinstance(atlas.labels, dict)
        assert len(atlas.labels) == 100
        assert all(isinstance(k, int) for k in atlas.labels.keys())
        assert all(isinstance(v, str) for v in atlas.labels.values())


class TestAtlasRegistryWithAnalysis:
    """Test atlas registry integration with analysis modules."""

    @pytest.fixture
    def synthetic_lesion(self):
        """Create a synthetic lesion in MNI152NLin6Asym space."""
        # Create lesion data with known affine for MNI152NLin6Asym 2mm
        affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Create small lesion in middle of brain
        data = np.zeros((91, 109, 91), dtype=np.float32)
        data[40:50, 50:60, 40:50] = 1.0

        img = nib.Nifti1Image(data, affine)

        return MaskData(mask_img=img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

    def test_atlas_aggregation_with_bundled_atlas(self, synthetic_lesion):
        """Test atlas aggregation using bundled atlas from registry."""
        # Use Schaefer100 atlas (1mm resolution, same space as lesion)
        analysis = AtlasAggregation(
            source="mask_img", aggregation="mean", atlas_names=["Schaefer2018_100Parcels7Networks"]
        )

        result = analysis.run(synthetic_lesion)

        # Check results
        assert "AtlasAggregation" in result.results
        aggregation_results = result.results["AtlasAggregation"]

        # Should have AtlasAggregationResult for Schaefer100
        assert "Schaefer2018_100Parcels7Networks" in aggregation_results
        roi_result = aggregation_results["Schaefer2018_100Parcels7Networks"]
        region_data = roi_result.get_data()

        # Should have multiple regions
        assert len(region_data) > 0

        # Values should be floats between 0 and 1 (mean of binary lesion)
        for key, value in region_data.items():
            assert isinstance(value, (int, float))
            assert 0 <= value <= 1

    def test_atlas_aggregation_with_multiple_atlases(self, synthetic_lesion):
        """Test atlas aggregation with multiple atlases simultaneously."""
        analysis = AtlasAggregation(
            source="mask_img",
            aggregation="percent",
            atlas_names=["Schaefer2018_100Parcels7Networks", "Schaefer2018_200Parcels7Networks"],
        )

        result = analysis.run(synthetic_lesion)

        aggregation_results = result.results["AtlasAggregation"]

        # Should have results from both atlases
        assert "Schaefer2018_100Parcels7Networks" in aggregation_results
        assert "Schaefer2018_200Parcels7Networks" in aggregation_results

        # Each should have region data
        schaefer100_data = aggregation_results["Schaefer2018_100Parcels7Networks"].get_data()
        schaefer200_data = aggregation_results["Schaefer2018_200Parcels7Networks"].get_data()

        assert len(schaefer100_data) > 0
        assert len(schaefer200_data) > 0

    def test_atlas_aggregation_auto_uses_all_atlases(self, synthetic_lesion):
        """Test that atlas aggregation uses all compatible atlases when none specified."""
        # Use only MNI152NLin6Asym atlases (filter out HCP which is in different space)
        analysis = AtlasAggregation(
            source="mask_img",
            aggregation="mean",
            atlas_names=[
                "Schaefer2018_100Parcels7Networks",
                "Schaefer2018_200Parcels7Networks",
                "TianSubcortex_3TS1",
            ],
        )

        result = analysis.run(synthetic_lesion)

        aggregation_results = result.results["AtlasAggregation"]

        # Should have results from all three atlases
        assert "Schaefer2018_100Parcels7Networks" in aggregation_results
        assert "Schaefer2018_200Parcels7Networks" in aggregation_results
        assert "TianSubcortex_3TS1" in aggregation_results
        has_schaefer200 = any(
            "Schaefer2018_200Parcels7Networks" in k for k in aggregation_results.keys()
        )
        has_tian = any("TianSubcortex_3TS1" in k for k in aggregation_results.keys())

        assert has_schaefer100, "Should have Schaefer100 results"
        assert has_schaefer200, "Should have Schaefer200 results"
        assert has_tian, "Should have Tian results"


class TestCustomAtlasRegistration:
    """Test user registration of custom atlases."""

    @pytest.fixture
    def custom_atlas_files(self, tmp_path):
        """Create custom atlas files for testing."""
        # Create simple atlas image
        affine = np.eye(4)
        affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0  # 2mm resolution

        data = np.zeros((91, 109, 91), dtype=np.int16)
        data[30:40, 40:50, 30:40] = 1  # Region 1
        data[50:60, 60:70, 50:60] = 2  # Region 2

        atlas_img = nib.Nifti1Image(data, affine)
        atlas_path = tmp_path / "custom_atlas.nii.gz"
        nib.save(atlas_img, atlas_path)

        # Create labels file
        labels_path = tmp_path / "custom_atlas_labels.txt"
        labels_path.write_text("1 Region_One\n2 Region_Two\n")

        return atlas_path, labels_path

    def test_register_custom_atlas(self, custom_atlas_files):
        """Test registering a custom atlas."""
        atlas_path, labels_path = custom_atlas_files

        # Register custom atlas
        register_atlas_from_files(
            name="CustomTestAtlas",
            atlas_path=str(atlas_path),
            labels_path=str(labels_path),
            space="MNI152NLin6Asym",
            resolution=2,
            description="Test custom atlas",
        )

        try:
            # Check it's in the registry
            assert "CustomTestAtlas" in ATLAS_REGISTRY

            # Load it
            atlas = load_atlas("CustomTestAtlas")
            assert atlas.metadata.name == "CustomTestAtlas"
            assert atlas.metadata.space == "MNI152NLin6Asym"
            assert atlas.metadata.resolution == 2
            assert len(atlas.labels) == 2

        finally:
            # Cleanup
            unregister_atlas("CustomTestAtlas")

    def test_use_custom_atlas_in_analysis(self, custom_atlas_files, synthetic_mask_data):
        """Test using a custom registered atlas in analysis."""
        atlas_path, labels_path = custom_atlas_files

        # Use the lesion from the fixture
        synthetic_lesion = synthetic_mask_data

        # Register custom atlas
        register_atlas_from_files(
            name="CustomAnalysisAtlas",
            atlas_path=str(atlas_path),
            labels_path=str(labels_path),
            space="MNI152NLin6Asym",
            resolution=2,
            description="Test atlas for analysis",
        )

        try:
            # Use in analysis
            analysis = AtlasAggregation(
                source="mask_img", aggregation="mean", atlas_names=["CustomAnalysisAtlas"]
            )

            result = analysis.run(synthetic_lesion)

            # Check results
            aggregation_results = result.results["AtlasAggregation"]
            assert "CustomAnalysisAtlas" in aggregation_results

            custom_data = aggregation_results["CustomAnalysisAtlas"].get_data()
            assert len(custom_data) == 2  # Two regions

        finally:
            # Cleanup
            unregister_atlas("CustomAnalysisAtlas")

    def test_register_atlases_from_directory(self, tmp_path):
        """Test bulk registration from directory."""
        # Create two atlas files with BIDS naming
        for i, name in enumerate(["atlas1", "atlas2"]):
            affine = np.eye(4)
            affine[0, 0] = affine[1, 1] = affine[2, 2] = 1.0

            data = np.zeros((91, 109, 91), dtype=np.int16)
            data[30:40, 40:50, 30:40] = 1

            atlas_img = nib.Nifti1Image(data, affine)
            atlas_path = tmp_path / f"tpl-MNI152NLin6Asym_res-01_atlas-{name}_dseg.nii.gz"
            nib.save(atlas_img, atlas_path)

            labels_path = tmp_path / f"tpl-MNI152NLin6Asym_res-01_atlas-{name}_dseg_labels.txt"
            labels_path.write_text(f"1 Region_One_{i}\n")

        # Register from directory
        registered = register_atlases_from_directory(
            tmp_path, space="MNI152NLin6Asym", resolution=1
        )

        try:
            # Should have registered both
            assert len(registered) == 2

            # Check they're in registry
            for name in registered:
                assert name.startswith("tpl-MNI152NLin6Asym_res-01_atlas-")
                assert name in ATLAS_REGISTRY

        finally:
            # Cleanup
            for name in registered:
                if name in ATLAS_REGISTRY:
                    unregister_atlas(name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
