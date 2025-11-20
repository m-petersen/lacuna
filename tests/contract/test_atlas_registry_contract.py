"""Contract tests for atlas registry and loader.

Tests the expected interface and behavior of the atlas registry system,
ensuring bundled atlases are properly registered with metadata and can be
loaded via the DataAssetManager integration.
"""

from dataclasses import is_dataclass

import nibabel as nib
import pytest


class TestAtlasMetadataContract:
    """Contract tests for AtlasMetadata dataclass."""

    def test_atlas_metadata_is_dataclass(self):
        """AtlasMetadata must be a dataclass."""
        from lacuna.assets.atlases.registry import AtlasMetadata

        assert is_dataclass(AtlasMetadata)

    def test_atlas_metadata_required_fields(self):
        """AtlasMetadata must have required fields."""
        from lacuna.assets.atlases.registry import AtlasMetadata

        # Should be able to create with minimal required fields
        metadata = AtlasMetadata(
            name="TestAtlas",
            space="MNI152NLin6Asym",
            resolution=1,
            description="Test atlas",
            atlas_filename="test.nii.gz",
            labels_filename="test_labels.txt",
        )

        assert metadata.name == "TestAtlas"
        assert metadata.space == "MNI152NLin6Asym"
        assert metadata.resolution == 1
        assert metadata.description == "Test atlas"
        assert metadata.atlas_filename == "test.nii.gz"
        assert metadata.labels_filename == "test_labels.txt"

    def test_atlas_metadata_optional_fields(self):
        """AtlasMetadata should have optional citation and network fields."""
        from lacuna.assets.atlases.registry import AtlasMetadata

        metadata = AtlasMetadata(
            name="TestAtlas",
            space="MNI152NLin6Asym",
            resolution=1,
            description="Test atlas",
            atlas_filename="test.nii.gz",
            labels_filename="test_labels.txt",
            citation="Test et al. (2024)",
            networks=["Visual", "Motor"],
            n_regions=100,
        )

        assert metadata.citation == "Test et al. (2024)"
        assert metadata.networks == ["Visual", "Motor"]
        assert metadata.n_regions == 100


class TestAtlasRegistryContract:
    """Contract tests for ATLAS_REGISTRY."""

    def test_atlas_registry_exists(self):
        """ATLAS_REGISTRY constant must exist."""
        from lacuna.assets.atlases.registry import ATLAS_REGISTRY

        assert ATLAS_REGISTRY is not None
        assert isinstance(ATLAS_REGISTRY, dict)

    def test_registry_contains_bundled_atlases(self):
        """Registry must contain all bundled atlases."""
        from lacuna.assets.atlases.registry import ATLAS_REGISTRY

        # Based on src/lacuna/data/atlases/ contents
        expected_atlases = [
            "Schaefer2018_100Parcels7Networks",
            "Schaefer2018_200Parcels7Networks",
            "Schaefer2018_400Parcels7Networks",
            "Schaefer2018_1000Parcels7Networks",
            "TianSubcortex_3TS1",
            "TianSubcortex_3TS2",
            "TianSubcortex_3TS3",
            "HCP1065_thr0p1",
        ]

        for atlas_name in expected_atlases:
            assert atlas_name in ATLAS_REGISTRY, f"Missing atlas: {atlas_name}"

    def test_registry_entries_have_metadata(self):
        """Each registry entry must be an AtlasMetadata instance."""
        from lacuna.assets.atlases.registry import ATLAS_REGISTRY, AtlasMetadata

        for atlas_name, metadata in ATLAS_REGISTRY.items():
            assert isinstance(
                metadata, AtlasMetadata
            ), f"{atlas_name} metadata is not AtlasMetadata"
            assert metadata.name == atlas_name
            assert metadata.space is not None
            assert metadata.resolution is not None
            assert metadata.atlas_filename is not None
            assert metadata.labels_filename is not None

    def test_schaefer_atlases_metadata(self):
        """Schaefer atlases must have correct metadata."""
        from lacuna.assets.atlases.registry import ATLAS_REGISTRY

        schaefer_100 = ATLAS_REGISTRY["Schaefer2018_100Parcels7Networks"]
        assert schaefer_100.space == "MNI152NLin6Asym"
        assert schaefer_100.resolution == 1
        assert schaefer_100.n_regions == 100
        assert "7" in schaefer_100.name or len(schaefer_100.networks) == 7

        schaefer_400 = ATLAS_REGISTRY["Schaefer2018_400Parcels7Networks"]
        assert schaefer_400.n_regions == 400

    def test_tian_atlases_metadata(self):
        """Tian subcortical atlases must have correct metadata."""
        from lacuna.assets.atlases.registry import ATLAS_REGISTRY

        tian_s1 = ATLAS_REGISTRY["TianSubcortex_3TS1"]
        assert tian_s1.space == "MNI152NLin6Asym"
        assert tian_s1.resolution == 1
        assert "subcort" in tian_s1.description.lower()

    def test_hcp_atlas_metadata(self):
        """HCP white matter atlas must have correct metadata."""
        from lacuna.assets.atlases.registry import ATLAS_REGISTRY

        hcp = ATLAS_REGISTRY["HCP1065_thr0p1"]
        assert hcp.space == "MNI152NLin2009aAsym"
        assert hcp.resolution == 1
        assert "white matter" in hcp.description.lower() or "tract" in hcp.description.lower()


class TestAtlasLoaderContract:
    """Contract tests for atlas loading functionality."""

    def test_load_atlas_function_exists(self):
        """load_atlas() function must exist."""
        from lacuna.assets.atlases.loader import load_atlas

        assert callable(load_atlas)

    def test_load_atlas_by_name(self):
        """load_atlas() must accept atlas name from registry."""
        from lacuna.assets.atlases.loader import load_atlas

        # Should not raise exception for valid atlas
        atlas = load_atlas("Schaefer2018_100Parcels7Networks")

        assert atlas is not None
        assert hasattr(atlas, "image"), "Atlas must have 'image' attribute"
        assert hasattr(atlas, "labels"), "Atlas must have 'labels' attribute"
        assert hasattr(atlas, "metadata"), "Atlas must have 'metadata' attribute"

    def test_loaded_atlas_image_is_nifti(self):
        """Loaded atlas image must be a nibabel Nifti1Image."""
        from lacuna.assets.atlases.loader import load_atlas

        atlas = load_atlas("Schaefer2018_100Parcels7Networks")

        assert isinstance(atlas.image, nib.Nifti1Image)
        assert atlas.image.shape[0] > 0  # Has valid dimensions

    def test_loaded_atlas_labels_is_dict(self):
        """Loaded atlas labels must be a dict mapping region_id -> name."""
        from lacuna.assets.atlases.loader import load_atlas

        atlas = load_atlas("Schaefer2018_100Parcels7Networks")

        assert isinstance(atlas.labels, dict)
        assert len(atlas.labels) > 0
        # Check format: int keys, string values
        for region_id, region_name in atlas.labels.items():
            assert isinstance(region_id, int)
            assert isinstance(region_name, str)

    def test_loaded_atlas_metadata_matches_registry(self):
        """Loaded atlas metadata must match registry entry."""
        from lacuna.assets.atlases.loader import load_atlas
        from lacuna.assets.atlases.registry import ATLAS_REGISTRY

        atlas_name = "Schaefer2018_100Parcels7Networks"
        atlas = load_atlas(atlas_name)
        expected_metadata = ATLAS_REGISTRY[atlas_name]

        assert atlas.metadata.name == expected_metadata.name
        assert atlas.metadata.space == expected_metadata.space
        assert atlas.metadata.resolution == expected_metadata.resolution

    def test_load_atlas_invalid_name_raises_error(self):
        """load_atlas() must raise KeyError for unknown atlas."""
        from lacuna.assets.atlases.loader import load_atlas

        with pytest.raises((KeyError, ValueError)):
            load_atlas("NonexistentAtlas")

    def test_load_atlas_uses_asset_manager(self):
        """load_atlas() should work without requiring external dependencies."""
        from lacuna.assets.atlases.loader import load_atlas

        # Should load bundled atlas without any external parameters
        atlas = load_atlas("Schaefer2018_100Parcels7Networks")

        assert atlas is not None


class TestAtlasListingContract:
    """Contract tests for atlas discovery and listing."""

    def test_list_atlases_function_exists(self):
        """list_atlases() function must exist."""
        from lacuna.assets.atlases.registry import list_atlases

        assert callable(list_atlases)

    def test_list_atlases_returns_all_registered(self):
        """list_atlases() must return all registered atlases by default."""
        from lacuna.assets.atlases.registry import list_atlases

        atlases = list_atlases()

        assert len(atlases) >= 8  # At least the bundled atlases
        assert all(hasattr(a, "name") for a in atlases)

    def test_list_atlases_filter_by_space(self):
        """list_atlases() must support filtering by space."""
        from lacuna.assets.atlases.registry import list_atlases

        mni6_atlases = list_atlases(space="MNI152NLin6Asym")

        assert len(mni6_atlases) > 0
        assert all(a.space == "MNI152NLin6Asym" for a in mni6_atlases)

    def test_list_atlases_filter_by_resolution(self):
        """list_atlases() must support filtering by resolution."""
        from lacuna.assets.atlases.registry import list_atlases

        res1_atlases = list_atlases(resolution=1)

        assert len(res1_atlases) > 0
        assert all(a.resolution == 1 for a in res1_atlases)

    def test_list_atlases_combined_filters(self):
        """list_atlases() must support multiple filters."""
        from lacuna.assets.atlases.registry import list_atlases

        filtered = list_atlases(space="MNI152NLin6Asym", resolution=1)

        assert all(a.space == "MNI152NLin6Asym" and a.resolution == 1 for a in filtered)


class TestAtlasRegistrationContract:
    """Contract tests for user atlas registration."""

    def test_register_atlas_function_exists(self):
        """register_atlas() function must exist."""
        from lacuna.assets.atlases.registry import register_atlas

        assert callable(register_atlas)

    def test_register_atlas_from_files_function_exists(self):
        """register_atlas_from_files() function must exist."""
        from lacuna.assets.atlases.registry import register_atlas_from_files

        assert callable(register_atlas_from_files)

    def test_unregister_atlas_function_exists(self):
        """unregister_atlas() function must exist."""
        from lacuna.assets.atlases.registry import unregister_atlas

        assert callable(unregister_atlas)

    def test_register_atlas_adds_to_registry(self, tmp_path):
        """register_atlas() must add atlas to ATLAS_REGISTRY."""
        from lacuna.assets.atlases.registry import (
            ATLAS_REGISTRY,
            AtlasMetadata,
            register_atlas,
            unregister_atlas,
        )

        # Create dummy files
        atlas_file = tmp_path / "test_atlas.nii.gz"
        labels_file = tmp_path / "test_labels.txt"
        atlas_file.touch()
        labels_file.touch()

        initial_count = len(ATLAS_REGISTRY)

        metadata = AtlasMetadata(
            name="TestAtlas_Registration",
            space="MNI152NLin6Asym",
            resolution=1,
            description="Test atlas for registration",
            atlas_filename=str(atlas_file),
            labels_filename=str(labels_file),
        )

        try:
            register_atlas(metadata)
            assert "TestAtlas_Registration" in ATLAS_REGISTRY
            assert len(ATLAS_REGISTRY) == initial_count + 1
        finally:
            # Cleanup
            if "TestAtlas_Registration" in ATLAS_REGISTRY:
                unregister_atlas("TestAtlas_Registration")

    def test_register_atlas_duplicate_name_raises_error(self):
        """register_atlas() must raise ValueError for duplicate names."""
        from lacuna.assets.atlases.registry import AtlasMetadata, register_atlas

        # Try to register with name that already exists
        metadata = AtlasMetadata(
            name="Schaefer2018_100Parcels7Networks",  # Already exists
            space="MNI152NLin6Asym",
            resolution=1,
            description="Duplicate",
            atlas_filename="/fake/path.nii.gz",
            labels_filename="/fake/labels.txt",
        )

        with pytest.raises(ValueError, match="already registered"):
            register_atlas(metadata)

    def test_unregister_atlas_removes_from_registry(self, tmp_path):
        """unregister_atlas() must remove atlas from registry."""
        from lacuna.assets.atlases.registry import (
            ATLAS_REGISTRY,
            AtlasMetadata,
            register_atlas,
            unregister_atlas,
        )

        # Create and register
        atlas_file = tmp_path / "test_atlas2.nii.gz"
        labels_file = tmp_path / "test_labels2.txt"
        atlas_file.touch()
        labels_file.touch()

        metadata = AtlasMetadata(
            name="TestAtlas_Unregister",
            space="MNI152NLin6Asym",
            resolution=1,
            description="Test",
            atlas_filename=str(atlas_file),
            labels_filename=str(labels_file),
        )

        register_atlas(metadata)
        assert "TestAtlas_Unregister" in ATLAS_REGISTRY

        # Unregister
        unregister_atlas("TestAtlas_Unregister")
        assert "TestAtlas_Unregister" not in ATLAS_REGISTRY

    def test_unregister_nonexistent_atlas_raises_error(self):
        """unregister_atlas() must raise KeyError for unknown atlas."""
        from lacuna.assets.atlases.registry import unregister_atlas

        with pytest.raises(KeyError):
            unregister_atlas("NonexistentAtlas_12345")
