"""Contract tests for atlas registry and loader.

Tests the expected interface and behavior of the atlas registry system,
ensuring bundled atlases are properly registered with metadata and can be
loaded via the DataAssetManager integration.
"""

from dataclasses import is_dataclass

import nibabel as nib
import pytest


class TestParcellationMetadataContract:
    """Contract tests for ParcellationMetadata dataclass."""

    def test_atlas_metadata_is_dataclass(self):
        """ParcellationMetadata must be a dataclass."""
        from lacuna.assets.parcellations.registry import ParcellationMetadata

        assert is_dataclass(ParcellationMetadata)

    def test_atlas_metadata_required_fields(self):
        """ParcellationMetadata must have required fields."""
        from lacuna.assets.parcellations.registry import ParcellationMetadata

        # Should be able to create with minimal required fields
        metadata = ParcellationMetadata(
            name="TestAtlas",
            space="MNI152NLin6Asym",
            resolution=1,
            description="Test atlas",
            parcellation_filename="test.nii.gz",
            labels_filename="test_labels.txt",
        )

        assert metadata.name == "TestAtlas"
        assert metadata.space == "MNI152NLin6Asym"
        assert metadata.resolution == 1
        assert metadata.description == "Test atlas"
        assert metadata.parcellation_filename == "test.nii.gz"
        assert metadata.labels_filename == "test_labels.txt"

    def test_atlas_metadata_optional_fields(self):
        """ParcellationMetadata should have optional citation and network fields."""
        from lacuna.assets.parcellations.registry import ParcellationMetadata

        metadata = ParcellationMetadata(
            name="TestAtlas",
            space="MNI152NLin6Asym",
            resolution=1,
            description="Test atlas",
            parcellation_filename="test.nii.gz",
            labels_filename="test_labels.txt",
            citation="Test et al. (2024)",
            networks=["Visual", "Motor"],
            n_regions=100,
        )

        assert metadata.citation == "Test et al. (2024)"
        assert metadata.networks == ["Visual", "Motor"]
        assert metadata.n_regions == 100


class TestAtlasRegistryContract:
    """Contract tests for PARCELLATION_REGISTRY."""

    def test_atlas_registry_exists(self):
        """PARCELLATION_REGISTRY constant must exist."""
        from lacuna.assets.parcellations.registry import PARCELLATION_REGISTRY

        assert PARCELLATION_REGISTRY is not None
        assert isinstance(PARCELLATION_REGISTRY, dict)

    def test_registry_contains_bundled_atlases(self):
        """Registry must contain all bundled atlases."""
        from lacuna.assets.parcellations.registry import PARCELLATION_REGISTRY

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
            assert atlas_name in PARCELLATION_REGISTRY, f"Missing atlas: {atlas_name}"

    def test_registry_entries_have_metadata(self):
        """Each registry entry must be an ParcellationMetadata instance."""
        from lacuna.assets.parcellations.registry import PARCELLATION_REGISTRY, ParcellationMetadata

        for atlas_name, metadata in PARCELLATION_REGISTRY.items():
            assert isinstance(
                metadata, ParcellationMetadata
            ), f"{atlas_name} metadata is not ParcellationMetadata"
            assert metadata.name == atlas_name
            assert metadata.space is not None
            assert metadata.resolution is not None
            assert metadata.parcellation_filename is not None
            assert metadata.labels_filename is not None

    def test_schaefer_atlases_metadata(self):
        """Schaefer atlases must have correct metadata."""
        from lacuna.assets.parcellations.registry import PARCELLATION_REGISTRY

        schaefer_100 = PARCELLATION_REGISTRY["Schaefer2018_100Parcels7Networks"]
        assert schaefer_100.space == "MNI152NLin6Asym"
        assert schaefer_100.resolution == 1
        assert schaefer_100.n_regions == 100
        assert "7" in schaefer_100.name or len(schaefer_100.networks) == 7

        schaefer_400 = PARCELLATION_REGISTRY["Schaefer2018_400Parcels7Networks"]
        assert schaefer_400.n_regions == 400

    def test_tian_atlases_metadata(self):
        """Tian subcortical atlases must have correct metadata."""
        from lacuna.assets.parcellations.registry import PARCELLATION_REGISTRY

        tian_s1 = PARCELLATION_REGISTRY["TianSubcortex_3TS1"]
        assert tian_s1.space == "MNI152NLin6Asym"
        assert tian_s1.resolution == 1
        assert "subcort" in tian_s1.description.lower()

    def test_hcp_atlas_metadata(self):
        """HCP white matter atlas must have correct metadata."""
        from lacuna.assets.parcellations.registry import PARCELLATION_REGISTRY

        hcp = PARCELLATION_REGISTRY["HCP1065_thr0p1"]
        assert hcp.space == "MNI152NLin2009aAsym"
        assert hcp.resolution == 1
        assert "white matter" in hcp.description.lower() or "tract" in hcp.description.lower()


class TestAtlasLoaderContract:
    """Contract tests for atlas loading functionality."""

    def test_load_parcellation_function_exists(self):
        """load_parcellation() function must exist."""
        from lacuna.assets.parcellations.loader import load_parcellation

        assert callable(load_parcellation)

    def test_load_parcellation_by_name(self):
        """load_parcellation() must accept atlas name from registry."""
        from lacuna.assets.parcellations.loader import load_parcellation

        # Should not raise exception for valid atlas
        atlas = load_parcellation("Schaefer2018_100Parcels7Networks")

        assert atlas is not None
        assert hasattr(atlas, "image"), "Atlas must have 'image' attribute"
        assert hasattr(atlas, "labels"), "Atlas must have 'labels' attribute"
        assert hasattr(atlas, "metadata"), "Atlas must have 'metadata' attribute"

    def test_loaded_atlas_image_is_nifti(self):
        """Loaded atlas image must be a nibabel Nifti1Image."""
        from lacuna.assets.parcellations.loader import load_parcellation

        atlas = load_parcellation("Schaefer2018_100Parcels7Networks")

        assert isinstance(atlas.image, nib.Nifti1Image)
        assert atlas.image.shape[0] > 0  # Has valid dimensions

    def test_loaded_atlas_labels_is_dict(self):
        """Loaded atlas labels must be a dict mapping region_id -> name."""
        from lacuna.assets.parcellations.loader import load_parcellation

        atlas = load_parcellation("Schaefer2018_100Parcels7Networks")

        assert isinstance(atlas.labels, dict)
        assert len(atlas.labels) > 0
        # Check format: int keys, string values
        for region_id, region_name in atlas.labels.items():
            assert isinstance(region_id, int)
            assert isinstance(region_name, str)

    def test_loaded_atlas_metadata_matches_registry(self):
        """Loaded atlas metadata must match registry entry."""
        from lacuna.assets.parcellations.loader import load_parcellation
        from lacuna.assets.parcellations.registry import PARCELLATION_REGISTRY

        atlas_name = "Schaefer2018_100Parcels7Networks"
        atlas = load_parcellation(atlas_name)
        expected_metadata = PARCELLATION_REGISTRY[atlas_name]

        assert atlas.metadata.name == expected_metadata.name
        assert atlas.metadata.space == expected_metadata.space
        assert atlas.metadata.resolution == expected_metadata.resolution

    def test_load_parcellation_invalid_name_raises_error(self):
        """load_parcellation() must raise KeyError for unknown atlas."""
        from lacuna.assets.parcellations.loader import load_parcellation

        with pytest.raises((KeyError, ValueError)):
            load_parcellation("NonexistentAtlas")

    def test_load_parcellation_uses_asset_manager(self):
        """load_parcellation() should work without requiring external dependencies."""
        from lacuna.assets.parcellations.loader import load_parcellation

        # Should load bundled atlas without any external parameters
        atlas = load_parcellation("Schaefer2018_100Parcels7Networks")

        assert atlas is not None


class TestAtlasListingContract:
    """Contract tests for atlas discovery and listing."""

    def test_list_parcellations_function_exists(self):
        """list_parcellations() function must exist."""
        from lacuna.assets.parcellations.registry import list_parcellations

        assert callable(list_parcellations)

    def test_list_parcellations_returns_all_registered(self):
        """list_parcellations() must return all registered atlases by default."""
        from lacuna.assets.parcellations.registry import list_parcellations

        atlases = list_parcellations()

        assert len(atlases) >= 8  # At least the bundled atlases
        assert all(hasattr(a, "name") for a in atlases)

    def test_list_parcellations_filter_by_space(self):
        """list_parcellations() must support filtering by space."""
        from lacuna.assets.parcellations.registry import list_parcellations

        mni6_atlases = list_parcellations(space="MNI152NLin6Asym")

        assert len(mni6_atlases) > 0
        assert all(a.space == "MNI152NLin6Asym" for a in mni6_atlases)

    def test_list_parcellations_filter_by_resolution(self):
        """list_parcellations() must support filtering by resolution."""
        from lacuna.assets.parcellations.registry import list_parcellations

        res1_atlases = list_parcellations(resolution=1)

        assert len(res1_atlases) > 0
        assert all(a.resolution == 1 for a in res1_atlases)

    def test_list_parcellations_combined_filters(self):
        """list_parcellations() must support multiple filters."""
        from lacuna.assets.parcellations.registry import list_parcellations

        filtered = list_parcellations(space="MNI152NLin6Asym", resolution=1)

        assert all(a.space == "MNI152NLin6Asym" and a.resolution == 1 for a in filtered)


class TestAtlasRegistrationContract:
    """Contract tests for user atlas registration."""

    def test_register_parcellation_function_exists(self):
        """register_parcellation() function must exist."""
        from lacuna.assets.parcellations.registry import register_parcellation

        assert callable(register_parcellation)

    def test_register_parcellation_from_files_function_exists(self):
        """register_parcellation_from_files() function must exist."""
        from lacuna.assets.parcellations.registry import register_parcellation_from_files

        assert callable(register_parcellation_from_files)

    def test_unregister_parcellation_function_exists(self):
        """unregister_parcellation() function must exist."""
        from lacuna.assets.parcellations.registry import unregister_parcellation

        assert callable(unregister_parcellation)

    def test_register_parcellation_adds_to_registry(self, tmp_path):
        """register_parcellation() must add atlas to PARCELLATION_REGISTRY."""
        from lacuna.assets.parcellations.registry import (
            PARCELLATION_REGISTRY,
            ParcellationMetadata,
            register_parcellation,
            unregister_parcellation,
        )

        # Create dummy files
        atlas_file = tmp_path / "test_atlas.nii.gz"
        labels_file = tmp_path / "test_labels.txt"
        atlas_file.touch()
        labels_file.touch()

        initial_count = len(PARCELLATION_REGISTRY)

        metadata = ParcellationMetadata(
            name="TestAtlas_Registration",
            space="MNI152NLin6Asym",
            resolution=1,
            description="Test atlas for registration",
            parcellation_filename=str(atlas_file),
            labels_filename=str(labels_file),
        )

        try:
            register_parcellation(metadata)
            assert "TestAtlas_Registration" in PARCELLATION_REGISTRY
            assert len(PARCELLATION_REGISTRY) == initial_count + 1
        finally:
            # Cleanup
            if "TestAtlas_Registration" in PARCELLATION_REGISTRY:
                unregister_parcellation("TestAtlas_Registration")

    def test_register_parcellation_duplicate_name_raises_error(self):
        """register_parcellation() must raise ValueError for duplicate names."""
        from lacuna.assets.parcellations.registry import ParcellationMetadata, register_parcellation

        # Try to register with name that already exists
        metadata = ParcellationMetadata(
            name="Schaefer2018_100Parcels7Networks",  # Already exists
            space="MNI152NLin6Asym",
            resolution=1,
            description="Duplicate",
            parcellation_filename="/fake/path.nii.gz",
            labels_filename="/fake/labels.txt",
        )

        with pytest.raises(ValueError, match="already registered"):
            register_parcellation(metadata)

    def test_unregister_parcellation_removes_from_registry(self, tmp_path):
        """unregister_parcellation() must remove atlas from registry."""
        from lacuna.assets.parcellations.registry import (
            PARCELLATION_REGISTRY,
            ParcellationMetadata,
            register_parcellation,
            unregister_parcellation,
        )

        # Create and register
        atlas_file = tmp_path / "test_atlas2.nii.gz"
        labels_file = tmp_path / "test_labels2.txt"
        atlas_file.touch()
        labels_file.touch()

        metadata = ParcellationMetadata(
            name="TestAtlas_Unregister",
            space="MNI152NLin6Asym",
            resolution=1,
            description="Test",
            parcellation_filename=str(atlas_file),
            labels_filename=str(labels_file),
        )

        register_parcellation(metadata)
        assert "TestAtlas_Unregister" in PARCELLATION_REGISTRY

        # Unregister
        unregister_parcellation("TestAtlas_Unregister")
        assert "TestAtlas_Unregister" not in PARCELLATION_REGISTRY

    def test_unregister_nonexistent_atlas_raises_error(self):
        """unregister_parcellation() must raise KeyError for unknown atlas."""
        from lacuna.assets.parcellations.registry import unregister_parcellation

        with pytest.raises(KeyError):
            unregister_parcellation("NonexistentAtlas_12345")
