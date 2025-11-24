"""TDD tests for atlas->parcellation refactoring.

These tests define the expected API after refactoring:
- Module: src/lacuna/assets/parcellations/
- Classes: Parcellation, ParcellationMetadata
- Functions: load_parcellation(), list_parcellations(), register_parcellation()
- Constants: PARCELLATION_REGISTRY, BUNDLED_PARCELLATIONS_DIR

All tests will fail initially - implementation follows in T162-T163.
"""


from pathlib import Path

import nibabel as nib
import numpy as np
import pytest


class TestParcellationModuleImports:
    """Test that parcellation module exists with correct structure."""

    def test_parcellation_module_exists(self):
        """Test that assets.parcellations module can be imported."""
        from lacuna.assets import parcellations

        assert parcellations is not None

    def test_parcellation_class_exists(self):
        """Test that Parcellation class exists."""
        from lacuna.assets.parcellations import Parcellation

        assert Parcellation is not None

    def test_parcellation_metadata_class_exists(self):
        """Test that ParcellationMetadata class exists."""
        from lacuna.assets.parcellations import ParcellationMetadata

        assert ParcellationMetadata is not None

    def test_load_parcellation_function_exists(self):
        """Test that load_parcellation() function exists."""
        from lacuna.assets.parcellations import load_parcellation

        assert callable(load_parcellation)

    def test_list_parcellations_function_exists(self):
        """Test that list_parcellations() function exists."""
        from lacuna.assets.parcellations import list_parcellations

        assert callable(list_parcellations)

    def test_register_parcellation_function_exists(self):
        """Test that register_parcellation() function exists."""
        from lacuna.assets.parcellations import register_parcellation

        assert callable(register_parcellation)

    def test_parcellation_registry_constant_exists(self):
        """Test that PARCELLATION_REGISTRY exists."""
        from lacuna.assets.parcellations import PARCELLATION_REGISTRY

        assert PARCELLATION_REGISTRY is not None
        assert isinstance(PARCELLATION_REGISTRY, dict)

    def test_bundled_parcellations_dir_exists(self):
        """Test that BUNDLED_PARCELLATIONS_DIR exists."""
        from lacuna.assets.parcellations import BUNDLED_PARCELLATIONS_DIR

        assert BUNDLED_PARCELLATIONS_DIR is not None
        assert isinstance(BUNDLED_PARCELLATIONS_DIR, Path)


class TestParcellationMetadataStructure:
    """Test ParcellationMetadata class structure."""

    def test_parcellation_metadata_has_required_fields(self):
        """Test that ParcellationMetadata has all required fields."""
        from lacuna.assets.parcellations import ParcellationMetadata

        metadata = ParcellationMetadata(
            name="TestParcellation",
            space="MNI152NLin6Asym",
            resolution=2.0,
            description="Test parcellation",
            parcellation_filename="test.nii.gz",
            labels_filename="test_labels.txt",
            region_labels=["Region1", "Region2"],
        )

        assert metadata.name == "TestParcellation"
        assert metadata.space == "MNI152NLin6Asym"
        assert metadata.resolution == 2.0
        assert metadata.description == "Test parcellation"
        assert metadata.parcellation_filename == "test.nii.gz"
        assert metadata.labels_filename == "test_labels.txt"
        assert metadata.region_labels == ["Region1", "Region2"]

    def test_parcellation_metadata_supports_4d(self):
        """Test that ParcellationMetadata supports is_4d field."""
        from lacuna.assets.parcellations import ParcellationMetadata

        metadata_4d = ParcellationMetadata(
            name="Test4D",
            space="MNI152NLin6Asym",
            resolution=2.0,
            description="4D parcellation",
            parcellation_filename="test_4d.nii.gz",
            labels_filename="test_4d_labels.txt",
            region_labels=["Tract1", "Tract2"],
            is_4d=True,
        )

        assert metadata_4d.is_4d is True


class TestParcellationFunctionSignatures:
    """Test function signatures match expected API."""

    def test_load_parcellation_signature(self, tmp_path):
        """Test load_parcellation() accepts name or path."""
        from lacuna.assets.parcellations import load_parcellation, register_parcellation

        # Create minimal test parcellation
        data = np.arange(1, 28).reshape(3, 3, 3)
        img = nib.Nifti1Image(data, affine=np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(img, path)

        # Register it
        register_parcellation(
            name="TestLoad",
            file_path=path,
            space="MNI152NLin6Asym",
            resolution="2mm",
            description="Test",
            regions=["R1", "R2"],
        )

        # Load by name
        parc_by_name = load_parcellation("TestLoad")
        assert parc_by_name is not None

        # Load by path
        parc_by_path = load_parcellation(str(path))
        assert parc_by_path is not None

    def test_register_parcellation_signature(self, tmp_path):
        """Test register_parcellation() has correct parameters."""
        from lacuna.assets.parcellations import register_parcellation

        data = np.arange(1, 28).reshape(3, 3, 3)
        img = nib.Nifti1Image(data, affine=np.eye(4))
        path = tmp_path / "test_register.nii.gz"
        nib.save(img, path)

        # Should accept these parameters
        register_parcellation(
            name="TestRegister",
            file_path=path,
            space="MNI152NLin6Asym",
            resolution="2mm",
            description="Test parcellation",
            regions=["Region1", "Region2", "Region3"],
        )

        # Verify it's registered
        from lacuna.assets.parcellations import PARCELLATION_REGISTRY

        assert "TestRegister" in PARCELLATION_REGISTRY

    def test_list_parcellations_returns_names(self):
        """Test list_parcellations() returns list of metadata objects."""
        from lacuna.assets.parcellations import ParcellationMetadata, list_parcellations

        parcellations = list_parcellations()
        assert isinstance(parcellations, list)
        assert len(parcellations) > 0  # Should have bundled parcellations
        assert all(isinstance(p, ParcellationMetadata) for p in parcellations)


class TestParcellationClassFunctionality:
    """Test Parcellation class functionality."""

    def test_parcellation_has_data_attribute(self, tmp_path):
        """Test Parcellation class provides access to image data."""
        from lacuna.assets.parcellations import Parcellation, ParcellationMetadata

        data = np.arange(1, 28).reshape(3, 3, 3)
        img = nib.Nifti1Image(data, affine=np.eye(4))

        metadata = ParcellationMetadata(
            name="TestData",
            space="MNI152NLin6Asym",
            resolution=2.0,
            description="Test",
            parcellation_filename="test.nii.gz",
            labels_filename="test_labels.txt",
            region_labels=["R1"],
        )

        parc = Parcellation(img=img, labels={1: "R1"}, metadata=metadata)

        assert hasattr(parc, "image")
        assert parc.image.shape == (3, 3, 3)
        # Test data access through image
        assert hasattr(parc.image, "get_fdata")

    def test_parcellation_has_metadata_attribute(self, tmp_path):
        """Test Parcellation class provides metadata access."""
        from lacuna.assets.parcellations import Parcellation, ParcellationMetadata

        data = np.arange(1, 28).reshape(3, 3, 3)
        img = nib.Nifti1Image(data, affine=np.eye(4))

        metadata = ParcellationMetadata(
            name="TestMeta",
            space="MNI152NLin6Asym",
            resolution=2.0,
            description="Test metadata",
            parcellation_filename="test_meta.nii.gz",
            labels_filename="test_meta_labels.txt",
            region_labels=["R1", "R2"],
        )

        parc = Parcellation(img=img, labels={1: "R1", 2: "R2"}, metadata=metadata)

        assert hasattr(parc, "metadata")
        assert parc.metadata.name == "TestMeta"
        assert parc.metadata.region_labels == ["R1", "R2"]


class TestRegionalDamageLogLevel:
    """Test RegionalDamage accepts log_level parameter (T164)."""

    def test_regional_damage_accepts_log_level(self):
        """Test RegionalDamage __init__ accepts log_level parameter."""
        from lacuna.analysis import RegionalDamage

        # Should not raise
        analysis = RegionalDamage(threshold=0.5, log_level=2)
        assert analysis.log_level == 2

    def test_regional_damage_log_level_defaults_to_1(self):
        """Test RegionalDamage log_level defaults to 1."""
        from lacuna.analysis import RegionalDamage

        analysis = RegionalDamage(threshold=0.5)
        assert analysis.log_level == 1

    def test_regional_damage_passes_log_level_to_parent(self):
        """Test RegionalDamage passes log_level to ParcelAggregation."""
        from lacuna.analysis import RegionalDamage

        analysis = RegionalDamage(threshold=0.5, log_level=0)
        # Verify it's accessible (ParcelAggregation inherits from BaseAnalysis)
        assert hasattr(analysis, "log_level")
        assert analysis.log_level == 0


class TestParcelAggregationParcellationParam:
    """Test ParcelAggregation uses parcellation_names parameter (T163)."""

    def test_parcel_aggregation_accepts_parcellation_names(self):
        """Test ParcelAggregation accepts parcellation_names parameter."""
        from lacuna.analysis import ParcelAggregation

        # Should not raise
        analysis = ParcelAggregation(
            source="mask_img",
            aggregation="mean",
            parcellation_names=["Schaefer2018_100Parcels_7Networks"],
        )

        assert hasattr(analysis, "parcellation_names")
        assert analysis.parcellation_names == ["Schaefer2018_100Parcels_7Networks"]

    def test_parcel_aggregation_parcellation_names_optional(self):
        """Test parcellation_names is optional (defaults to all)."""
        from lacuna.analysis import ParcelAggregation

        analysis = ParcelAggregation(source="mask_img", aggregation="mean")

        # Should use all available parcellations
        assert hasattr(analysis, "parcellation_names")


class TestNetworkMappingParcellationParam:
    """Test network mapping analyses use parcellation_name parameter (T165)."""

    def test_structural_network_mapping_parcellation_name(self, tmp_path):
        """Test StructuralNetworkMapping accepts parcellation_name."""
        from lacuna.analysis import StructuralNetworkMapping

        # Create minimal test connectome
        connectome_path = tmp_path / "test.h5"
        connectome_path.write_text("")  # Minimal file

        # Should accept parcellation_name parameter
        analysis = StructuralNetworkMapping(
            connectome_name="test",
            parcellation_name="Schaefer2018_100Parcels_7Networks",
        )

        assert hasattr(analysis, "parcellation_name")
        assert analysis.parcellation_name == "Schaefer2018_100Parcels_7Networks"

    def test_functional_network_mapping_parcellation_name(self):
        """Test FunctionalNetworkMapping accepts parcellation_name."""
        from lacuna.analysis import FunctionalNetworkMapping

        # Should accept parcellation_name parameter
        analysis = FunctionalNetworkMapping(
            parcellation_name="Schaefer2018_100Parcels_7Networks",
        )

        assert hasattr(analysis, "parcellation_name")
        assert analysis.parcellation_name == "Schaefer2018_100Parcels_7Networks"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
