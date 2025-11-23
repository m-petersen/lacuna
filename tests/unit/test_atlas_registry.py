"""Unit tests for atlas registry functionality.

Tests atlas metadata, registry operations, and atlas listing/filtering.
"""

import pytest

from lacuna.assets.parcellations import (
    PARCELLATION_REGISTRY,
    ParcellationMetadata,
    list_parcellations,
    register_parcellation,
)


def test_atlas_metadata_creation():
    """Test creating ParcellationMetadata with required fields."""
    metadata = ParcellationMetadata(
        name="TestAtlas",
        space="MNI152NLin6Asym",
        resolution=2,
        atlas_filename="test_atlas.nii.gz",
        labels_filename="test_atlas_labels.txt",
        n_regions=50,
        description="Test atlas for unit testing",
    )

    assert metadata.name == "TestAtlas"
    assert metadata.description == "Test atlas for unit testing"
    assert metadata.space == "MNI152NLin6Asym"
    assert metadata.resolution == 2
    assert metadata.n_regions == 50
    assert metadata.atlas_filename == "test_atlas.nii.gz"
    assert metadata.labels_filename == "test_atlas_labels.txt"


def test_atlas_metadata_bundled():
    """Test creating bundled atlas metadata."""
    metadata = ParcellationMetadata(
        name="BundledAtlas",
        space="MNI152NLin6Asym",
        resolution=1,
        atlas_filename="bundled_atlas.nii.gz",
        labels_filename="bundled_atlas_labels.txt",
        n_regions=100,
        description="Bundled atlas",
    )

    assert metadata.name == "BundledAtlas"
    assert metadata.description == "Bundled atlas"
    assert metadata.n_regions == 100


def test_atlas_registry_has_bundled_atlases():
    """Test that registry contains expected bundled atlases."""
    # Check for Schaefer atlases (actual names with 2018 and full parcel count)
    assert "Schaefer2018_100Parcels7Networks" in PARCELLATION_REGISTRY
    assert "Schaefer2018_200Parcels7Networks" in PARCELLATION_REGISTRY
    assert "Schaefer2018_400Parcels7Networks" in PARCELLATION_REGISTRY
    assert "Schaefer2018_1000Parcels7Networks" in PARCELLATION_REGISTRY

    # Check for Tian atlases (actual names with full details)
    assert "TianSubcortex_3TS1" in PARCELLATION_REGISTRY
    assert "TianSubcortex_3TS2" in PARCELLATION_REGISTRY
    assert "TianSubcortex_3TS3" in PARCELLATION_REGISTRY

    # Check for HCP atlas (actual name with threshold)
    assert "HCP1065_thr0p1" in PARCELLATION_REGISTRY


def test_atlas_registry_metadata_validity():
    """Test that all registered atlases have valid metadata."""
    for name, metadata in PARCELLATION_REGISTRY.items():
        assert isinstance(metadata, ParcellationMetadata)
        assert metadata.name == name
        assert len(metadata.description) > 0
        assert len(metadata.space) > 0
        assert metadata.resolution > 0
        assert metadata.n_regions is None or metadata.n_regions > 0
        assert isinstance(metadata.is_4d, bool)


def test_list_parcellations_all():
    """Test listing all atlases without filters."""
    atlases = list_parcellations()

    assert len(atlases) >= 8  # At least 8 bundled atlases
    assert all(isinstance(a, ParcellationMetadata) for a in atlases)

    # Check they're sorted by name
    names = [a.name for a in atlases]
    assert names == sorted(names)


def test_list_parcellations_filter_by_space():
    """Test filtering atlases by coordinate space."""
    nlin6_atlases = list_parcellations(space="MNI152NLin6Asym")

    assert len(nlin6_atlases) > 0
    assert all(a.space == "MNI152NLin6Asym" for a in nlin6_atlases)

    # Should include Schaefer atlases
    names = [a.name for a in nlin6_atlases]
    assert "Schaefer2018_100Parcels7Networks" in names


def test_list_parcellations_filter_by_resolution():
    """Test filtering atlases by resolution."""
    res1_atlases = list_parcellations(resolution=1)

    assert len(res1_atlases) > 0
    assert all(a.resolution == 1 for a in res1_atlases)

    # Schaefer atlases are at 1mm resolution
    names = [a.name for a in res1_atlases]
    assert any("Schaefer" in name for name in names)


def test_list_parcellations_check_region_counts():
    """Test that atlases have region count information."""
    atlases = list_parcellations()

    # Check that region count info exists for some atlases
    has_region_info = [a for a in atlases if a.n_regions is not None]
    assert len(has_region_info) > 0

    # Verify counts are reasonable
    for atlas in has_region_info:
        assert atlas.n_regions > 0
        assert atlas.n_regions < 10000  # Sanity check


def test_list_parcellations_combined_filters():
    """Test combining multiple filters."""
    # Bundled atlases in NLin6Asym space at 1mm resolution
    filtered = list_parcellations(space="MNI152NLin6Asym", resolution=1)

    # Should find bundled atlases that match both filters
    assert len(filtered) > 0
    assert all(a.space == "MNI152NLin6Asym" and a.resolution == 1 for a in filtered)

    # Should include Schaefer atlases
    names = {a.name for a in filtered}
    assert any("Schaefer" in name for name in names)


def test_register_parcellation():
    """Test registering a custom atlas."""
    # Create custom atlas metadata
    custom = ParcellationMetadata(
        name="CustomTestAtlas",
        space="MNI152NLin6Asym",
        resolution=2,
        atlas_filename="custom_test_atlas.nii.gz",
        labels_filename="custom_test_atlas_labels.txt",
        n_regions=25,
        description="Custom atlas for testing",
    )

    # Register it
    register_parcellation(custom)

    # Verify it's in the registry
    assert "CustomTestAtlas" in PARCELLATION_REGISTRY
    assert PARCELLATION_REGISTRY["CustomTestAtlas"] == custom

    # Verify it appears in listings
    atlases = list_parcellations()
    names = [a.name for a in atlases]
    assert "CustomTestAtlas" in names

    # Clean up
    del PARCELLATION_REGISTRY["CustomTestAtlas"]


def test_register_parcellation_overwrites_with_warning():
    """Test that registering an existing atlas name raises ValueError."""
    # Create atlas with same name as existing one
    duplicate = ParcellationMetadata(
        name="Schaefer2018_100Parcels7Networks",
        space="MNI152NLin6Asym",
        resolution=2,
        atlas_filename="duplicate_atlas.nii.gz",
        labels_filename="duplicate_labels.txt",
        n_regions=100,
        description="Duplicate for testing",
    )

    # Register should raise ValueError
    with pytest.raises(ValueError, match="already registered"):
        register_parcellation(duplicate)


def test_schaefer_atlas_metadata():
    """Test specific Schaefer atlas metadata."""
    schaefer400 = PARCELLATION_REGISTRY["Schaefer2018_400Parcels7Networks"]

    assert schaefer400.name == "Schaefer2018_400Parcels7Networks"
    assert "Schaefer" in schaefer400.atlas_filename
    assert schaefer400.space == "MNI152NLin6Asym"
    assert schaefer400.resolution == 1
    assert schaefer400.n_regions == 400
    assert schaefer400.atlas_filename is not None
    assert "Schaefer2018" in schaefer400.atlas_filename


def test_tian_atlas_metadata():
    """Test specific Tian atlas metadata."""
    tian_s2 = PARCELLATION_REGISTRY["TianSubcortex_3TS2"]

    assert tian_s2.name == "TianSubcortex_3TS2"
    assert "Tian" in tian_s2.atlas_filename
    assert tian_s2.space == "MNI152NLin6Asym"
    assert tian_s2.resolution == 1
    assert tian_s2.n_regions == 32
    assert tian_s2.atlas_filename is not None


def test_hcp_atlas_metadata():
    """Test HCP1065 atlas metadata."""
    hcp = PARCELLATION_REGISTRY["HCP1065_thr0p1"]

    assert hcp.name == "HCP1065_thr0p1"
    assert "HCP" in hcp.atlas_filename or "White Matter" in hcp.atlas_filename
    assert hcp.space == "MNI152NLin2009aAsym"
    assert hcp.resolution == 1
    assert hcp.n_regions == 64  # HCP white matter atlas has 64 tracts
    assert hcp.atlas_filename is not None
    assert hcp.is_4d is True  # HCP is a 4D probabilistic atlas


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
