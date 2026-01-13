"""Contract tests for structural connectome registry.

These tests define the expected behavior of structural connectome management.
All tests should pass after implementation is complete.

Note: TDI is no longer pre-provided - it's computed on-the-fly during analysis
with optional caching (cache_tdi parameter in StructuralNetworkMapping).
"""

from pathlib import Path

import pytest


def test_structural_connectome_can_import():
    """Test that structural connectome functions can be imported."""
    from lacuna.assets.connectomes import (
        list_structural_connectomes,
        load_structural_connectome,
        register_structural_connectome,
        unregister_structural_connectome,
    )

    assert callable(register_structural_connectome)
    assert callable(unregister_structural_connectome)
    assert callable(list_structural_connectomes)
    assert callable(load_structural_connectome)


def test_register_structural_connectome_with_valid_files(tmp_path):
    """Test registering a structural connectome with valid files."""
    from lacuna.assets.connectomes import (
        list_structural_connectomes,
        register_structural_connectome,
        unregister_structural_connectome,
    )

    # Create test files
    tractogram = tmp_path / "test_tractogram.tck"
    tractogram.touch()

    # Register connectome (TDI computed on-the-fly during analysis)
    register_structural_connectome(
        name="test_structural",
        space="MNI152NLin2009cAsym",
        tractogram_path=tractogram,
        description="Test structural connectome",
    )

    try:
        # Check it was registered
        connectomes = list_structural_connectomes()
        names = [c.name for c in connectomes]
        assert "test_structural" in names
    finally:
        unregister_structural_connectome("test_structural")


def test_register_structural_connectome_missing_tractogram(tmp_path):
    """Test that registration fails if tractogram file doesn't exist."""
    from lacuna.assets.connectomes import register_structural_connectome

    with pytest.raises(FileNotFoundError, match="Tractogram file not found"):
        register_structural_connectome(
            name="test_missing_tractogram",
            space="MNI152NLin2009cAsym",
            tractogram_path=tmp_path / "missing.tck",
        )


def test_register_structural_connectome_invalid_tractogram_extension(tmp_path):
    """Test that registration fails if tractogram has wrong extension."""
    from lacuna.assets.connectomes import register_structural_connectome

    # Create file with wrong extension
    tractogram = tmp_path / "test_tractogram.txt"
    tractogram.touch()

    with pytest.raises(ValueError, match="Expected .tck file"):
        register_structural_connectome(
            name="test_wrong_ext",
            space="MNI152NLin2009cAsym",
            tractogram_path=tractogram,
        )


def test_register_structural_connectome_with_template(tmp_path):
    """Test registering a structural connectome with optional template."""
    from lacuna.assets.connectomes import (
        load_structural_connectome,
        register_structural_connectome,
        unregister_structural_connectome,
    )

    # Create test files
    tractogram = tmp_path / "test_tractogram.tck"
    template = tmp_path / "test_template.nii.gz"
    tractogram.touch()
    template.touch()

    # Register with template
    register_structural_connectome(
        name="test_with_template",
        space="MNI152NLin2009cAsym",
        tractogram_path=tractogram,
        template_path=template,
    )

    try:
        # Load and check template was included
        connectome = load_structural_connectome("test_with_template")
        assert connectome.template_path is not None
        assert connectome.template_path.exists()
    finally:
        unregister_structural_connectome("test_with_template")


def test_list_structural_connectomes_returns_list(tmp_path):
    """Test that list_structural_connectomes returns a list."""
    from lacuna.assets.connectomes import list_structural_connectomes

    result = list_structural_connectomes()

    assert isinstance(result, list)


def test_list_structural_connectomes_filter_by_space(tmp_path):
    """Test filtering structural connectomes by space."""
    from lacuna.assets.connectomes import (
        list_structural_connectomes,
        register_structural_connectome,
        unregister_structural_connectome,
    )

    # Create test file
    tractogram = tmp_path / "test_tractogram.tck"
    tractogram.touch()

    # Register connectomes in different spaces
    register_structural_connectome(
        name="test_space_nlin6",
        space="MNI152NLin6Asym",
        tractogram_path=tractogram,
    )

    register_structural_connectome(
        name="test_space_nlin2009",
        space="MNI152NLin2009cAsym",
        tractogram_path=tractogram,
    )

    try:
        # Filter by space
        nlin6_connectomes = list_structural_connectomes(space="MNI152NLin6Asym")

        assert len(nlin6_connectomes) > 0
        assert all(c.space == "MNI152NLin6Asym" for c in nlin6_connectomes)
    finally:
        unregister_structural_connectome("test_space_nlin6")
        unregister_structural_connectome("test_space_nlin2009")


def test_load_structural_connectome_returns_correct_type(tmp_path):
    """Test that load_structural_connectome returns StructuralConnectome."""
    from lacuna.assets.connectomes import (
        load_structural_connectome,
        register_structural_connectome,
        unregister_structural_connectome,
    )
    from lacuna.assets.connectomes.structural import StructuralConnectome

    # Create and register
    tractogram = tmp_path / "test_tractogram.tck"
    tractogram.touch()

    register_structural_connectome(
        name="test_load",
        space="MNI152NLin2009cAsym",
        tractogram_path=tractogram,
    )

    try:
        # Load
        connectome = load_structural_connectome("test_load")

        assert isinstance(connectome, StructuralConnectome)
        assert isinstance(connectome.tractogram_path, Path)
    finally:
        unregister_structural_connectome("test_load")


def test_load_structural_connectome_has_correct_paths(tmp_path):
    """Test that loaded connectome has correct file paths."""
    from lacuna.assets.connectomes import (
        load_structural_connectome,
        register_structural_connectome,
        unregister_structural_connectome,
    )

    # Create and register
    tractogram = tmp_path / "test_tractogram.tck"
    tractogram.touch()

    register_structural_connectome(
        name="test_paths",
        space="MNI152NLin2009cAsym",
        tractogram_path=tractogram,
    )

    try:
        # Load and check paths
        connectome = load_structural_connectome("test_paths")

        assert connectome.tractogram_path == tractogram
        assert connectome.tractogram_path.exists()
    finally:
        unregister_structural_connectome("test_paths")


def test_load_structural_connectome_raises_on_invalid_name():
    """Test that load_structural_connectome raises KeyError for invalid name."""
    from lacuna.assets.connectomes import load_structural_connectome

    with pytest.raises(KeyError, match="not found"):
        load_structural_connectome("NonexistentConnectome12345")


def test_unregister_structural_connectome(tmp_path):
    """Test unregistering a structural connectome."""
    from lacuna.assets.connectomes import (
        list_structural_connectomes,
        register_structural_connectome,
        unregister_structural_connectome,
    )

    # Create and register
    tractogram = tmp_path / "test_tractogram.tck"
    tractogram.touch()

    register_structural_connectome(
        name="test_unregister",
        space="MNI152NLin2009cAsym",
        tractogram_path=tractogram,
    )

    # Verify it exists
    connectomes = list_structural_connectomes()
    names = [c.name for c in connectomes]
    assert "test_unregister" in names

    # Unregister
    unregister_structural_connectome("test_unregister")

    # Verify it's gone
    connectomes = list_structural_connectomes()
    names = [c.name for c in connectomes]
    assert "test_unregister" not in names


def test_structural_connectome_metadata_has_required_fields(tmp_path):
    """Test that metadata has all required fields."""
    from lacuna.assets.connectomes import (
        list_structural_connectomes,
        register_structural_connectome,
        unregister_structural_connectome,
    )

    # Create and register
    tractogram = tmp_path / "test_tractogram.tck"
    tractogram.touch()

    register_structural_connectome(
        name="test_metadata",
        space="MNI152NLin2009cAsym",
        tractogram_path=tractogram,
        description="Test connectome",
    )

    try:
        connectomes = list_structural_connectomes()
        metadata = next(c for c in connectomes if c.name == "test_metadata")

        # Check required fields
        assert hasattr(metadata, "name")
        assert hasattr(metadata, "description")
        assert hasattr(metadata, "space")
        assert hasattr(metadata, "resolution")
        assert hasattr(metadata, "tractogram_path")
        assert hasattr(metadata, "template_path")

        # Check values
        assert metadata.name == "test_metadata"
        assert metadata.space == "MNI152NLin2009cAsym"
        # Note: Structural connectomes (tractograms) don't have inherent voxel resolution
        # The resolution is set to 0.0 as a placeholder; output resolution is controlled
        # by StructuralNetworkMapping.output_resolution parameter
        assert metadata.resolution == 0.0
        assert metadata.tractogram_path == tractogram
    finally:
        unregister_structural_connectome("test_metadata")
