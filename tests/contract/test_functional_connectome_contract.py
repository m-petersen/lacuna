"""Contract tests for functional connectome registry.

These tests define the expected behavior of functional connectome management.
All tests should pass after implementation is complete.
"""

import pytest
from pathlib import Path
import h5py
import numpy as np


@pytest.fixture
def valid_h5_file(tmp_path):
    """Create a valid HDF5 connectome file."""
    h5_path = tmp_path / "test_connectome.h5"
    
    with h5py.File(h5_path, "w") as f:
        # Create required datasets
        f.create_dataset("timeseries", data=np.random.rand(10, 100, 1000))
        f.create_dataset("mask_indices", data=np.random.randint(0, 91, size=(3, 1000)))
        f.create_dataset("mask_affine", data=np.eye(4))
        
        # Required attribute
        f.attrs["mask_shape"] = (91, 109, 91)
    
    return h5_path


@pytest.fixture
def invalid_h5_file(tmp_path):
    """Create an invalid HDF5 file (missing required datasets)."""
    h5_path = tmp_path / "invalid_connectome.h5"
    
    with h5py.File(h5_path, "w") as f:
        # Only create timeseries, missing other required datasets
        f.create_dataset("timeseries", data=np.random.rand(10, 100, 1000))
    
    return h5_path


def test_functional_connectome_can_import():
    """Test that functional connectome functions can be imported."""
    from lacuna.assets.connectomes import (
        register_functional_connectome,
        unregister_functional_connectome,
        list_functional_connectomes,
        load_functional_connectome,
    )
    
    assert callable(register_functional_connectome)
    assert callable(unregister_functional_connectome)
    assert callable(list_functional_connectomes)
    assert callable(load_functional_connectome)


def test_register_functional_connectome_with_valid_file(valid_h5_file):
    """Test registering a functional connectome with valid HDF5 file."""
    from lacuna.assets.connectomes import register_functional_connectome, list_functional_connectomes
    
    register_functional_connectome(
        name="test_functional",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=valid_h5_file,
        n_subjects=10,
        description="Test functional connectome",
    )
    
    # Check it was registered
    connectomes = list_functional_connectomes()
    names = [c.name for c in connectomes]
    
    assert "test_functional" in names


def test_register_functional_connectome_missing_file(tmp_path):
    """Test that registration fails if HDF5 file doesn't exist."""
    from lacuna.assets.connectomes import register_functional_connectome
    
    with pytest.raises(FileNotFoundError, match="Data path not found"):
        register_functional_connectome(
            name="test_missing",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=tmp_path / "missing.h5",
            n_subjects=10,
        )


def test_register_functional_connectome_invalid_extension(tmp_path):
    """Test that registration fails if file has wrong extension."""
    from lacuna.assets.connectomes import register_functional_connectome
    
    # Create file with wrong extension
    wrong_file = tmp_path / "test.txt"
    wrong_file.touch()
    
    with pytest.raises(ValueError, match="Expected .h5 file"):
        register_functional_connectome(
            name="test_wrong_ext",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=wrong_file,
            n_subjects=10,
        )


def test_register_functional_connectome_invalid_structure(invalid_h5_file):
    """Test that registration fails if HDF5 structure is invalid."""
    from lacuna.assets.connectomes import register_functional_connectome
    
    with pytest.raises(ValueError, match="missing required datasets"):
        register_functional_connectome(
            name="test_invalid",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=invalid_h5_file,
            n_subjects=10,
        )


def test_register_functional_connectome_batched_directory(tmp_path):
    """Test registering a functional connectome with batched directory."""
    from lacuna.assets.connectomes import register_functional_connectome, load_functional_connectome
    
    # Create batch directory with valid HDF5 files
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    
    for i in range(3):
        h5_path = batch_dir / f"batch_{i}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("timeseries", data=np.random.rand(10, 100, 1000))
            f.create_dataset("mask_indices", data=np.random.randint(0, 91, size=(3, 1000)))
            f.create_dataset("mask_affine", data=np.eye(4))
            f.attrs["mask_shape"] = (91, 109, 91)
    
    # Register batched
    register_functional_connectome(
        name="test_batched",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=batch_dir,
        n_subjects=30,
        description="Batched functional connectome",
    )
    
    # Load and verify is_batched
    connectome = load_functional_connectome("test_batched")
    assert connectome.is_batched is True
    assert connectome.data_path.is_dir()


def test_register_functional_connectome_batched_empty_directory(tmp_path):
    """Test that registration fails if batch directory has no .h5 files."""
    from lacuna.assets.connectomes import register_functional_connectome
    
    # Create empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    with pytest.raises(ValueError, match="No .h5 files found"):
        register_functional_connectome(
            name="test_empty_batch",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=empty_dir,
            n_subjects=10,
        )


def test_list_functional_connectomes_returns_list():
    """Test that list_functional_connectomes returns a list."""
    from lacuna.assets.connectomes import list_functional_connectomes
    
    result = list_functional_connectomes()
    
    assert isinstance(result, list)


def test_list_functional_connectomes_filter_by_space(valid_h5_file):
    """Test filtering functional connectomes by space."""
    from lacuna.assets.connectomes import register_functional_connectome, list_functional_connectomes
    
    # Register connectomes in different spaces
    register_functional_connectome(
        name="test_space_nlin6",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=valid_h5_file,
        n_subjects=10,
    )
    
    # Register another in different space (reusing same file for test)
    register_functional_connectome(
        name="test_space_nlin2009",
        space="MNI152NLin2009cAsym",
        resolution=2.0,
        data_path=valid_h5_file,
        n_subjects=10,
    )
    
    # Filter by space
    nlin6_connectomes = list_functional_connectomes(space="MNI152NLin6Asym")
    
    assert len(nlin6_connectomes) > 0
    assert all(c.space == "MNI152NLin6Asym" for c in nlin6_connectomes)


def test_load_functional_connectome_returns_correct_type(valid_h5_file):
    """Test that load_functional_connectome returns FunctionalConnectome."""
    from lacuna.assets.connectomes import register_functional_connectome, load_functional_connectome
    from lacuna.assets.connectomes.functional import FunctionalConnectome
    
    # Register
    register_functional_connectome(
        name="test_load",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=valid_h5_file,
        n_subjects=10,
    )
    
    # Load
    connectome = load_functional_connectome("test_load")
    
    assert isinstance(connectome, FunctionalConnectome)
    assert isinstance(connectome.data_path, Path)
    assert isinstance(connectome.is_batched, bool)


def test_load_functional_connectome_has_correct_path(valid_h5_file):
    """Test that loaded connectome has correct file path."""
    from lacuna.assets.connectomes import register_functional_connectome, load_functional_connectome
    
    # Register
    register_functional_connectome(
        name="test_path",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=valid_h5_file,
        n_subjects=10,
    )
    
    # Load and check path
    connectome = load_functional_connectome("test_path")
    
    assert connectome.data_path == valid_h5_file
    assert connectome.data_path.exists()
    assert connectome.is_batched is False


def test_load_functional_connectome_raises_on_invalid_name():
    """Test that load_functional_connectome raises KeyError for invalid name."""
    from lacuna.assets.connectomes import load_functional_connectome
    
    with pytest.raises(KeyError, match="not found"):
        load_functional_connectome("NonexistentConnectome12345")


def test_unregister_functional_connectome(valid_h5_file):
    """Test unregistering a functional connectome."""
    from lacuna.assets.connectomes import (
        register_functional_connectome,
        unregister_functional_connectome,
        list_functional_connectomes,
    )
    
    # Register
    register_functional_connectome(
        name="test_unregister",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=valid_h5_file,
        n_subjects=10,
    )
    
    # Verify it exists
    connectomes = list_functional_connectomes()
    names = [c.name for c in connectomes]
    assert "test_unregister" in names
    
    # Unregister
    unregister_functional_connectome("test_unregister")
    
    # Verify it's gone
    connectomes = list_functional_connectomes()
    names = [c.name for c in connectomes]
    assert "test_unregister" not in names


def test_functional_connectome_metadata_has_required_fields(valid_h5_file):
    """Test that metadata has all required fields."""
    from lacuna.assets.connectomes import register_functional_connectome, list_functional_connectomes
    
    # Register
    register_functional_connectome(
        name="test_metadata",
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=valid_h5_file,
        n_subjects=10,
        description="Test connectome",
    )
    
    connectomes = list_functional_connectomes()
    metadata = next(c for c in connectomes if c.name == "test_metadata")
    
    # Check required fields
    assert hasattr(metadata, "name")
    assert hasattr(metadata, "description")
    assert hasattr(metadata, "space")
    assert hasattr(metadata, "resolution")
    assert hasattr(metadata, "n_subjects")
    assert hasattr(metadata, "data_path")
    assert hasattr(metadata, "is_batched")
    
    # Check values
    assert metadata.name == "test_metadata"
    assert metadata.space == "MNI152NLin6Asym"
    assert metadata.resolution == 2.0
    assert metadata.n_subjects == 10
    assert metadata.data_path == valid_h5_file
    assert metadata.is_batched is False


def test_functional_connectome_hdf5_validation_checks_mask_shape(tmp_path):
    """Test that HDF5 validation checks for mask_shape attribute."""
    from lacuna.assets.connectomes import register_functional_connectome
    
    # Create HDF5 with required datasets but missing mask_shape attribute
    h5_path = tmp_path / "no_mask_shape.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("timeseries", data=np.random.rand(10, 100, 1000))
        f.create_dataset("mask_indices", data=np.random.randint(0, 91, size=(3, 1000)))
        f.create_dataset("mask_affine", data=np.eye(4))
        # Missing: f.attrs["mask_shape"] = (91, 109, 91)
    
    with pytest.raises(ValueError, match="mask_shape"):
        register_functional_connectome(
            name="test_no_mask_shape",
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=h5_path,
            n_subjects=10,
        )
