"""Contract tests for transform registry.

These tests define the expected behavior of the transform asset management system.
All tests should pass after implementation is complete.
"""

import pytest


def test_transform_registry_can_import():
    """Test that transform registry can be imported."""
    from lacuna.assets.transforms import list_transforms, load_transform

    assert callable(list_transforms)
    assert callable(load_transform)


def test_list_transforms_returns_list():
    """Test that list_transforms returns a list."""
    from lacuna.assets.transforms import list_transforms

    transforms = list_transforms()

    assert isinstance(transforms, list)
    assert len(transforms) > 0


def test_list_transforms_includes_bidirectional():
    """Test that list includes forward and reverse transforms."""
    from lacuna.assets.transforms import list_transforms

    transforms = list_transforms()
    names = [t.name for t in transforms]

    # Should have transforms in both directions
    has_forward = any("MNI152NLin6Asym_to_MNI152NLin2009cAsym" in name for name in names)
    has_reverse = any("MNI152NLin2009cAsym_to_MNI152NLin6Asym" in name for name in names)

    assert has_forward, "Missing forward transform NLin6Asym -> NLin2009cAsym"
    assert has_reverse, "Missing reverse transform NLin2009cAsym -> NLin6Asym"


def test_list_transforms_filter_by_from_space():
    """Test that transforms can be filtered by source space."""
    from lacuna.assets.transforms import list_transforms

    transforms = list_transforms(from_space="MNI152NLin6Asym")

    assert len(transforms) > 0
    assert all(t.from_space == "MNI152NLin6Asym" for t in transforms)


def test_list_transforms_filter_by_to_space():
    """Test that transforms can be filtered by target space."""
    from lacuna.assets.transforms import list_transforms

    transforms = list_transforms(to_space="MNI152NLin2009cAsym")

    assert len(transforms) > 0
    assert all(t.to_space == "MNI152NLin2009cAsym" for t in transforms)


def test_list_transforms_combined_filters():
    """Test that multiple filters can be combined."""
    from lacuna.assets.transforms import list_transforms

    transforms = list_transforms(
        from_space="MNI152NLin6Asym",
        to_space="MNI152NLin2009cAsym",
    )

    assert len(transforms) > 0
    for t in transforms:
        assert t.from_space == "MNI152NLin6Asym"
        assert t.to_space == "MNI152NLin2009cAsym"


def test_transform_metadata_has_required_fields():
    """Test that transform metadata has all required fields."""
    from lacuna.assets.transforms import list_transforms

    transforms = list_transforms()
    transform = transforms[0]

    # Check required fields exist
    assert hasattr(transform, "name")
    assert hasattr(transform, "description")
    assert hasattr(transform, "from_space")
    assert hasattr(transform, "to_space")
    assert hasattr(transform, "transform_type")
    assert hasattr(transform, "source")

    # Check types
    assert isinstance(transform.name, str)
    assert isinstance(transform.description, str)
    assert isinstance(transform.from_space, str)
    assert isinstance(transform.to_space, str)
    assert isinstance(transform.transform_type, str)
    assert isinstance(transform.source, str)

    # Check values
    assert transform.source == "templateflow"
    assert transform.transform_type in ["nonlinear", "affine", "composite"]


def test_load_transform_returns_path(tmp_path, monkeypatch):
    """Test that load_transform returns a Path object."""
    import importlib
    import sys
    from pathlib import Path

    # Save original modules to restore later
    saved_modules = {}
    modules_to_save = []
    for k in list(sys.modules.keys()):
        if "templateflow" in k or "lacuna.assets.transforms" in k:
            saved_modules[k] = sys.modules[k]
            modules_to_save.append(k)

    try:
        # Mock templateflow to return test path
        def mock_tflow_get(*args, **kwargs):
            test_file = tmp_path / "test_transform.h5"
            test_file.write_bytes(b"mock transform data")  # Write some content
            return str(test_file)

        # Create mock templateflow.api module
        mock_tflow_module = type("MockTemplateFlow", (), {"get": mock_tflow_get})()

        # Clear existing imports to ensure mock is picked up
        for mod in modules_to_save:
            if mod in sys.modules:
                del sys.modules[mod]

        sys.modules["templateflow"] = type("MockTemplateFlow", (), {"api": mock_tflow_module})()
        sys.modules["templateflow.api"] = mock_tflow_module

        # Must import after mocking
        from lacuna.assets.transforms.loader import load_transform

        result = load_transform("MNI152NLin6Asym_to_MNI152NLin2009cAsym")

        assert isinstance(result, Path)
        assert result.exists()
    finally:
        # Restore original modules
        for mod_name in saved_modules:
            sys.modules[mod_name] = saved_modules[mod_name]
        # Clean up any mock modules added
        mocked_keys = [
            k for k in sys.modules.keys() if "templateflow" in k and k not in saved_modules
        ]
        for k in mocked_keys:
            del sys.modules[k]
        # Reload the loader to restore correct state
        if "lacuna.assets.transforms.loader" in sys.modules:
            importlib.reload(sys.modules["lacuna.assets.transforms.loader"])


def test_load_transform_raises_on_invalid_name():
    """Test that load_transform raises KeyError for invalid transform name."""
    from lacuna.assets.transforms import load_transform

    with pytest.raises(KeyError, match="Transform.*not found"):
        load_transform("NonexistentTransform12345")


def test_is_transform_cached_returns_bool(tmp_path, monkeypatch):
    """Test that is_transform_cached returns boolean."""
    import importlib
    import sys

    # Save original modules to restore later
    saved_modules = {}
    modules_to_save = []
    for k in list(sys.modules.keys()):
        if "templateflow" in k or "lacuna.assets.transforms" in k:
            saved_modules[k] = sys.modules[k]
            modules_to_save.append(k)

    try:
        # Mock the cache check
        def mock_tflow_get(*args, **kwargs):
            test_file = tmp_path / "cached_transform.h5"
            test_file.write_bytes(b"mock transform data")  # Write some content
            return str(test_file)

        # Create mock templateflow.api module
        mock_tflow_module = type("MockTemplateFlow", (), {"get": mock_tflow_get})()

        # Clear existing imports to ensure mock is picked up
        for mod in modules_to_save:
            if mod in sys.modules:
                del sys.modules[mod]

        sys.modules["templateflow"] = type("MockTemplateFlow", (), {"api": mock_tflow_module})()
        sys.modules["templateflow.api"] = mock_tflow_module

        # Must import after mocking
        from lacuna.assets.transforms.loader import is_transform_cached

        result = is_transform_cached("MNI152NLin6Asym_to_MNI152NLin2009cAsym")

        assert isinstance(result, bool)
    finally:
        # Restore original modules
        for mod_name in saved_modules:
            sys.modules[mod_name] = saved_modules[mod_name]
        # Clean up any mock modules added
        mocked_keys = [
            k for k in sys.modules.keys() if "templateflow" in k and k not in saved_modules
        ]
        for k in mocked_keys:
            del sys.modules[k]
        # Reload the loader to restore correct state
        if "lacuna.assets.transforms.loader" in sys.modules:
            importlib.reload(sys.modules["lacuna.assets.transforms.loader"])


def test_transform_names_follow_convention():
    """Test that transform names follow expected naming convention."""
    from lacuna.assets.transforms import list_transforms

    transforms = list_transforms()

    # Names should include from_space and to_space
    for t in transforms:
        assert "_to_" in t.name, f"Transform name should contain '_to_': {t.name}"
        assert t.from_space in t.name, f"Transform name should contain from_space: {t.name}"
        assert t.to_space in t.name, f"Transform name should contain to_space: {t.name}"


def test_transform_validate_method():
    """Test that transforms have a validate method."""
    from lacuna.assets.transforms import list_transforms

    transforms = list_transforms()
    transform = transforms[0]

    # Should have validate method
    assert hasattr(transform, "validate")
    assert callable(transform.validate)

    # Should not raise for valid metadata
    transform.validate()
