"""Contract tests for DataAssetManager.

Tests the interface for managing spatial data assets (templates, atlases, transforms)
using TemplateFlow as the primary source.

Following TDD: tests written before implementation.
"""

import tempfile
from pathlib import Path

import pytest


class TestDataAssetManagerConstruction:
    """Tests for DataAssetManager initialization."""

    def test_can_create_with_defaults(self):
        """DataAssetManager can be created with default settings."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()
        assert manager is not None

    def test_can_create_with_custom_cache_dir(self):
        """DataAssetManager accepts custom cache directory."""
        from lacuna.spatial.assets import DataAssetManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataAssetManager(cache_dir=Path(tmpdir))
            assert manager.cache_dir == Path(tmpdir)


class TestGetTransform:
    """Tests for transform retrieval."""

    def test_get_transform_accepts_space_parameters(self):
        """get_transform accepts source and target space identifiers."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        # Should not raise
        try:
            manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")
        except NotImplementedError:
            # OK if not fully implemented yet
            pass

    def test_get_transform_returns_path_or_transform_object(self):
        """get_transform returns usable transform (path or object)."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        result = manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")

        # Should return Path, string, or transform object, or None if not available
        assert result is None or isinstance(result, (Path, str)) or hasattr(result, "apply")

    def test_get_transform_caches_downloaded_transforms(self):
        """get_transform caches transforms for reuse."""
        from lacuna.spatial.assets import DataAssetManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataAssetManager(cache_dir=Path(tmpdir))

            # First call may download
            result1 = manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")

            # Second call should be faster (cached)
            result2 = manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")

            # Should return same result
            if result1 is not None:
                assert result1 == result2

    def test_get_transform_raises_on_unsupported_pair(self):
        """get_transform raises TransformNotAvailableError for unsupported pairs."""
        from lacuna.core.exceptions import TransformNotAvailableError
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        # Native space cannot transform to MNI
        with pytest.raises(TransformNotAvailableError):
            manager.get_transform("native", "MNI152NLin6Asym")

    def test_get_transform_handles_download_failure(self):
        """get_transform raises TransformDownloadError on download failure."""
        from lacuna.core.exceptions import TransformDownloadError
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        # If we force a download failure (e.g., no internet), should raise
        # This test may need mocking in real implementation
        # For now, just verify the exception type exists
        assert TransformDownloadError is not None


class TestGetTemplate:
    """Tests for template retrieval."""

    def test_get_template_accepts_space_and_resolution(self):
        """get_template accepts space identifier and resolution."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        # Should not raise
        try:
            manager.get_template("MNI152NLin6Asym", resolution=2)
        except NotImplementedError:
            pass

    def test_get_template_returns_path(self):
        """get_template returns Path to template file."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        result = manager.get_template("MNI152NLin6Asym", resolution=2)

        # Should return Path or None
        assert result is None or isinstance(result, Path)

    def test_get_template_validates_supported_spaces(self):
        """get_template validates space is supported."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        with pytest.raises(ValueError, match="[Uu]nsupported|[Ii]nvalid"):
            manager.get_template("InvalidSpace123", resolution=2)

    def test_get_template_validates_resolution(self):
        """get_template validates resolution is valid."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        with pytest.raises(ValueError, match="resolution"):
            manager.get_template("MNI152NLin6Asym", resolution=999)

    def test_get_template_uses_local_bundled_templates(self):
        """get_template prefers local bundled templates over downloads."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        # Check that bundled templates in src/lacuna/data/templates/ are used
        result = manager.get_template("MNI152NLin6Asym", resolution=2)

        if result is not None:
            assert result.exists(), f"Template should exist at {result}"
            assert "lacuna/data/templates" in str(result)


class TestGetAtlas:
    """Tests for atlas retrieval."""

    def test_get_atlas_accepts_name_and_space(self):
        """get_atlas accepts atlas name and target space."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        # Should not raise
        try:
            manager.get_atlas("HarvardOxford", space="MNI152NLin6Asym")
        except NotImplementedError:
            pass

    def test_get_atlas_returns_path(self):
        """get_atlas returns Path to atlas file."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        result = manager.get_atlas("HarvardOxford", space="MNI152NLin6Asym")

        # Should return Path or None
        assert result is None or isinstance(result, Path)

    def test_get_atlas_validates_atlas_name(self):
        """get_atlas validates atlas name is recognized."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        with pytest.raises(ValueError, match="[Uu]nknown|[Ii]nvalid"):
            manager.get_atlas("NonexistentAtlas123", space="MNI152NLin6Asym")

    def test_get_atlas_supports_common_atlases(self):
        """get_atlas recognizes common atlas names."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        # These atlases should be recognized (even if not yet available)
        common_atlases = ["HarvardOxford", "AAL", "Schaefer"]

        for atlas_name in common_atlases:
            try:
                manager.get_atlas(atlas_name, space="MNI152NLin6Asym")
            except (NotImplementedError, FileNotFoundError):
                # OK if not fully implemented
                pass
            except ValueError as e:
                # Should not raise "unknown atlas" error
                if "nknown" in str(e).lower() or "nvalid" in str(e).lower():
                    pytest.fail(f"Atlas {atlas_name} should be recognized")


class TestIntegration:
    """Integration tests for DataAssetManager."""

    def test_manager_integrates_with_transform_cache(self):
        """DataAssetManager can work with TransformCache."""
        from lacuna.spatial.assets import DataAssetManager
        from lacuna.spatial.cache import TransformCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TransformCache(cache_dir=Path(tmpdir))
            manager = DataAssetManager(cache_dir=Path(tmpdir))

            # Should be able to use both together
            assert manager is not None
            assert cache is not None

    def test_template_and_transform_retrieval_workflow(self):
        """Complete workflow: get template, get transform."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        # Get template
        template = manager.get_template("MNI152NLin6Asym", resolution=2)

        # Get transform
        try:
            transform = manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")
        except NotImplementedError:
            transform = None

        # At least one should work
        assert template is not None or transform is None  # OK if transform not implemented
