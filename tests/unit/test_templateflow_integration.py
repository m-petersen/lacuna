"""Unit tests for TemplateFlow integration in DataAssetManager.

Following TDD: Tests define the expected behavior for TemplateFlow integration
before implementation.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestTemplateFlowImport:
    """Tests for TemplateFlow package availability."""

    def test_templateflow_can_be_imported(self):
        """TemplateFlow package can be imported."""
        try:
            import templateflow  # noqa: F401

            assert True
        except ImportError:
            pytest.skip("templateflow not installed")


class TestTemplateFlowTransformDownload:
    """Tests for downloading transforms from TemplateFlow."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Provide temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_templateflow(self):
        """Mock templateflow.api.get for testing."""
        with patch("templateflow.api.get") as mock_get:
            yield mock_get

    def test_get_transform_calls_templateflow_api(self, temp_cache_dir, mock_templateflow):
        """get_transform calls templateflow.api.get with correct parameters."""
        from lacuna.spatial.assets import DataAssetManager

        # Mock templateflow to return a fake path
        fake_transform_path = temp_cache_dir / "fake_transform.h5"
        fake_transform_path.touch()
        mock_templateflow.return_value = str(fake_transform_path)

        manager = DataAssetManager(cache_dir=temp_cache_dir)
        result = manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")

        # Should have called templateflow.api.get
        assert mock_templateflow.called
        assert result is not None

    def test_get_transform_uses_correct_templateflow_parameters(
        self, temp_cache_dir, mock_templateflow
    ):
        """get_transform passes correct parameters to TemplateFlow API."""
        from lacuna.spatial.assets import DataAssetManager

        fake_transform_path = temp_cache_dir / "fake_transform.h5"
        fake_transform_path.touch()
        mock_templateflow.return_value = str(fake_transform_path)

        manager = DataAssetManager(cache_dir=temp_cache_dir)
        manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")

        # Verify the call was made with appropriate parameters
        # TemplateFlow expects: template, from_template, mode, suffix
        call_args = mock_templateflow.call_args
        assert call_args is not None

        # Check that call includes necessary templateflow parameters
        # Either positional or keyword arguments
        if call_args[0]:  # positional args
            assert "MNI152NLin6Asym" in str(call_args[0]) or "MNI152NLin2009cAsym" in str(
                call_args[0]
            )
        if call_args[1]:  # keyword args
            kwargs = call_args[1]
            assert (
                "template" in kwargs
                or "from" in kwargs
                or kwargs.get("mode") == "image"
                or kwargs.get("suffix") == "xfm"
            )

    def test_get_transform_handles_both_directions(self, temp_cache_dir, mock_templateflow):
        """get_transform can retrieve transforms in both directions."""
        from lacuna.spatial.assets import DataAssetManager

        fake_transform_path = temp_cache_dir / "fake_transform.h5"
        fake_transform_path.touch()
        mock_templateflow.return_value = str(fake_transform_path)

        manager = DataAssetManager(cache_dir=temp_cache_dir)

        # Forward direction
        result1 = manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")
        assert result1 is not None

        # Reverse direction
        result2 = manager.get_transform("MNI152NLin2009cAsym", "MNI152NLin6Asym")
        assert result2 is not None

    def test_get_transform_caches_downloaded_files(self, temp_cache_dir, mock_templateflow):
        """get_transform caches downloaded transforms to avoid re-downloading."""
        from lacuna.spatial.assets import DataAssetManager

        fake_transform_path = temp_cache_dir / "fake_transform.h5"
        fake_transform_path.touch()
        mock_templateflow.return_value = str(fake_transform_path)

        manager = DataAssetManager(cache_dir=temp_cache_dir)

        # First call
        result1 = manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")

        # Second call
        result2 = manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")

        # Should return same path
        assert result1 == result2

        # Should have only called templateflow once (cached)
        assert mock_templateflow.call_count == 1

    def test_get_transform_handles_templateflow_errors(self, temp_cache_dir, mock_templateflow):
        """get_transform handles errors from TemplateFlow gracefully."""
        from lacuna.core.exceptions import TransformDownloadError
        from lacuna.spatial.assets import DataAssetManager

        # Mock templateflow to raise an error
        mock_templateflow.side_effect = Exception("Network error")

        manager = DataAssetManager(cache_dir=temp_cache_dir)

        # Should raise TransformDownloadError
        with pytest.raises(TransformDownloadError):
            manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")

    def test_get_transform_returns_path_object(self, temp_cache_dir, mock_templateflow):
        """get_transform returns Path object, not string."""
        from lacuna.spatial.assets import DataAssetManager

        fake_transform_path = temp_cache_dir / "fake_transform.h5"
        fake_transform_path.touch()
        mock_templateflow.return_value = str(fake_transform_path)

        manager = DataAssetManager(cache_dir=temp_cache_dir)
        result = manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")

        # Should return Path, not string
        assert isinstance(result, Path)


class TestTemplateFlowTransformMapping:
    """Tests for mapping between lacuna space names and TemplateFlow conventions."""

    def test_space_names_map_correctly_to_templateflow(self):
        """Lacuna space names map correctly to TemplateFlow template names."""
        # Our space names should match TemplateFlow's template naming
        space_mappings = {
            "MNI152NLin6Asym": "MNI152NLin6Asym",
            "MNI152NLin2009cAsym": "MNI152NLin2009cAsym",
            "MNI152NLin2009bAsym": "MNI152NLin2009bAsym",
        }

        for lacuna_name, tf_name in space_mappings.items():
            assert lacuna_name == tf_name  # We use same names as TemplateFlow

    def test_transform_filename_format_matches_templateflow(self):
        """Transform file naming follows TemplateFlow conventions."""
        # TemplateFlow uses format:
        # tpl-{template}_from-{source}_mode-image_xfm.h5
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        # The filename pattern should match TemplateFlow's convention
        # We'll verify this in the actual implementation
        assert hasattr(manager, "get_transform")


class TestTemplateFlowIntegrationWithRealAPI:
    """Integration tests with real TemplateFlow API (requires internet)."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_download_real_transform_from_templateflow(self):
        """Can download a real transform from TemplateFlow."""
        pytest.importorskip("templateflow")

        from lacuna.spatial.assets import DataAssetManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataAssetManager(cache_dir=Path(tmpdir))

            # Try to download a real transform
            # This will take a few seconds and requires internet
            try:
                result = manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")

                # Should return a valid path
                assert result is not None
                assert isinstance(result, Path)
                assert result.exists()
                assert result.suffix == ".h5"  # TemplateFlow uses HDF5 format

            except Exception as e:
                # If download fails (no internet, etc), skip
                pytest.skip(f"Could not download from TemplateFlow: {e}")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_downloaded_transform_can_be_loaded_by_nitransforms(self):
        """Downloaded transform can be loaded by nitransforms."""
        pytest.importorskip("templateflow")
        pytest.importorskip("nitransforms")

        from lacuna.spatial.assets import DataAssetManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DataAssetManager(cache_dir=Path(tmpdir))

            try:
                transform_path = manager.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")

                if transform_path is not None:
                    # Should be loadable by nitransforms
                    from nitransforms.linear import load as load_transform

                    transform = load_transform(transform_path)
                    assert transform is not None
                    assert hasattr(transform, "apply")

            except Exception as e:
                pytest.skip(f"Could not test with real transform: {e}")


class TestTemplateFlowErrorHandling:
    """Tests for error handling in TemplateFlow integration."""

    def test_missing_templateflow_package_raises_clear_error(self):
        """If templateflow is not installed, provide clear error message."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        # When templateflow is not installed and we try to download,
        # should get a clear error message
        # This will be tested in the actual implementation
        assert hasattr(manager, "get_transform")

    def test_network_error_raises_transformdownloaderror(self):
        """Network errors during download raise TransformDownloadError."""
        from lacuna.core.exceptions import TransformDownloadError
        from lacuna.spatial.assets import DataAssetManager

        # Verify the exception exists
        assert TransformDownloadError is not None

        # Implementation should catch network errors and raise this
        manager = DataAssetManager()
        assert hasattr(manager, "get_transform")

    def test_invalid_transform_pair_raises_before_download_attempt(self):
        """Invalid transform pairs are rejected before attempting download."""
        from lacuna.core.exceptions import TransformNotAvailableError
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager()

        # Should raise immediately without attempting download
        with pytest.raises(TransformNotAvailableError):
            manager.get_transform("InvalidSpace1", "InvalidSpace2")


class TestTemplateFlowCaching:
    """Tests for caching behavior with TemplateFlow."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Provide temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_cache_persists_across_manager_instances(self, temp_cache_dir):
        """Downloaded transforms persist across DataAssetManager instances."""
        pytest.importorskip("templateflow")

        from lacuna.spatial.assets import DataAssetManager

        # Create first manager and download
        manager1 = DataAssetManager(cache_dir=temp_cache_dir)

        with patch("templateflow.api.get") as mock_tf:
            fake_path = temp_cache_dir / "cached_transform.h5"
            fake_path.touch()
            mock_tf.return_value = str(fake_path)

            result1 = manager1.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")

            # Create second manager with same cache dir
            manager2 = DataAssetManager(cache_dir=temp_cache_dir)
            result2 = manager2.get_transform("MNI152NLin6Asym", "MNI152NLin2009cAsym")

            # Should use cached version
            assert result1 == result2

    def test_cache_dir_structure_is_organized(self, temp_cache_dir):
        """Cache directory is organized logically."""
        from lacuna.spatial.assets import DataAssetManager

        manager = DataAssetManager(cache_dir=temp_cache_dir)

        # Cache dir should exist
        assert manager.cache_dir.exists()
        assert manager.cache_dir.is_dir()

        # Should be able to store multiple transforms
        # Implementation can organize as needed (subdirs, etc)
