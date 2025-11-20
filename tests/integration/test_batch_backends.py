"""
Integration tests for batch processing with different backends.

These tests verify that batch processing works correctly with threading and loky
backends using real (synthetic) lesion data.
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lacuna import LesionData, batch_process
from lacuna.analysis import RegionalDamage
from lacuna.batch.strategies import ParallelStrategy


@pytest.fixture
def synthetic_lesions():
    """Create synthetic lesion data for testing."""
    lesions = []
    # Reduced from 5 to 3 subjects for faster testing
    for i in range(3):
        # Create simple 10x10x10 binary lesion
        data = np.zeros((10, 10, 10), dtype=np.float32)
        data[3:7, 3:7, 3:7] = 1  # Small cube lesion

        affine = np.eye(4)
        img = nib.Nifti1Image(data, affine)

        lesion = LesionData(
            img,
            metadata={
                "subject_id": f"sub-{i:03d}",
                "space": "MNI152NLin6Asym",
                "resolution": 2
            }
        )
        lesions.append(lesion)

    return lesions


@pytest.fixture
def test_atlas_dir(tmp_path):
    """Create a minimal test atlas directory with only one small atlas."""
    atlas_dir = tmp_path / "test_atlases"
    atlas_dir.mkdir()

    # Create a minimal 10x10x10 atlas with 3 regions
    data = np.zeros((10, 10, 10), dtype=np.int32)
    data[0:3, 0:3, 0:3] = 1  # Region 1
    data[3:6, 3:6, 3:6] = 2  # Region 2
    data[6:9, 6:9, 6:9] = 3  # Region 3

    affine = np.eye(4)
    atlas_img = nib.Nifti1Image(data, affine)
    nib.save(atlas_img, atlas_dir / "test_atlas.nii.gz")

    # Create labels file
    labels_file = atlas_dir / "test_atlas_labels.txt"
    labels_file.write_text("1\tRegion1\n2\tRegion2\n3\tRegion3\n")

    return atlas_dir


@pytest.fixture
def regional_damage_analysis(test_atlas_dir):
    """Create RegionalDamage analysis instance with minimal test atlas."""
    from lacuna.assets.atlases.registry import register_atlases_from_directory
    
    # Register the test atlas
    register_atlases_from_directory(test_atlas_dir, space="MNI152NLin6Asym", resolution=2)
    
    # Create analysis without atlas_dir parameter
    return RegionalDamage()


class TestThreadingBackend:
    """Test threading backend for Jupyter compatibility."""

    def test_threading_backend_processes_all_subjects(
        self, synthetic_lesions, regional_damage_analysis
    ):
        """Threading backend should process all subjects successfully."""
        results = batch_process(
            lesion_data_list=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=2,
            show_progress=False,
            backend="threading",
        )

        assert len(results) == len(synthetic_lesions)
        assert all(isinstance(r, LesionData) for r in results)

    def test_threading_backend_adds_results(self, synthetic_lesions, regional_damage_analysis):
        """Threading backend should add analysis results to lesion data."""
        results = batch_process(
            lesion_data_list=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=2,
            show_progress=False,
            backend="threading",
        )

        for result in results:
            assert len(result.results) > 0
            assert "RegionalDamage" in result.results

    def test_threading_backend_with_single_worker(
        self, synthetic_lesions, regional_damage_analysis
    ):
        """Threading backend should work with n_jobs=1."""
        results = batch_process(
            lesion_data_list=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=1,
            show_progress=False,
            backend="threading",
        )

        assert len(results) == len(synthetic_lesions)

    def test_threading_backend_with_all_cores(self, synthetic_lesions, regional_damage_analysis):
        """Threading backend should work with n_jobs=-1."""
        results = batch_process(
            lesion_data_list=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=-1,
            show_progress=False,
            backend="threading",
        )

        assert len(results) == len(synthetic_lesions)


class TestLokyBackend:
    """Test loky backend for standalone scripts."""

    def test_loky_backend_processes_all_subjects(self, synthetic_lesions, regional_damage_analysis):
        """Loky backend should process all subjects successfully."""
        results = batch_process(
            lesion_data_list=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=2,
            show_progress=False,
            backend="loky",
        )

        assert len(results) == len(synthetic_lesions)
        assert all(isinstance(r, LesionData) for r in results)

    def test_loky_backend_adds_results(self, synthetic_lesions, regional_damage_analysis):
        """Loky backend should add analysis results to lesion data."""
        results = batch_process(
            lesion_data_list=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=2,
            show_progress=False,
            backend="loky",
        )

        for result in results:
            assert len(result.results) > 0
            assert "RegionalDamage" in result.results

    def test_loky_backend_is_default(self, synthetic_lesions, regional_damage_analysis):
        """Loky should be the default backend."""
        # Don't specify backend - should default to loky
        results = batch_process(
            lesion_data_list=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=2,
            show_progress=False,
        )

        assert len(results) == len(synthetic_lesions)


class TestMultiprocessingBackend:
    """Test multiprocessing backend."""

    @pytest.mark.skip(
        reason="Standard multiprocessing backend has pickling issues with nested functions. Use 'loky' or 'threading' instead."
    )
    def test_multiprocessing_backend_works(self, synthetic_lesions, regional_damage_analysis):
        """Multiprocessing backend should work for batch processing."""
        results = batch_process(
            lesion_data_list=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=2,
            show_progress=False,
            backend="multiprocessing",
        )

        assert len(results) == len(synthetic_lesions)
        assert all(isinstance(r, LesionData) for r in results)


class TestBackendComparison:
    """Compare results across different backends."""

    def test_backends_produce_same_results(self, synthetic_lesions, regional_damage_analysis):
        """Different backends should produce equivalent results."""
        # Process with threading
        results_threading = batch_process(
            lesion_data_list=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=1,  # Use 1 to ensure deterministic order
            show_progress=False,
            backend="threading",
        )

        # Process with loky
        results_loky = batch_process(
            lesion_data_list=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=1,  # Use 1 to ensure deterministic order
            show_progress=False,
            backend="loky",
        )

        # Compare results
        assert len(results_threading) == len(results_loky)

        for r_thread, r_loky in zip(results_threading, results_loky, strict=False):
            assert r_thread.metadata["subject_id"] == r_loky.metadata["subject_id"]
            assert set(r_thread.results.keys()) == set(r_loky.results.keys())


class TestParallelStrategyBackend:
    """Test ParallelStrategy directly with different backends."""

    def test_parallel_strategy_threading_backend(self, synthetic_lesions, regional_damage_analysis):
        """ParallelStrategy should work with threading backend."""
        strategy = ParallelStrategy(n_jobs=2, backend="threading")
        results = strategy.execute(
            lesion_data_list=synthetic_lesions, analysis=regional_damage_analysis
        )

        assert len(results) == len(synthetic_lesions)
        assert strategy.backend == "threading"

    def test_parallel_strategy_loky_backend(self, synthetic_lesions, regional_damage_analysis):
        """ParallelStrategy should work with loky backend."""
        strategy = ParallelStrategy(n_jobs=2, backend="loky")
        results = strategy.execute(
            lesion_data_list=synthetic_lesions, analysis=regional_damage_analysis
        )

        assert len(results) == len(synthetic_lesions)
        assert strategy.backend == "loky"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
