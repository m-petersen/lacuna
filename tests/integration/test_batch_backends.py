"""
Integration tests for batch processing with different backends.

These tests verify that batch processing works correctly with threading and loky
backends using real (synthetic) lesion data.

NOTE on loky backend tests:
--------------------------
Loky spawns new processes that don't inherit the parent process's memory,
including any dynamically registered parcellations. However, loky DOES work
with bundled parcellations (like Schaefer atlases) because workers can
discover and load them from the package paths.

Threading backend tests use dynamically registered test atlases (faster).
Loky backend tests use bundled Schaefer atlas (works across processes).
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from lacuna import SubjectData, batch_process
from lacuna.analysis import RegionalDamage
from lacuna.assets.parcellations import load_parcellation
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

        affine = np.eye(4)  # 1mm isotropic
        img = nib.Nifti1Image(data, affine)

        lesion = SubjectData(
            img,
            space="MNI152NLin6Asym",
            resolution=1.0,  # Match np.eye(4) affine
            metadata={"subject_id": f"sub-{i:03d}"},
        )
        lesions.append(lesion)

    return lesions


@pytest.fixture
def test_atlas_dir(tmp_path):
    """Create a minimal test atlas directory with only one small atlas.

    Uses a unique atlas name based on tmp_path to avoid conflicts with
    pytest-xdist parallel execution.
    """
    atlas_dir = tmp_path / "test_atlases"
    atlas_dir.mkdir()

    # Create unique atlas name based on tmp_path hash to avoid xdist conflicts
    atlas_name = f"test_atlas_{hash(str(tmp_path)) % 100000}"

    # Create a minimal 10x10x10 atlas with 3 regions
    data = np.zeros((10, 10, 10), dtype=np.int32)
    data[0:3, 0:3, 0:3] = 1  # Region 1
    data[3:6, 3:6, 3:6] = 2  # Region 2
    data[6:9, 6:9, 6:9] = 3  # Region 3

    affine = np.eye(4)
    atlas_img = nib.Nifti1Image(data, affine)
    nib.save(atlas_img, atlas_dir / f"{atlas_name}.nii.gz")

    # Create labels file
    labels_file = atlas_dir / f"{atlas_name}_labels.txt"
    labels_file.write_text("1\tRegion1\n2\tRegion2\n3\tRegion3\n")

    # Return both the directory and the atlas name
    return atlas_dir, atlas_name


@pytest.fixture
def regional_damage_analysis(test_atlas_dir):
    """Create RegionalDamage analysis instance with minimal test atlas."""
    from lacuna.assets.parcellations.registry import register_parcellations_from_directory

    atlas_dir, atlas_name = test_atlas_dir

    # Register the test atlas with resolution=1 to match synthetic_lesions fixture
    # (synthetic lesions use np.eye(4) affine = 1mm isotropic)
    register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=1)

    # Create analysis with explicit parcel_names to avoid bundled atlases that require TemplateFlow
    return RegionalDamage(parcel_names=[atlas_name])


# ==== Loky-specific fixtures ====
# Loky tests use bundled atlases since dynamically registered ones don't
# transfer to worker processes. These fixtures create data matching the
# bundled Schaefer atlas dimensions.


@pytest.fixture
def mni_synthetic_lesions():
    """Create synthetic lesion data matching MNI152NLin6Asym space.

    Uses 1mm resolution to match bundled Schaefer atlases.
    Shape: (182, 218, 182) with same affine as Schaefer atlases.
    """
    # Get the actual atlas affine and shape for matching
    parcel = load_parcellation("Schaefer2018_100Parcels7Networks")
    shape = parcel.image.shape
    affine = parcel.image.affine

    lesions = []
    for i in range(3):
        # Create lesion data matching atlas dimensions
        data = np.zeros(shape, dtype=np.float32)
        # Create a lesion in different locations for each subject
        # Centering around a common brain region
        cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
        offset = i * 5
        data[cx - 5 + offset : cx + 5 + offset, cy - 5 : cy + 5, cz - 5 : cz + 5] = 1

        img = nib.Nifti1Image(data, affine)
        lesion = SubjectData(
            img,
            metadata={"subject_id": f"sub-{i:03d}", "space": "MNI152NLin6Asym", "resolution": 1},
        )
        lesions.append(lesion)

    return lesions


@pytest.fixture
def bundled_atlas_analysis():
    """Create RegionalDamage analysis using bundled atlas.

    Uses Schaefer2018_100Parcels7Networks which is bundled with the package
    and can be loaded by loky worker processes.
    """
    return RegionalDamage(parcel_names=["Schaefer2018_100Parcels7Networks"])


class TestThreadingBackend:
    """Test threading backend for Jupyter compatibility."""

    def test_threading_backend_processes_all_subjects(
        self, synthetic_lesions, regional_damage_analysis
    ):
        """Threading backend should process all subjects successfully."""
        results = batch_process(
            inputs=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=2,
            show_progress=False,
            backend="threading",
        )

        assert len(results) == len(synthetic_lesions)
        assert all(isinstance(r, SubjectData) for r in results)

    def test_threading_backend_with_single_worker(
        self, synthetic_lesions, regional_damage_analysis
    ):
        """Threading backend should work with n_jobs=1."""
        results = batch_process(
            inputs=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=1,
            show_progress=False,
            backend="threading",
        )

        assert len(results) == len(synthetic_lesions)

    def test_threading_backend_with_all_cores(self, synthetic_lesions, regional_damage_analysis):
        """Threading backend should work with n_jobs=-1."""
        results = batch_process(
            inputs=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=-1,
            show_progress=False,
            backend="threading",
        )

        assert len(results) == len(synthetic_lesions)


class TestLokyBackend:
    """Test loky backend for standalone scripts.

    These tests use bundled Schaefer atlas which loky workers can discover
    and load from package paths. This is the real-world use case for loky.
    """

    def test_loky_backend_processes_all_subjects(
        self, mni_synthetic_lesions, bundled_atlas_analysis
    ):
        """Loky backend should process all subjects successfully."""
        results = batch_process(
            inputs=mni_synthetic_lesions,
            analysis=bundled_atlas_analysis,
            n_jobs=2,
            show_progress=False,
            backend="loky",
        )

        assert len(results) == len(mni_synthetic_lesions)
        assert all(isinstance(r, SubjectData) for r in results)

    def test_loky_backend_adds_results(self, mni_synthetic_lesions, bundled_atlas_analysis):
        """Loky backend should add analysis results to lesion data."""
        results = batch_process(
            inputs=mni_synthetic_lesions,
            analysis=bundled_atlas_analysis,
            n_jobs=2,
            show_progress=False,
            backend="loky",
        )

        for result in results:
            assert len(result.results) > 0
            assert "RegionalDamage" in result.results

    def test_loky_backend_is_default(self, mni_synthetic_lesions, bundled_atlas_analysis):
        """Loky should be the default backend."""
        # Don't specify backend - should default to loky
        results = batch_process(
            inputs=mni_synthetic_lesions,
            analysis=bundled_atlas_analysis,
            n_jobs=2,
            show_progress=False,
        )

        assert len(results) == len(mni_synthetic_lesions)


class TestMultiprocessingBackend:
    """Test multiprocessing backend."""

    def test_multiprocessing_backend_works(self, synthetic_lesions, regional_damage_analysis):
        """Multiprocessing backend should work for batch processing."""
        results = batch_process(
            inputs=synthetic_lesions,
            analysis=regional_damage_analysis,
            n_jobs=2,
            show_progress=False,
            backend="multiprocessing",
        )

        assert len(results) == len(synthetic_lesions)
        assert all(isinstance(r, SubjectData) for r in results)


class TestBackendComparison:
    """Compare results across different backends."""

    def test_backends_produce_same_results(self, mni_synthetic_lesions, bundled_atlas_analysis):
        """Different backends should produce equivalent results.

        Uses bundled atlas so both threading and loky can access it.
        """
        # Process with threading
        results_threading = batch_process(
            inputs=mni_synthetic_lesions,
            analysis=bundled_atlas_analysis,
            n_jobs=1,  # Use 1 to ensure deterministic order
            show_progress=False,
            backend="threading",
        )

        # Process with loky
        results_loky = batch_process(
            inputs=mni_synthetic_lesions,
            analysis=bundled_atlas_analysis,
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
        results = strategy.execute(inputs=synthetic_lesions, analysis=regional_damage_analysis)

        assert len(results) == len(synthetic_lesions)
        assert strategy.backend == "threading"

    def test_parallel_strategy_loky_backend(self, mni_synthetic_lesions, bundled_atlas_analysis):
        """ParallelStrategy should work with loky backend.

        Uses bundled atlas so loky workers can access it.
        """
        strategy = ParallelStrategy(n_jobs=2, backend="loky")
        results = strategy.execute(inputs=mni_synthetic_lesions, analysis=bundled_atlas_analysis)

        assert len(results) == len(mni_synthetic_lesions)
        assert strategy.backend == "loky"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
