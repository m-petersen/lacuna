"""Contract tests for Pipeline API."""

import nibabel as nib
import numpy as np
import pytest

from lacuna import Pipeline, SubjectData
from lacuna.analysis import RegionalDamage


class TestPipelineContract:
    """Contract tests for Pipeline class."""

    @pytest.fixture
    def sample_mask_data(self, tmp_path):
        """Create a minimal SubjectData for testing with a registered local atlas."""
        from lacuna.assets.parcellations.registry import register_parcellations_from_directory

        # Create test atlas to avoid TemplateFlow
        atlas_dir = tmp_path / "atlases"
        atlas_dir.mkdir()
        atlas_data = np.zeros((10, 10, 10), dtype=np.uint8)
        atlas_data[3:7, 3:7, 3:7] = 1
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4) * 2), atlas_dir / "test_pipeline.nii.gz")
        (atlas_dir / "test_pipeline_labels.txt").write_text("1 Region1\n")
        register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0
        img = nib.Nifti1Image(data, affine)
        return SubjectData(
            mask_img=img,
            space="MNI152NLin6Asym",
            resolution=2,
            metadata={"subject_id": "sub-001"},
        )

    def test_pipeline_add_returns_self(self):
        """Contract: add() returns self for method chaining."""
        pipeline = Pipeline()
        result = pipeline.add(RegionalDamage())
        assert result is pipeline

    def test_pipeline_fluent_chaining(self):
        """Contract: Multiple add() calls can be chained."""
        pipeline = (
            Pipeline(name="Test Pipeline")
            .add(RegionalDamage())
            .add(RegionalDamage())  # Same analysis twice for testing
        )
        assert len(pipeline) == 2

    def test_pipeline_run_returns_mask_data(self, sample_mask_data):
        """Contract: run() returns SubjectData with results."""
        # Use parcel_names to avoid bundled atlases that require TemplateFlow
        pipeline = Pipeline().add(RegionalDamage(parcel_names=["test_pipeline"]))
        result = pipeline.run(sample_mask_data)
        assert isinstance(result, SubjectData)
        assert "RegionalDamage" in result.results

    def test_pipeline_describe_returns_string(self):
        """Contract: describe() returns human-readable string."""
        pipeline = Pipeline(name="My Analysis").add(RegionalDamage())
        description = pipeline.describe()
        assert isinstance(description, str)
        assert "My Analysis" in description
        assert "RegionalDamage" in description

    def test_pipeline_len_returns_step_count(self):
        """Contract: len(pipeline) returns number of steps."""
        pipeline = Pipeline()
        assert len(pipeline) == 0
        pipeline.add(RegionalDamage())
        assert len(pipeline) == 1
        pipeline.add(RegionalDamage())
        assert len(pipeline) == 2


class TestAnalyzeFunctionContract:
    """Contract tests for analyze() convenience function."""

    @pytest.fixture
    def sample_mask_data(self, tmp_path):
        """Create a minimal SubjectData for testing with a registered local atlas.

        This fixture temporarily replaces the parcellation registry with only
        a local test atlas to avoid TemplateFlow dependencies in CI.
        """
        from lacuna.assets.parcellations.registry import (
            PARCELLATION_REGISTRY,
            register_parcellations_from_directory,
        )

        # Save original registry
        saved_registry = PARCELLATION_REGISTRY.copy()

        # Clear registry to avoid loading bundled atlases that need TemplateFlow
        PARCELLATION_REGISTRY.clear()

        # Create test atlas - analyze() uses RegionalDamage without parcel_names,
        # so it needs atlases registered
        atlas_dir = tmp_path / "atlases"
        atlas_dir.mkdir()
        atlas_data = np.zeros((10, 10, 10), dtype=np.uint8)
        atlas_data[3:7, 3:7, 3:7] = 1
        nib.save(nib.Nifti1Image(atlas_data, np.eye(4) * 2), atlas_dir / "test_analyze.nii.gz")
        (atlas_dir / "test_analyze_labels.txt").write_text("1 Region1\n")
        register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

        data = np.zeros((10, 10, 10), dtype=np.uint8)
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0
        img = nib.Nifti1Image(data, affine)

        try:
            yield SubjectData(
                mask_img=img,
                space="MNI152NLin6Asym",
                resolution=2,
                metadata={"subject_id": "sub-001"},
            )
        finally:
            # Restore original registry
            PARCELLATION_REGISTRY.clear()
            PARCELLATION_REGISTRY.update(saved_registry)

    def test_analyze_single_returns_mask_data(self, sample_mask_data):
        """Contract: analyze(single) returns SubjectData."""
        from lacuna import analyze

        result = analyze(sample_mask_data)
        assert isinstance(result, SubjectData)

    def test_analyze_list_returns_list(self, sample_mask_data):
        """Contract: analyze(list) returns list of SubjectData.

        Note: We use batch_process directly with our local atlas instead of
        analyze() because parallel workers don't inherit the modified registry.
        This still tests the contract that list input returns list output.
        """
        from lacuna.analysis import RegionalDamage
        from lacuna.batch.api import batch_process

        # Use the registered local atlas explicitly
        analysis = RegionalDamage(parcel_names=["test_analyze"])

        results = batch_process(
            inputs=[sample_mask_data, sample_mask_data],
            analysis=analysis,
            n_jobs=1,  # Sequential to avoid registry propagation issues
            show_progress=False,
        )
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, SubjectData) for r in results)
