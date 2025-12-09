"""Unit tests for provenance tracking."""

import nibabel as nib
import numpy as np
import pytest

from lacuna.assets.parcellations.registry import (
    register_parcellations_from_directory,
    unregister_parcellation,
)
from lacuna.core.provenance import TransformationRecord


class TestTransformationRecord:
    """Tests for TransformationRecord validation."""

    def test_transformation_record_creation(self):
        """TransformationRecord should store all required fields."""
        record = TransformationRecord(
            source_space="MNI152NLin6Asym",
            source_resolution=2.0,
            target_space="MNI152NLin2009cAsym",
            target_resolution=1.0,
            method="nitransforms",
            interpolation="linear",
        )

        assert record.source_space == "MNI152NLin6Asym"
        assert record.source_resolution == 2.0
        assert record.target_space == "MNI152NLin2009cAsym"
        assert record.target_resolution == 1.0
        assert record.method == "nitransforms"
        assert record.interpolation == "linear"
        assert record.timestamp is not None  # Auto-generated

    def test_transformation_record_with_optional_fields(self):
        """TransformationRecord should accept optional fields."""
        record = TransformationRecord(
            source_space="MNI152NLin6Asym",
            source_resolution=2.0,
            target_space="MNI152NLin2009cAsym",
            target_resolution=1.0,
            method="nitransforms",
            interpolation="linear",
            rationale="Better alignment with HCP dataset",
            transform_file="/path/to/transform.h5",
        )

        assert record.rationale == "Better alignment with HCP dataset"
        assert record.transform_file == "/path/to/transform.h5"

    def test_transformation_record_to_dict(self):
        """TransformationRecord should convert to dictionary."""
        record = TransformationRecord(
            source_space="MNI152NLin6Asym",
            source_resolution=2.0,
            target_space="MNI152NLin2009cAsym",
            target_resolution=1.0,
            method="nitransforms",
            interpolation="linear",
        )

        result = record.to_dict()

        assert result["type"] == "spatial_transformation"
        assert result["source_space"] == "MNI152NLin6Asym"
        assert result["source_resolution"] == 2.0
        assert result["target_space"] == "MNI152NLin2009cAsym"
        assert result["target_resolution"] == 1.0
        assert result["method"] == "nitransforms"
        assert result["interpolation"] == "linear"
        assert "timestamp" in result

    def test_transformation_record_dict_includes_optional_fields(self):
        """TransformationRecord.to_dict() should include optional fields if provided."""
        record = TransformationRecord(
            source_space="MNI152NLin6Asym",
            source_resolution=2.0,
            target_space="MNI152NLin2009cAsym",
            target_resolution=1.0,
            method="nitransforms",
            interpolation="linear",
            rationale="Test reason",
            transform_file="/path/to/file.h5",
        )

        result = record.to_dict()

        assert result["rationale"] == "Test reason"
        assert result["transform_file"] == "/path/to/file.h5"

    def test_transformation_record_dict_omits_none_optional_fields(self):
        """TransformationRecord.to_dict() should omit None optional fields."""
        record = TransformationRecord(
            source_space="MNI152NLin6Asym",
            source_resolution=2.0,
            target_space="MNI152NLin2009cAsym",
            target_resolution=1.0,
            method="nitransforms",
            interpolation="linear",
            rationale=None,
            transform_file=None,
        )

        result = record.to_dict()

        assert "rationale" not in result
        assert "transform_file" not in result


class TestAnalysisProvenance:
    """Tests for analysis provenance tracking with package version."""

    @pytest.fixture
    def local_test_atlas(self, tmp_path):
        """Create and register a local test atlas for provenance tests.

        This avoids TemplateFlow downloads for CI.
        """
        atlas_dir = tmp_path / "atlases"
        atlas_dir.mkdir()

        # Use same dimensions as synthetic_mask_img (91, 109, 91)
        shape = (91, 109, 91)
        affine = np.array(
            [
                [-2.0, 0.0, 0.0, 90.0],
                [0.0, 2.0, 0.0, -126.0],
                [0.0, 0.0, 2.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Create atlas with 5 regions
        atlas_data = np.zeros(shape, dtype=np.int16)
        atlas_data[20:40, 30:50, 20:40] = 1
        atlas_data[40:60, 30:50, 20:40] = 2
        atlas_data[20:40, 50:70, 20:40] = 3
        atlas_data[40:60, 50:70, 20:40] = 4
        atlas_data[30:50, 40:60, 40:60] = 5

        atlas_img = nib.Nifti1Image(atlas_data, affine)
        atlas_path = atlas_dir / "test_provenance_atlas.nii.gz"
        nib.save(atlas_img, atlas_path)

        # Create labels file
        labels_path = atlas_dir / "test_provenance_atlas_labels.txt"
        labels_path.write_text("1 Region_A\n2 Region_B\n3 Region_C\n4 Region_D\n5 Region_E\n")

        # Register the atlas
        register_parcellations_from_directory(atlas_dir, space="MNI152NLin6Asym", resolution=2)

        yield "test_provenance_atlas"

        # Cleanup
        try:
            unregister_parcellation("test_provenance_atlas")
        except KeyError:
            pass

    def test_analysis_provenance_includes_package_version(
        self, synthetic_mask_img, local_test_atlas
    ):
        """BaseAnalysis._create_provenance() should include package version."""
        from lacuna import MaskData, __version__
        from lacuna.analysis.regional_damage import RegionalDamage

        mask_data = MaskData(
            mask_img=synthetic_mask_img,
            metadata={"space": "MNI152NLin6Asym", "resolution": 2},
        )

        analysis = RegionalDamage(parcel_names=[local_test_atlas])
        result = analysis.run(mask_data)

        # Check provenance was created
        assert hasattr(result, "provenance")
        assert len(result.provenance) > 0

        # Last provenance record should be from this analysis
        latest_prov = result.provenance[-1]

        # Should include version from package
        assert latest_prov["version"] == __version__
        assert latest_prov["version"] != "0.1.0"  # Not hardcoded
        assert "RegionalDamage" in latest_prov["function"]

    def test_all_analyses_use_package_version(self, synthetic_mask_img, local_test_atlas):
        """All analysis classes should use package version in provenance."""
        from lacuna import MaskData, __version__
        from lacuna.analysis.parcel_aggregation import ParcelAggregation
        from lacuna.analysis.regional_damage import RegionalDamage

        mask_data = MaskData(
            mask_img=synthetic_mask_img,
            metadata={"space": "MNI152NLin6Asym", "resolution": 2},
        )

        # Test RegionalDamage
        rd = RegionalDamage(parcel_names=[local_test_atlas])
        rd_result = rd.run(mask_data)
        assert rd_result.provenance[-1]["version"] == __version__
        assert "RegionalDamage" in rd_result.provenance[-1]["function"]

        # Test ParcelAggregation
        pa = ParcelAggregation(parcel_names=[local_test_atlas])
        pa_result = pa.run(mask_data)
        assert pa_result.provenance[-1]["version"] == __version__
        assert "ParcelAggregation" in pa_result.provenance[-1]["function"]
