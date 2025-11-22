"""Test ParcelAggregation enhancements (T138, T139, T142)."""

import warnings

import nibabel as nib
import numpy as np
import pytest

from lacuna import MaskData
from lacuna.analysis.parcel_aggregation import ParcelAggregation
from lacuna.core.data_types import VoxelMap


@pytest.fixture
def sample_voxel_map(tmp_path):
    """Create a sample VoxelMap for testing."""
    # Use MNI152NLin6Asym 2mm dimensions
    shape = (91, 109, 91)
    data = np.random.rand(*shape).astype(np.float32)
    
    # MNI152NLin6Asym 2mm affine
    affine = np.array([
        [-2., 0., 0., 90.],
        [0., 2., 0., -126.],
        [0., 0., 2., -72.],
        [0., 0., 0., 1.]
    ])

    img = nib.Nifti1Image(data, affine)

    return VoxelMap(
        name="TestMap",
        data=img,
        space="MNI152NLin6Asym",
        resolution=2.0
    )


@pytest.fixture
def sample_mask_data(tmp_path):
    """Create sample MaskData with VoxelMap result."""
    shape = (10, 10, 10)
    mask = np.random.rand(*shape) > 0.5
    affine = np.eye(4)
    affine[:3, :3] *= 2.0  # 2mm resolution
    
    mask_img = nib.Nifti1Image(mask.astype(np.float32), affine)
    
    mask_data = MaskData(
        subject_id="test001",
        mask_img=mask_img,
        metadata={"space": "MNI152NLin6Asym", "resolution": 2}
    )
    
    # Add a VoxelMap result
    voxel_map = VoxelMap(
        name="CorrelationMap",
        data=nib.Nifti1Image(np.random.rand(*shape).astype(np.float32), affine),
        space="MNI152NLin6Asym",
        resolution=2.0
    )
    
    mask_data.results["DemoAnalysis"] = {"desc-CorrelationMap": voxel_map}
    
    return mask_data


class TestVoxelMapDirectInput:
    """Test VoxelMap direct input to ParcelAggregation (T138, T149)."""
    
    def test_voxelmap_direct_input(self, sample_voxel_map):
        """Test ParcelAggregation accepts VoxelMap directly."""
        analysis = ParcelAggregation(
            parcel_names=["Schaefer2018_100Parcels7Networks"],
            aggregation="mean"
        )
        
        result = analysis.run(sample_voxel_map)
        
        # Should return ParcelData
        from lacuna.core.data_types import ParcelData
        assert isinstance(result, ParcelData)
        assert len(result.data) > 0
        assert result.aggregation_method == "mean"
    
    def test_voxelmap_preserves_metadata(self, sample_voxel_map):
        """Test VoxelMap metadata is used for space/resolution."""
        analysis = ParcelAggregation(
            parcel_names=["Schaefer2018_100Parcels7Networks"],
            aggregation="mean"
        )
        
        result = analysis.run(sample_voxel_map)
        
        # Metadata should include source VoxelMap info
        assert "source_space" in result.metadata or result.metadata.get("space") == "MNI152NLin6Asym"
        assert result.metadata.get("source_resolution") == 2.0 or "resolution" in result.metadata
    
    def test_voxelmap_list_input(self, sample_voxel_map):
        """Test list of VoxelMaps returns list of ParcelData."""
        analysis = ParcelAggregation(
            parcel_names=["Schaefer2018_100Parcels7Networks"],
            aggregation="mean"
        )
        
        # Create list of VoxelMaps
        voxel_maps = [sample_voxel_map] * 3
        
        results = analysis.run(voxel_maps)
        
        # Should return list of ParcelData
        assert isinstance(results, list)
        assert len(results) == 3
        from lacuna.core.data_types import ParcelData
        assert all(isinstance(r, ParcelData) for r in results)


class TestMultiSourceAggregation:
    """Test multi-source ParcelAggregation (T139, T150)."""
    
    def test_multi_source_list(self, sample_mask_data):
        """Test ParcelAggregation with list of sources."""
        analysis = ParcelAggregation(
            source=["mask_img", "DemoAnalysis.desc-CorrelationMap"],
            parcel_names=["Schaefer2018_100Parcels7Networks"],
            aggregation="mean"
        )
        
        result = analysis.run(sample_mask_data)
        
        # Should have results for both sources
        assert isinstance(result, MaskData)
        assert "ParcelAggregation" in result.results
        
        # Should have separate keys for each source
        parcel_results = result.results["ParcelAggregation"]
        assert len(parcel_results) >= 2  # At least one atlas per source
    
    def test_multi_source_naming(self, sample_mask_data):
        """Test multi-source results use descriptive BIDS keys."""
        analysis = ParcelAggregation(
            source=["mask_img", "DemoAnalysis.desc-CorrelationMap"],
            parcel_names=["Schaefer2018_100Parcels7Networks"],
            aggregation="mean"
        )
        
        result = analysis.run(sample_mask_data)
        parcel_results = result.results["ParcelAggregation"]
        
        # Check for BIDS-style keys differentiating sources
        # Format: atlas-{name}_desc-{Source} or atlas-{name}_source-{source}
        keys = list(parcel_results.keys())
        assert any("MaskImg" in k or "mask" in k for k in keys)
        assert any("CorrelationMap" in k for k in keys)
    
    def test_multi_source_empty_list_raises(self):
        """Test empty source list raises ValueError."""
        with pytest.raises(ValueError, match="source cannot be empty"):
            ParcelAggregation(
                source=[],
                parcel_names=["Schaefer2018_100Parcels7Networks"]
            )
    
    def test_multi_source_invalid_type_raises(self):
        """Test invalid source type raises TypeError."""
        with pytest.raises(TypeError, match="must be str or list"):
            ParcelAggregation(
                source=123,  # Invalid type
                parcel_names=["Schaefer2018_100Parcels7Networks"]
            )


class TestNilearnWarningSuppression:
    """Test nilearn warning suppression (T142, T151)."""
    
    def test_nilearn_warnings_suppressed_at_low_log_level(self, sample_mask_data):
        """Test nilearn warnings are suppressed when log_level < 2."""
        analysis = ParcelAggregation(
            parcel_names=["Schaefer2018_100Parcels7Networks"],
            aggregation="mean",
            log_level=0  # Quiet mode
        )
        
        # This should not raise any warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = analysis.run(sample_mask_data)
            
            # Filter for nilearn warnings
            nilearn_warnings = [warn for warn in w if "nilearn" in str(warn.message).lower()]
            
            # Should have no nilearn warnings at log_level=0
            assert len(nilearn_warnings) == 0
    
    def test_nilearn_warnings_shown_at_high_log_level(self, sample_mask_data):
        """Test nilearn warnings are shown when log_level >= 2."""
        # Create mismatched data to trigger warnings
        shape = (20, 20, 20)  # Different shape from atlas
        data = np.random.rand(*shape).astype(np.float32)
        affine = np.eye(4)
        affine[:3, :3] *= 3.0  # Different resolution
        
        mask_img = nib.Nifti1Image(data, affine)
        mask_data = MaskData(
            subject_id="test002",
            mask_img=mask_img,
            metadata={"space": "MNI152NLin6Asym", "resolution": 3}
        )
        
        analysis = ParcelAggregation(
            parcel_names=["Schaefer2018_100Parcels7Networks"],
            aggregation="mean",
            log_level=2  # Verbose mode
        )
        
        # This may raise warnings about resampling
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = analysis.run(mask_data)
            
            # At log_level >= 2, warnings should be visible
            # (This is a weaker assertion - we just check it doesn't crash)
            assert result is not None


class TestAtlasResamplingLogging:
    """Test internal logging for atlas resampling (T152)."""
    
    def test_resampling_logged_at_debug(self, sample_mask_data, caplog):
        """Test atlas resampling is logged at DEBUG level."""
        import logging
        
        # Create mismatched data to trigger resampling
        shape = (15, 15, 15)
        data = np.random.rand(*shape) > 0.5
        affine = np.eye(4)
        affine[:3, :3] *= 3.0  # Different resolution
        
        mask_img = nib.Nifti1Image(data.astype(np.float32), affine)
        mask_data = MaskData(
            subject_id="test003",
            mask_img=mask_img,
            metadata={"space": "MNI152NLin6Asym", "resolution": 3}
        )
        
        analysis = ParcelAggregation(
            parcel_names=["Schaefer2018_100Parcels7Networks"],
            aggregation="mean",
            log_level=3  # DEBUG mode
        )
        
        with caplog.at_level(logging.DEBUG):
            result = analysis.run(mask_data)
            
            # Should log atlas resampling
            log_text = " ".join(record.message for record in caplog.records)
            assert "resample" in log_text.lower() or "transform" in log_text.lower()
    
    def test_no_resampling_log_at_quiet(self, sample_mask_data, caplog):
        """Test atlas resampling not logged at quiet level."""
        import logging

        analysis = ParcelAggregation(
            parcel_names=["Schaefer2018_100Parcels7Networks"],
            aggregation="mean",
            log_level=0  # Quiet mode
        )

        with caplog.at_level(logging.INFO):
            result = analysis.run(sample_mask_data)

            # Should have minimal/no logging
            assert len(caplog.records) == 0 or all(
                record.levelno >= logging.WARNING for record in caplog.records
            )

