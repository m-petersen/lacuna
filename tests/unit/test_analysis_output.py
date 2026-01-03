"""Unit tests for polymorphic analysis output classes."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from lacuna.core.data_types import (
    ConnectivityMatrix,
    ParcelData,
    ScalarMetric,
    SurfaceMesh,
    Tractogram,
    VoxelMap,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_nifti():
    """Create a sample NIfTI image."""
    data = np.random.randn(91, 109, 91)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0
    return nib.Nifti1Image(data, affine)


# ============================================================================
# VoxelMap Tests (Simplified API)
# ============================================================================


def test_voxel_map_result_initialization(sample_nifti):
    """Test basic VoxelMap initialization with simplified API."""
    result = VoxelMap(name="rmap", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0)

    assert result.name == "rmap"
    assert result.data is sample_nifti
    assert result.space == "MNI152NLin6Asym"
    assert result.resolution == 2.0
    assert result.metadata == {}
    assert result.data_type == "VoxelMap"


def test_voxel_map_string_space_storage(sample_nifti):
    """Test that space is stored as string."""
    result = VoxelMap(
        name="test_map", data=sample_nifti, space="MNI152NLin2009cAsym", resolution=1.0
    )

    assert isinstance(result.space, str)
    assert result.space == "MNI152NLin2009cAsym"
    assert isinstance(result.resolution, float)
    assert result.resolution == 1.0


def test_voxel_map_get_data_returns_original(sample_nifti):
    """Test get_data returns the original data without transformation."""
    result = VoxelMap(name="test_map", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0)

    # get_data should simply return the data
    assert result.get_data() is sample_nifti


def test_voxel_map_metadata_storage(sample_nifti):
    """Test metadata storage and retrieval."""
    metadata = {"analysis_date": "2024-01-01", "parameter": 42, "nested": {"key": "value"}}

    result = VoxelMap(
        name="test_map",
        data=sample_nifti,
        space="MNI152NLin6Asym",
        resolution=2.0,
        metadata=metadata,
    )

    assert result.metadata == metadata
    assert result.metadata["analysis_date"] == "2024-01-01"


def test_voxel_map_summary(sample_nifti):
    """Test summary string generation."""
    result = VoxelMap(name="rmap", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0)

    summary = result.summary()
    assert "rmap" in summary
    assert "91, 109, 91" in summary
    assert "MNI152NLin6Asym" in summary
    assert "2.0mm" in summary


def test_voxel_map_repr(sample_nifti):
    """Test string representation."""
    result = VoxelMap(name="test_map", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0)

    repr_str = repr(result)
    assert "VoxelMap" in repr_str
    assert "test_map" in repr_str
    assert "91, 109, 91" in repr_str


def test_voxel_map_with_different_resolutions(sample_nifti):
    """Test VoxelMap with different resolution values."""
    result_1mm = VoxelMap(
        name="test_1mm", data=sample_nifti, space="MNI152NLin6Asym", resolution=1.0
    )

    result_2mm = VoxelMap(
        name="test_2mm", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0
    )

    assert result_1mm.resolution == 1.0
    assert result_2mm.resolution == 2.0


def test_voxel_map_with_different_spaces(sample_nifti):
    """Test VoxelMap with different space identifiers."""
    result_nlin6 = VoxelMap(
        name="test_nlin6", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0
    )

    result_nlin2009 = VoxelMap(
        name="test_nlin2009", data=sample_nifti, space="MNI152NLin2009cAsym", resolution=2.0
    )

    assert result_nlin6.space == "MNI152NLin6Asym"
    assert result_nlin2009.space == "MNI152NLin2009cAsym"


def test_voxel_map_optional_metadata(sample_nifti):
    """Test that metadata is optional."""
    result = VoxelMap(name="test_map", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0)

    assert result.metadata == {}
    assert isinstance(result.metadata, dict)


# ============================================================================
# ParcelData Tests
# ============================================================================


def test_roi_result_initialization():
    """Test basic ParcelData initialization."""
    data = {"Schaefer400_1": 0.5, "Schaefer400_2": 0.3, "Tian_Frontal_L": 0.8}

    result = ParcelData(
        name="atlas_aggregation",
        data=data,
        parcel_names=["Schaefer400", "Tian"],
        aggregation_method="mean",
    )

    assert result.name == "atlas_aggregation"
    assert result.data == data
    assert result.parcel_names == ["Schaefer400", "Tian"]
    assert result.aggregation_method == "mean"
    assert result.data_type == "ParcelData"


def test_roi_result_get_data_no_filter():
    """Test getting all ROI data without filter."""
    data = {"roi1": 0.5, "roi2": 0.3, "roi3": 0.8}
    result = ParcelData(name="test", data=data)

    assert result.get_data() == data


def test_roi_result_get_data_with_atlas_filter():
    """Test filtering ROI data by atlas name."""
    data = {
        "Schaefer400_1": 0.5,
        "Schaefer400_2": 0.3,
        "Tian_Frontal_L": 0.8,
        "Tian_Temporal_R": 0.6,
    }

    result = ParcelData(name="test", data=data)

    schaefer_data = result.get_data(atlas_filter="Schaefer400")
    assert len(schaefer_data) == 2
    assert "Schaefer400_1" in schaefer_data
    assert "Tian_Frontal_L" not in schaefer_data

    tian_data = result.get_data(atlas_filter="Tian")
    assert len(tian_data) == 2
    assert "Tian_Frontal_L" in tian_data


def test_roi_result_get_top_regions():
    """Test getting top N regions by value."""
    data = {"roi1": 0.5, "roi2": 0.9, "roi3": 0.2, "roi4": 0.7, "roi5": 0.3}

    result = ParcelData(name="test", data=data)

    # Top 3 descending
    top3 = result.get_top_regions(n=3)
    assert len(top3) == 3
    assert list(top3.keys()) == ["roi2", "roi4", "roi1"]
    assert list(top3.values()) == [0.9, 0.7, 0.5]

    # Top 2 ascending
    bottom2 = result.get_top_regions(n=2, ascending=True)
    assert len(bottom2) == 2
    assert list(bottom2.keys()) == ["roi3", "roi5"]


def test_roi_result_summary():
    """Test summary string generation."""
    data = {"roi1": 0.5, "roi2": 0.3, "roi3": 0.8}
    result = ParcelData(
        name="atlas_aggregation",
        data=data,
        parcel_names=["Atlas1", "Atlas2"],
        aggregation_method="mean",
    )

    summary = result.summary()
    assert "atlas_aggregation" in summary
    assert "3 regions" in summary
    assert "2 atlases" in summary
    assert "method=mean" in summary


def test_roi_result_repr():
    """Test string representation."""
    data = {"roi1": 0.5, "roi2": 0.3}
    result = ParcelData(name="test", data=data)

    repr_str = repr(result)
    assert "ParcelData" in repr_str
    assert "test" in repr_str
    assert "n_regions=2" in repr_str


# ============================================================================
# ConnectivityMatrix Tests
# ============================================================================


def test_connectivity_matrix_result_initialization():
    """Test basic ConnectivityMatrix initialization."""
    matrix = np.random.rand(10, 10)
    labels = [f"region_{i}" for i in range(10)]

    result = ConnectivityMatrix(
        name="structural_connectivity",
        matrix=matrix,
        region_labels=labels,
        matrix_type="structural",
    )

    assert result.name == "structural_connectivity"
    assert np.array_equal(result.matrix, matrix)
    assert result.region_labels == labels
    assert result.matrix_type == "structural"
    assert result.data_type == "ConnectivityMatrix"


def test_connectivity_matrix_validation_not_2d():
    """Test validation rejects non-2D matrices."""
    matrix = np.random.rand(10, 10, 10)

    with pytest.raises(ValueError, match="Matrix must be 2D"):
        ConnectivityMatrix(name="test", matrix=matrix)


def test_connectivity_matrix_validation_not_square():
    """Test validation rejects non-square matrices."""
    matrix = np.random.rand(10, 15)

    with pytest.raises(ValueError, match="Matrix must be square"):
        ConnectivityMatrix(name="test", matrix=matrix)


def test_connectivity_matrix_validation_lesioned_shape_mismatch():
    """Test that ConnectivityMatrix no longer supports lesioned_matrix parameter."""
    matrix = np.random.rand(10, 10)
    lesioned = np.random.rand(8, 8)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ConnectivityMatrix(name="test", matrix=matrix, lesioned_matrix=lesioned)


def test_connectivity_matrix_validation_labels_length_mismatch():
    """Test validation rejects labels with wrong length."""
    matrix = np.random.rand(10, 10)
    labels = [f"region_{i}" for i in range(8)]

    with pytest.raises(ValueError, match="must match matrix size"):
        ConnectivityMatrix(name="test", matrix=matrix, region_labels=labels)


def test_connectivity_matrix_get_data_intact():
    """Test getting connectivity matrix."""
    matrix = np.random.rand(10, 10)
    result = ConnectivityMatrix(name="test", matrix=matrix)

    assert np.array_equal(result.get_data(), matrix)


# Connectivity matrix tests - updated for simplified API (no lesioned_matrix, no compute_disconnection)


def test_connectivity_matrix_summary():
    """Test summary string generation."""
    matrix = np.random.rand(10, 10)

    result = ConnectivityMatrix(
        name="structural_connectivity",
        matrix=matrix,
        matrix_type="structural",
    )

    summary = result.summary()
    assert "structural_connectivity" in summary
    assert "10x10" in summary
    assert "type=structural" in summary


def test_connectivity_matrix_repr():
    """Test string representation."""
    matrix = np.random.rand(10, 10)
    result = ConnectivityMatrix(name="test", matrix=matrix)

    repr_str = repr(result)
    assert "ConnectivityMatrix" in repr_str
    assert "test" in repr_str
    assert "(10, 10)" in repr_str


# ============================================================================
# SurfaceMesh Tests
# ============================================================================


def test_surface_result_initialization():
    """Test basic SurfaceMesh initialization."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 3))
    vertex_data = np.random.rand(1000)

    result = SurfaceMesh(
        name="pial_surface",
        vertices=vertices,
        faces=faces,
        vertex_data=vertex_data,
        hemisphere="L",
        surface_type="pial",
    )

    assert result.name == "pial_surface"
    assert np.array_equal(result.vertices, vertices)
    assert np.array_equal(result.faces, faces)
    assert np.array_equal(result.vertex_data, vertex_data)
    assert result.hemisphere == "L"
    assert result.surface_type == "pial"
    assert result.data_type == "SurfaceMesh"


def test_surface_result_validation_vertices_not_nx3():
    """Test validation rejects vertices not N x 3."""
    vertices = np.random.rand(1000, 2)  # Wrong shape
    faces = np.random.randint(0, 1000, (2000, 3))

    with pytest.raises(ValueError, match="Vertices must be N x 3"):
        SurfaceMesh(name="test", vertices=vertices, faces=faces)


def test_surface_result_validation_faces_not_mx3():
    """Test validation rejects faces not M x 3."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 4))  # Wrong shape

    with pytest.raises(ValueError, match="Faces must be M x 3"):
        SurfaceMesh(name="test", vertices=vertices, faces=faces)


def test_surface_result_validation_vertex_data_length_mismatch():
    """Test validation rejects vertex data with wrong length."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 3))
    vertex_data = np.random.rand(900)  # Wrong length

    with pytest.raises(ValueError, match="must match number of vertices"):
        SurfaceMesh(name="test", vertices=vertices, faces=faces, vertex_data=vertex_data)


def test_surface_result_get_data():
    """Test getting vertex data."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 3))
    vertex_data = np.random.rand(1000)

    result = SurfaceMesh(name="test", vertices=vertices, faces=faces, vertex_data=vertex_data)

    assert np.array_equal(result.get_data(), vertex_data)


def test_surface_result_get_data_no_vertex_data():
    """Test error when getting data without vertex data."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 3))

    result = SurfaceMesh(name="test", vertices=vertices, faces=faces)

    with pytest.raises(ValueError, match="No vertex data available"):
        result.get_data()


def test_surface_result_get_mesh():
    """Test getting mesh geometry."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 3))

    result = SurfaceMesh(name="test", vertices=vertices, faces=faces)

    verts, faces_out = result.get_mesh()
    assert np.array_equal(verts, vertices)
    assert np.array_equal(faces_out, faces)


def test_surface_result_summary():
    """Test summary string generation."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 3))
    vertex_data = np.random.rand(1000)

    result = SurfaceMesh(
        name="pial_surface",
        vertices=vertices,
        faces=faces,
        vertex_data=vertex_data,
        hemisphere="L",
        surface_type="pial",
    )

    summary = result.summary()
    assert "pial_surface" in summary
    assert "1000 vertices" in summary
    assert "2000 faces" in summary
    assert "with vertex data" in summary
    assert "hemisphere=L" in summary
    assert "type=pial" in summary


def test_surface_result_repr():
    """Test string representation."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 3))

    result = SurfaceMesh(name="test", vertices=vertices, faces=faces)

    repr_str = repr(result)
    assert "SurfaceMesh" in repr_str
    assert "test" in repr_str
    assert "n_vertices=1000" in repr_str
    assert "n_faces=2000" in repr_str


# ============================================================================
# Tractogram Tests
# ============================================================================


def test_tractogram_result_initialization_with_streamlines():
    """Test Tractogram no longer accepts streamlines parameter - requires path."""
    streamlines = [np.random.rand(100, 3), np.random.rand(150, 3), np.random.rand(80, 3)]

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        Tractogram(name="tract_streamlines", streamlines=streamlines)


def test_tractogram_result_initialization_with_path():
    """Test Tractogram initialization with file path."""
    path = Path("/data/tractogram.tck")

    result = Tractogram(name="tract_streamlines", tractogram_path=path)

    assert result.tractogram_path == path
    assert result.streamlines is None  # No in-memory data


def test_tractogram_result_validation_no_data():
    """Test validation requires tractogram_path."""
    with pytest.raises(
        TypeError, match="missing 1 required positional argument: 'tractogram_path'"
    ):
        Tractogram(name="test")


def test_tractogram_result_get_data_from_memory(tmp_path):
    """Test removed: Tractogram no longer stores streamlines in memory."""
    pytest.skip("Tractogram simplified to path-only storage in T015")


def test_tractogram_result_get_data_from_file_not_implemented():
    """Test removed: get_data() now expected to load from file."""
    pytest.skip("Tractogram.get_data() implementation moved to T030")


def test_tractogram_result_get_data_no_load():
    """Test that get_data returns path when streamlines not loaded."""
    path = Path("/data/tractogram.tck")
    result = Tractogram(name="test", tractogram_path=path)

    # Should return path when streamlines not in memory
    data = result.get_data()
    assert data == path
    assert isinstance(data, Path)


def test_tractogram_result_summary(tmp_path):
    """Test summary string generation."""
    # Create dummy path
    tck_path = tmp_path / "tracts.tck"
    tck_path.touch()

    streamlines = [np.random.rand(100, 3) for _ in range(50)]
    result = Tractogram(name="tract", tractogram_path=tck_path, streamlines=streamlines)

    summary = result.summary()
    assert "tract" in summary
    assert "in-memory" in summary  # Has streamlines in memory
    assert "tracts.tck" in summary  # Path included


def test_tractogram_result_repr():
    """Test string representation."""
    path = Path("/data/tractogram.tck")
    result = Tractogram(name="test", tractogram_path=path)

    repr_str = repr(result)
    assert "Tractogram" in repr_str
    assert "test" in repr_str
    assert "in_memory=False" in repr_str


# ============================================================================
# ScalarMetric Tests
# ============================================================================


def test_misc_result_initialization_scalar():
    """Test ScalarMetric with scalar value."""
    result = ScalarMetric(name="mean_correlation", data=0.42)

    assert result.name == "mean_correlation"
    assert result.data == 0.42
    assert result.data_type == "scalar"


def test_misc_result_initialization_dict():
    """Test ScalarMetric with dictionary."""
    data = {"mean": 0.5, "std": 0.1, "count": 100}
    result = ScalarMetric(name="summary_stats", data=data)

    assert result.data == data
    assert result.data_type == "dictionary"


def test_misc_result_initialization_list():
    """Test ScalarMetric with list."""
    data = [1, 2, 3, 4, 5]
    result = ScalarMetric(name="values", data=data)

    assert result.data == data
    assert result.data_type == "sequence"


def test_misc_result_explicit_data_type():
    """Test ScalarMetric with explicit data_type."""
    result = ScalarMetric(name="custom", data={"key": "value"}, data_type="metadata")

    assert result.data_type == "metadata"


def test_misc_result_get_data():
    """Test getting data from ScalarMetric."""
    data = {"key": "value"}
    result = ScalarMetric(name="test", data=data)

    assert result.get_data() == data


def test_misc_result_summary_scalar():
    """Test summary for scalar data."""
    result = ScalarMetric(name="mean_value", data=3.14)

    summary = result.summary()
    assert "mean_value" in summary
    assert "3.14" in summary
    assert "float" in summary


def test_misc_result_summary_dict():
    """Test summary for dictionary data."""
    data = {"a": 1, "b": 2, "c": 3}
    result = ScalarMetric(name="stats", data=data)

    summary = result.summary()
    assert "stats" in summary
    assert "dict with 3 keys" in summary


def test_misc_result_summary_sequence():
    """Test summary for sequence data."""
    data = [1, 2, 3, 4, 5]
    result = ScalarMetric(name="values", data=data)

    summary = result.summary()
    assert "values" in summary
    assert "list with 5 items" in summary


def test_misc_result_repr():
    """Test string representation."""
    result = ScalarMetric(name="test", data=42)

    repr_str = repr(result)
    assert "ScalarMetric" in repr_str
    assert "test" in repr_str
    assert "scalar" in repr_str
