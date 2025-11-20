"""Unit tests for polymorphic analysis output classes."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from lacuna.core.output import (
    ConnectivityMatrixResult,
    MiscResult,
    AtlasAggregationResult,
    SurfaceResult,
    TractogramResult,
    VoxelMapResult,
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
# VoxelMapResult Tests (Simplified API)
# ============================================================================


def test_voxel_map_result_initialization(sample_nifti):
    """Test basic VoxelMapResult initialization with simplified API."""
    result = VoxelMapResult(
        name="correlation_map", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0
    )

    assert result.name == "correlation_map"
    assert result.data is sample_nifti
    assert result.space == "MNI152NLin6Asym"
    assert result.resolution == 2.0
    assert result.metadata == {}
    assert result.result_type == "VoxelMapResult"


def test_voxel_map_string_space_storage(sample_nifti):
    """Test that space is stored as string."""
    result = VoxelMapResult(
        name="test_map", data=sample_nifti, space="MNI152NLin2009cAsym", resolution=1.0
    )

    assert isinstance(result.space, str)
    assert result.space == "MNI152NLin2009cAsym"
    assert isinstance(result.resolution, float)
    assert result.resolution == 1.0


def test_voxel_map_get_data_returns_original(sample_nifti):
    """Test get_data returns the original data without transformation."""
    result = VoxelMapResult(
        name="test_map", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0
    )

    # get_data should simply return the data
    assert result.get_data() is sample_nifti


def test_voxel_map_metadata_storage(sample_nifti):
    """Test metadata storage and retrieval."""
    metadata = {"analysis_date": "2024-01-01", "parameter": 42, "nested": {"key": "value"}}

    result = VoxelMapResult(
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
    result = VoxelMapResult(
        name="correlation_map", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0
    )

    summary = result.summary()
    assert "correlation_map" in summary
    assert "91, 109, 91" in summary
    assert "MNI152NLin6Asym" in summary
    assert "2.0mm" in summary


def test_voxel_map_repr(sample_nifti):
    """Test string representation."""
    result = VoxelMapResult(
        name="test_map", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0
    )

    repr_str = repr(result)
    assert "VoxelMapResult" in repr_str
    assert "test_map" in repr_str
    assert "91, 109, 91" in repr_str


def test_voxel_map_with_different_resolutions(sample_nifti):
    """Test VoxelMapResult with different resolution values."""
    result_1mm = VoxelMapResult(
        name="test_1mm", data=sample_nifti, space="MNI152NLin6Asym", resolution=1.0
    )

    result_2mm = VoxelMapResult(
        name="test_2mm", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0
    )

    assert result_1mm.resolution == 1.0
    assert result_2mm.resolution == 2.0


def test_voxel_map_with_different_spaces(sample_nifti):
    """Test VoxelMapResult with different space identifiers."""
    result_nlin6 = VoxelMapResult(
        name="test_nlin6", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0
    )

    result_nlin2009 = VoxelMapResult(
        name="test_nlin2009", data=sample_nifti, space="MNI152NLin2009cAsym", resolution=2.0
    )

    assert result_nlin6.space == "MNI152NLin6Asym"
    assert result_nlin2009.space == "MNI152NLin2009cAsym"


def test_voxel_map_optional_metadata(sample_nifti):
    """Test that metadata is optional."""
    result = VoxelMapResult(
        name="test_map", data=sample_nifti, space="MNI152NLin6Asym", resolution=2.0
    )

    assert result.metadata == {}
    assert isinstance(result.metadata, dict)


# ============================================================================
# AtlasAggregationResult Tests
# ============================================================================


def test_roi_result_initialization():
    """Test basic AtlasAggregationResult initialization."""
    data = {"Schaefer400_1": 0.5, "Schaefer400_2": 0.3, "AAL_Frontal_L": 0.8}

    result = AtlasAggregationResult(
        name="atlas_aggregation",
        data=data,
        atlas_names=["Schaefer400", "AAL"],
        aggregation_method="mean",
    )

    assert result.name == "atlas_aggregation"
    assert result.data == data
    assert result.atlas_names == ["Schaefer400", "AAL"]
    assert result.aggregation_method == "mean"
    assert result.result_type == "AtlasAggregationResult"


def test_roi_result_get_data_no_filter():
    """Test getting all ROI data without filter."""
    data = {"roi1": 0.5, "roi2": 0.3, "roi3": 0.8}
    result = AtlasAggregationResult(name="test", data=data)

    assert result.get_data() == data


def test_roi_result_get_data_with_atlas_filter():
    """Test filtering ROI data by atlas name."""
    data = {"Schaefer400_1": 0.5, "Schaefer400_2": 0.3, "AAL_Frontal_L": 0.8, "AAL_Temporal_R": 0.6}

    result = AtlasAggregationResult(name="test", data=data)

    schaefer_data = result.get_data(atlas_filter="Schaefer400")
    assert len(schaefer_data) == 2
    assert "Schaefer400_1" in schaefer_data
    assert "AAL_Frontal_L" not in schaefer_data

    aal_data = result.get_data(atlas_filter="AAL")
    assert len(aal_data) == 2
    assert "AAL_Frontal_L" in aal_data


def test_roi_result_get_top_regions():
    """Test getting top N regions by value."""
    data = {"roi1": 0.5, "roi2": 0.9, "roi3": 0.2, "roi4": 0.7, "roi5": 0.3}

    result = AtlasAggregationResult(name="test", data=data)

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
    result = AtlasAggregationResult(
        name="atlas_aggregation",
        data=data,
        atlas_names=["Atlas1", "Atlas2"],
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
    result = AtlasAggregationResult(name="test", data=data)

    repr_str = repr(result)
    assert "AtlasAggregationResult" in repr_str
    assert "test" in repr_str
    assert "n_regions=2" in repr_str


# ============================================================================
# ConnectivityMatrixResult Tests
# ============================================================================


def test_connectivity_matrix_result_initialization():
    """Test basic ConnectivityMatrixResult initialization."""
    matrix = np.random.rand(10, 10)
    labels = [f"region_{i}" for i in range(10)]

    result = ConnectivityMatrixResult(
        name="structural_connectivity",
        matrix=matrix,
        region_labels=labels,
        matrix_type="structural",
    )

    assert result.name == "structural_connectivity"
    assert np.array_equal(result.matrix, matrix)
    assert result.region_labels == labels
    assert result.matrix_type == "structural"
    assert result.result_type == "ConnectivityMatrixResult"


def test_connectivity_matrix_validation_not_2d():
    """Test validation rejects non-2D matrices."""
    matrix = np.random.rand(10, 10, 10)

    with pytest.raises(ValueError, match="Matrix must be 2D"):
        ConnectivityMatrixResult(name="test", matrix=matrix)


def test_connectivity_matrix_validation_not_square():
    """Test validation rejects non-square matrices."""
    matrix = np.random.rand(10, 15)

    with pytest.raises(ValueError, match="Matrix must be square"):
        ConnectivityMatrixResult(name="test", matrix=matrix)


def test_connectivity_matrix_validation_lesioned_shape_mismatch():
    """Test validation rejects lesioned matrix with wrong shape."""
    matrix = np.random.rand(10, 10)
    lesioned = np.random.rand(8, 8)

    with pytest.raises(ValueError, match="must match intact matrix shape"):
        ConnectivityMatrixResult(name="test", matrix=matrix, lesioned_matrix=lesioned)


def test_connectivity_matrix_validation_labels_length_mismatch():
    """Test validation rejects labels with wrong length."""
    matrix = np.random.rand(10, 10)
    labels = [f"region_{i}" for i in range(8)]

    with pytest.raises(ValueError, match="must match matrix size"):
        ConnectivityMatrixResult(name="test", matrix=matrix, region_labels=labels)


def test_connectivity_matrix_get_data_intact():
    """Test getting intact connectivity matrix."""
    matrix = np.random.rand(10, 10)
    result = ConnectivityMatrixResult(name="test", matrix=matrix)

    assert np.array_equal(result.get_data(), matrix)
    assert np.array_equal(result.get_data(lesioned=False), matrix)


def test_connectivity_matrix_get_data_lesioned():
    """Test getting lesioned connectivity matrix."""
    matrix = np.random.rand(10, 10)
    lesioned = np.random.rand(10, 10)

    result = ConnectivityMatrixResult(name="test", matrix=matrix, lesioned_matrix=lesioned)

    assert np.array_equal(result.get_data(lesioned=True), lesioned)


def test_connectivity_matrix_get_data_lesioned_not_available():
    """Test error when requesting lesioned matrix that doesn't exist."""
    matrix = np.random.rand(10, 10)
    result = ConnectivityMatrixResult(name="test", matrix=matrix)

    with pytest.raises(ValueError, match="No lesioned matrix available"):
        result.get_data(lesioned=True)


def test_connectivity_matrix_compute_disconnection_absolute():
    """Test absolute disconnection computation."""
    intact = np.array([[1.0, 0.8], [0.8, 1.0]])
    lesioned = np.array([[1.0, 0.3], [0.3, 1.0]])

    result = ConnectivityMatrixResult(name="test", matrix=intact, lesioned_matrix=lesioned)

    disconnection = result.compute_disconnection(method="absolute")
    expected = np.array([[0.0, 0.5], [0.5, 0.0]])

    np.testing.assert_array_almost_equal(disconnection, expected)


def test_connectivity_matrix_compute_disconnection_relative():
    """Test relative disconnection computation."""
    intact = np.array([[1.0, 0.8], [0.8, 1.0]])
    lesioned = np.array([[1.0, 0.4], [0.4, 1.0]])

    result = ConnectivityMatrixResult(name="test", matrix=intact, lesioned_matrix=lesioned)

    disconnection = result.compute_disconnection(method="relative")
    expected = np.array([[0.0, 0.5], [0.5, 0.0]])  # (0.8-0.4)/0.8 = 0.5

    np.testing.assert_array_almost_equal(disconnection, expected)


def test_connectivity_matrix_compute_disconnection_percent():
    """Test percent disconnection computation."""
    intact = np.array([[1.0, 0.8], [0.8, 1.0]])
    lesioned = np.array([[1.0, 0.4], [0.4, 1.0]])

    result = ConnectivityMatrixResult(name="test", matrix=intact, lesioned_matrix=lesioned)

    disconnection = result.compute_disconnection(method="percent")
    expected = np.array([[0.0, 50.0], [50.0, 0.0]])

    np.testing.assert_array_almost_equal(disconnection, expected)


def test_connectivity_matrix_compute_disconnection_normalized():
    """Test normalized disconnection computation."""
    intact = np.array([[1.0, 0.8, 0.6], [0.8, 1.0, 0.4], [0.6, 0.4, 1.0]])
    lesioned = np.array([[1.0, 0.2, 0.6], [0.2, 1.0, 0.4], [0.6, 0.4, 1.0]])

    result = ConnectivityMatrixResult(name="test", matrix=intact, lesioned_matrix=lesioned)

    disconnection = result.compute_disconnection(method="absolute", normalize=True)

    # Should be normalized to [0, 1]
    assert disconnection.min() == 0.0
    assert disconnection.max() == 1.0


def test_connectivity_matrix_compute_disconnection_no_lesioned():
    """Test error when computing disconnection without lesioned matrix."""
    matrix = np.random.rand(10, 10)
    result = ConnectivityMatrixResult(name="test", matrix=matrix)

    with pytest.raises(ValueError, match="Cannot compute disconnection"):
        result.compute_disconnection()


def test_connectivity_matrix_compute_disconnection_invalid_method():
    """Test error with invalid disconnection method."""
    intact = np.random.rand(10, 10)
    lesioned = np.random.rand(10, 10)

    result = ConnectivityMatrixResult(name="test", matrix=intact, lesioned_matrix=lesioned)

    with pytest.raises(ValueError, match="Invalid method"):
        result.compute_disconnection(method="invalid")


def test_connectivity_matrix_summary():
    """Test summary string generation."""
    matrix = np.random.rand(10, 10)
    lesioned = np.random.rand(10, 10)

    result = ConnectivityMatrixResult(
        name="structural_connectivity",
        matrix=matrix,
        lesioned_matrix=lesioned,
        matrix_type="structural",
    )

    summary = result.summary()
    assert "structural_connectivity" in summary
    assert "10x10" in summary
    assert "with lesioned version" in summary
    assert "type=structural" in summary


def test_connectivity_matrix_repr():
    """Test string representation."""
    matrix = np.random.rand(10, 10)
    result = ConnectivityMatrixResult(name="test", matrix=matrix)

    repr_str = repr(result)
    assert "ConnectivityMatrixResult" in repr_str
    assert "test" in repr_str
    assert "(10, 10)" in repr_str
    assert "has_lesioned=False" in repr_str


# ============================================================================
# SurfaceResult Tests
# ============================================================================


def test_surface_result_initialization():
    """Test basic SurfaceResult initialization."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 3))
    vertex_data = np.random.rand(1000)

    result = SurfaceResult(
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
    assert result.result_type == "SurfaceResult"


def test_surface_result_validation_vertices_not_nx3():
    """Test validation rejects vertices not N x 3."""
    vertices = np.random.rand(1000, 2)  # Wrong shape
    faces = np.random.randint(0, 1000, (2000, 3))

    with pytest.raises(ValueError, match="Vertices must be N x 3"):
        SurfaceResult(name="test", vertices=vertices, faces=faces)


def test_surface_result_validation_faces_not_mx3():
    """Test validation rejects faces not M x 3."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 4))  # Wrong shape

    with pytest.raises(ValueError, match="Faces must be M x 3"):
        SurfaceResult(name="test", vertices=vertices, faces=faces)


def test_surface_result_validation_vertex_data_length_mismatch():
    """Test validation rejects vertex data with wrong length."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 3))
    vertex_data = np.random.rand(900)  # Wrong length

    with pytest.raises(ValueError, match="must match number of vertices"):
        SurfaceResult(name="test", vertices=vertices, faces=faces, vertex_data=vertex_data)


def test_surface_result_get_data():
    """Test getting vertex data."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 3))
    vertex_data = np.random.rand(1000)

    result = SurfaceResult(name="test", vertices=vertices, faces=faces, vertex_data=vertex_data)

    assert np.array_equal(result.get_data(), vertex_data)


def test_surface_result_get_data_no_vertex_data():
    """Test error when getting data without vertex data."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 3))

    result = SurfaceResult(name="test", vertices=vertices, faces=faces)

    with pytest.raises(ValueError, match="No vertex data available"):
        result.get_data()


def test_surface_result_get_mesh():
    """Test getting mesh geometry."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 3))

    result = SurfaceResult(name="test", vertices=vertices, faces=faces)

    verts, faces_out = result.get_mesh()
    assert np.array_equal(verts, vertices)
    assert np.array_equal(faces_out, faces)


def test_surface_result_summary():
    """Test summary string generation."""
    vertices = np.random.rand(1000, 3)
    faces = np.random.randint(0, 1000, (2000, 3))
    vertex_data = np.random.rand(1000)

    result = SurfaceResult(
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

    result = SurfaceResult(name="test", vertices=vertices, faces=faces)

    repr_str = repr(result)
    assert "SurfaceResult" in repr_str
    assert "test" in repr_str
    assert "n_vertices=1000" in repr_str
    assert "n_faces=2000" in repr_str


# ============================================================================
# TractogramResult Tests
# ============================================================================


def test_tractogram_result_initialization_with_streamlines():
    """Test TractogramResult initialization with in-memory streamlines."""
    streamlines = [np.random.rand(100, 3), np.random.rand(150, 3), np.random.rand(80, 3)]

    result = TractogramResult(name="tract_streamlines", streamlines=streamlines)

    assert result.name == "tract_streamlines"
    assert result.streamlines == streamlines
    assert result.n_streamlines == 3
    assert result.result_type == "TractogramResult"


def test_tractogram_result_initialization_with_path():
    """Test TractogramResult initialization with file path."""
    path = Path("/data/tractogram.tck")

    result = TractogramResult(name="tract_streamlines", tractogram_path=path, n_streamlines=5000)

    assert result.tractogram_path == path
    assert result.n_streamlines == 5000
    assert result.streamlines is None


def test_tractogram_result_validation_no_data():
    """Test validation requires either streamlines or path."""
    with pytest.raises(ValueError, match="Must provide either streamlines or tractogram_path"):
        TractogramResult(name="test")


def test_tractogram_result_get_data_from_memory():
    """Test getting streamlines from memory."""
    streamlines = [np.random.rand(100, 3) for _ in range(10)]
    result = TractogramResult(name="test", streamlines=streamlines)

    data = result.get_data()
    assert data == streamlines


def test_tractogram_result_get_data_from_file_not_implemented():
    """Test that loading from file raises NotImplementedError."""
    path = Path("/data/tractogram.tck")
    result = TractogramResult(name="test", tractogram_path=path, n_streamlines=100)

    with pytest.raises(
        NotImplementedError, match="Loading streamlines from file not yet implemented"
    ):
        result.get_data(load_if_needed=True)


def test_tractogram_result_get_data_no_load():
    """Test error when data not available and load_if_needed=False."""
    path = Path("/data/tractogram.tck")
    result = TractogramResult(name="test", tractogram_path=path, n_streamlines=100)

    with pytest.raises(ValueError, match="No streamlines available"):
        result.get_data(load_if_needed=False)


def test_tractogram_result_summary():
    """Test summary string generation."""
    streamlines = [np.random.rand(100, 3) for _ in range(50)]
    result = TractogramResult(name="tract", streamlines=streamlines)

    summary = result.summary()
    assert "tract" in summary
    assert "50 streamlines" in summary
    assert "in memory" in summary


def test_tractogram_result_repr():
    """Test string representation."""
    path = Path("/data/tractogram.tck")
    result = TractogramResult(name="test", tractogram_path=path, n_streamlines=1000)

    repr_str = repr(result)
    assert "TractogramResult" in repr_str
    assert "test" in repr_str
    assert "n_streamlines=1000" in repr_str
    assert "in_memory=False" in repr_str


# ============================================================================
# MiscResult Tests
# ============================================================================


def test_misc_result_initialization_scalar():
    """Test MiscResult with scalar value."""
    result = MiscResult(name="mean_correlation", data=0.42)

    assert result.name == "mean_correlation"
    assert result.data == 0.42
    assert result.data_type == "scalar"
    assert result.result_type == "MiscResult"


def test_misc_result_initialization_dict():
    """Test MiscResult with dictionary."""
    data = {"mean": 0.5, "std": 0.1, "count": 100}
    result = MiscResult(name="summary_stats", data=data)

    assert result.data == data
    assert result.data_type == "dictionary"


def test_misc_result_initialization_list():
    """Test MiscResult with list."""
    data = [1, 2, 3, 4, 5]
    result = MiscResult(name="values", data=data)

    assert result.data == data
    assert result.data_type == "sequence"


def test_misc_result_explicit_data_type():
    """Test MiscResult with explicit data_type."""
    result = MiscResult(name="custom", data={"key": "value"}, data_type="metadata")

    assert result.data_type == "metadata"


def test_misc_result_get_data():
    """Test getting data from MiscResult."""
    data = {"key": "value"}
    result = MiscResult(name="test", data=data)

    assert result.get_data() == data


def test_misc_result_summary_scalar():
    """Test summary for scalar data."""
    result = MiscResult(name="mean_value", data=3.14)

    summary = result.summary()
    assert "mean_value" in summary
    assert "3.14" in summary
    assert "float" in summary


def test_misc_result_summary_dict():
    """Test summary for dictionary data."""
    data = {"a": 1, "b": 2, "c": 3}
    result = MiscResult(name="stats", data=data)

    summary = result.summary()
    assert "stats" in summary
    assert "dict with 3 keys" in summary


def test_misc_result_summary_sequence():
    """Test summary for sequence data."""
    data = [1, 2, 3, 4, 5]
    result = MiscResult(name="values", data=data)

    summary = result.summary()
    assert "values" in summary
    assert "list with 5 items" in summary


def test_misc_result_repr():
    """Test string representation."""
    result = MiscResult(name="test", data=42)

    repr_str = repr(result)
    assert "MiscResult" in repr_str
    assert "test" in repr_str
    assert "scalar" in repr_str
