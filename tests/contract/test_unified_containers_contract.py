"""
Contract tests for unified data type containers.

These tests validate that the unified container pattern works correctly,
ensuring classes can be used as both inputs and outputs for analyses.
"""

import nibabel as nib
import numpy as np

from lacuna.core.data_types import (
    ConnectivityMatrix,
    ParcelData,
    ScalarMetric,
    SurfaceMesh,
    Tractogram,
    VoxelMap,
)


def test_voxel_map_creation():
    """Test VoxelMap can be created with required attributes."""
    data = nib.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))
    vmap = VoxelMap(name="test_map", data=data, space="MNI152NLin6Asym", resolution=2.0)
    assert vmap.name == "test_map"
    assert vmap.space == "MNI152NLin6Asym"
    assert vmap.resolution == 2.0
    assert isinstance(vmap.get_data(), nib.Nifti1Image)


def test_parcel_data_creation():
    """Test ParcelData can be created with required attributes."""
    data = {"region_1": 0.5, "region_2": 0.8}
    pdata = ParcelData(
        name="test_parcel",
        data=data,
        parcel_names=["Schaefer2018_100Parcels7Networks"],
        aggregation_method="mean",
    )

    assert pdata.name == "test_parcel"
    assert pdata.parcel_names == ["Schaefer2018_100Parcels7Networks"]
    assert pdata.aggregation_method == "mean"
    assert pdata.get_data() == data


def test_connectivity_matrix_creation():
    """Test ConnectivityMatrix can be created with required attributes."""
    matrix = np.random.rand(10, 10)
    labels = [f"region_{i}" for i in range(10)]

    cmatrix = ConnectivityMatrix(
        name="test_matrix", matrix=matrix, region_labels=labels, matrix_type="structural"
    )

    assert cmatrix.name == "test_matrix"
    assert cmatrix.matrix_type == "structural"
    assert len(cmatrix.region_labels) == 10
    assert np.array_equal(cmatrix.get_data(), matrix)


def test_tractogram_creation(tmp_path):
    """Test Tractogram can be created with path."""
    tck_path = tmp_path / "test.tck"
    tck_path.write_text("dummy tractogram")

    tractogram = Tractogram(name="test_tractogram", tractogram_path=tck_path)

    assert tractogram.name == "test_tractogram"
    assert tractogram.tractogram_path == tck_path


def test_surface_mesh_creation():
    """Test SurfaceMesh can be created with vertices and faces."""
    vertices = np.random.rand(100, 3)
    faces = np.random.randint(0, 100, (50, 3))

    mesh = SurfaceMesh(name="test_mesh", vertices=vertices, faces=faces)

    assert mesh.name == "test_mesh"
    assert mesh.vertices.shape == (100, 3)
    assert mesh.faces.shape == (50, 3)


def test_scalar_metric_creation():
    """Test ScalarMetric can be created with any data type."""
    metric = ScalarMetric(name="test_metric", data=0.75, data_type="scalar")

    assert metric.name == "test_metric"
    assert metric.data == 0.75
    assert metric.data_type == "scalar"


def test_voxel_map_works_as_input_and_output():
    """Test VoxelMap can be used as both input and output (unified pattern)."""
    # Create a VoxelMap as if it were analysis output
    data = nib.Nifti1Image(np.random.rand(10, 10, 10), np.eye(4))
    vmap = VoxelMap(
        name="network_map",
        data=data,
        space="MNI152NLin6Asym",
        resolution=2.0,
        metadata={"analysis": "FunctionalNetworkMapping"},
    )

    # Now use it as input (e.g., to ParcelAggregation)
    # The fact that we can pass it around and access its attributes
    # demonstrates the unified pattern
    assert isinstance(vmap, VoxelMap)
    assert vmap.space == "MNI152NLin6Asym"
    assert vmap.resolution == 2.0

    # Can extract data for use in another analysis
    extracted_data = vmap.get_data()
    assert isinstance(extracted_data, nib.Nifti1Image)


def test_parcel_data_metadata_preservation():
    """Test ParcelData preserves metadata from analysis."""
    data = {"region_1": 0.5, "region_2": 0.8}
    pdata = ParcelData(
        name="test_parcel",
        data=data,
        parcel_names=["Schaefer2018_100Parcels7Networks"],
        aggregation_method="mean",
        metadata={"source": "network_map", "threshold": 0.0, "n_regions": 2},
    )

    assert "source" in pdata.metadata
    assert pdata.metadata["source"] == "network_map"
    assert pdata.metadata["threshold"] == 0.0
