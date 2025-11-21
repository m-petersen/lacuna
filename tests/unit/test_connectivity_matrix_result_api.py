"""Tests for ConnectivityMatrix API usage.

Tests ensure correct initialization of ConnectivityMatrix with proper field names.
"""

import numpy as np
import pytest

from lacuna.core.data_types import ConnectivityMatrix


class TestConnectivityMatrixResultAPI:
    """Test correct API usage for ConnectivityMatrix."""

    def test_connectivity_matrix_result_correct_initialization(self):
        """ConnectivityMatrix should use 'matrix' parameter, not 'data'."""
        # Create a simple 3x3 connectivity matrix
        matrix = np.array([[0, 10, 5], [10, 0, 8], [5, 8, 0]], dtype=np.float32)

        labels = ["region_A", "region_B", "region_C"]

        # CORRECT API: Use 'matrix' parameter
        result = ConnectivityMatrix(
            name="test_connectivity",
            matrix=matrix,
            region_labels=labels,
            matrix_type="structural",
            metadata={"atlas": "TestAtlas"},
        )

        # Verify initialization
        assert result.name == "test_connectivity"
        assert np.array_equal(result.matrix, matrix)
        assert result.region_labels == labels
        assert result.matrix_type == "structural"
        assert result.metadata["atlas"] == "TestAtlas"

        # Verify get_data() returns matrix
        retrieved_matrix = result.get_data()
        assert np.array_equal(retrieved_matrix, matrix)

    def test_connectivity_matrix_result_with_lesioned_matrix(self):
        """ConnectivityMatrix no longer supports lesioned_matrix parameter."""
        intact_matrix = np.array([[0, 10, 5], [10, 0, 8], [5, 8, 0]], dtype=np.float32)

        lesioned_matrix = np.array([[0, 5, 2], [5, 0, 4], [2, 4, 0]], dtype=np.float32)

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            ConnectivityMatrix(
                name="test_connectivity_lesioned",
                matrix=intact_matrix,
                lesioned_matrix=lesioned_matrix,
                region_labels=["A", "B", "C"],
            )

    def test_connectivity_matrix_result_incorrect_parameter_name(self):
        """ConnectivityMatrix should fail with 'data' parameter."""
        matrix = np.array([[0, 1], [1, 0]], dtype=np.float32)

        # INCORRECT API: Using 'data' instead of 'matrix' should fail
        with pytest.raises(TypeError, match="unexpected keyword argument 'data'"):
            ConnectivityMatrix(
                name="test", data=matrix, region_labels=["A", "B"]  # Wrong parameter name
            )

    def test_connectivity_matrix_result_incorrect_label_parameter(self):
        """ConnectivityMatrix should fail with 'row_labels'/'column_labels'."""
        matrix = np.array([[0, 1], [1, 0]], dtype=np.float32)
        labels = ["A", "B"]

        # INCORRECT API: Using 'row_labels' instead of 'region_labels'
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            ConnectivityMatrix(
                name="test",
                matrix=matrix,
                row_labels=labels,  # Wrong parameter name
                column_labels=labels,  # Wrong parameter name
            )

    def test_connectivity_matrix_result_validation(self):
        """ConnectivityMatrix should validate matrix shape."""
        # Non-square matrix should fail
        non_square = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        with pytest.raises(ValueError, match="Matrix must be square"):
            ConnectivityMatrix(name="test", matrix=non_square)

        # 3D matrix should fail
        matrix_3d = np.random.rand(3, 3, 3).astype(np.float32)

        with pytest.raises(ValueError, match="Matrix must be 2D"):
            ConnectivityMatrix(name="test", matrix=matrix_3d)

    def test_connectivity_matrix_result_labels_validation(self):
        """ConnectivityMatrix should validate label count."""
        matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=np.float32)

        # Wrong number of labels should fail
        with pytest.raises(ValueError, match="Number of labels.*must match matrix size"):
            ConnectivityMatrix(
                name="test", matrix=matrix, region_labels=["A", "B"]  # Only 2 labels for 3x3 matrix
            )

    def test_connectivity_matrix_result_disconnection_computation(self):
        """Test removed: lesioned_matrix parameter no longer exists (removed in T014)."""
        # This functionality moved to StructuralNetworkMapping analysis module
        pytest.skip("lesioned_matrix parameter removed from ConnectivityMatrix in T014")
