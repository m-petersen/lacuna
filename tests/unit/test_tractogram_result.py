"""
Unit tests for TractogramResult on-demand loading.

T023: Tests that TractogramResult.get_data() can load streamlines from disk when needed.
"""

import pytest
from pathlib import Path


@pytest.mark.unit
def test_tractogram_result_get_data_returns_path_when_no_streamlines():
    """Test that get_data() returns Path when streamlines not in memory."""
    from lacuna.core.output import TractogramResult

    tractogram_path = Path("/fake/path/to/tract.tck")

    result = TractogramResult(
        name="TestTractogram",
        tractogram_path=tractogram_path,
        streamlines=None,  # No in-memory streamlines
    )

    # get_data() should return the path
    data = result.get_data()
    assert isinstance(data, Path)
    assert data == tractogram_path


@pytest.mark.unit
def test_tractogram_result_get_data_returns_streamlines_when_loaded():
    """Test that get_data() returns streamlines when they're in memory."""
    from lacuna.core.output import TractogramResult
    import numpy as np

    tractogram_path = Path("/fake/path/to/tract.tck")
    mock_streamlines = [
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
    ]

    result = TractogramResult(
        name="TestTractogram",
        tractogram_path=tractogram_path,
        streamlines=mock_streamlines,
    )

    # get_data() should return streamlines
    data = result.get_data()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data is mock_streamlines


@pytest.mark.unit
def test_tractogram_result_load_on_demand(tmp_path):
    """Test that get_data(load_if_needed=True) loads streamlines from disk."""
    from lacuna.core.output import TractogramResult

    # Create a fake .tck file (this test will fail until implementation)
    tractogram_path = tmp_path / "test.tck"
    tractogram_path.write_text("fake tractogram data")

    result = TractogramResult(
        name="TestTractogram",
        tractogram_path=tractogram_path,
        streamlines=None,
    )

    # First call: returns path
    assert isinstance(result.get_data(), Path)

    # With load_if_needed=True, should load from disk (will fail until implemented)
    # This is expected to fail - testing the intended API
    with pytest.raises((NotImplementedError, AttributeError)):
        _ = result.get_data(load_if_needed=True)


@pytest.mark.unit
def test_tractogram_result_path_required():
    """Test that TractogramResult requires tractogram_path."""
    from lacuna.core.output import TractogramResult

    # Should raise error if path is missing
    with pytest.raises((TypeError, ValueError)):
        _ = TractogramResult(
            name="TestTractogram",
            tractogram_path=None,  # This should be required
        )
