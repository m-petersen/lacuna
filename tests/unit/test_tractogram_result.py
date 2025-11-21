"""
Unit tests for Tractogram on-demand loading.

T023: Tests that Tractogram.get_data() can load streamlines from disk when needed.
"""

from pathlib import Path

import pytest


@pytest.mark.unit
def test_tractogram_result_get_data_returns_path_when_no_streamlines():
    """Test that get_data() returns Path when streamlines not in memory."""
    from lacuna.core.data_types import Tractogram

    tractogram_path = Path("/fake/path/to/tract.tck")

    result = Tractogram(
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
    import numpy as np

    from lacuna.core.data_types import Tractogram

    tractogram_path = Path("/fake/path/to/tract.tck")
    mock_streamlines = [
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
    ]

    result = Tractogram(
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
    """Test removed: get_data() no longer supports load_if_needed parameter."""
    pytest.skip(
        "Tractogram.get_data() simplified in T030 - returns path or cached streamlines only"
    )


@pytest.mark.unit
def test_tractogram_result_path_required():
    """Test that Tractogram requires tractogram_path."""
    from lacuna.core.data_types import Tractogram

    # Should raise error if path is missing
    with pytest.raises((TypeError, ValueError)):
        _ = Tractogram(
            name="TestTractogram",
            tractogram_path=None,  # This should be required
        )
