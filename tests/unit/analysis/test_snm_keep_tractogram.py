"""Test that StructuralNetworkMapping can expose the filtered tractogram."""

from pathlib import Path

import pytest

from lacuna.analysis import StructuralNetworkMapping
from lacuna.assets.connectomes import (
    register_structural_connectome,
    unregister_structural_connectome,
)


@pytest.fixture
def fake_connectome(tmp_path):
    """Register a fake structural connectome for testing."""
    tck_path = tmp_path / "fake.tck"
    tck_path.touch()
    name = "test_keep_filtered"
    register_structural_connectome(
        name=name,
        space="MNI152NLin2009cAsym",
        tractogram_path=tck_path,
        description="Fake connectome for testing",
    )
    yield name
    unregister_structural_connectome(name)


class TestKeepFilteredTractogram:
    def test_parameter_accepted(self, fake_connectome):
        """Verify the parameter is accepted without error."""
        snm = StructuralNetworkMapping(
            connectome_name=fake_connectome,
            keep_filtered_tractogram=True,
            check_dependencies=False,
        )
        assert snm.keep_filtered_tractogram is True

    def test_default_is_false(self, fake_connectome):
        snm = StructuralNetworkMapping(
            connectome_name=fake_connectome,
            check_dependencies=False,
        )
        assert snm.keep_filtered_tractogram is False
