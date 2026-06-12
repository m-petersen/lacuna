"""StructuralNetworkMapping dependency handling — runs without MRtrix3 installed.

The existing SNM instantiation tests are all gated behind ``requires_mrtrix`` and
therefore skipped in environments without MRtrix3, leaving the dependency-check
path untested. These mock the MRtrix availability check so the user-facing error
(and the ``check_dependencies=False`` escape hatch) are exercised everywhere.
"""

import tempfile
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def temp_connectome():
    """Register a throwaway structural connectome (tractogram never read here)."""
    from lacuna.assets.connectomes import (
        register_structural_connectome,
        unregister_structural_connectome,
    )

    name = f"test_snm_dep_{uuid.uuid4().hex[:8]}"
    with tempfile.NamedTemporaryFile(suffix=".tck", delete=False) as f:
        tck = Path(f.name)
    register_structural_connectome(
        name=name,
        space="MNI152NLin2009cAsym",
        tractogram_path=tck,
        description="dummy",
    )
    yield name
    unregister_structural_connectome(name)
    tck.unlink(missing_ok=True)


def _raise_mrtrix():
    from lacuna.utils.mrtrix import MRtrixError

    raise MRtrixError("mrtrix3 not found on PATH")


def test_snm_raises_clear_error_without_mrtrix(temp_connectome, monkeypatch):
    import lacuna.analysis.structural_network_mapping as snm_mod
    from lacuna.utils.mrtrix import MRtrixError

    monkeypatch.setattr(snm_mod, "check_mrtrix_available", _raise_mrtrix)

    with pytest.raises(MRtrixError, match="MRtrix3 is required for StructuralNetworkMapping"):
        snm_mod.StructuralNetworkMapping(connectome_name=temp_connectome)


def test_snm_skips_check_when_dependencies_disabled(temp_connectome, monkeypatch):
    import lacuna.analysis.structural_network_mapping as snm_mod

    # Even if MRtrix would report missing, check_dependencies=False must not invoke it.
    monkeypatch.setattr(snm_mod, "check_mrtrix_available", _raise_mrtrix)

    analysis = snm_mod.StructuralNetworkMapping(
        connectome_name=temp_connectome, check_dependencies=False
    )
    assert analysis.tractogram_space == "MNI152NLin2009cAsym"
