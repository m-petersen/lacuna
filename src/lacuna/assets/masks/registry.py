"""Brain mask registry.

Binary brain masks per coordinate space and resolution. The masks are the
TemplateFlow ``desc-brain`` masks, redistributed via OSF (https://osf.io/yz9mb/)
and downloaded (with SHA-256 verification) and cached on first use. They define
the in-brain voxels for operations that need one — e.g. extracting the GSP1000
functional connectome on the MNI152NLin6Asym 2 mm grid.
"""

from __future__ import annotations

from dataclasses import dataclass

from lacuna.assets.base import AssetRegistry, SpatialAssetMetadata


@dataclass(frozen=True)
class BrainMaskMetadata(SpatialAssetMetadata):
    """Metadata for a binary brain mask.

    Attributes
    ----------
    space : str
        Coordinate space (e.g. "MNI152NLin6Asym").
    resolution : float
        Voxel resolution in mm (1.0 or 2.0).
    url : str
        Download URL (fetched and cached on first use).
    sha256 : str
        Expected SHA-256 of the file (verified on download).
    source : str
        Origin of the mask data.
    """

    url: str = ""
    sha256: str = ""
    source: str = "templateflow"


def mask_name(space: str, resolution: float) -> str:
    """Registry key for a brain mask, e.g. ``MNI152NLin6Asym_2mm``."""
    return f"{space}_{resolution:g}mm"


BRAIN_MASK_REGISTRY = AssetRegistry[BrainMaskMetadata]("brain mask")

# TemplateFlow desc-brain masks, redistributed via OSF (https://osf.io/yz9mb/).
# URL<->file mapping is verified by content hash, not by filename/label.
_KNOWN_MASKS = [
    BrainMaskMetadata(
        name=mask_name("MNI152NLin6Asym", 1.0),
        description="MNI152NLin6Asym brain mask, 1 mm",
        space="MNI152NLin6Asym",
        resolution=1.0,
        url="https://osf.io/2bjz3/download",
        sha256="2214902e77ef40ce44a66c8db1a55235e254843d30ce242767b135e2c046d77b",
    ),
    BrainMaskMetadata(
        name=mask_name("MNI152NLin6Asym", 2.0),
        description="MNI152NLin6Asym brain mask, 2 mm",
        space="MNI152NLin6Asym",
        resolution=2.0,
        url="https://osf.io/nb7rt/download",
        sha256="61b4ae898807264ea026a6a8f4f478ab58ae817eb97181f8bd62bb5546ca611c",
    ),
    BrainMaskMetadata(
        name=mask_name("MNI152NLin2009cAsym", 1.0),
        description="MNI152NLin2009cAsym brain mask, 1 mm",
        space="MNI152NLin2009cAsym",
        resolution=1.0,
        url="https://osf.io/4mtvj/download",
        sha256="5ef11991a77ad081690bc4e272ecc9153bd35e60952641468aed7fb6d8a9a064",
    ),
    BrainMaskMetadata(
        name=mask_name("MNI152NLin2009cAsym", 2.0),
        description="MNI152NLin2009cAsym brain mask, 2 mm",
        space="MNI152NLin2009cAsym",
        resolution=2.0,
        url="https://osf.io/6jkdb/download",
        sha256="9ec2c641c1e2d1b7a94a57e779197874bfedba4c6dbc55b092f5fbf12c750fb9",
    ),
]

for _mask in _KNOWN_MASKS:
    BRAIN_MASK_REGISTRY.register(_mask)


__all__ = ["BrainMaskMetadata", "BRAIN_MASK_REGISTRY", "mask_name"]
