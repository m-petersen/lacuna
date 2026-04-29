"""Local Neurotransmitter Fingerprinting (lntf).

Scores NT atlas values directly within the lesion mask.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lacuna.analysis.base import BaseAnalysis
from lacuna.atlas.config import resolve_targets
from lacuna.atlas.scoring import score_focal
from lacuna.atlas.store import load_atlas
from lacuna.core.data_types import ParcelData
from lacuna.core.keys import build_result_key
from lacuna.core.subject_data import SubjectData

logger = logging.getLogger(__name__)


class LocalNeurotransmitterFingerprinting(BaseAnalysis):
    """Local neurotransmitter fingerprinting: NT scores within the lesion footprint.

    Computes, for each NT target, the mean z-scored NT density within the
    lesion mask (excluding zero-valued voxels).

    Provide exactly one of ``atlas_cache_dir`` (static NT atlas from
    ``lacuna fetch ntatlas``) or ``ace_cache_dir`` (ACE-enriched atlas
    from ``lacuna prepare ace``). The scoring is identical in both
    cases — only the underlying atlas values differ.

    Parameters
    ----------
    atlas_cache_dir : Path or None
        Directory with the prepared NT atlas (output of ``lacuna fetch ntatlas``).
    ace_cache_dir : Path or None
        Directory with the ACE cache (output of ``lacuna prepare ace``).
    targets : str or list[str]
        Target selection. Preset name ("all", "dopaminergic", etc.) or
        explicit list of target names. Default "all".
    parcel_atlases : list[str] or None
        Atlas names for regional scoring.
    aggregation : str
        "mean" or "sum". Default "mean".
    verbose : bool
        Enable verbose logging.
    keep_intermediate : bool
        Keep intermediate results.
    """

    TARGET_SPACE = None
    TARGET_RESOLUTION = None
    batch_strategy = "sequential"

    def __init__(
        self,
        atlas_cache_dir: str | Path | None = None,
        ace_cache_dir: str | Path | None = None,
        targets: str | list[str] = "all",
        parcel_atlases: list[str] | None = None,
        aggregation: str = "mean",
        verbose: bool = False,
        keep_intermediate: bool = False,
    ):
        super().__init__(verbose=verbose, keep_intermediate=keep_intermediate)
        if (atlas_cache_dir is None) == (ace_cache_dir is None):
            raise ValueError(
                "Provide exactly one of atlas_cache_dir or ace_cache_dir."
            )
        self.atlas_cache_dir = Path(atlas_cache_dir) if atlas_cache_dir else None
        self.ace_cache_dir = Path(ace_cache_dir) if ace_cache_dir else None
        self._target_spec = targets
        self.parcel_atlases = parcel_atlases
        self.aggregation = aggregation

    @property
    def enriched(self) -> bool:
        """Whether the analysis is sourcing from an ACE cache."""
        return self.ace_cache_dir is not None

    def _resolve_atlas_dir(self) -> Path:
        """Return the directory the VoxelAtlas should be loaded from."""
        if self.ace_cache_dir is not None:
            return self.ace_cache_dir / "stage2_atlas"
        return self.atlas_cache_dir

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """Validate that the atlas cache exists and targets are available."""
        atlas = load_atlas(self._resolve_atlas_dir())
        self._atlas = atlas
        self._resolved_targets = resolve_targets(self._target_spec, atlas.targets)

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, Any]:
        """Compute local NT scores. Returns a single ParcelData with one row per target."""
        atlas = self._atlas

        # Resample atlas to mask grid if needed
        mask_img = mask_data.mask_img
        atlas_shape = atlas.get_map(atlas.targets[0]).shape[:3]
        if atlas_shape != mask_img.shape[:3]:
            atlas = atlas.resample_to(mask_img.affine, mask_img.shape[:3])

        atlas = atlas.subset(self._resolved_targets)
        lesion_mask = mask_img.get_fdata().astype(bool)
        scores = score_focal(atlas, lesion_mask, aggregation=self.aggregation)

        parcel_data = ParcelData(
            name="neurotransmitter",
            data={target: float(score) for target, score in scores.items()},
            region_labels=list(scores.keys()),
            parcel_names=["neurotransmitter"],
            aggregation_method=self.aggregation,
            metadata={
                "analysis": "lntf",
                "enriched": self.enriched,
                "systems": atlas.metadata.get("systems"),
            },
        )
        key = build_result_key(
            atlas="neurotransmitter",
            source="LocalNeurotransmitterFingerprinting",
            desc="lntfscores",
        )
        return {key: parcel_data}

    def _get_parameters(self) -> dict:
        return {
            "atlas_cache_dir": str(self.atlas_cache_dir) if self.atlas_cache_dir else None,
            "ace_cache_dir": str(self.ace_cache_dir) if self.ace_cache_dir else None,
            "targets": self._target_spec,
            "aggregation": self.aggregation,
            "parcel_atlases": self.parcel_atlases,
        }
