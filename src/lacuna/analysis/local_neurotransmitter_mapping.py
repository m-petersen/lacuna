"""Local Neurotransmitter Mapping (lntm).

Scores NT atlas values directly within the lesion mask.
Answers: "what neurotransmitter landscape did the lesion wipe out?"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lacuna.analysis.base import BaseAnalysis
from lacuna.atlas.config import resolve_targets
from lacuna.atlas.scoring import score_focal
from lacuna.atlas.store import load_atlas
from lacuna.core.data_types import ScalarMetric
from lacuna.core.subject_data import SubjectData

logger = logging.getLogger(__name__)


class LocalNeurotransmitterMapping(BaseAnalysis):
    """Local neurotransmitter mapping: NT scores within the lesion footprint.

    Computes, for each NT target, the mean z-scored NT density within the
    lesion mask (excluding zero-valued voxels).

    Parameters
    ----------
    atlas_cache_dir : Path
        Directory containing the prepared NT atlas (from `lacuna prepare lntm`).
    targets : str or list[str]
        Target selection. Preset name ("all", "dopaminergic", etc.) or
        explicit list of target names. Default "all".
    enriched : bool
        If True, use REACT-enriched atlas instead of static.
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
        atlas_cache_dir: str | Path,
        targets: str | list[str] = "all",
        enriched: bool = False,
        parcel_atlases: list[str] | None = None,
        aggregation: str = "mean",
        verbose: bool = False,
        keep_intermediate: bool = False,
    ):
        super().__init__(verbose=verbose, keep_intermediate=keep_intermediate)
        self.atlas_cache_dir = Path(atlas_cache_dir)
        self._target_spec = targets
        self.enriched = enriched
        self.parcel_atlases = parcel_atlases
        self.aggregation = aggregation

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """Validate that the atlas cache exists and targets are available."""
        atlas = load_atlas(self.atlas_cache_dir)
        self._atlas = atlas
        self._resolved_targets = resolve_targets(self._target_spec, atlas.targets)

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, Any]:
        """Compute local NT scores."""
        atlas = self._atlas

        # Resample atlas to mask space if needed
        mask_img = mask_data.mask_img
        mask_affine = mask_img.affine
        mask_shape = mask_img.shape[:3]

        atlas_shape = atlas.get_map(atlas.targets[0]).shape[:3]
        if atlas_shape != mask_shape:
            atlas = atlas.resample_to(mask_affine, mask_shape)

        # Subset to requested targets
        atlas = atlas.subset(self._resolved_targets)

        lesion_mask = mask_img.get_fdata().astype(bool)

        # Global scores
        global_scores = score_focal(atlas, lesion_mask, aggregation=self.aggregation)

        results = {}
        for target, score in global_scores.items():
            results[target] = ScalarMetric(
                name=target,
                data=score,
                data_type="scalar",
                metadata={"analysis": "lntm", "aggregation": self.aggregation},
            )

        return results

    def _get_parameters(self) -> dict:
        return {
            "atlas_cache_dir": str(self.atlas_cache_dir),
            "targets": self._target_spec,
            "enriched": self.enriched,
            "aggregation": self.aggregation,
            "parcel_atlases": self.parcel_atlases,
        }
