"""Local Neurotransmitter Fingerprinting (lntf).

Scores NT atlas values directly within the lesion mask.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lacuna.analysis.base import BaseAnalysis
from lacuna.atlas.config import resolve_targets
from lacuna.atlas.scoring import score_focal
from lacuna.atlas.store import load_atlas
from lacuna.core.data_types import LabeledScalars
from lacuna.core.keys import build_result_key
from lacuna.core.subject_data import SubjectData

logger = logging.getLogger(__name__)


class LocalNeurotransmitterFingerprinting(BaseAnalysis):
    """Local neurotransmitter fingerprinting: NT scores within the lesion footprint.

    Computes, for each NT target, the mean z-scored NT density within the
    lesion mask (excluding zero-valued voxels).

    Provide exactly one of ``ntatlas_dir`` (static NT atlas from
    ``lacuna fetch ntatlas``) or ``ace_dir`` (ACE-enriched atlas
    from ``lacuna prepare ace``). The scoring is identical in both
    cases — only the underlying atlas values differ.

    Parameters
    ----------
    ntatlas_dir : Path or None
        Directory with the prepared NT atlas (output of ``lacuna fetch ntatlas``).
    ace_dir : Path or None
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
        ntatlas_dir: str | Path | None = None,
        ace_dir: str | Path | None = None,
        targets: str | list[str] = "all",
        parcel_atlases: list[str] | None = None,
        aggregation: str = "mean",
        verbose: bool = False,
        keep_intermediate: bool = False,
    ):
        super().__init__(verbose=verbose, keep_intermediate=keep_intermediate)
        if (ntatlas_dir is None) == (ace_dir is None):
            raise ValueError(
                "Provide exactly one of ntatlas_dir or ace_dir."
            )
        self.ntatlas_dir = Path(ntatlas_dir) if ntatlas_dir else None
        self.ace_dir = Path(ace_dir) if ace_dir else None
        self._target_spec = targets
        self.parcel_atlases = parcel_atlases
        self.aggregation = aggregation

    @property
    def enriched(self) -> bool:
        """Whether the analysis is sourcing from an ACE cache."""
        return self.ace_dir is not None

    def _resolve_atlas_dir(self) -> Path:
        """Return the directory the VoxelAtlas should be loaded from."""
        if self.ace_dir is not None:
            return self.ace_dir / "stage2_atlas"
        return self.ntatlas_dir

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """Validate that the atlas cache exists and targets are available."""
        atlas = load_atlas(self._resolve_atlas_dir())
        self._atlas = atlas
        self._resolved_targets = resolve_targets(self._target_spec, atlas.targets)

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, Any]:
        """Compute local NT scores. Returns a single ParcelData with one row per target."""
        atlas = self._atlas

        # Resample atlas to mask grid if shape OR affine differs
        mask_img = mask_data.mask_img
        atlas_img = atlas.get_map(atlas.targets[0])
        if atlas_img.shape[:3] != mask_img.shape[:3] or not np.allclose(
            atlas_img.affine, mask_img.affine
        ):
            atlas = atlas.resample_to(mask_img.affine, mask_img.shape[:3])

        atlas = atlas.subset(self._resolved_targets)
        lesion_mask = mask_img.get_fdata().astype(bool)
        scores = score_focal(atlas, lesion_mask, aggregation=self.aggregation)

        fingerprint = LabeledScalars(
            name="neurotransmitter",
            data={target: float(score) for target, score in scores.items()},
            label_kind="target",
            aggregation_method=self.aggregation,
            metadata={
                "analysis": "lntf",
                "mode": "enriched" if self.enriched else "static",
                "systems": self._atlas.metadata.get("systems"),
            },
        )
        desc = "enriched" if self.enriched else "static"
        key = build_result_key(
            atlas="neurotransmitter",
            source="LocalNeurotransmitterFingerprinting",
            desc=desc,
        )
        return {key: fingerprint}

    def _get_parameters(self) -> dict:
        return {
            "ntatlas_dir": str(self.ntatlas_dir) if self.ntatlas_dir else None,
            "ace_dir": str(self.ace_dir) if self.ace_dir else None,
            "targets": self._target_spec,
            "aggregation": self.aggregation,
            "parcel_atlases": self.parcel_atlases,
        }
