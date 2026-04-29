"""Functional Neurotransmitter Fingerprinting (fntf).

Scores NT atlas values weighted by lesion functional connectivity.
Static mode: NT atlas x fLNM z-map.
ACE-enriched mode: global = temporal correlation with atlas timeseries,
                   regional = ACE atlas x fLNM z-map.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lacuna.analysis.base import BaseAnalysis
from lacuna.atlas.config import resolve_targets
from lacuna.atlas.scoring import score_ace_temporal, score_functional_overlap
from lacuna.atlas.store import load_atlas
from lacuna.core.data_types import LabeledScalars
from lacuna.core.keys import build_result_key
from lacuna.core.subject_data import SubjectData

logger = logging.getLogger(__name__)


class FunctionalNeurotransmitterFingerprinting(BaseAnalysis):
    """Functional neurotransmitter fingerprinting: NT scores via functional connectivity.

    Computes functional connectivity of the lesion (using FunctionalNetworkMapping
    internally), then scores the resulting z-map against the NT atlas.

    Provide exactly one of ``atlas_cache_dir`` (static mode: NT atlas
    × fLNM z-map) or ``ace_cache_dir`` (enriched mode: temporal
    correlation of lesion BOLD with stage-1 NT timeseries).

    Parameters
    ----------
    connectome_name : str
        Name of the functional connectome (e.g., "GSP1000").
    atlas_cache_dir : Path or None
        Directory with the prepared NT atlas (output of ``lacuna fetch ntatlas``).
        Triggers static-mode scoring.
    ace_cache_dir : Path or None
        Directory with the ACE cache (output of ``lacuna prepare ace``).
        Triggers enriched-mode scoring.
    targets : str or list[str]
        Target selection. Default "all".
    parcel_atlases : list[str] or None
        Atlas names for regional scoring.
    method : str
        Lesion timeseries extraction method ("boes" or "pini").
    n_jobs : int
        Number of parallel jobs.
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
        connectome_name: str,
        atlas_cache_dir: str | Path | None = None,
        ace_cache_dir: str | Path | None = None,
        targets: str | list[str] = "all",
        parcel_atlases: list[str] | None = None,
        method: str = "boes",
        n_jobs: int = 1,
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
        self.connectome_name = connectome_name
        self._target_spec = targets
        self.parcel_atlases = parcel_atlases
        self.method = method
        self.n_jobs = n_jobs

    @property
    def enriched(self) -> bool:
        """Whether the analysis is sourcing from an ACE cache."""
        return self.ace_cache_dir is not None

    def _resolve_atlas_dir(self) -> Path:
        if self.ace_cache_dir is not None:
            return self.ace_cache_dir / "stage2_atlas"
        return self.atlas_cache_dir

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """Validate atlas and connectome inputs."""
        atlas = load_atlas(self._resolve_atlas_dir())
        self._atlas = atlas
        self._resolved_targets = resolve_targets(self._target_spec, atlas.targets)

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, Any]:
        """Compute functional NT scores. Returns a single ParcelData with one row per target."""
        atlas = self._atlas.subset(self._resolved_targets)

        if self.enriched:
            scores = self._run_enriched(mask_data, atlas)
            mode = "enriched"
        else:
            z_map = self._compute_functional_connectivity(mask_data)
            scores = self._run_static(atlas, z_map)
            mode = "static"

        fingerprint = LabeledScalars(
            name="neurotransmitter",
            data={target: float(score) for target, score in scores.items()},
            label_kind="target",
            aggregation_method=mode,
            metadata={
                "analysis": "fntf",
                "mode": mode,
                "systems": atlas.metadata.get("systems"),
            },
        )
        key = build_result_key(
            atlas="neurotransmitter",
            source="FunctionalNeurotransmitterFingerprinting",
            desc=mode,
        )
        return {key: fingerprint}

    def _run_static(self, atlas, z_map) -> dict[str, float]:
        """Static mode: NT atlas x fLNM z-map. Returns target → score."""
        z_shape = z_map.shape[:3]
        atlas_shape = atlas.get_map(atlas.targets[0]).shape[:3]
        if atlas_shape != z_shape:
            atlas = atlas.resample_to(z_map.affine, z_shape)
        return score_functional_overlap(atlas, z_map)

    def _run_enriched(self, mask_data, atlas) -> dict[str, float]:
        """ACE-enriched mode: temporal correlation for global scoring. Returns target → score."""
        ace_data = self._load_ace_data()
        lesion_ts = self._extract_lesion_timeseries(mask_data)

        # Average ACE stage 1 atlas timeseries across subjects
        all_stage1 = ace_data["stage1_timeseries"]
        avg_stage1 = np.mean(all_stage1, axis=0)  # (n_timepoints, n_targets)

        # Build target-keyed timeseries dict
        nt_timeseries = {}
        ace_targets = ace_data["stage2_atlas"].targets
        for i, target in enumerate(ace_targets):
            if target in self._resolved_targets:
                nt_timeseries[target] = avg_stage1[:, i]

        return score_ace_temporal(nt_timeseries, lesion_ts)

    def _compute_functional_connectivity(self, mask_data):
        """Compute fLNM z-map using FunctionalNetworkMapping logic."""
        from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

        fnm = FunctionalNetworkMapping(
            connectome_name=self.connectome_name,
            method=self.method,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            compute_p_map=False,
            fdr_alpha=None,
            return_in_input_space=False,
        )
        result = fnm.run(mask_data)
        z_map_result = result.results["FunctionalNetworkMapping"]["zmap"]
        return z_map_result.data  # nib.Nifti1Image

    def _extract_lesion_timeseries(self, mask_data):
        """Extract mean lesion BOLD timeseries from connectome data.

        Reuses the extraction logic from FunctionalNetworkMapping.
        """
        raise NotImplementedError(
            "Lesion timeseries extraction for ACE enriched mode "
            "requires refactoring FNM internals into shared utility."
        )

    def _load_ace_data(self):
        """Load ACE stage 1 timeseries and stage 2 atlas from cache."""
        stage2_atlas = load_atlas(self.ace_cache_dir / "stage2_atlas")

        # Load stage 1 timeseries
        stage1_dir = self.ace_cache_dir / "stage1_timeseries"
        stage1_list = []
        for ts_file in sorted(stage1_dir.glob("*.npy")):
            stage1_list.append(np.load(ts_file))

        return {
            "stage2_atlas": stage2_atlas,
            "stage1_timeseries": stage1_list,
        }

    def _get_parameters(self) -> dict:
        return {
            "atlas_cache_dir": str(self.atlas_cache_dir) if self.atlas_cache_dir else None,
            "ace_cache_dir": str(self.ace_cache_dir) if self.ace_cache_dir else None,
            "connectome_name": self.connectome_name,
            "targets": self._target_spec,
            "method": self.method,
        }
