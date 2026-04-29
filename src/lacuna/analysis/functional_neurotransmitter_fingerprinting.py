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
from lacuna.core.data_types import ScalarMetric, VoxelMap
from lacuna.core.subject_data import SubjectData

logger = logging.getLogger(__name__)


class FunctionalNeurotransmitterFingerprinting(BaseAnalysis):
    """Functional neurotransmitter fingerprinting: NT scores via functional connectivity.

    Computes functional connectivity of the lesion (using FunctionalNetworkMapping
    internally), then scores the resulting z-map against the NT atlas.

    Parameters
    ----------
    atlas_cache_dir : Path
        Directory containing the prepared NT atlas.
    connectome_name : str
        Name of the functional connectome (e.g., "GSP1000").
    targets : str or list[str]
        Target selection. Default "all".
    enriched : bool
        If True, use ACE-enriched scoring.
    ace_cache_dir : Path or None
        Directory with ACE outputs (required if enriched=True).
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
        atlas_cache_dir: str | Path,
        connectome_name: str,
        targets: str | list[str] = "all",
        enriched: bool = False,
        ace_cache_dir: str | Path | None = None,
        parcel_atlases: list[str] | None = None,
        method: str = "boes",
        n_jobs: int = 1,
        verbose: bool = False,
        keep_intermediate: bool = False,
    ):
        super().__init__(verbose=verbose, keep_intermediate=keep_intermediate)
        self.atlas_cache_dir = Path(atlas_cache_dir)
        self.connectome_name = connectome_name
        self._target_spec = targets
        self.enriched = enriched
        self.ace_cache_dir = Path(ace_cache_dir) if ace_cache_dir else None
        self.parcel_atlases = parcel_atlases
        self.method = method
        self.n_jobs = n_jobs

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """Validate atlas, connectome, and ACE data if enriched."""
        atlas = load_atlas(self.atlas_cache_dir)
        self._atlas = atlas
        self._resolved_targets = resolve_targets(self._target_spec, atlas.targets)

        if self.enriched and self.ace_cache_dir is None:
            raise ValueError(
                "ACE cache directory required for enriched mode. "
                "Run 'lacuna prepare ace' first."
            )

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, Any]:
        """Compute functional NT scores."""
        atlas = self._atlas.subset(self._resolved_targets)

        # Compute functional connectivity z-map internally
        z_map = self._compute_functional_connectivity(mask_data)

        if self.enriched:
            return self._run_enriched(mask_data, atlas, z_map)
        else:
            return self._run_static(atlas, z_map)

    def _run_static(self, atlas, z_map):
        """Static mode: NT atlas x fLNM z-map."""
        z_shape = z_map.shape[:3]
        atlas_shape = atlas.get_map(atlas.targets[0]).shape[:3]
        if atlas_shape != z_shape:
            atlas = atlas.resample_to(z_map.affine, z_shape)

        scores = score_functional_overlap(atlas, z_map)

        results = {}
        for target, score in scores.items():
            results[target] = ScalarMetric(
                name=target,
                data=score,
                data_type="scalar",
                metadata={"analysis": "fntf", "mode": "static"},
            )
        return results

    def _run_enriched(self, mask_data, atlas, z_map):
        """ACE-enriched mode: temporal correlation for global scoring."""
        results = {}

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

        temporal_scores = score_ace_temporal(nt_timeseries, lesion_ts)

        for target, score in temporal_scores.items():
            results[target] = ScalarMetric(
                name=target,
                data=score,
                data_type="scalar",
                metadata={"analysis": "fntf", "mode": "enriched"},
            )

        return results

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
            "atlas_cache_dir": str(self.atlas_cache_dir),
            "connectome_name": self.connectome_name,
            "targets": self._target_spec,
            "enriched": self.enriched,
            "method": self.method,
        }
