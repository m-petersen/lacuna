"""Functional Neurotransmitter Fingerprinting (fntf).

Two scoring modes:
* Static (``ntatlas_dir``): voxelwise NT atlas × lesion-FC z-map.
* ACE-enriched (``ace_dir``): per-subject Pearson correlation between the
  ACE stage-1 NT-weighted timeseries and the lesion BOLD timeseries from
  the same connectome subject, Fisher-z transformed, then averaged across
  subjects to get one Fisher-z per target.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lacuna.analysis.base import BaseAnalysis
from lacuna.atlas.config import resolve_targets
from lacuna.atlas.scoring import score_functional_overlap
from lacuna.atlas.store import load_atlas
from lacuna.core.data_types import LabeledScalars
from lacuna.core.keys import build_result_key
from lacuna.core.subject_data import SubjectData

logger = logging.getLogger(__name__)


class FunctionalNeurotransmitterFingerprinting(BaseAnalysis):
    """Functional neurotransmitter fingerprinting: NT scores via functional connectivity.

    Provide exactly one of ``ntatlas_dir`` or ``ace_dir``:

    * ``ntatlas_dir`` (static): the lesion's functional connectivity z-map
      (computed via ``FunctionalNetworkMapping`` internally) is dot-producted
      against each NT atlas map.
    * ``ace_dir`` (enriched): for each connectome subject ``s`` and target
      ``T``, we compute Pearson r between the lesion BOLD ts from subject
      ``s`` and the ACE stage-1 NT-weighted ts ``stage1[s, :, T]``,
      Fisher-z transform, then average across subjects. The output is one
      Fisher-z value per target.

    The ACE cache must have been built from the same connectome that
    ``connectome_name`` refers to. Two checks at runtime guard alignment:
    the cache's shared envelope (``ace_dir/lacuna_asset.json``) is walked
    by :func:`lacuna.assets.envelope.validate_requires` to re-fingerprint
    the recorded ntatlas + connectome inputs, and a dimension check
    matches the stage-1 .npy file count against the connectome HDF5
    batches.

    Parameters
    ----------
    connectome_name : str
        Name of the registered functional connectome (e.g., "GSP1000").
    ntatlas_dir : Path or None
        Directory with the prepared NT atlas (output of ``lacuna fetch ntatlas``).
    ace_dir : Path or None
        Directory with the ACE cache. Expected layout:
        ``stage2_atlas/`` + ``stage1_timeseries/*.npy`` (one .npy per
        connectome subject, sorted to match connectome subject order;
        each .npy has shape ``(n_timepoints, n_targets)``) +
        ``lacuna_asset.json`` (the shared envelope).
    targets : str or list[str]
        Target selection. Default "all".
    n_jobs : int
        Number of parallel jobs for the internal FNM run.
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
        ntatlas_dir: str | Path | None = None,
        ace_dir: str | Path | None = None,
        targets: str | list[str] = "all",
        n_jobs: int = 1,
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
        self.connectome_name = connectome_name
        self._target_spec = targets
        self.n_jobs = n_jobs

    @property
    def enriched(self) -> bool:
        """Whether the analysis is sourcing from an ACE cache."""
        return self.ace_dir is not None

    def _resolve_atlas_dir(self) -> Path:
        if self.ace_dir is not None:
            return self.ace_dir / "stage2_atlas"
        return self.ntatlas_dir

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """Load the atlas (sets ``self._atlas``) and resolve targets."""
        atlas = load_atlas(self._resolve_atlas_dir())
        self._atlas = atlas
        self._resolved_targets = resolve_targets(self._target_spec, atlas.targets)

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, Any]:
        """Compute functional NT scores. Returns a single LabeledScalars with one row per target."""
        if self.enriched:
            scores = self._run_enriched(mask_data)
            mode = "enriched"
        else:
            atlas = self._atlas.subset(self._resolved_targets)
            z_map = self._compute_functional_connectivity(mask_data)
            scores = self._score_overlap(atlas, z_map)
            mode = "static"

        fingerprint = LabeledScalars(
            name="neurotransmitter",
            data={target: float(score) for target, score in scores.items()},
            label_kind="target",
            aggregation_method=mode,
            metadata={
                "analysis": "fntf",
                "mode": mode,
                "systems": self._atlas.metadata.get("systems"),
            },
        )
        key = build_result_key(
            atlas="neurotransmitter",
            source="FunctionalNeurotransmitterFingerprinting",
            desc=mode,
        )
        return {key: fingerprint}

    def _score_overlap(self, atlas, z_map) -> dict[str, float]:
        """Voxelwise NT-atlas × z-map scoring on a shared grid."""
        atlas_img = atlas.get_map(atlas.targets[0])
        if atlas_img.shape[:3] != z_map.shape[:3] or not np.allclose(
            atlas_img.affine, z_map.affine
        ):
            atlas = atlas.resample_to(z_map.affine, z_map.shape[:3])
        return score_functional_overlap(atlas, z_map)

    def _run_enriched(self, mask_data: SubjectData) -> dict[str, float]:
        """Subject-level Fisher-z mean of ACE-stage1 vs. lesion-BOLD correlations."""
        from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

        fnm = FunctionalNetworkMapping(
            connectome_name=self.connectome_name,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            compute_p_map=False,
            fdr_alpha=None,
            return_in_input_space=False,
        )

        self._verify_cache_against_connectome(fnm)

        in_connectome_space = fnm._ensure_target_space(mask_data)
        per_subject_lesion_ts = self._collect_per_subject_lesion_ts(
            fnm, in_connectome_space
        )

        stage1 = self._load_ace_stage1_array()
        self._verify_dimension_alignment(per_subject_lesion_ts, stage1)

        return self._fisher_z_mean_per_target(per_subject_lesion_ts, stage1)

    def _collect_per_subject_lesion_ts(
        self, fnm, mask_in_connectome_space: SubjectData
    ) -> np.ndarray:
        """Per-subject lesion BOLD timeseries (HDF5 fancy-index read)."""
        per_batch: list[np.ndarray] = []
        for _, mask_ts in fnm._iter_batch_lesion_timeseries(
            mask_in_connectome_space, full_batch=False
        ):
            per_batch.append(mask_ts)
        if not per_batch:
            with __import__("h5py").File(fnm._get_connectome_files()[0], "r") as hf:
                n_timepoints = int(hf["timeseries"].shape[1])
            return np.zeros((0, n_timepoints), dtype=np.float64)
        return np.vstack(per_batch)

    def _load_ace_stage1_array(self) -> np.ndarray:
        """Stack ACE stage-1 timeseries: shape (n_subjects, n_timepoints, n_targets)."""
        stage1_dir = self.ace_dir / "stage1_timeseries"
        files = sorted(stage1_dir.glob("*.npy"))
        if not files:
            raise FileNotFoundError(
                f"No stage-1 timeseries (*.npy) found in {stage1_dir}.\n"
                "Run 'lacuna prepare ace' first to populate the cache."
            )
        return np.stack([np.load(f) for f in files], axis=0)

    def _verify_cache_against_connectome(self, fnm) -> None:
        """Verify the cache's recorded inputs still match the runtime inputs.

        Only the connectome is a runtime requirement: FNTF in enriched mode
        consumes ``<ace_dir>/stage2_atlas`` directly, and the source ntatlas
        the cache was built from is recorded only as build-time provenance
        (``envelope.provenance.source_ntatlas_*``).
        """
        from lacuna.assets.envelope import read_envelope, validate_requires

        env = read_envelope(self.ace_dir)
        validate_requires(env, {"connectome": Path(fnm.connectome_path)})

    def _verify_dimension_alignment(
        self, lesion_ts: np.ndarray, stage1: np.ndarray
    ) -> None:
        """Catch mismatched subject counts or timepoint counts."""
        n_lesion_subj, n_lesion_t = lesion_ts.shape
        n_stage1_subj, n_stage1_t, _ = stage1.shape
        if n_lesion_subj != n_stage1_subj:
            raise ValueError(
                "ACE stage-1 subject count does not match the connectome.\n"
                f"  stage1 .npy files: {n_stage1_subj}\n"
                f"  connectome subjects: {n_lesion_subj}\n"
                "Re-run 'lacuna prepare ace' against the current connectome."
            )
        if n_lesion_t != n_stage1_t:
            raise ValueError(
                "ACE stage-1 timepoint count does not match the connectome.\n"
                f"  stage1 timepoints: {n_stage1_t}\n"
                f"  connectome timepoints: {n_lesion_t}"
            )

    def _fisher_z_mean_per_target(
        self, lesion_ts: np.ndarray, stage1: np.ndarray
    ) -> dict[str, float]:
        """Vectorized per-subject Pearson r → Fisher z → mean across subjects.

        ``lesion_ts``: (n_subjects, n_timepoints).
        ``stage1``:    (n_subjects, n_timepoints, n_targets_in_cache).
        """
        if lesion_ts.shape[0] == 0:
            return {target: 0.0 for target in self._resolved_targets}

        # Center along the timepoint axis.
        L = lesion_ts - lesion_ts.mean(axis=1, keepdims=True)
        S = stage1 - stage1.mean(axis=1, keepdims=True)

        # Pearson r per (subject, target):
        #   numerator[s, T]   = Σ_t L[s,t] * S[s,t,T]
        #   denominator[s, T] = ||L[s]|| * ||S[s,:,T]||
        numerator = np.einsum("st,stT->sT", L, S)
        l_norm = np.linalg.norm(L, axis=1)
        s_norm = np.linalg.norm(S, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = numerator / (l_norm[:, None] * s_norm)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

        # Fisher z, then mean across subjects per target.
        r = np.clip(r, -1.0 + 1e-9, 1.0 - 1e-9)
        z = np.arctanh(r)
        z_mean = z.mean(axis=0)  # (n_targets_in_cache,)

        full_targets = self._atlas.targets
        resolved = set(self._resolved_targets)
        return {
            target: float(z_mean[i])
            for i, target in enumerate(full_targets)
            if target in resolved
        }

    def _compute_functional_connectivity(self, mask_data):
        """Compute fLNM z-map using FunctionalNetworkMapping."""
        from lacuna.analysis.functional_network_mapping import FunctionalNetworkMapping

        fnm = FunctionalNetworkMapping(
            connectome_name=self.connectome_name,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            compute_p_map=False,
            fdr_alpha=None,
            return_in_input_space=False,
        )
        result = fnm.run(mask_data)
        z_map_result = result.results["FunctionalNetworkMapping"]["zmap"]
        return z_map_result.data  # nib.Nifti1Image

    def _get_parameters(self) -> dict:
        return {
            "ntatlas_dir": str(self.ntatlas_dir) if self.ntatlas_dir else None,
            "ace_dir": str(self.ace_dir) if self.ace_dir else None,
            "connectome_name": self.connectome_name,
            "targets": self._target_spec,
            "n_jobs": self.n_jobs,
        }


