"""Structural Neurotransmitter Fingerprinting (sntf).

Scores NT atlas values at endpoints of lesion-disconnected streamlines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lacuna.analysis.base import BaseAnalysis
from lacuna.assets.connectomes import load_structural_connectome
from lacuna.atlas.config import resolve_targets
from lacuna.atlas.store import load_atlas
from lacuna.core.data_types import LabeledScalars, VoxelMap
from lacuna.core.keys import build_result_key
from lacuna.core.subject_data import SubjectData

logger = logging.getLogger(__name__)


class StructuralNeurotransmitterFingerprinting(BaseAnalysis):
    """Structural neurotransmitter fingerprinting: NT scores via structural disconnection.

    For each lesion-intersecting streamline, computes the mean NT value at its
    two endpoints, then sums across all intersecting streamlines.

    Provide exactly one of ``ntatlas_dir`` (static NT atlas from
    ``lacuna fetch ntatlas``) or ``ace_dir`` (ACE-enriched atlas
    from ``lacuna prepare ace``).

    Requires a prepared (atlas, tractogram) cache produced by
    ``lacuna prepare sntf``. The cache is the canonical input — there
    is no on-the-fly fallback. This avoids subtle numerical drift
    between repeated `tckedit`/`tckresample` runs.

    Parameters
    ----------
    connectome_name : str
        Name of the structural connectome (e.g., "dTOR-985").
    precomputed_weights_dir : Path
        Directory with the prepared endpoint NT weights cache
        (output of ``lacuna prepare sntf``). Required.
    ntatlas_dir : Path or None
        Directory with the prepared NT atlas.
    ace_dir : Path or None
        Directory with the ACE cache.
    targets : str or list[str]
        Target selection. Default "all".
    check_dependencies : bool
        Check for MRtrix3 availability.
    n_jobs : int
        Number of parallel jobs for MRtrix.
    verbose : bool
        Enable verbose logging.
    keep_intermediate : bool
        Keep intermediate results.

    References
    ----------
    - Koch, Frey et al. (2025): https://doi.org/10.1093/brain/awaf185
    """

    TARGET_SPACE = None
    TARGET_RESOLUTION = None
    batch_strategy = "sequential"
    # The mask itself isn't the analysis input — sntf works off the
    # tractogram filtered by the lesion. No need to emit a default
    # analysis_mask intermediate.
    EMIT_ANALYSIS_MASK = False

    def __init__(
        self,
        connectome_name: str,
        precomputed_weights_dir: str | Path,
        ntatlas_dir: str | Path | None = None,
        ace_dir: str | Path | None = None,
        targets: str | list[str] = "all",
        endpoint_combine: str = "mean",
        aggregation: str = "sum",
        check_dependencies: bool = True,
        n_jobs: int = 1,
        verbose: bool = False,
        keep_intermediate: bool = False,
    ):
        super().__init__(verbose=verbose, keep_intermediate=keep_intermediate)
        if (ntatlas_dir is None) == (ace_dir is None):
            raise ValueError(
                "Provide exactly one of ntatlas_dir or ace_dir."
            )
        if endpoint_combine not in ("mean", "sum", "product"):
            raise ValueError(
                f"endpoint_combine must be 'mean', 'sum', or 'product'; got '{endpoint_combine}'"
            )
        if aggregation not in ("sum", "mean"):
            raise ValueError(
                f"aggregation must be 'sum' or 'mean'; got '{aggregation}'"
            )
        self.ntatlas_dir = Path(ntatlas_dir) if ntatlas_dir else None
        self.ace_dir = Path(ace_dir) if ace_dir else None
        self.connectome_name = connectome_name
        self._target_spec = targets
        self.precomputed_weights_dir = Path(precomputed_weights_dir)
        self.endpoint_combine = endpoint_combine
        self.aggregation = aggregation
        self.n_jobs = n_jobs

        # Load connectome metadata
        connectome = load_structural_connectome(connectome_name)
        self.tractogram_path = connectome.tractogram_path
        self.tractogram_space = connectome.metadata.space

        if check_dependencies:
            from lacuna.utils.mrtrix import check_mrtrix_available

            check_mrtrix_available()

    @property
    def enriched(self) -> bool:
        """Whether the analysis is sourcing from an ACE cache."""
        return self.ace_dir is not None

    def _resolve_atlas_dir(self) -> Path:
        if self.ace_dir is not None:
            return self.ace_dir / "stage2_atlas"
        return self.ntatlas_dir

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """Validate atlas, connectome, and resolve targets."""
        atlas = load_atlas(self._resolve_atlas_dir())
        self._atlas = atlas
        self._resolved_targets = resolve_targets(self._target_spec, atlas.targets)

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, Any]:
        """Compute structural NT scores. Returns a single LabeledScalars with one row per target."""
        atlas = self._atlas.subset(self._resolved_targets)
        scores, count, endpoint_density = self._score_from_cache(mask_data, atlas)

        fingerprint = LabeledScalars(
            name="neurotransmitter",
            data={target: float(score) for target, score in scores.items()},
            label_kind="target",
            aggregation_method=f"{self.endpoint_combine}_endpoints/{self.aggregation}",
            extras={"streamline_count": int(count)},
            metadata={
                "analysis": "sntf",
                "mode": "enriched" if self.enriched else "static",
                "endpoint_combine": self.endpoint_combine,
                "aggregation": self.aggregation,
                "streamline_count": int(count),
                "systems": atlas.metadata.get("systems"),
            },
        )
        desc = "enriched" if self.enriched else "static"
        key = build_result_key(
            atlas="neurotransmitter",
            source="StructuralNeurotransmitterFingerprinting",
            desc=desc,
        )
        results: dict[str, Any] = {key: fingerprint}
        if endpoint_density is not None:
            results["endpointdensity"] = endpoint_density
        return results

    def _validate_connectome_matches_cache(self, cache: Path) -> None:
        """Reject caches built from a different tractogram than the one supplied at run time.

        Without this check, ``tckedit`` may not flag a count mismatch between
        the streamline-index file and the user's tractogram, and even when
        counts coincide the per-streamline weight arrays describe a different
        connectome — yielding silently random scores.
        """
        import json

        from lacuna.utils.tractogram_id import (
            compute_tractogram_fingerprint,
            fingerprints_match,
        )

        meta_path = cache / "connectome_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Precomputed weights cache at {cache} is missing 'connectome_meta.json'.\n"
                "This file records which tractogram the cache was built from. "
                "Re-run 'lacuna prepare sntf --connectome-path ... --cache-dir "
                f"{cache}' to refresh the cache."
            )
        expected = json.loads(meta_path.read_text())
        actual = compute_tractogram_fingerprint(self.tractogram_path)
        if not fingerprints_match(expected, actual):
            raise ValueError(
                "Precomputed weights cache connectome does not match the "
                "tractogram supplied at run time.\n"
                f"  cache:    {expected.get('path')} "
                f"({expected.get('size_bytes')} bytes, "
                f"sha256[:1MiB]={expected.get('sha256_first_mib', '')[:12]}…)\n"
                f"  runtime:  {actual['path']} "
                f"({actual['size_bytes']} bytes, "
                f"sha256[:1MiB]={actual['sha256_first_mib'][:12]}…)\n"
                "Re-run 'lacuna prepare sntf' with the runtime tractogram or "
                "point '--connectome-path' at the same tractogram used during prepare."
            )

    def _score_from_cache(
        self, mask_data: SubjectData, atlas
    ) -> tuple[dict[str, float], int, VoxelMap | None]:
        """Score using a precomputed (atlas, tractogram) cache.

        Filters the full tractogram by the lesion using ``tckedit -include`` while
        passing through float-encoded streamline indices (``-tck_weights_in``)
        so that ``-tck_weights_out`` returns the surviving original streamline IDs.
        Then indexes the precomputed (n_targets, n_streamlines) start/end weight
        matrices and applies the requested endpoint_combine + aggregation.
        """
        import shutil

        import nibabel as nib

        from lacuna.utils.cache import get_temp_dir
        from lacuna.utils.mrtrix import run_mrtrix_command

        cache = self.precomputed_weights_dir
        for required in ("start_weights.npy", "end_weights.npy", "targets.txt", "streamline_indices.txt"):
            if not (cache / required).exists():
                raise FileNotFoundError(
                    f"Precomputed weights cache missing '{required}' in {cache}.\n"
                    f"Run 'lacuna prepare sntf --connectome-path ... --cache-dir {cache}' first."
                )

        self._validate_connectome_matches_cache(cache)

        # Load cache contents and validate target alignment.
        cached_targets = (cache / "targets.txt").read_text().splitlines()
        target_index = {t: i for i, t in enumerate(cached_targets)}
        missing = [t for t in self._resolved_targets if t not in target_index]
        if missing:
            raise ValueError(
                f"Cached weights are missing targets requested in this run: {missing}. "
                f"Re-run 'lacuna prepare sntf' against the current NT atlas."
            )
        start_weights = np.load(cache / "start_weights.npy", mmap_mode="r")
        end_weights = np.load(cache / "end_weights.npy", mmap_mode="r")

        # Filter the full tractogram by the lesion mask while propagating the
        # float-encoded streamline indices through to the output CSV.
        # Workspace lives under LACUNA_TEMP_DIR (or ~/.cache/lacuna/tmp/) for
        # consistency with the rest of the package.
        subject_id = mask_data.metadata.get("subject_id", "subject")
        tmp = get_temp_dir(prefix=f"sntf_{subject_id}_")
        try:
            mask_path = tmp / "lesion_mask.nii.gz"
            nib.save(mask_data.mask_img, str(mask_path))
            run_mrtrix_command(
                [
                    "tckedit",
                    str(self.tractogram_path),
                    str(tmp / "filtered.tck"),
                    "-include", str(mask_path),
                    "-tck_weights_in", str(cache / "streamline_indices.txt"),
                    "-tck_weights_out", str(tmp / "surviving.csv"),
                    "-force",
                ],
                verbose=self.verbose,
            )
            surviving_text = (tmp / "surviving.csv").read_text().split()
        finally:
            if not self.keep_intermediate:
                shutil.rmtree(tmp, ignore_errors=True)

        if not surviving_text:
            density = self._build_endpoint_density_from_cache(np.array([], dtype=np.int64), atlas)
            return {target: 0.0 for target in self._resolved_targets}, 0, (
                density if self.keep_intermediate else None
            )
        surviving_ids = np.asarray(surviving_text, dtype=np.float64).round().astype(np.int64)
        count = int(surviving_ids.size)

        # Pull the relevant rows of the cache for the requested targets.
        target_rows = np.array(
            [target_index[t] for t in self._resolved_targets], dtype=np.int64
        )
        start_vals = np.asarray(start_weights[target_rows][:, surviving_ids])
        end_vals = np.asarray(end_weights[target_rows][:, surviving_ids])

        if self.endpoint_combine == "mean":
            per_streamline = (start_vals + end_vals) / 2.0
        elif self.endpoint_combine == "sum":
            per_streamline = start_vals + end_vals
        else:  # product
            per_streamline = start_vals * end_vals

        if self.aggregation == "sum":
            agg = per_streamline.sum(axis=1)
        else:  # mean
            agg = per_streamline.mean(axis=1) if count else np.zeros(len(target_rows))

        scores = {t: float(agg[i]) for i, t in enumerate(self._resolved_targets)}

        endpoint_density = (
            self._build_endpoint_density_from_cache(surviving_ids, atlas)
            if self.keep_intermediate
            else None
        )
        return scores, count, endpoint_density

    def _build_endpoint_density_from_cache(
        self, surviving_ids: np.ndarray, atlas
    ) -> VoxelMap:
        """Endpoint density on the atlas grid for cache-based runs."""
        import nibabel as nib

        ref_img = atlas.get_map(atlas.targets[0])
        shape = np.array(ref_img.shape[:3])
        density = np.zeros(shape, dtype=np.int32)

        cache = self.precomputed_weights_dir
        # Endpoint coords are derivable from the cached endpoints.tck. Re-derive
        # cheaply by reading the streamlines (one-time cost).
        streamlines = nib.streamlines.load(str(cache / "endpoints.tck")).streamlines
        inv_aff = np.linalg.inv(ref_img.affine)
        for sid in surviving_ids:
            sl = streamlines[int(sid)]
            for pt in (sl[0], sl[-1]):
                v = np.clip(
                    (inv_aff[:3, :3] @ pt + inv_aff[:3, 3]).astype(np.int32),
                    0, shape - 1,
                )
                density[v[0], v[1], v[2]] += 1

        return VoxelMap(
            name="endpointdensity",
            data=nib.Nifti1Image(density.astype(np.float32), ref_img.affine),
            space=atlas.space,
            resolution=atlas.resolution,
            metadata={"description": "Endpoint counts of lesion-disconnected streamlines"},
        )

    def _get_parameters(self) -> dict:
        return {
            "ntatlas_dir": str(self.ntatlas_dir) if self.ntatlas_dir else None,
            "ace_dir": str(self.ace_dir) if self.ace_dir else None,
            "connectome_name": self.connectome_name,
            "precomputed_weights_dir": str(self.precomputed_weights_dir),
            "endpoint_combine": self.endpoint_combine,
            "aggregation": self.aggregation,
            "targets": self._target_spec,
        }
