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
from lacuna.atlas.scoring import score_structural_endpoints
from lacuna.atlas.store import load_atlas
from lacuna.core.data_types import LabeledScalars, Tractogram, VoxelMap
from lacuna.core.keys import build_result_key
from lacuna.core.subject_data import SubjectData

logger = logging.getLogger(__name__)


class StructuralNeurotransmitterFingerprinting(BaseAnalysis):
    """Structural neurotransmitter fingerprinting: NT scores via structural disconnection.

    For each lesion-intersecting streamline, computes the mean NT value at its
    two endpoints, then sums across all intersecting streamlines.

    Provide exactly one of ``atlas_cache_dir`` (static NT atlas from
    ``lacuna fetch ntatlas``) or ``ace_cache_dir`` (ACE-enriched atlas
    from ``lacuna prepare ace``).

    Parameters
    ----------
    connectome_name : str
        Name of the structural connectome (e.g., "dTOR-985").
    atlas_cache_dir : Path or None
        Directory with the prepared NT atlas.
    ace_cache_dir : Path or None
        Directory with the ACE cache.
    targets : str or list[str]
        Target selection. Default "all".
    parcel_atlases : list[str] or None
        Atlas names for regional scoring.
    precomputed_weights_dir : Path or None
        Directory with precomputed endpoint NT weights.
    check_dependencies : bool
        Check for MRtrix3 availability.
    n_jobs : int
        Number of parallel jobs for MRtrix.
    verbose : bool
        Enable verbose logging.
    keep_intermediate : bool
        Keep intermediate results.
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
        atlas_cache_dir: str | Path | None = None,
        ace_cache_dir: str | Path | None = None,
        targets: str | list[str] = "all",
        parcel_atlases: list[str] | None = None,
        precomputed_weights_dir: str | Path | None = None,
        endpoint_combine: str = "mean",
        aggregation: str = "sum",
        check_dependencies: bool = True,
        n_jobs: int = 1,
        verbose: bool = False,
        keep_intermediate: bool = False,
    ):
        super().__init__(verbose=verbose, keep_intermediate=keep_intermediate)
        if (atlas_cache_dir is None) == (ace_cache_dir is None):
            raise ValueError(
                "Provide exactly one of atlas_cache_dir or ace_cache_dir."
            )
        if endpoint_combine not in ("mean", "sum", "product"):
            raise ValueError(
                f"endpoint_combine must be 'mean', 'sum', or 'product'; got '{endpoint_combine}'"
            )
        if aggregation not in ("sum", "mean"):
            raise ValueError(
                f"aggregation must be 'sum' or 'mean'; got '{aggregation}'"
            )
        self.atlas_cache_dir = Path(atlas_cache_dir) if atlas_cache_dir else None
        self.ace_cache_dir = Path(ace_cache_dir) if ace_cache_dir else None
        self.connectome_name = connectome_name
        self._target_spec = targets
        self.parcel_atlases = parcel_atlases
        self.precomputed_weights_dir = (
            Path(precomputed_weights_dir) if precomputed_weights_dir else None
        )
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
        return self.ace_cache_dir is not None

    def _resolve_atlas_dir(self) -> Path:
        if self.ace_cache_dir is not None:
            return self.ace_cache_dir / "stage2_atlas"
        return self.atlas_cache_dir

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        """Validate atlas, connectome, and resolve targets."""
        atlas = load_atlas(self._resolve_atlas_dir())
        self._atlas = atlas
        self._resolved_targets = resolve_targets(self._target_spec, atlas.targets)

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, Any]:
        """Compute structural NT scores. Returns a single ParcelData with one row per target."""
        atlas = self._atlas.subset(self._resolved_targets)

        filtered_tck_path = self._find_or_compute_filtered_tractogram(mask_data)

        endpoint_density: VoxelMap | None = None
        if filtered_tck_path is None:
            scores = {target: 0.0 for target in self._resolved_targets}
            count = 0
        else:
            endpoints_start, endpoints_end, intersecting_ids = self._get_endpoint_data(
                mask_data, filtered_tck_path, atlas
            )
            scores, count = score_structural_endpoints(
                atlas,
                endpoints_start,
                endpoints_end,
                intersecting_ids,
                endpoint_combine=self.endpoint_combine,
                aggregation=self.aggregation,
            )
            if self.keep_intermediate:
                endpoint_density = self._build_endpoint_density(
                    endpoints_start, endpoints_end, atlas
                )

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

    def _build_endpoint_density(
        self,
        endpoints_start: np.ndarray,
        endpoints_end: np.ndarray,
        atlas,
    ) -> VoxelMap:
        """Voxelwise count of lesion-disconnected streamline endpoints on the atlas grid."""
        ref_img = atlas.get_map(atlas.targets[0])
        shape = ref_img.shape[:3]
        density = np.zeros(shape, dtype=np.int32)
        for vox in np.concatenate([endpoints_start, endpoints_end], axis=0):
            density[vox[0], vox[1], vox[2]] += 1
        import nibabel as nib

        return VoxelMap(
            name="endpointdensity",
            data=nib.Nifti1Image(density.astype(np.float32), ref_img.affine),
            space=atlas.space,
            resolution=atlas.resolution,
            metadata={"description": "Endpoint counts of lesion-disconnected streamlines"},
        )

    def _find_or_compute_filtered_tractogram(
        self, mask_data: SubjectData
    ) -> Path | None:
        """Find existing filtered tractogram or compute one via MRtrix.

        Checks:
        1. SubjectData.results for SNM filtered_tractogram
        2. Computes filtering via MRtrix tckedit
        """
        import tempfile

        import nibabel as nib

        from lacuna.utils.mrtrix import filter_tractogram_by_mask

        # Check SubjectData results from prior SNM run
        if "StructuralNetworkMapping" in mask_data.results:
            snm_results = mask_data.results["StructuralNetworkMapping"]
            if "filtered_tractogram" in snm_results:
                tck = snm_results["filtered_tractogram"]
                if isinstance(tck, Tractogram) and tck.tractogram_path.exists():
                    logger.info("Reusing filtered tractogram from SNM results")
                    return tck.tractogram_path

        # Compute via MRtrix
        logger.info("Computing filtered tractogram via MRtrix")
        tmp_dir = tempfile.mkdtemp(prefix="sntf_")
        mask_path = Path(tmp_dir) / "lesion_mask.nii.gz"
        nib.save(mask_data.mask_img, str(mask_path))

        filtered_path = Path(tmp_dir) / "filtered.tck"
        filter_tractogram_by_mask(
            tractogram_path=self.tractogram_path,
            mask=str(mask_path),
            output_path=str(filtered_path),
            n_jobs=self.n_jobs,
            force=True,
            verbose=self.verbose,
        )

        if not filtered_path.exists():
            return None
        return filtered_path

    def _get_endpoint_data(
        self, mask_data: SubjectData, filtered_tck_path: Path, atlas
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract endpoint coordinates from filtered tractogram.

        Returns
        -------
        tuple of (endpoints_start, endpoints_end, intersecting_ids)
        """
        import nibabel as nib

        from lacuna.utils.mrtrix import run_mrtrix_command

        tmp_dir = filtered_tck_path.parent

        # Get endpoints from filtered tractogram
        endpoints_tck = tmp_dir / "endpoints.tck"
        run_mrtrix_command(
            ["tckresample", str(filtered_tck_path), str(endpoints_tck), "-endpoints"],
            verbose=self.verbose,
        )

        # Load endpoints and convert to voxel coordinates
        endpoints_tractogram = nib.streamlines.load(str(endpoints_tck))
        streamlines = endpoints_tractogram.streamlines

        ref_img = atlas.get_map(atlas.targets[0])
        inv_affine = np.linalg.inv(ref_img.affine)

        n_streamlines = len(streamlines)
        endpoints_start = np.zeros((n_streamlines, 3), dtype=np.int32)
        endpoints_end = np.zeros((n_streamlines, 3), dtype=np.int32)

        for i, sl in enumerate(streamlines):
            start_world = sl[0]
            end_world = sl[-1]
            start_vox = (inv_affine[:3, :3] @ start_world + inv_affine[:3, 3]).astype(
                np.int32
            )
            end_vox = (inv_affine[:3, :3] @ end_world + inv_affine[:3, 3]).astype(
                np.int32
            )
            endpoints_start[i] = np.clip(
                start_vox, 0, np.array(ref_img.shape[:3]) - 1
            )
            endpoints_end[i] = np.clip(end_vox, 0, np.array(ref_img.shape[:3]) - 1)

        # All streamlines in the filtered tractogram are intersecting
        intersecting_ids = np.arange(n_streamlines)

        return endpoints_start, endpoints_end, intersecting_ids

    def _get_parameters(self) -> dict:
        return {
            "atlas_cache_dir": str(self.atlas_cache_dir) if self.atlas_cache_dir else None,
            "ace_cache_dir": str(self.ace_cache_dir) if self.ace_cache_dir else None,
            "connectome_name": self.connectome_name,
            "targets": self._target_spec,
        }
