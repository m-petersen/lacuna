"""Accelerated functional network mapping (afnm).

Implements the matrix-multiplication formulation of functional LNM from van
den Heuvel et al. (2026, *Nat Neurosci*): ``afnmap = m @ C``, where ``m`` is
a lesion-to-parcel weight vector and ``C`` is a precomputed group-level
parcel-level functional connectivity matrix (Fisher r-to-z values, produced
by ``lacuna prepare functional``).

Compared with voxel-level functional LNM this replaces a per-subject
correlation sweep over a voxelwise connectome with a single (N,) × (N, N)
product per subject, where ``N`` is the number of parcels.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img

from lacuna.analysis.base import BaseAnalysis
from lacuna.assets.parcellations import list_parcellations, load_parcellation
from lacuna.core.data_types import ParcelData
from lacuna.core.keys import build_result_key
from lacuna.core.subject_data import SubjectData
from lacuna.utils.logging import ConsoleLogger

if TYPE_CHECKING:
    from lacuna.core.data_types import DataContainer


_LESION_WEIGHTINGS = ("fractional", "binary", "voxel_count")


class AcceleratedFunctionalNetworkMapping(BaseAnalysis):
    """Accelerated functional LNM via matrix multiplication (``m @ C``).

    Parameters
    ----------
    matrix_path : str or Path
        Path to the group-FC matrix TSV produced by
        ``lacuna prepare functional``.
        Must be a square matrix with region labels as its row/column index.
    parcel_names : list[str]
        Registered parcellation name(s) matching the atlas used to build ``C``.
        Exactly one atlas must be supplied — its labels must align with the
        TSV row/column labels.
    lesion_weighting : {"fractional", "binary", "voxel_count"}, default="fractional"
        Scheme for the ``m`` row vector:

        - ``fractional``: each touched region gets ``1 / n_regions_touched``
        - ``binary``: 1 if the region is touched, else 0
        - ``voxel_count``: fraction of the parcel's voxels covered by the lesion
    verbose : bool, default=False
        If True, print progress messages.
    keep_intermediate : bool, default=False
        If True, include the ``m`` weight vector in results for inspection.
    """

    batch_strategy = "parallel"
    TARGET_SPACE = None
    TARGET_RESOLUTION = None

    def __init__(
        self,
        matrix_path: str | Path,
        parcel_names: list[str] | None = None,
        lesion_weighting: Literal["fractional", "binary", "voxel_count"] = "fractional",
        verbose: bool = False,
        keep_intermediate: bool = False,
    ):
        super().__init__(verbose=verbose, keep_intermediate=keep_intermediate)

        if lesion_weighting not in _LESION_WEIGHTINGS:
            raise ValueError(
                f"lesion_weighting must be one of {_LESION_WEIGHTINGS}, got "
                f"'{lesion_weighting}'"
            )

        self.matrix_path = Path(matrix_path)
        self.lesion_weighting = lesion_weighting

        if not parcel_names:
            raise ValueError(
                "parcel_names is required for AcceleratedFunctionalNetworkMapping "
                "(pass --parcel-atlases or --custom-parcellation)."
            )
        if isinstance(parcel_names, str):
            parcel_names = [parcel_names]
        if len(parcel_names) != 1:
            raise ValueError(
                "AcceleratedFunctionalNetworkMapping currently supports exactly one "
                f"parcellation (got {len(parcel_names)}). Run it per atlas."
            )
        registered = {p.name for p in list_parcellations()}
        if parcel_names[0] not in registered:
            raise KeyError(
                f"Parcellation '{parcel_names[0]}' is not registered. "
                f"Use list_parcellations() or register via --custom-parcellation."
            )
        self.parcel_names = parcel_names

        self.logger = ConsoleLogger(verbose=verbose, width=70)

        self._c_matrix: np.ndarray | None = None
        self._c_labels: list[str] | None = None
        self._c_metadata: dict | None = None

    def _get_parameters(self) -> dict:
        params = super()._get_parameters()
        params.update(
            {
                "matrix_path": str(self.matrix_path),
                "parcel_names": list(self.parcel_names),
                "lesion_weighting": self.lesion_weighting,
            }
        )
        return params

    def _load_c_matrix(self) -> None:
        if self._c_matrix is not None:
            return
        if not self.matrix_path.exists():
            raise FileNotFoundError(f"Matrix TSV not found: {self.matrix_path}")
        df = pd.read_csv(self.matrix_path, sep="\t", index_col=0)
        if df.shape[0] != df.shape[1]:
            raise ValueError(
                f"Expected a square matrix in {self.matrix_path}, got shape {df.shape}."
            )
        if list(df.index) != list(df.columns):
            raise ValueError(
                "Matrix TSV row/column labels disagree; the file does not look like "
                "a ConnectivityMatrix produced by 'lacuna prepare functional'."
            )
        self._c_matrix = df.values.astype(np.float64)
        self._c_labels = [str(x) for x in df.index]

        sidecar = self.matrix_path.with_suffix(".json")
        if sidecar.exists():
            import json

            with open(sidecar) as f:
                self._c_metadata = json.load(f)
        else:
            self._c_metadata = {}

    def _validate_inputs(self, mask_data: SubjectData) -> None:
        self._load_c_matrix()
        if mask_data.space is None:
            raise ValueError(
                "AcceleratedFunctionalNetworkMapping requires input with a known "
                "coordinate space."
            )

    def _build_weight_vector(
        self,
        mask_array: np.ndarray,
        atlas_values: np.ndarray,
        region_ids: list[int],
    ) -> tuple[np.ndarray, dict[int, int]]:
        """Return (m, voxel_counts_per_region) for the chosen weighting scheme."""
        flat_mask = mask_array.astype(bool).ravel()
        flat_atlas = atlas_values.ravel()

        voxels_per_region: dict[int, int] = {}
        hit_per_region: dict[int, int] = {}

        for rid in region_ids:
            region_mask = flat_atlas == rid
            voxels_per_region[rid] = int(region_mask.sum())
            hit_per_region[rid] = int(np.logical_and(region_mask, flat_mask).sum())

        touched = [rid for rid, hits in hit_per_region.items() if hits > 0]
        n_touched = len(touched)

        m = np.zeros(len(region_ids), dtype=np.float64)
        if n_touched == 0:
            return m, voxels_per_region

        if self.lesion_weighting == "binary":
            for col, rid in enumerate(region_ids):
                m[col] = 1.0 if hit_per_region[rid] > 0 else 0.0
        elif self.lesion_weighting == "fractional":
            w = 1.0 / n_touched
            for col, rid in enumerate(region_ids):
                if hit_per_region[rid] > 0:
                    m[col] = w
        else:  # voxel_count
            for col, rid in enumerate(region_ids):
                nvox = voxels_per_region[rid]
                if nvox > 0 and hit_per_region[rid] > 0:
                    m[col] = hit_per_region[rid] / nvox
        return m, voxels_per_region

    def _align_atlas_and_labels(
        self, mask_data: SubjectData
    ) -> tuple[np.ndarray, list[int], list[str]]:
        """Load atlas, resample to mask grid, and align with C's label order."""
        atlas_name = self.parcel_names[0]
        parc = load_parcellation(atlas_name)
        atlas_img = parc.image

        ref = nib.Nifti1Image(
            np.zeros(mask_data.mask_img.shape, dtype=np.int16),
            mask_data.mask_img.affine,
        )
        resampled = resample_to_img(
            atlas_img,
            ref,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        atlas_values = np.round(resampled.get_fdata()).astype(np.int64)

        label_to_rid: dict[str, int] = {}
        for rid, label in parc.labels.items():
            label_to_rid[str(label)] = int(rid)

        assert self._c_labels is not None
        region_ids: list[int] = []
        ordered_labels: list[str] = []
        missing: list[str] = []
        for lab in self._c_labels:
            if lab in label_to_rid:
                region_ids.append(label_to_rid[lab])
                ordered_labels.append(lab)
            else:
                missing.append(lab)
        if missing:
            raise ValueError(
                f"Atlas '{atlas_name}' is missing {len(missing)} label(s) present in "
                f"the C matrix (first few: {missing[:3]}). The atlas and matrix must "
                "match."
            )
        return atlas_values, region_ids, ordered_labels

    def _run_analysis(self, mask_data: SubjectData) -> dict[str, DataContainer]:
        self._load_c_matrix()
        assert self._c_matrix is not None and self._c_labels is not None

        atlas_values, region_ids, ordered_labels = self._align_atlas_and_labels(mask_data)
        mask_array = mask_data.mask_img.get_fdata()

        m, voxels_per_region = self._build_weight_vector(mask_array, atlas_values, region_ids)

        afnmap = m @ self._c_matrix  # (N,)
        atlas_name = self.parcel_names[0]

        results: dict[str, DataContainer] = {}

        n_touched = int((m > 0).sum())
        meta_base = {
            "atlas": atlas_name,
            "matrix_path": str(self.matrix_path),
            "lesion_weighting": self.lesion_weighting,
            "n_regions": len(ordered_labels),
            "n_regions_touched": n_touched,
        }

        afnmap_data = {lab: float(afnmap[i]) for i, lab in enumerate(ordered_labels)}
        afnmap_parc = ParcelData(
            name="afnmap",
            data=afnmap_data,
            region_labels=ordered_labels,
            parcel_names=[atlas_name],
            aggregation_method=f"afnm_{self.lesion_weighting}",
            metadata=dict(meta_base, description="Accelerated functional network map (m @ C)"),
        )
        results[
            build_result_key(
                atlas=atlas_name,
                source="AcceleratedFunctionalNetworkMapping",
                desc="afnmap",
            )
        ] = afnmap_parc

        if self.keep_intermediate:
            m_data = {lab: float(m[i]) for i, lab in enumerate(ordered_labels)}
            results[
                build_result_key(
                    atlas=atlas_name,
                    source="AcceleratedFunctionalNetworkMapping",
                    desc="afnmweights",
                )
            ] = ParcelData(
                name="afnmweights",
                data=m_data,
                region_labels=ordered_labels,
                parcel_names=[atlas_name],
                aggregation_method=f"afnm_{self.lesion_weighting}",
                metadata=dict(
                    meta_base,
                    description="Lesion-to-parcel weights m used for m @ C",
                    voxels_per_region={str(k): v for k, v in voxels_per_region.items()},
                ),
            )

        return results
