"""Reduce a whole-brain connectome to a parcellated NxN connectivity matrix.

Functional branch only for now: load a voxelwise fMRI HDF5 connectome (same
format consumed by ``lacuna run fnm``), project timeseries onto parcel means
using the atlas resampled to the connectome grid, compute a per-subject
Pearson correlation matrix, average across subjects via Fisher z, and write
the resulting group-level :class:`~lacuna.core.data_types.ConnectivityMatrix`
as a BIDS-style TSV + JSON sidecar (mirrors the layout produced by
``lacuna run snm`` for its disconnectivity matrix).

Exposed via the ``lacuna prepare afnm`` CLI target.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

from lacuna.assets.connectomes.functional_io import (
    list_connectome_batch_files,
    read_mask_info,
)
from lacuna.assets.parcellations.loader import _load_labels_file, load_parcellation
from lacuna.core.data_types import ConnectivityMatrix


@dataclass
class ResolvedParcellation:
    """An atlas ready for use by the parcellate pipeline."""

    short_name: str
    image: nib.Nifti1Image
    labels: dict[int, str]
    space: str | None
    source: str  # "registry" or "custom"


def resolve_parcellations(
    parcel_atlases: list[str] | None,
    custom_parcellation: list[list[str]] | None,
) -> list[ResolvedParcellation]:
    """Turn CLI ``--parcel-atlases`` / ``--custom-parcellation`` into objects."""
    resolved: list[ResolvedParcellation] = []

    for name in parcel_atlases or []:
        parc = load_parcellation(name)
        resolved.append(
            ResolvedParcellation(
                short_name=name,
                image=parc.image,
                labels=parc.labels,
                space=getattr(parc.metadata, "space", None),
                source="registry",
            )
        )

    for entry in custom_parcellation or []:
        short_name, nifti_path, labels_path, space = entry
        image = nib.load(nifti_path)
        labels = _load_labels_file(Path(labels_path))
        resolved.append(
            ResolvedParcellation(
                short_name=short_name,
                image=image,
                labels=labels,
                space=space,
                source="custom",
            )
        )

    if not resolved:
        raise ValueError("At least one of --parcel-atlases or --custom-parcellation must be given.")
    return resolved


def _resample_atlas_to_mask_grid(
    atlas_img: nib.Nifti1Image,
    mask_affine: np.ndarray,
    mask_shape: tuple[int, int, int],
) -> np.ndarray:
    """Resample a parcellation to the connectome mask grid with nearest-neighbour."""
    from nilearn.image import resample_to_img

    ref = nib.Nifti1Image(np.zeros(mask_shape, dtype=np.int16), mask_affine)
    resampled = resample_to_img(
        atlas_img, ref, interpolation="nearest", force_resample=True, copy_header=True
    )
    return np.round(resampled.get_fdata()).astype(np.int64)


def _build_parcel_weight_matrix(
    atlas_values: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Build an (n_vox, N) column-normalised parcel weight matrix.

    Columns correspond to ``sorted_region_ids`` (0 excluded). Each column sums
    to 1 so ``timeseries @ W`` yields the parcel-mean timeseries.
    """
    unique_ids = np.unique(atlas_values)
    region_ids = [int(r) for r in unique_ids if int(r) != 0]
    n_vox = atlas_values.shape[0]
    n_reg = len(region_ids)
    if n_reg == 0:
        raise ValueError("Resampled atlas has no non-zero regions inside the connectome mask.")

    W = np.zeros((n_vox, n_reg), dtype=np.float32)
    for col, rid in enumerate(region_ids):
        hits = atlas_values == rid
        count = int(hits.sum())
        if count == 0:
            continue
        W[hits, col] = 1.0 / count
    return W, region_ids


def _per_subject_correlation(parcel_ts: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix per subject from (n_sub, n_tp, N) timeseries.

    Returns (n_sub, N, N).
    """
    x = parcel_ts - parcel_ts.mean(axis=1, keepdims=True)
    s = np.sqrt((x * x).sum(axis=1, keepdims=True))
    s = np.where(s == 0, 1.0, s)
    xn = x / s
    return np.einsum("stn,stm->snm", xn, xn)


def parcellate_functional(
    connectome_path: Path,
    parcellations: list[ResolvedParcellation],
    output_dir: Path,
    *,
    overwrite: bool = False,
    verbose: bool = False,
) -> list[Path]:
    """Compute group-average parcellated FC matrices from a voxelwise HDF5 connectome.

    One output TSV+JSON pair is written per parcellation under ``output_dir``.
    Returns the list of TSV paths.
    """
    connectome_path = Path(connectome_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_files = list_connectome_batch_files(connectome_path)
    mask_info = read_mask_info(batch_files[0])
    mi = mask_info["mask_indices"]
    mask_affine = mask_info["mask_affine"]
    mask_shape = mask_info["mask_shape"]

    # Pre-resample every atlas once and build a weight matrix per parcellation.
    prepared: list[dict] = []
    for parc in parcellations:
        resampled = _resample_atlas_to_mask_grid(parc.image, mask_affine, mask_shape)
        atlas_values = resampled[mi[0], mi[1], mi[2]]
        W, region_ids = _build_parcel_weight_matrix(atlas_values)
        region_labels = [parc.labels.get(rid, f"Region{rid}") for rid in region_ids]
        N = len(region_ids)
        prepared.append(
            {
                "parc": parc,
                "W": W,
                "region_ids": region_ids,
                "region_labels": region_labels,
                "sum_z": np.zeros((N, N), dtype=np.float64),
                "n_subjects": 0,
            }
        )

    hasher = hashlib.sha256()
    for bf in batch_files:
        hasher.update(str(bf.name).encode())
        hasher.update(str(bf.stat().st_size).encode())

        if verbose:
            print(f"[parcellate] batch {bf.name}")
        with h5py.File(bf, "r") as hf:
            ts = hf["timeseries"][:]  # (n_sub, n_tp, n_vox)
        n_sub = ts.shape[0]

        for item in prepared:
            parcel_ts = ts @ item["W"]  # (n_sub, n_tp, N)
            corr = _per_subject_correlation(parcel_ts)
            corr = np.clip(corr, -0.999999, 0.999999)
            item["sum_z"] += np.arctanh(corr).sum(axis=0)
            item["n_subjects"] += n_sub
        del ts

    source_hash = hasher.hexdigest()[:16]
    written: list[Path] = []
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    for item in prepared:
        parc: ResolvedParcellation = item["parc"]
        mean_z = item["sum_z"] / max(item["n_subjects"], 1)
        np.fill_diagonal(mean_z, 0.0)
        mean_z_f32 = mean_z.astype(np.float32)

        matrix_r = np.tanh(mean_z).astype(np.float32)
        np.fill_diagonal(matrix_r, 1.0)

        shared_meta = {
            "connectome_path": str(connectome_path),
            "parcellation_name": parc.short_name,
            "parcellation_source": parc.source,
            "parcellation_space": parc.space,
            "n_subjects": item["n_subjects"],
            "n_batches": len(batch_files),
            "modality": "functional",
            "source_hash": source_hash,
            "timestamp": timestamp,
            "mask_shape": list(mask_shape),
        }

        outputs = [
            (
                matrix_r,
                f"groupFC_{parc.short_name}",
                "pearson_r",
                f"method-parcellate_atlas-{parc.short_name}" f"_desc-fcgroupr_connmatrix",
            ),
            (
                mean_z_f32,
                f"groupFCz_{parc.short_name}",
                "fisher_z",
                f"method-parcellate_atlas-{parc.short_name}" f"_desc-fcgroupz_connmatrix",
            ),
        ]

        for matrix, name, value_type, base in outputs:
            connmat = ConnectivityMatrix(
                name=name,
                matrix=matrix,
                region_labels=item["region_labels"],
                matrix_type="functional",
                metadata=dict(shared_meta, value_type=value_type),
            )

            tsv_path = output_dir / f"{base}.tsv"
            json_path = output_dir / f"{base}.json"

            if tsv_path.exists() and not overwrite:
                raise FileExistsError(f"Output exists: {tsv_path}. Pass --overwrite to replace.")

            df = pd.DataFrame(
                connmat.matrix,
                index=connmat.region_labels,
                columns=connmat.region_labels,
            )
            df.to_csv(tsv_path, sep="\t")

            sidecar = {
                "Description": connmat.name,
                "MatrixType": connmat.matrix_type,
                "ValueType": value_type,
                "RegionLabels": connmat.region_labels,
                "Shape": list(connmat.matrix.shape),
                "Metadata": connmat.metadata,
            }
            with open(json_path, "w") as f:
                json.dump(sidecar, f, indent=2)

            written.append(tsv_path)
            if verbose:
                print(f"[parcellate] wrote {tsv_path}")

    return written


def run_parcellate_functional_cli(args) -> int:
    """Entry point for ``lacuna prepare afnm``."""
    try:
        parcellations = resolve_parcellations(
            getattr(args, "parcel_atlases", None),
            getattr(args, "custom_parcellation", None),
        )
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        return 1

    try:
        parcellate_functional(
            connectome_path=args.connectome_path,
            parcellations=parcellations,
            output_dir=args.output,
            overwrite=getattr(args, "overwrite", False),
            verbose=getattr(args, "verbose_count", 0) > 0,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"Error: {e}", file=__import__("sys").stderr)
        return 1
    return 0
