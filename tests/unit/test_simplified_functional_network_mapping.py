"""Unit tests for SimplifiedFunctionalNetworkMapping analysis."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from lacuna.analysis import SimplifiedFunctionalNetworkMapping
from lacuna.assets.parcellations.registry import (
    PARCELLATION_REGISTRY,
    register_parcellation_from_files,
)
from lacuna.core.subject_data import SubjectData


MASK_SHAPE = (4, 4, 4)
MASK_AFFINE = np.eye(4)


def _write_tiny_atlas(tmp_path: Path, name: str) -> tuple[Path, Path, list[str]]:
    """Create a tiny atlas with 2 regions (R1 at x=0, R2 at x=1) plus labels file."""
    atlas_arr = np.zeros(MASK_SHAPE, dtype=np.int16)
    atlas_arr[0, 0, :3] = 1
    atlas_arr[1, 0, :3] = 2
    nifti = tmp_path / f"{name}.nii.gz"
    nib.save(nib.Nifti1Image(atlas_arr, MASK_AFFINE), nifti)

    labels = tmp_path / f"{name}_labels.txt"
    labels.write_text("1 R1\n2 R2\n")
    return nifti, labels, ["R1", "R2"]


def _write_c_matrix(tmp_path: Path, labels: list[str], values: np.ndarray) -> Path:
    tsv = tmp_path / "C.tsv"
    df = pd.DataFrame(values, index=labels, columns=labels)
    df.to_csv(tsv, sep="\t")
    sidecar = tsv.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "Description": "test_C",
                "MatrixType": "functional",
                "RegionLabels": labels,
                "Shape": list(values.shape),
                "Metadata": {"modality": "functional"},
            }
        )
    )
    return tsv


def _make_subject(mask_array: np.ndarray, *, space: str = "MNI152NLin6Asym") -> SubjectData:
    img = nib.Nifti1Image(mask_array.astype(np.uint8), MASK_AFFINE)
    return SubjectData(
        mask_img=img,
        space=space,
        resolution=1,
        metadata={"subject_id": "sub-001"},
    )


@pytest.fixture
def registered_atlas(tmp_path):
    nifti, labels_path, _labels = _write_tiny_atlas(tmp_path, "sfnm_tiny")
    name = "sfnm_tiny"
    register_parcellation_from_files(
        name=name,
        parcellation_path=nifti,
        labels_path=labels_path,
        space="MNI152NLin6Asym",
        resolution=1,
        description="Tiny test atlas",
    )
    yield name
    PARCELLATION_REGISTRY.pop(name, None)


def test_sfnm_fractional_two_region_overlap(tmp_path, registered_atlas):
    """Lesion touching both R1 and R2 → m = [0.5, 0.5]; lnm = m @ C."""
    C = np.array([[1.0, 0.5], [0.5, 1.0]])
    matrix_path = _write_c_matrix(tmp_path, ["R1", "R2"], C)

    mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
    mask[0, 0, 0] = 1  # in R1
    mask[1, 0, 0] = 1  # in R2
    subj = _make_subject(mask)

    analysis = SimplifiedFunctionalNetworkMapping(
        matrix_path=matrix_path,
        parcel_names=[registered_atlas],
        lesion_weighting="fractional",
    )
    out = analysis.run(subj)
    results = out.results["SimplifiedFunctionalNetworkMapping"]
    lnm_key = next(k for k in results if k.endswith("desc-sfnmap"))
    parc = results[lnm_key]
    assert parc.data["R1"] == pytest.approx(0.75)
    assert parc.data["R2"] == pytest.approx(0.75)


def test_sfnm_binary_single_region(tmp_path, registered_atlas):
    """Lesion in R1 only with binary weighting → m = [1, 0]; lnm = C row 0."""
    C = np.array([[0.9, 0.3], [0.3, 0.8]])
    matrix_path = _write_c_matrix(tmp_path, ["R1", "R2"], C)

    mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
    mask[0, 0, 0] = 1
    mask[0, 0, 1] = 1
    subj = _make_subject(mask)

    analysis = SimplifiedFunctionalNetworkMapping(
        matrix_path=matrix_path,
        parcel_names=[registered_atlas],
        lesion_weighting="binary",
    )
    out = analysis.run(subj)
    results = out.results["SimplifiedFunctionalNetworkMapping"]
    parc = next(v for k, v in results.items() if k.endswith("desc-sfnmap"))
    assert parc.data["R1"] == pytest.approx(0.9)
    assert parc.data["R2"] == pytest.approx(0.3)


def test_sfnm_voxel_count_weighting(tmp_path, registered_atlas):
    """voxel_count: m[i] = hits_in_region_i / n_voxels_region_i."""
    C = np.eye(2)
    matrix_path = _write_c_matrix(tmp_path, ["R1", "R2"], C)

    mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
    # R1 has 3 voxels; cover 2 of them → m[R1] = 2/3
    mask[0, 0, 0] = 1
    mask[0, 0, 1] = 1
    subj = _make_subject(mask)

    analysis = SimplifiedFunctionalNetworkMapping(
        matrix_path=matrix_path,
        parcel_names=[registered_atlas],
        lesion_weighting="voxel_count",
    )
    out = analysis.run(subj)
    results = out.results["SimplifiedFunctionalNetworkMapping"]
    parc = next(v for k, v in results.items() if k.endswith("desc-sfnmap"))
    assert parc.data["R1"] == pytest.approx(2 / 3)
    assert parc.data["R2"] == pytest.approx(0.0)


def test_sfnm_rejects_unknown_weighting(tmp_path, registered_atlas):
    matrix_path = _write_c_matrix(tmp_path, ["R1", "R2"], np.eye(2))
    with pytest.raises(ValueError):
        SimplifiedFunctionalNetworkMapping(
            matrix_path=matrix_path,
            parcel_names=[registered_atlas],
            lesion_weighting="nonsense",
        )


def test_sfnm_requires_parcel_names(tmp_path):
    matrix_path = _write_c_matrix(tmp_path, ["R1", "R2"], np.eye(2))
    with pytest.raises(ValueError):
        SimplifiedFunctionalNetworkMapping(matrix_path=matrix_path, parcel_names=None)


def test_sfnm_single_atlas_only(tmp_path, registered_atlas):
    matrix_path = _write_c_matrix(tmp_path, ["R1", "R2"], np.eye(2))
    with pytest.raises(ValueError, match="exactly one parcellation"):
        SimplifiedFunctionalNetworkMapping(
            matrix_path=matrix_path,
            parcel_names=[registered_atlas, registered_atlas],
        )


def test_sfnm_mismatched_labels_error(tmp_path, registered_atlas):
    """Matrix labels not present in atlas should raise clearly."""
    matrix_path = _write_c_matrix(tmp_path, ["R1", "OTHER"], np.eye(2))
    mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
    mask[0, 0, 0] = 1
    subj = _make_subject(mask)
    analysis = SimplifiedFunctionalNetworkMapping(
        matrix_path=matrix_path, parcel_names=[registered_atlas]
    )
    with pytest.raises(ValueError, match="missing"):
        analysis.run(subj)


def test_sfnm_empty_overlap_returns_zero_map(tmp_path, registered_atlas):
    """Lesion that misses every region → m is zero; sfnmap is zero; no crash."""
    C = np.array([[1.0, 0.5], [0.5, 1.0]])
    matrix_path = _write_c_matrix(tmp_path, ["R1", "R2"], C)
    mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
    mask[3, 3, 3] = 1  # background, outside both regions
    subj = _make_subject(mask)

    analysis = SimplifiedFunctionalNetworkMapping(
        matrix_path=matrix_path, parcel_names=[registered_atlas]
    )
    out = analysis.run(subj)
    results = out.results["SimplifiedFunctionalNetworkMapping"]
    parc = next(v for k, v in results.items() if k.endswith("desc-sfnmap"))
    assert parc.data["R1"] == pytest.approx(0.0)
    assert parc.data["R2"] == pytest.approx(0.0)
    assert parc.metadata["n_regions_touched"] == 0


def test_sfnm_keep_intermediate_emits_weights(tmp_path, registered_atlas):
    """keep_intermediate=True → sfnmweights ParcelData is emitted."""
    matrix_path = _write_c_matrix(tmp_path, ["R1", "R2"], np.eye(2))
    mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
    mask[0, 0, 0] = 1  # R1
    subj = _make_subject(mask)

    analysis = SimplifiedFunctionalNetworkMapping(
        matrix_path=matrix_path,
        parcel_names=[registered_atlas],
        lesion_weighting="binary",
        keep_intermediate=True,
    )
    out = analysis.run(subj)
    results = out.results["SimplifiedFunctionalNetworkMapping"]
    weights = next(v for k, v in results.items() if k.endswith("desc-sfnmweights"))
    assert weights.data["R1"] == pytest.approx(1.0)
    assert weights.data["R2"] == pytest.approx(0.0)
    assert "voxels_per_region" in weights.metadata


def test_sfnm_missing_matrix_file(tmp_path, registered_atlas):
    analysis = SimplifiedFunctionalNetworkMapping(
        matrix_path=tmp_path / "does_not_exist.tsv",
        parcel_names=[registered_atlas],
    )
    mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
    mask[0, 0, 0] = 1
    with pytest.raises(FileNotFoundError):
        analysis.run(_make_subject(mask))


def test_sfnm_non_square_matrix_rejected(tmp_path, registered_atlas):
    tsv = tmp_path / "nonsquare.tsv"
    pd.DataFrame(
        np.ones((2, 3)), index=["R1", "R2"], columns=["R1", "R2", "R3"]
    ).to_csv(tsv, sep="\t")
    analysis = SimplifiedFunctionalNetworkMapping(
        matrix_path=tsv, parcel_names=[registered_atlas]
    )
    mask = np.zeros(MASK_SHAPE, dtype=np.uint8)
    mask[0, 0, 0] = 1
    with pytest.raises(ValueError, match="square"):
        analysis.run(_make_subject(mask))


def test_sfnm_bids_filename_resolves_to_method_sfnm():
    """Guard against desc/source mapping drift: sfnm result key → method-sfnm."""
    from lacuna.core.keys import BidsFilename

    key = "atlas-schaefer400_source-SimplifiedFunctionalNetworkMapping_desc-sfnmap"
    bids = BidsFilename.from_result_key(
        key, suffix="values", namespace="SimplifiedFunctionalNetworkMapping"
    )
    rendered = str(bids)
    assert "method-sfnm_" in rendered
    assert "method-fnm_" not in rendered
    assert "desc-sfnmap" in rendered
    assert rendered.endswith("parcelstats")
