"""Tests for StructuralNeurotransmitterFingerprinting analysis."""

from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.structural_neurotransmitter_fingerprinting import (
    StructuralNeurotransmitterFingerprinting,
)
from lacuna.assets.connectomes import (
    register_structural_connectome,
    unregister_structural_connectome,
)
from lacuna.atlas.store import build_nt_atlas, load_atlas, save_atlas
from lacuna.core.data_types import LabeledScalars
from lacuna.core.subject_data import SubjectData
from lacuna.data.ntatlas import load_collection


def _write_synthetic_collection(src, targets):
    """Write synthetic NIfTIs for the given targets using collection map IDs."""
    coll = load_collection()
    target_to_map_id = {
        map_id.split("_", 1)[0][len("target-"):]: map_id
        for ids in coll["systems"].values()
        for map_id in ids
    }
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    shape = (91, 109, 91)
    for target in targets:
        map_id = target_to_map_id[target]
        rng = np.random.default_rng(hash(map_id) % 2**32)
        data = rng.random(shape).astype(np.float32) + 0.1
        fname = f"{map_id}_space-MNI152NLin6Asym_desc-proc.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(src / fname))


@pytest.fixture
def atlas_cache(tmp_path):
    pet_dir = tmp_path / "pet_raw"
    pet_dir.mkdir()
    _write_synthetic_collection(pet_dir, ["D1", "5HT1a"])
    atlas = build_nt_atlas(pet_dir)
    cache_dir = tmp_path / "cache"
    save_atlas(atlas, cache_dir)
    return cache_dir


_FAKE_TCK_N_STREAMLINES = 200


def _write_synthetic_tck(path, n_streamlines, seed):
    """Write a small synthetic .tck file used to exercise fingerprinting."""
    rng = np.random.default_rng(seed)
    streamlines = nib.streamlines.ArraySequence(
        [rng.random((2, 3)).astype(np.float32) for _ in range(n_streamlines)]
    )
    tractogram = nib.streamlines.Tractogram(
        streamlines, affine_to_rasmm=np.eye(4)
    )
    nib.streamlines.save(tractogram, str(path))


@pytest.fixture
def fake_connectome(tmp_path):
    """Register a fake structural connectome for testing."""
    tck_path = tmp_path / "fake.tck"
    _write_synthetic_tck(tck_path, _FAKE_TCK_N_STREAMLINES, seed=1)
    name = "test_sntf_connectome"
    register_structural_connectome(
        name=name,
        space="MNI152NLin2009cAsym",
        tractogram_path=tck_path,
        description="Fake connectome for sntf testing",
    )
    yield name
    unregister_structural_connectome(name)


@pytest.fixture
def fake_weights_cache(tmp_path, atlas_cache, fake_connectome):
    from lacuna.assets.connectomes import load_structural_connectome

    cache_dir = tmp_path / "weights_cache"
    atlas = load_atlas(atlas_cache)
    tck_path = load_structural_connectome(fake_connectome).tractogram_path
    _build_fake_weights_cache(
        cache_dir, atlas, _FAKE_TCK_N_STREAMLINES,
        tractogram_path=tck_path, atlas_dir=atlas_cache,
    )
    return cache_dir


class TestStructuralNTMConstruction:
    def test_basic_construction(self, atlas_cache, fake_connectome, fake_weights_cache):
        sntf = StructuralNeurotransmitterFingerprinting(
            ntatlas_dir=atlas_cache,
            connectome_name=fake_connectome,
            precomputed_weights_dir=fake_weights_cache,
            check_dependencies=False,
        )
        assert sntf._target_spec == "all"

    def test_targets_parameter(self, atlas_cache, fake_connectome, fake_weights_cache):
        sntf = StructuralNeurotransmitterFingerprinting(
            ntatlas_dir=atlas_cache,
            connectome_name=fake_connectome,
            precomputed_weights_dir=fake_weights_cache,
            targets=["D1"],
            check_dependencies=False,
        )
        assert sntf._target_spec == ["D1"]

    def test_precomputed_weights_dir_required(self, atlas_cache, fake_connectome):
        with pytest.raises(TypeError, match="precomputed_weights_dir"):
            StructuralNeurotransmitterFingerprinting(
                ntatlas_dir=atlas_cache,
                connectome_name=fake_connectome,
                check_dependencies=False,
            )


def _build_fake_weights_cache(cache_dir, atlas, n_streamlines, tractogram_path, atlas_dir):
    """Write a synthetic precomputed-weights cache and matching envelope."""
    from lacuna.assets.envelope import (
        AssetEnvelope,
        AssetType,
        RequiresEntry,
        fingerprint,
        write_envelope,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_targets = len(atlas.targets)
    start_w = rng.standard_normal((n_targets, n_streamlines)).astype(np.float32)
    end_w = rng.standard_normal((n_targets, n_streamlines)).astype(np.float32)
    np.save(cache_dir / "start_weights.npy", start_w)
    np.save(cache_dir / "end_weights.npy", end_w)
    (cache_dir / "targets.txt").write_text("\n".join(atlas.targets) + "\n")
    np.savetxt(
        cache_dir / "streamline_indices.txt",
        np.arange(n_streamlines, dtype=np.float32),
        fmt="%.0f",
    )
    affine = np.eye(4)
    streamlines = nib.streamlines.ArraySequence(
        [np.array([[i, 0, 0], [0, i, 0]], dtype=np.float32) for i in range(n_streamlines)]
    )
    tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=affine)
    nib.streamlines.save(tractogram, str(cache_dir / "endpoints.tck"))
    env = AssetEnvelope(
        asset_type=AssetType.SNTF_CACHE,
        identity=fingerprint(cache_dir, AssetType.SNTF_CACHE),
        requires=[
            RequiresEntry(
                role="tractogram",
                asset_type=AssetType.STRUCTURAL_CONNECTOME,
                identity=fingerprint(tractogram_path, AssetType.STRUCTURAL_CONNECTOME),
                path_hint=str(tractogram_path),
            ),
            RequiresEntry(
                role="ntatlas",
                asset_type=AssetType.NTATLAS,
                identity=fingerprint(atlas_dir, AssetType.NTATLAS),
                path_hint=str(atlas_dir),
            ),
        ],
        data={"targets": list(atlas.targets)},
    )
    write_envelope(env, cache_dir)
    return start_w, end_w


class TestStructuralNTFCache:
    """Tests for the --precomputed-weights-dir code path."""

    def _lesion_subject(self):
        affine = np.eye(4) * 2
        affine[3, 3] = 1
        mask_data = np.zeros((91, 109, 91), dtype=np.int8)
        mask_data[45:50, 55:60, 45:50] = 1
        return SubjectData(
            mask_img=nib.Nifti1Image(mask_data, affine),
            space="MNI152NLin2009cAsym",
            resolution=2.0,
            metadata={"subject_id": "sub-01"},
        )

    def test_score_from_cache_matches_manual(self, atlas_cache, fake_connectome, tmp_path):
        from lacuna.assets.connectomes import load_structural_connectome

        atlas = load_atlas(atlas_cache)
        weights_dir = tmp_path / "weights"
        n_streamlines = _FAKE_TCK_N_STREAMLINES
        tck_path = load_structural_connectome(fake_connectome).tractogram_path
        start_w, end_w = _build_fake_weights_cache(
            weights_dir, atlas, n_streamlines,
            tractogram_path=tck_path, atlas_dir=atlas_cache,
        )

        # Surviving IDs we want tckedit to "find"
        rng = np.random.default_rng(42)
        surviving = rng.choice(n_streamlines, size=50, replace=False)
        surviving_csv = "\n".join(f"{i:.0f}" for i in surviving)

        sntf = StructuralNeurotransmitterFingerprinting(
            connectome_name=fake_connectome,
            ntatlas_dir=atlas_cache,
            precomputed_weights_dir=weights_dir,
            check_dependencies=False,
            endpoint_combine="mean",
            aggregation="sum",
        )
        # Trigger validation (loads atlas into self._atlas, resolves targets)
        sntf._validate_inputs(self._lesion_subject())

        # Patch tckedit so it just writes the chosen surviving IDs
        def fake_run_mrtrix(cmd, **_):
            out_path = cmd[cmd.index("-tck_weights_out") + 1]
            from pathlib import Path as _P
            _P(out_path).write_text(surviving_csv)

        with patch(
            "lacuna.utils.mrtrix.run_mrtrix_command",
            side_effect=fake_run_mrtrix,
        ):
            scores, count, _ = sntf._score_from_cache(self._lesion_subject(), atlas)

        assert count == len(surviving)
        # Manual recompute: mean of two endpoints, summed across surviving streamlines
        for ti, target in enumerate(atlas.targets):
            expected = float(((start_w[ti, surviving] + end_w[ti, surviving]) / 2.0).sum())
            assert scores[target] == pytest.approx(expected, rel=1e-5, abs=1e-5)

    def test_cache_missing_raises(self, atlas_cache, fake_connectome, tmp_path):
        sntf = StructuralNeurotransmitterFingerprinting(
            connectome_name=fake_connectome,
            ntatlas_dir=atlas_cache,
            precomputed_weights_dir=tmp_path / "nonexistent_cache",
            check_dependencies=False,
        )
        sntf._validate_inputs(self._lesion_subject())
        with pytest.raises(FileNotFoundError, match="Precomputed weights cache missing"):
            sntf._score_from_cache(self._lesion_subject(), load_atlas(atlas_cache))

    def test_cache_built_for_different_tractogram_raises(
        self, atlas_cache, tmp_path
    ):
        """Cache prepared for tractogram A must not be silently usable with B.

        Reproduces the case where ``lacuna prepare sntf`` is run against one
        tractogram and ``lacuna run sntf`` is then pointed at a different
        tractogram via ``--connectome-path`` while reusing the same
        ``--precomputed-weights-dir``.
        """
        atlas = load_atlas(atlas_cache)
        tck_a = tmp_path / "tractogram_a.tck"
        tck_b = tmp_path / "tractogram_b.tck"
        # Same streamline count but different content — tckedit may not
        # raise on its own, so lacuna must catch this.
        _write_synthetic_tck(tck_a, _FAKE_TCK_N_STREAMLINES, seed=11)
        _write_synthetic_tck(tck_b, _FAKE_TCK_N_STREAMLINES, seed=22)

        weights_dir = tmp_path / "weights"
        _build_fake_weights_cache(
            weights_dir, atlas, _FAKE_TCK_N_STREAMLINES,
            tractogram_path=tck_a, atlas_dir=atlas_cache,
        )

        name = "test_sntf_mismatch_connectome"
        register_structural_connectome(
            name=name,
            space="MNI152NLin2009cAsym",
            tractogram_path=tck_b,
            description="Different tractogram than the cache was built for",
        )
        try:
            sntf = StructuralNeurotransmitterFingerprinting(
                connectome_name=name,
                ntatlas_dir=atlas_cache,
                precomputed_weights_dir=weights_dir,
                check_dependencies=False,
            )
            sntf._validate_inputs(self._lesion_subject())
            with pytest.raises(ValueError, match="does not match"):
                sntf._score_from_cache(self._lesion_subject(), atlas)
        finally:
            unregister_structural_connectome(name)

    def test_cache_without_envelope_raises(
        self, atlas_cache, fake_connectome, tmp_path
    ):
        """Caches written without the envelope (older lacuna or corrupted) must not silently load."""
        from lacuna.assets.connectomes import load_structural_connectome
        from lacuna.assets.envelope import ENVELOPE_FILENAME

        atlas = load_atlas(atlas_cache)
        weights_dir = tmp_path / "weights"
        tck_path = load_structural_connectome(fake_connectome).tractogram_path
        _build_fake_weights_cache(
            weights_dir, atlas, _FAKE_TCK_N_STREAMLINES,
            tractogram_path=tck_path, atlas_dir=atlas_cache,
        )
        (weights_dir / ENVELOPE_FILENAME).unlink()

        sntf = StructuralNeurotransmitterFingerprinting(
            connectome_name=fake_connectome,
            ntatlas_dir=atlas_cache,
            precomputed_weights_dir=weights_dir,
            check_dependencies=False,
        )
        sntf._validate_inputs(self._lesion_subject())
        with pytest.raises(FileNotFoundError, match=ENVELOPE_FILENAME):
            sntf._score_from_cache(self._lesion_subject(), atlas)
