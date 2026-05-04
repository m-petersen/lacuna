"""Tests for FunctionalNeurotransmitterFingerprinting analysis."""

from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pytest

from lacuna.analysis.functional_neurotransmitter_fingerprinting import (
    FunctionalNeurotransmitterFingerprinting,
)
from lacuna.assets.connectomes import (
    register_functional_connectome,
    unregister_functional_connectome,
)
from lacuna.atlas.store import build_nt_atlas, load_atlas, save_atlas
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


class TestFunctionalNTMConstruction:
    def test_basic_construction(self, atlas_cache):
        fntf = FunctionalNeurotransmitterFingerprinting(
            ntatlas_dir=atlas_cache,
            connectome_name="GSP1000",
        )
        assert fntf._target_spec == "all"
        assert fntf.enriched is False

    def test_enriched_via_ace_dir(self, tmp_path, atlas_cache):
        ace_dir = tmp_path / "ace"
        (ace_dir / "stage2_atlas").mkdir(parents=True)
        (ace_dir / "stage1_timeseries").mkdir()
        for child in atlas_cache.iterdir():
            target = ace_dir / "stage2_atlas" / child.name
            if child.is_dir():
                target.mkdir(exist_ok=True)
                for f in child.iterdir():
                    target.joinpath(f.name).write_bytes(f.read_bytes())
            else:
                target.write_bytes(child.read_bytes())

        fntf = FunctionalNeurotransmitterFingerprinting(
            ace_dir=ace_dir,
            connectome_name="GSP1000",
        )
        assert fntf.enriched is True

    def test_xor_atlas_and_ace(self, atlas_cache):
        with pytest.raises(ValueError, match="exactly one"):
            FunctionalNeurotransmitterFingerprinting(
                ntatlas_dir=atlas_cache,
                ace_dir=atlas_cache,
                connectome_name="GSP1000",
            )
        with pytest.raises(ValueError, match="exactly one"):
            FunctionalNeurotransmitterFingerprinting(
                connectome_name="GSP1000",
            )

    def test_targets_parameter(self, atlas_cache):
        fntf = FunctionalNeurotransmitterFingerprinting(
            ntatlas_dir=atlas_cache,
            connectome_name="GSP1000",
            targets=["D1"],
        )
        assert fntf._target_spec == ["D1"]


# ----- ACE-enriched fixtures and tests -----


_CONNECTOME_AFFINE = np.array(
    [
        [-2.0, 0.0, 0.0, 90.0],
        [0.0, 2.0, 0.0, -126.0],
        [0.0, 0.0, 2.0, -72.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
_CONNECTOME_MASK_SHAPE = (91, 109, 91)


def _write_fake_connectome(path, n_subjects, n_timepoints, n_voxels, seed):
    """Write a single-batch HDF5 connectome with sequential mask coords."""
    rng = np.random.default_rng(seed)
    timeseries = rng.standard_normal((n_subjects, n_timepoints, n_voxels)).astype(
        np.float32
    )
    mask_indices = np.array(
        [
            np.arange(n_voxels) // 100,
            (np.arange(n_voxels) // 10) % 10,
            np.arange(n_voxels) % 10,
        ],
        dtype=np.int64,
    )
    with h5py.File(path, "w") as f:
        f.create_dataset("timeseries", data=timeseries)
        f.create_dataset("mask_indices", data=mask_indices)
        f.create_dataset("mask_affine", data=_CONNECTOME_AFFINE)
        f.attrs["mask_shape"] = _CONNECTOME_MASK_SHAPE
    return timeseries, mask_indices


@pytest.fixture
def fake_functional_connectome(tmp_path):
    path = tmp_path / "fake_fc.h5"
    _write_fake_connectome(path, n_subjects=4, n_timepoints=30, n_voxels=200, seed=1)
    name = "test_fntf_connectome"
    register_functional_connectome(
        name=name,
        space="MNI152NLin6Asym",
        resolution=2.0,
        data_path=path,
        n_subjects=4,
        description="Fake functional connectome for fntf testing",
    )
    yield name, path
    unregister_functional_connectome(name)


def _seed_ace_cache(tmp_path, atlas_cache_dir, connectome_path,
                    n_subjects, n_timepoints, *, write_envelope: bool = True):
    """Build an ACE cache aligned with a connectome.

    Layout matches what `lacuna prepare ace` produces:
    - <ace_dir>/stage2_atlas/      (an ntatlas, copied from atlas_cache_dir)
    - <ace_dir>/stage1_timeseries/ (n_subjects subject-NNNN.npy files)
    - <ace_dir>/lacuna_asset.json  (envelope; only when write_envelope=True)

    Returns (ace_dir, stage1_array). When write_envelope is False, the
    cache directory is left without lacuna_asset.json so callers can
    test the missing-envelope error path.
    """
    from lacuna.assets.envelope import (
        AssetEnvelope,
        AssetType,
        RequiresEntry,
        fingerprint,
        write_envelope as _write_env,
    )

    ace_dir = tmp_path / "ace"
    (ace_dir / "stage2_atlas").mkdir(parents=True)
    stage1_dir = ace_dir / "stage1_timeseries"
    stage1_dir.mkdir()
    for child in atlas_cache_dir.iterdir():
        target = ace_dir / "stage2_atlas" / child.name
        if child.is_dir():
            target.mkdir(exist_ok=True)
            for f in child.iterdir():
                target.joinpath(f.name).write_bytes(f.read_bytes())
        else:
            target.write_bytes(child.read_bytes())

    n_targets = len(load_atlas(ace_dir / "stage2_atlas").targets)
    rng = np.random.default_rng(7)
    stage1 = rng.standard_normal((n_subjects, n_timepoints, n_targets)).astype(
        np.float32
    )
    for s in range(n_subjects):
        np.save(stage1_dir / f"sub-{s:02d}.npy", stage1[s])

    if write_envelope:
        env = AssetEnvelope(
            asset_type=AssetType.ACE_CACHE,
            identity=fingerprint(ace_dir, AssetType.ACE_CACHE),
            requires=[
                RequiresEntry(
                    role="connectome",
                    asset_type=AssetType.FUNCTIONAL_CONNECTOME,
                    identity=fingerprint(
                        Path(connectome_path), AssetType.FUNCTIONAL_CONNECTOME,
                    ),
                    path_hint=str(Path(connectome_path).resolve()),
                ),
            ],
            provenance={
                "source_ntatlas_path": str((ace_dir / "stage2_atlas").resolve()),
                "source_ntatlas_identity": fingerprint(
                    ace_dir / "stage2_atlas", AssetType.NTATLAS
                ).to_dict(),
            },
            data={"n_targets": n_targets, "n_timepoints": n_timepoints},
        )
        _write_env(env, ace_dir)

    return ace_dir, stage1


class TestFunctionalNTFEnriched:
    """Tests for the ACE-enriched scoring path."""

    def _lesion_subject(self):
        # Mask in connectome space; overlap with mask_indices via voxels
        # near (0,0,0) up to (1,1,9) which the connectome's mask includes.
        mask_data = np.zeros(_CONNECTOME_MASK_SHAPE, dtype=np.int8)
        mask_data[0:2, 0:2, 0:2] = 1
        return SubjectData(
            mask_img=nib.Nifti1Image(mask_data, _CONNECTOME_AFFINE),
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": "sub-01"},
        )

    def test_fisher_z_mean_matches_manual(
        self, tmp_path, atlas_cache, fake_functional_connectome
    ):
        connectome_name, connectome_path = fake_functional_connectome
        ace_dir, stage1 = _seed_ace_cache(
            tmp_path, atlas_cache, connectome_path,
            n_subjects=4, n_timepoints=30,
        )

        fntf = FunctionalNeurotransmitterFingerprinting(
            ace_dir=ace_dir,
            connectome_name=connectome_name,
        )
        fntf._validate_inputs(self._lesion_subject())
        scores = fntf._run_enriched(self._lesion_subject())

        # Hand-recompute: use boes (mean) lesion ts per subject from the same
        # connectome, do per-subject Pearson r vs stage1[s,:,T], Fisher-z, mean.
        with h5py.File(connectome_path, "r") as hf:
            ts = hf["timeseries"][:]
        # Mask indices 0..7 correspond to voxels (0,0,0)..(0,0,7) in the
        # connectome's brain mask layout; the lesion's bool overlap is the
        # first few of those.
        lesion = self._lesion_subject().mask_img.get_fdata().astype(bool)
        from lacuna.assets.connectomes.functional_io import read_mask_info
        mi = read_mask_info(connectome_path)["mask_indices"]
        flat_lesion_idx = [
            i for i in range(len(mi[0])) if lesion[mi[0][i], mi[1][i], mi[2][i]]
        ]
        per_subj_lesion_ts = ts[:, :, flat_lesion_idx].mean(axis=2)  # (n_sub, n_t)

        atlas = load_atlas(ace_dir / "stage2_atlas")
        for i, target in enumerate(atlas.targets):
            r_per_sub = []
            for s in range(per_subj_lesion_ts.shape[0]):
                r = np.corrcoef(per_subj_lesion_ts[s], stage1[s, :, i])[0, 1]
                r_per_sub.append(r)
            expected_z = float(np.mean(np.arctanh(np.clip(r_per_sub, -1 + 1e-9, 1 - 1e-9))))
            assert scores[target] == pytest.approx(expected_z, rel=1e-4, abs=1e-4)

    def test_fingerprint_mismatch_raises(
        self, tmp_path, atlas_cache, fake_functional_connectome
    ):
        """Cache built against connectome A, FNTF run against connectome B → AssetMismatchError."""
        connectome_name, connectome_path = fake_functional_connectome
        ace_dir, _ = _seed_ace_cache(
            tmp_path, atlas_cache, connectome_path,
            n_subjects=4, n_timepoints=30,
        )

        # Write a different connectome (different shape ⇒ different fingerprint)
        # and register it under a fresh name. The cache's envelope still pins
        # the ORIGINAL connectome's fingerprint, so swapping at run time MUST
        # raise AssetMismatchError.
        other_path = tmp_path / "other_fc.h5"
        _write_fake_connectome(other_path, n_subjects=8, n_timepoints=30,
                               n_voxels=200, seed=2)
        other_name = "test_fntf_other_connectome"
        register_functional_connectome(
            name=other_name,
            space="MNI152NLin6Asym",
            resolution=2.0,
            data_path=other_path,
            n_subjects=8,
            description="Different functional connectome for mismatch test",
        )
        try:
            fntf = FunctionalNeurotransmitterFingerprinting(
                ace_dir=ace_dir,
                connectome_name=other_name,
            )
            fntf._validate_inputs(self._lesion_subject())
            with pytest.raises(ValueError, match="connectome"):
                fntf._run_enriched(self._lesion_subject())
        finally:
            unregister_functional_connectome(other_name)

    def test_missing_envelope_raises(
        self, tmp_path, atlas_cache, fake_functional_connectome
    ):
        """ACE cache without lacuna_asset.json → FileNotFoundError with envelope filename."""
        from lacuna.assets.envelope import ENVELOPE_FILENAME

        connectome_name, connectome_path = fake_functional_connectome
        ace_dir, _ = _seed_ace_cache(
            tmp_path, atlas_cache, connectome_path,
            n_subjects=4, n_timepoints=30, write_envelope=False,
        )
        fntf = FunctionalNeurotransmitterFingerprinting(
            ace_dir=ace_dir,
            connectome_name=connectome_name,
        )
        fntf._validate_inputs(self._lesion_subject())
        with pytest.raises(FileNotFoundError, match=ENVELOPE_FILENAME):
            fntf._run_enriched(self._lesion_subject())

    def test_subject_count_mismatch_raises(
        self, tmp_path, atlas_cache, fake_functional_connectome
    ):
        connectome_name, connectome_path = fake_functional_connectome
        # Connectome has 4 subjects; stage1 has 5 → dimension mismatch.
        ace_dir, _ = _seed_ace_cache(
            tmp_path, atlas_cache, connectome_path,
            n_subjects=5, n_timepoints=30,
        )
        fntf = FunctionalNeurotransmitterFingerprinting(
            ace_dir=ace_dir,
            connectome_name=connectome_name,
        )
        fntf._validate_inputs(self._lesion_subject())
        with pytest.raises(ValueError, match="subject count"):
            fntf._run_enriched(self._lesion_subject())

    def test_timepoint_count_mismatch_raises(
        self, tmp_path, atlas_cache, fake_functional_connectome
    ):
        connectome_name, connectome_path = fake_functional_connectome
        # Connectome has 30 timepoints; stage1 has 25 → dimension mismatch.
        ace_dir, _ = _seed_ace_cache(
            tmp_path, atlas_cache, connectome_path,
            n_subjects=4, n_timepoints=25,
        )
        fntf = FunctionalNeurotransmitterFingerprinting(
            ace_dir=ace_dir,
            connectome_name=connectome_name,
        )
        fntf._validate_inputs(self._lesion_subject())
        with pytest.raises(ValueError, match="timepoint count"):
            fntf._run_enriched(self._lesion_subject())

    def _lesion_subject_b(self):
        """Second lesion with a different mask shape — exercises real cross-lesion math."""
        mask_data = np.zeros(_CONNECTOME_MASK_SHAPE, dtype=np.int8)
        mask_data[3:5, 0:2, 0:3] = 1
        return SubjectData(
            mask_img=nib.Nifti1Image(mask_data, _CONNECTOME_AFFINE),
            space="MNI152NLin6Asym",
            resolution=2.0,
            metadata={"subject_id": "sub-02"},
        )

    def test_run_batch_enriched_matches_per_lesion_results(
        self, tmp_path, atlas_cache, fake_functional_connectome
    ):
        """Batched enriched mode produces the same per-target scores as per-lesion."""
        connectome_name, connectome_path = fake_functional_connectome
        ace_dir, _ = _seed_ace_cache(
            tmp_path, atlas_cache, connectome_path,
            n_subjects=4, n_timepoints=30,
        )
        m1 = self._lesion_subject()
        m2 = self._lesion_subject_b()

        # Per-lesion baseline
        analysis_a = FunctionalNeurotransmitterFingerprinting(
            ace_dir=ace_dir, connectome_name=connectome_name,
        )
        scores_per_lesion = []
        for m in (m1, m2):
            analysis_a._validate_inputs(m)
            scores_per_lesion.append(analysis_a._run_enriched(m))

        # Batched
        analysis_b = FunctionalNeurotransmitterFingerprinting(
            ace_dir=ace_dir, connectome_name=connectome_name,
        )
        results = analysis_b.run_batch([m1, m2])

        # Compare per-target scores
        from lacuna.core.data_types import LabeledScalars
        for li, m in enumerate((m1, m2)):
            fp = next(
                v for v in results[li].results[
                    "FunctionalNeurotransmitterFingerprinting"
                ].values()
                if isinstance(v, LabeledScalars)
            )
            for target, expected in scores_per_lesion[li].items():
                assert fp.data[target] == pytest.approx(expected, rel=1e-6, abs=1e-6)
