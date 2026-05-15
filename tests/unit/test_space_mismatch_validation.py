"""
Unit tests for space mismatch validation.

Tests that the system correctly detects and rejects mismatches between:
- Declared space (--mask-space or space= kwarg) and image affine
- Auto-detection from image affine when no space is declared

Also tests the "trust user with warning" path when affine detection fails.
"""

import json
import logging

import nibabel as nib
import numpy as np
import pytest

from lacuna.core.spaces import REFERENCE_AFFINES
from lacuna.core.subject_data import SubjectData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mask_img(space: str, resolution: int = 2) -> nib.Nifti1Image:
    """Create a binary mask with the canonical affine for the given space."""
    affine = REFERENCE_AFFINES[(space, resolution)].copy()
    shape = {
        ("MNI152NLin6Asym", 1): (182, 218, 182),
        ("MNI152NLin6Asym", 2): (91, 109, 91),
        ("MNI152NLin2009cAsym", 1): (193, 229, 193),
        ("MNI152NLin2009cAsym", 2): (97, 115, 97),
    }[(space, resolution)]
    data = np.zeros(shape, dtype=np.uint8)
    data[10:15, 10:15, 10:15] = 1
    return nib.Nifti1Image(data, affine)


def _make_mask_img_nonstandard_affine() -> nib.Nifti1Image:
    """Create a mask with an affine that doesn't match any known reference."""
    # 2mm voxels but with a completely different origin
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-50.0, -60.0, -30.0]  # Not any known template origin
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)
    data[20:30, 20:30, 20:30] = 1
    return nib.Nifti1Image(data, affine)


# ===========================================================================
# Test: SubjectData.__init__ space validation
# ===========================================================================


class TestSubjectDataSpaceValidation:
    """Tests for space detection and cross-validation in SubjectData.__init__."""

    @pytest.mark.fast
    def test_declared_matches_detected(self):
        """When declared space matches the affine, accept silently."""
        img = _make_mask_img("MNI152NLin6Asym", 2)
        sd = SubjectData(mask_img=img, space="MNI152NLin6Asym", resolution=2)
        assert sd.space == "MNI152NLin6Asym"

    @pytest.mark.fast
    def test_declared_contradicts_detected_raises(self):
        """When declared space doesn't match the affine, raise ValueError."""
        img = _make_mask_img("MNI152NLin6Asym", 1)  # NLin6 affine
        with pytest.raises(ValueError, match="Space mismatch"):
            SubjectData(
                mask_img=img,
                space="MNI152NLin2009cAsym",  # Wrong!
                resolution=1,
            )

    @pytest.mark.fast
    def test_autodetect_from_affine(self):
        """When no space is declared, detect from the affine."""
        img = _make_mask_img("MNI152NLin2009cAsym", 2)
        sd = SubjectData(mask_img=img, resolution=2)
        assert sd.space == "MNI152NLin2009cAsym"

    @pytest.mark.fast
    def test_unrecognized_affine_trusts_declared_with_warning(self, caplog):
        """When affine doesn't match any reference, trust declared space but warn."""
        img = _make_mask_img_nonstandard_affine()
        with caplog.at_level(logging.WARNING, logger="lacuna"):
            sd = SubjectData(
                mask_img=img,
                space="MNI152NLin6Asym",
                resolution=2,
            )
        assert sd.space == "MNI152NLin6Asym"
        assert "Cannot verify" in caplog.text

    @pytest.mark.fast
    def test_unrecognized_affine_no_declared_raises(self):
        """When affine doesn't match and no space is declared, raise ValueError."""
        img = _make_mask_img_nonstandard_affine()
        with pytest.raises(ValueError, match="Coordinate space must be specified"):
            SubjectData(mask_img=img, resolution=2)

    @pytest.mark.fast
    def test_affine_slightly_off_still_detected(self):
        """An affine within 1e-3 tolerance should still be detected correctly."""
        affine = REFERENCE_AFFINES[("MNI152NLin6Asym", 2)].copy()
        affine[0, 3] += 0.0005  # Within tolerance
        shape = (91, 109, 91)
        data = np.zeros(shape, dtype=np.uint8)
        data[10:15, 10:15, 10:15] = 1
        img = nib.Nifti1Image(data, affine)

        sd = SubjectData(mask_img=img, space="MNI152NLin6Asym", resolution=2)
        assert sd.space == "MNI152NLin6Asym"

    @pytest.mark.fast
    def test_affine_outside_tolerance_triggers_warning(self, caplog):
        """An affine just outside 1e-3 tolerance loses detection → warning path."""
        affine = REFERENCE_AFFINES[("MNI152NLin6Asym", 2)].copy()
        affine[0, 3] += 5.0  # Outside FOV-center tolerance (resolution + 1e-3)
        shape = (91, 109, 91)
        data = np.zeros(shape, dtype=np.uint8)
        data[10:15, 10:15, 10:15] = 1
        img = nib.Nifti1Image(data, affine)

        with caplog.at_level(logging.WARNING, logger="lacuna"):
            sd = SubjectData(
                mask_img=img,
                space="MNI152NLin6Asym",
                resolution=2,
            )
        assert sd.space == "MNI152NLin6Asym"
        assert "Cannot verify" in caplog.text


# ===========================================================================
# Test: SubjectData.from_nifti space cross-validation
# ===========================================================================


class TestFromNiftiSpaceCrossValidation:
    """Tests for space kwarg vs image affine validation in from_nifti."""

    @pytest.mark.fast
    def test_space_kwarg_matches_filename_entity(self, tmp_path):
        """Space kwarg matching image affine is accepted."""
        img = _make_mask_img("MNI152NLin6Asym", 2)
        path = tmp_path / "sub-01_space-MNI152NLin6Asym_mask.nii.gz"
        nib.save(img, path)

        sd = SubjectData.from_nifti(path, space="MNI152NLin6Asym", resolution=2)
        assert sd.space == "MNI152NLin6Asym"

    @pytest.mark.fast
    def test_space_kwarg_contradicts_affine_raises(self, tmp_path):
        """Space kwarg contradicting image affine raises ValueError."""
        img = _make_mask_img("MNI152NLin6Asym", 1)
        path = tmp_path / "sub-01_space-MNI152NLin6Asym_mask.nii.gz"
        nib.save(img, path)

        # Affine is NLin6Asym but we declare NLin2009cAsym — caught by affine check
        with pytest.raises(ValueError, match="Space mismatch"):
            SubjectData.from_nifti(
                path,
                space="MNI152NLin2009cAsym",
                resolution=1,
            )

    @pytest.mark.fast
    def test_space_kwarg_no_filename_entity_accepted(self, tmp_path):
        """Space kwarg with no filename entity is accepted normally."""
        img = _make_mask_img("MNI152NLin6Asym", 2)
        # Filename has no space- entity
        path = tmp_path / "sub-01_desc-lesion_mask.nii.gz"
        nib.save(img, path)

        sd = SubjectData.from_nifti(path, space="MNI152NLin6Asym", resolution=2)
        assert sd.space == "MNI152NLin6Asym"


# ===========================================================================
# Test: load_bids_dataset space cross-validation
# ===========================================================================


class TestBidsLoaderSpaceCrossValidation:
    """Tests for --mask-space vs image affine in load_bids_dataset."""

    def _make_bids_dataset(self, tmp_path, space_in_filename: str) -> tuple:
        """Create a minimal BIDS dataset with a space entity in the filename."""
        dataset_root = tmp_path / "bids_dataset"
        dataset_root.mkdir()

        desc = {"Name": "Test", "BIDSVersion": "1.6.0", "DatasetType": "raw"}
        with open(dataset_root / "dataset_description.json", "w") as f:
            json.dump(desc, f)

        sub_dir = dataset_root / "sub-01" / "anat"
        sub_dir.mkdir(parents=True)

        # Use canonical affine for the filename space
        img = _make_mask_img(space_in_filename, 1)
        fname = f"sub-01_space-{space_in_filename}_desc-lesion_mask.nii.gz"
        nib.save(img, sub_dir / fname)

        return dataset_root, fname

    @pytest.mark.fast
    def test_matching_cli_and_filename_space(self, tmp_path):
        """CLI --mask-space matching image affine loads successfully."""
        from lacuna.io.bids import load_bids_dataset

        root, _ = self._make_bids_dataset(tmp_path, "MNI152NLin6Asym")
        entries = load_bids_dataset(root, space="MNI152NLin6Asym")
        assert len(entries) == 1
        sd = entries[0].load()
        assert sd.space == "MNI152NLin6Asym"

    @pytest.mark.fast
    def test_contradicting_cli_space_vs_affine_rejected(self, tmp_path):
        """CLI --mask-space contradicting image affine is caught at materialisation.

        Discovery is lazy now, so the contradiction surfaces on ``entry.load()``
        (SubjectData.__init__ checks the affine against the declared space),
        not at discovery time.
        """
        from lacuna.io.bids import load_bids_dataset

        root, _ = self._make_bids_dataset(tmp_path, "MNI152NLin6Asym")

        # Image affine is NLin6Asym but CLI says NLin2009cAsym
        entries = load_bids_dataset(root, space="MNI152NLin2009cAsym")
        assert len(entries) == 1
        with pytest.raises(ValueError):
            entries[0].load()

    @pytest.mark.fast
    def test_cli_space_with_no_filename_entity(self, tmp_path):
        """CLI --mask-space with files that have no space entity works normally."""
        from lacuna.io.bids import load_bids_dataset

        dataset_root = tmp_path / "bids_dataset"
        dataset_root.mkdir()

        desc = {"Name": "Test", "BIDSVersion": "1.6.0", "DatasetType": "raw"}
        with open(dataset_root / "dataset_description.json", "w") as f:
            json.dump(desc, f)

        sub_dir = dataset_root / "sub-01" / "anat"
        sub_dir.mkdir(parents=True)

        # No space entity in filename
        img = _make_mask_img("MNI152NLin6Asym", 2)
        nib.save(img, sub_dir / "sub-01_desc-lesion_mask.nii.gz")

        result = load_bids_dataset(dataset_root, space="MNI152NLin6Asym")
        assert len(result) == 1

    @pytest.mark.fast
    def test_no_cli_space_uses_affine_detection(self, tmp_path):
        """Without CLI --mask-space, space is auto-detected from affine."""
        from lacuna.io.bids import load_bids_dataset

        root, _ = self._make_bids_dataset(tmp_path, "MNI152NLin6Asym")
        entries = load_bids_dataset(root)
        assert len(entries) == 1
        sd = entries[0].load()
        assert sd.space == "MNI152NLin6Asym"
