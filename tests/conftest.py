"""
Shared test fixtures for lesion decoding toolkit tests.

Provides common fixtures used across contract, integration, and unit tests.
"""

import json

import nibabel as nib
import numpy as np
import pytest


def _check_mrtrix_available():
    """Check if MRtrix3 is available for tests."""
    try:
        from lacuna.utils.mrtrix import check_mrtrix_available

        check_mrtrix_available()
        return True
    except (ImportError, Exception):
        return False


def _check_templateflow_available():
    """Check if TemplateFlow is available and usable.

    Returns True only if TemplateFlow can be imported and at least one
    template is already cached. This prevents CI failures when
    TemplateFlow tries to download files in parallel tests.
    """
    try:
        import importlib.util
        import os
        from pathlib import Path

        # Check if templateflow is available without importing it
        if importlib.util.find_spec("templateflow") is None:
            return False

        home = Path.home()
        templateflow_home = home / ".cache" / "templateflow"

        # Also check TEMPLATEFLOW_HOME env var
        if "TEMPLATEFLOW_HOME" in os.environ:
            templateflow_home = Path(os.environ["TEMPLATEFLOW_HOME"])

        # Check if at least one common template is cached
        cached_templates = list(templateflow_home.glob("tpl-MNI152*"))
        if not cached_templates:
            return False

        return True
    except (ImportError, Exception):
        return False


# Cache the availability checks to avoid repeated calls
_MRTRIX_AVAILABLE = None
_TEMPLATEFLOW_AVAILABLE = None


def pytest_configure(config):
    """Cache availability checks at pytest startup."""
    global _MRTRIX_AVAILABLE, _TEMPLATEFLOW_AVAILABLE
    _MRTRIX_AVAILABLE = _check_mrtrix_available()
    _TEMPLATEFLOW_AVAILABLE = _check_templateflow_available()


def pytest_collection_modifyitems(config, items):
    """Skip tests that require unavailable dependencies."""
    for item in items:
        # Skip tests marked with requires_mrtrix if MRtrix3 is not available
        if "requires_mrtrix" in [m.name for m in item.iter_markers()]:
            if not _MRTRIX_AVAILABLE:
                item.add_marker(pytest.mark.skip(reason="MRtrix3 not available"))

        # Skip tests marked with requires_templateflow if TemplateFlow is not cached
        if "requires_templateflow" in [m.name for m in item.iter_markers()]:
            if not _TEMPLATEFLOW_AVAILABLE:
                item.add_marker(pytest.mark.skip(reason="TemplateFlow not available or not cached"))


# Export helper functions for use in test files (for backward compatibility)
requires_mrtrix_skipif = pytest.mark.skipif(
    not _check_mrtrix_available(), reason="MRtrix3 not available"
)
requires_templateflow_skipif = pytest.mark.skipif(
    not _check_templateflow_available(), reason="TemplateFlow not available or not cached"
)


@pytest.fixture
def synthetic_mask_img():
    """Create a synthetic 3D lesion mask for testing."""
    shape = (64, 64, 64)
    data = np.zeros(shape, dtype=np.uint8)

    # Create small spherical lesion
    center = (32, 32, 32)
    radius = 5
    for x in range(max(0, center[0] - radius), min(shape[0], center[0] + radius + 1)):
        for y in range(max(0, center[1] - radius), min(shape[1], center[1] + radius + 1)):
            for z in range(max(0, center[2] - radius), min(shape[2], center[2] + radius + 1)):
                dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
                if dist <= radius:
                    data[x, y, z] = 1

    affine = np.eye(4)
    affine[0, 0] = 2.0  # 2mm voxels
    affine[1, 1] = 2.0
    affine[2, 2] = 2.0

    return nib.Nifti1Image(data, affine)


@pytest.fixture
def synthetic_anatomical_img():
    """Create a synthetic anatomical (T1w) image for testing."""
    shape = (64, 64, 64)
    # Create brain-like structure with higher intensity in center
    data = np.random.rand(*shape).astype(np.float32) * 100

    # Add "brain" with ellipsoid
    center = np.array(shape) // 2
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                dist = np.sqrt(
                    ((x - center[0]) / 20) ** 2
                    + ((y - center[1]) / 25) ** 2
                    + ((z - center[2]) / 20) ** 2
                )
                if dist < 1.0:
                    data[x, y, z] += 500 * (1 - dist)

    affine = np.eye(4)
    affine[0, 0] = 2.0
    affine[1, 1] = 2.0
    affine[2, 2] = 2.0

    return nib.Nifti1Image(data, affine)


@pytest.fixture
def synthetic_4d_img():
    """Create a synthetic 4D image (should be rejected by validation)."""
    shape = (64, 64, 64, 10)
    data = np.random.rand(*shape).astype(np.float32)
    affine = np.eye(4)
    return nib.Nifti1Image(data, affine)


@pytest.fixture
def simple_bids_dataset(tmp_path):
    """
    Create a minimal BIDS dataset for testing.

    Structure:
        dataset/
        ├── dataset_description.json
        ├── sub-001/
        │   └── anat/
        │       ├── sub-001_T1w.nii.gz
        │       └── sub-001_desc-lesion_mask.nii.gz
        └── sub-002/
            └── anat/
                └── sub-002_desc-lesion_mask.nii.gz
    """
    dataset_root = tmp_path / "bids_dataset"
    dataset_root.mkdir()

    # Create dataset_description.json

    desc = {
        "Name": "Test Lesion Dataset",
        "BIDSVersion": "1.6.0",
        "DatasetType": "raw",
    }
    with open(dataset_root / "dataset_description.json", "w") as f:
        json.dump(desc, f)

    # Create synthetic lesion masks
    shape = (64, 64, 64)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    # Subject 1 (with anatomical)
    sub1_dir = dataset_root / "sub-001" / "anat"
    sub1_dir.mkdir(parents=True)

    # Lesion mask for sub-001
    data1 = np.zeros(shape, dtype=np.uint8)
    data1[30:35, 30:35, 30:35] = 1
    lesion1 = nib.Nifti1Image(data1, affine)
    nib.save(lesion1, sub1_dir / "sub-001_desc-lesion_mask.nii.gz")

    # Anatomical for sub-001
    anat1_data = np.random.rand(*shape).astype(np.float32) * 1000
    anat1 = nib.Nifti1Image(anat1_data, affine)
    nib.save(anat1, sub1_dir / "sub-001_T1w.nii.gz")

    # Subject 2 (lesion only, no anatomical)
    sub2_dir = dataset_root / "sub-002" / "anat"
    sub2_dir.mkdir(parents=True)

    data2 = np.zeros(shape, dtype=np.uint8)
    data2[25:30, 25:30, 25:30] = 1
    lesion2 = nib.Nifti1Image(data2, affine)
    nib.save(lesion2, sub2_dir / "sub-002_desc-lesion_mask.nii.gz")

    return dataset_root


@pytest.fixture
def multisession_bids_dataset(tmp_path):
    """
    Create a multi-session BIDS dataset for testing.

    Structure:
        dataset/
        ├── dataset_description.json
        └── sub-001/
            ├── ses-01/
            │   └── anat/
            │       └── sub-001_ses-01_desc-lesion_mask.nii.gz
            └── ses-02/
                └── anat/
                    └── sub-001_ses-02_desc-lesion_mask.nii.gz
    """
    dataset_root = tmp_path / "bids_multisession"
    dataset_root.mkdir()

    # Create dataset_description.json

    desc = {
        "Name": "Test Multi-Session Lesion Dataset",
        "BIDSVersion": "1.6.0",
        "DatasetType": "raw",
    }
    with open(dataset_root / "dataset_description.json", "w") as f:
        json.dump(desc, f)

    # Create synthetic lesion masks
    shape = (64, 64, 64)
    affine = np.eye(4)
    affine[0, 0] = affine[1, 1] = affine[2, 2] = 2.0

    # Session 1
    ses1_dir = dataset_root / "sub-001" / "ses-01" / "anat"
    ses1_dir.mkdir(parents=True)

    data1 = np.zeros(shape, dtype=np.uint8)
    data1[30:35, 30:35, 30:35] = 1
    lesion1 = nib.Nifti1Image(data1, affine)
    nib.save(lesion1, ses1_dir / "sub-001_ses-01_desc-lesion_mask.nii.gz")

    # Session 2
    ses2_dir = dataset_root / "sub-001" / "ses-02" / "anat"
    ses2_dir.mkdir(parents=True)

    data2 = np.zeros(shape, dtype=np.uint8)
    data2[25:30, 25:30, 25:30] = 1
    lesion2 = nib.Nifti1Image(data2, affine)
    nib.save(lesion2, ses2_dir / "sub-001_ses-02_desc-lesion_mask.nii.gz")

    return dataset_root


@pytest.fixture
def synthetic_mask_data(synthetic_mask_img):
    """Create a MaskData object from synthetic lesion image."""
    import sys

    sys.path.insert(0, "/home/marvin/projects/lacuna/src")
    from lacuna.core.mask_data import MaskData

    return MaskData(
        mask_img=synthetic_mask_img,
        metadata={
            "subject_id": "sub-test",
            "source": "synthetic",
            "space": "MNI152NLin6Asym",
            "resolution": 2,
        },
    )


@pytest.fixture
def batch_mask_data_list(synthetic_mask_img):
    """Create a list of MaskData objects for batch testing."""
    import sys

    sys.path.insert(0, "/home/marvin/projects/lacuna/src")
    from lacuna.core.mask_data import MaskData

    lesion_list = []
    for i in range(1, 4):  # Create 3 test subjects
        # Create slightly different lesion for each subject
        data = synthetic_mask_img.get_fdata().copy()
        # Shift lesion slightly for each subject
        data = np.roll(data, shift=i * 2, axis=0)

        mask_img = nib.Nifti1Image(data.astype(np.uint8), synthetic_mask_img.affine)
        mask_data = MaskData(
            mask_img=mask_img,
            metadata={
                "subject_id": f"sub-{i:03d}",
                "source": "synthetic_batch",
                "space": "MNI152NLin6Asym",
                "resolution": 2,
            },
        )
        lesion_list.append(mask_data)

    return lesion_list


@pytest.fixture(autouse=True)
def clean_atlas_registry():
    """Clean up atlas registry after each test to prevent cross-contamination.

    This fixture automatically runs after every test to remove any atlases
    registered during the test. This prevents tests from interfering with
    each other when they register temporary atlases.
    """
    # Store bundled atlas names before test
    # (these are the atlases pre-registered in the module)
    from lacuna.assets.parcellations.registry import PARCELLATION_REGISTRY

    # On first call, save the bundled atlas names
    if not hasattr(clean_atlas_registry, "_bundled_names"):
        clean_atlas_registry._bundled_names = set(PARCELLATION_REGISTRY.keys())

    # Run the test
    yield

    # After test: remove any atlases that weren't bundled
    # (i.e., remove test-registered atlases)
    bundled_names = clean_atlas_registry._bundled_names
    to_remove = [name for name in PARCELLATION_REGISTRY.keys() if name not in bundled_names]
    for name in to_remove:
        del PARCELLATION_REGISTRY[name]
