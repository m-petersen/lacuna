"""
Test fixtures for lesion decoding toolkit.

This module provides synthetic NIfTI data for testing without requiring
large real neuroimaging files.
"""

from pathlib import Path

import nibabel as nib
import numpy as np


def create_synthetic_lesion_mask(
    shape=(64, 64, 64), voxel_size=(2.0, 2.0, 2.0), lesion_coords=None, lesion_radius=5
):
    """
    Create a synthetic binary lesion mask.

    Parameters
    ----------
    shape : tuple
        Image dimensions (x, y, z)
    voxel_size : tuple
        Voxel size in mm (x, y, z)
    lesion_coords : tuple, optional
        Center coordinates of lesion (x, y, z). If None, uses center of image.
    lesion_radius : int
        Radius of spherical lesion in voxels

    Returns
    -------
    nibabel.Nifti1Image
        Synthetic lesion mask
    """
    # Create empty mask
    data = np.zeros(shape, dtype=np.uint8)

    # Determine lesion center
    if lesion_coords is None:
        lesion_coords = tuple(s // 2 for s in shape)

    # Create spherical lesion
    cx, cy, cz = lesion_coords
    for x in range(max(0, cx - lesion_radius), min(shape[0], cx + lesion_radius + 1)):
        for y in range(max(0, cy - lesion_radius), min(shape[1], cy + lesion_radius + 1)):
            for z in range(max(0, cz - lesion_radius), min(shape[2], cz + lesion_radius + 1)):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                if dist <= lesion_radius:
                    data[x, y, z] = 1

    # Create affine matrix (RAS+ orientation)
    affine = np.eye(4)
    affine[0, 0] = voxel_size[0]
    affine[1, 1] = voxel_size[1]
    affine[2, 2] = voxel_size[2]
    # Center the image
    affine[0, 3] = -shape[0] * voxel_size[0] / 2
    affine[1, 3] = -shape[1] * voxel_size[1] / 2
    affine[2, 3] = -shape[2] * voxel_size[2] / 2

    return nib.Nifti1Image(data, affine)


def create_test_fixtures(output_dir=None):
    """
    Create all test fixtures and save to disk.

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory to save fixtures. Defaults to tests/fixtures/
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create various test lesions
    fixtures = {
        "sample_lesion.nii.gz": create_synthetic_lesion_mask(),
        "sample_lesion_small.nii.gz": create_synthetic_lesion_mask(lesion_radius=3),
        "sample_lesion_large.nii.gz": create_synthetic_lesion_mask(lesion_radius=10),
        "sample_lesion_1mm.nii.gz": create_synthetic_lesion_mask(
            shape=(128, 128, 128), voxel_size=(1.0, 1.0, 1.0)
        ),
    }

    for filename, img in fixtures.items():
        filepath = output_dir / filename
        nib.save(img, filepath)
        print(f"Created: {filepath}")


if __name__ == "__main__":
    create_test_fixtures()
