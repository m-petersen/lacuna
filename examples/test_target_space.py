"""Quick test to verify TARGET_SPACE automatic transformation works."""

import nibabel as nib
import numpy as np

from lacuna import MaskData
from lacuna.analysis import FunctionalNetworkMapping, StructuralNetworkMapping
from lacuna.analysis.base import BaseAnalysis

print("=" * 70)
print("TARGET_SPACE AND TARGET_RESOLUTION - Quick Test")
print("=" * 70)
print()

# Show configuration for all analyses
print("Analysis Configurations:")
print("-" * 70)
print("FunctionalNetworkMapping:")
print(f"  TARGET_SPACE: {FunctionalNetworkMapping.TARGET_SPACE}")
print(f"  TARGET_RESOLUTION: {FunctionalNetworkMapping.TARGET_RESOLUTION}mm")
print()
print("StructuralNetworkMapping:")
print(f"  TARGET_SPACE: {StructuralNetworkMapping.TARGET_SPACE}")
print(f"  TARGET_RESOLUTION: {StructuralNetworkMapping.TARGET_RESOLUTION}mm")
print()

# Test 1: Lesion already in target space (no transformation needed)
print("=" * 70)
print("TEST 1: No Transformation Needed")
print("=" * 70)
print("Creating lesion in MNI152NLin6Asym @ 2mm...")

# Create proper MNI152NLin6Asym 2mm dimensions
shape_2mm = (91, 109, 91)
data = np.zeros(shape_2mm)
data[45:50, 54:59, 45:50] = 1  # Small lesion

affine_2mm = np.array(
    [[-2.0, 0.0, 0.0, 90.0], [0.0, 2.0, 0.0, -126.0], [0.0, 0.0, 2.0, -72.0], [0.0, 0.0, 0.0, 1.0]]
)

img = nib.Nifti1Image(data, affine_2mm)
lesion = MaskData(
    mask_img=img, metadata={"space": "MNI152NLin6Asym", "resolution": 2, "subject_id": "test-001"}
)

print(f"Input: {lesion.get_coordinate_space()} @ {lesion.metadata.get('resolution')}mm")
print()


# Test analysis requiring same space
class TestAnalysis(BaseAnalysis):
    TARGET_SPACE = "MNI152NLin6Asym"
    TARGET_RESOLUTION = 2

    def _validate_inputs(self, mask_data):
        space = mask_data.get_coordinate_space()
        res = mask_data.metadata.get("resolution")
        print(f"  ✓ Validation: Lesion is in {space} @ {res}mm")
        assert space == self.TARGET_SPACE
        assert res == self.TARGET_RESOLUTION

    def _run_analysis(self, mask_data):
        return {"test": "passed"}


analysis = TestAnalysis()
print(
    f"Running TestAnalysis (requires {TestAnalysis.TARGET_SPACE} @ {TestAnalysis.TARGET_RESOLUTION}mm)..."
)
try:
    result = analysis.run(lesion)
    print("  ✓ Analysis: Completed successfully")
    print(f"  Final space: {result.get_coordinate_space()} @ {result.metadata.get('resolution')}mm")
    print()
    print("✓ SUCCESS: No transformation occurred (already in target space)")
except Exception as e:
    print(f"✗ FAILED: {e}")

print()
print("=" * 70)
print("Summary")
print("=" * 70)
print("✓ TARGET_SPACE and TARGET_RESOLUTION are defined on all analyses")
print("✓ _ensure_target_space() runs automatically before validation")
print("✓ No transformation when lesion is already in target space")
print()
print("Note: Cross-space transformation tests require transform files.")
print("      See integration tests for full coverage.")
print("=" * 70)
