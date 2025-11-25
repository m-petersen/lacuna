"""Quick demo showing TARGET_SPACE automatic transformation."""

import nibabel as nib
import numpy as np

from lacuna import MaskData
from lacuna.analysis.base import BaseAnalysis


class DemoAnalysis(BaseAnalysis):
    """Demo analysis requiring MNI152NLin6Asym @ 2mm."""

    TARGET_SPACE = "MNI152NLin6Asym"
    TARGET_RESOLUTION = 2

    def _validate_inputs(self, mask_data):
        space = mask_data.space
        resolution = mask_data.resolution
        print(f"  ✓ Validation: Lesion is now in {space} @ {resolution}mm")

        # This should always pass because _ensure_target_space ran first
        assert space == self.TARGET_SPACE
        assert resolution == self.TARGET_RESOLUTION

    def _run_analysis(self, mask_data):
        space = mask_data.space
        resolution = mask_data.resolution
        shape = mask_data.mask_img.shape
        return {"space": space, "resolution": resolution, "shape": shape}


print("\n" + "=" * 60)
print("TARGET_SPACE Automatic Transformation Demo")
print("=" * 60)

# Create lesion in MNI152NLin6Asym @ 2mm
data = np.zeros((91, 109, 91))
data[45:50, 54:59, 45:50] = 1

affine = np.array(
    [[-2.0, 0.0, 0.0, 90.0], [0.0, 2.0, 0.0, -126.0], [0.0, 0.0, 2.0, -72.0], [0.0, 0.0, 0.0, 1.0]]
)

img = nib.Nifti1Image(data, affine)
lesion = MaskData(mask_img=img, metadata={"space": "MNI152NLin6Asym", "resolution": 2})

print(f"\n1. Input lesion: {lesion.metadata.get('space')} @ {lesion.metadata.get('resolution')}mm")
print(f"   Shape: {lesion.mask_img.shape}")

print(
    f"\n2. DemoAnalysis requires: {DemoAnalysis.TARGET_SPACE} @ {DemoAnalysis.TARGET_RESOLUTION}mm"
)

print("\n3. Running analysis...")
analysis = DemoAnalysis()
result = analysis.run(lesion)

print("\n4. Result:")
print(f"   Space: {result.results['DemoAnalysis']['space']}")
print(f"   Resolution: {result.results['DemoAnalysis']['resolution']}mm")
print(f"   Shape: {result.results['DemoAnalysis']['shape']}")

print("\n✓ SUCCESS: Analysis ran with lesion in target space!")

print("\n" + "-" * 60)
print("How it works:")
print("-" * 60)
print("1. Analysis declares TARGET_SPACE and TARGET_RESOLUTION")
print("2. BaseAnalysis.run() calls _ensure_target_space()")
print("3. If lesion not in target space, automatic transformation happens")
print("4. Then _validate_inputs() is called (lesion already transformed)")
print("5. Finally _run_analysis() executes (lesion guaranteed in target space)")

print("\n" + "=" * 60)
print("Checking real analysis classes...")
print("=" * 60)

from lacuna.analysis import (
    FunctionalNetworkMapping,
    ParcelAggregation,
    RegionalDamage,
    StructuralNetworkMapping,
)

analyses = [
    ("FunctionalNetworkMapping", FunctionalNetworkMapping),
    ("StructuralNetworkMapping", StructuralNetworkMapping),
    ("ParcelAggregation", ParcelAggregation),
    ("RegionalDamage", RegionalDamage),
]

for name, cls in analyses:
    space = cls.TARGET_SPACE
    res = cls.TARGET_RESOLUTION
    print(f"\n{name}:")
    print(f"  TARGET_SPACE: {space}")
    print(f"  TARGET_RESOLUTION: {res}")
    if space == "atlas":
        print("  → Adaptive (uses atlas's native space)")
    else:
        print(f"  → Fixed (transforms to {space} @ {res}mm)")

print("\n" + "=" * 60)
print("✓ All analyses have TARGET_SPACE/TARGET_RESOLUTION defined!")
print("=" * 60 + "\n")
