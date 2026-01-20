"""Contract tests for BIDS-style naming and result key conventions.

Verify that result keys follow BIDS-style naming conventions:
- Format: atlas-{atlas}_source-{source}[_desc-{description}]
- Source class uses abbreviated names (e.g., InputMask for SubjectData)
- For InputMask source, desc is omitted (the mask itself is the data)
- Use build_result_key() to construct keys programmatically
"""

import nibabel as nib
import numpy as np

from lacuna import SubjectData
from lacuna.analysis import ParcelAggregation, RegionalDamage


class TestBIDSStyleResultKeys:
    """Test that result keys follow BIDS key-value naming patterns."""

    def test_parcel_aggregation_uses_bids_style_keys(self):
        """ParcelAggregation should generate BIDS-style keys.

        Format: atlas-{atlas}_source-{source}[_desc-{description}]
        Example: atlas-Schaefer2018_100Parcels7Networks_source-InputMask
        """
        # Create simple test mask
        data = np.zeros((10, 10, 10))
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0  # 2mm resolution
        mask_img = nib.Nifti1Image(data.astype(np.uint8), affine)

        mask_data = SubjectData(
            mask_img=mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2.0}
        )

        # Run analysis
        analysis = ParcelAggregation(
            source="maskimg",
            aggregation="percent",
            parcel_names=["Schaefer2018_100Parcels7Networks"],
        )

        result = analysis.run(mask_data)

        # Check BIDS-style naming
        assert "ParcelAggregation" in result.results
        parcel_results = result.results["ParcelAggregation"]

        # Should have at least one key
        assert len(parcel_results) > 0

        # Check key format: atlas-{name}_source-{source}[_desc-{key}]
        for key in parcel_results.keys():
            # BIDS-style should start with entity-label pattern
            assert key.startswith("atlas-"), f"Expected key to start with 'atlas-', got: {key}"
            assert "_source-" in key, f"Expected '_source-' in key, got: {key}"
            # desc is optional (omitted for InputMask source)
            # Should NOT have old format
            assert "_from_" not in key, f"Old format '_from_' found in key: {key}"
            assert not key.startswith("parc-"), f"Old format 'parc-' found in key: {key}"

    def test_regional_damage_uses_bids_style_keys(self):
        """RegionalDamage should generate BIDS-style keys.

        Format: atlas-{atlas}_source-InputMask
        """
        # Create simple test mask
        data = np.zeros((10, 10, 10))
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0  # 2mm resolution
        mask_img = nib.Nifti1Image(data.astype(np.uint8), affine)

        mask_data = SubjectData(
            mask_img=mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2.0}
        )

        # Run analysis
        analysis = RegionalDamage(parcel_names=["Schaefer2018_100Parcels7Networks"])

        result = analysis.run(mask_data)

        # Check BIDS-style naming
        assert "RegionalDamage" in result.results
        damage_results = result.results["RegionalDamage"]

        # Check key format
        for key in damage_results.keys():
            assert key.startswith("atlas-"), f"Expected key to start with 'atlas-', got: {key}"
            assert "_source-" in key, f"Expected '_source-' in key, got: {key}"
            # desc is optional (omitted for InputMask source)
            assert "_from_" not in key, f"Old format '_from_' found in key: {key}"


class TestResultKeyComponents:
    """Test that result key components follow conventions."""

    def test_source_class_uses_pascal_case(self):
        """Source class names in keys should use PascalCase (e.g., 'SubjectData')."""
        # Create simple test mask
        data = np.zeros((10, 10, 10))
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0  # 2mm resolution
        mask_img = nib.Nifti1Image(data.astype(np.uint8), affine)

        mask_data = SubjectData(
            mask_img=mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2.0}
        )

        # Test ParcelAggregation
        analysis = ParcelAggregation(
            source="maskimg", parcel_names=["Schaefer2018_100Parcels7Networks"]
        )
        result = analysis.run(mask_data)

        # Check that source uses PascalCase (SubjectData, not mask_data)
        parcel_results = result.results["ParcelAggregation"]
        for key in parcel_results.keys():
            # Extract source from key
            source_match = key.split("_source-")[1].split("_desc-")[0]
            # Should be PascalCase
            assert source_match[0].isupper(), f"Expected PascalCase source, got: {source_match}"
            assert "_" not in source_match, f"Source should not contain underscore: {source_match}"

    def test_desc_uses_snake_case(self):
        """Description portion of keys should use snake_case (e.g., 'mask_img')."""
        # Create simple test mask
        data = np.zeros((10, 10, 10))
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0  # 2mm resolution
        mask_img = nib.Nifti1Image(data.astype(np.uint8), affine)

        mask_data = SubjectData(
            mask_img=mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2.0}
        )

        # Test ParcelAggregation
        analysis = ParcelAggregation(
            source="maskimg", parcel_names=["Schaefer2018_100Parcels7Networks"]
        )
        result = analysis.run(mask_data)

        # Check that desc uses snake_case when present
        parcel_results = result.results["ParcelAggregation"]
        for key in parcel_results.keys():
            # desc is optional for InputMask source
            if "_desc-" in key:
                # Extract desc from key
                desc_part = key.split("_desc-")[1]
                # Should be snake_case (lowercase with underscores or just lowercase)
                assert (
                    desc_part.islower() or "_" in desc_part
                ), f"Expected snake_case desc, got: {desc_part}"

    def test_no_old_format_patterns(self):
        """Verify no old-style format patterns like '_from_' or 'parc-' prefix."""
        # Create simple test mask
        data = np.zeros((10, 10, 10))
        data[4:6, 4:6, 4:6] = 1
        affine = np.eye(4)
        affine[:3, :3] *= 2.0  # 2mm resolution
        mask_img = nib.Nifti1Image(data.astype(np.uint8), affine)

        mask_data = SubjectData(
            mask_img=mask_img, metadata={"space": "MNI152NLin6Asym", "resolution": 2.0}
        )

        # Test ParcelAggregation
        analysis = ParcelAggregation(
            source="maskimg", parcel_names=["Schaefer2018_100Parcels7Networks"]
        )
        result = analysis.run(mask_data)

        # Check that result keys don't use old format
        parcel_results = result.results["ParcelAggregation"]
        for key in parcel_results.keys():
            # Should not have old patterns
            assert "_from_" not in key, f"Found old pattern '_from_' in: {key}"
            assert not key.startswith("parc-"), f"Found old pattern 'parc-' in: {key}"
            # Should use new format
            assert key.startswith("atlas-"), f"Expected new format starting with 'atlas-': {key}"
