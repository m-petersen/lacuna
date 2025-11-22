"""Contract tests for BIDS-style naming and PascalCase result values.

Verify that result keys follow BIDS-style naming conventions
and result values use PascalCase.
"""

import nibabel as nib
import numpy as np
import pytest

from lacuna import MaskData
from lacuna.analysis import ParcelAggregation, RegionalDamage


class TestBIDSStyleResultKeys:
    """Test that result keys follow BIDS key-value naming patterns."""

    def test_parcel_aggregation_uses_bids_style_keys(self):
        """ParcelAggregation should generate keys like 'atlas-{name}_desc-{source}' with PascalCase."""
        # Create simple test mask
        data = np.zeros((10, 10, 10))
        data[4:6, 4:6, 4:6] = 1
        mask_img = nib.Nifti1Image(data.astype(np.uint8), np.eye(4))

        mask_data = MaskData(
            mask_img=mask_img,
            metadata={"space": "MNI152NLin6Asym", "resolution": 2.0}
        )

        # Run analysis
        analysis = ParcelAggregation(
            source="mask_img",
            aggregation="percent",
            parcel_names=["Schaefer2018_100Parcels7Networks"]
        )

        result = analysis.run(mask_data)

        # Check BIDS-style naming with PascalCase
        assert "ParcelAggregation" in result.results
        parcel_results = result.results["ParcelAggregation"]

        # Should have at least one key
        assert len(parcel_results) > 0

        # Check key format: atlas-{name}_desc-{Source} (PascalCase)
        for key in parcel_results.keys():
            # BIDS-style should start with entity-label pattern
            assert "atlas-" in key, f"Expected 'atlas-' in key, got: {key}"
            assert "_desc-" in key or "_source-" in key, f"Expected '_desc-' or '_source-' in key, got: {key}"
            # Should NOT have old format
            assert "_from_" not in key, f"Old format '_from_' found in key: {key}"
            # Value should be PascalCase (starts with capital letter after desc-)
            if "_desc-" in key:
                desc_value = key.split("_desc-")[1].split("_")[0]
                assert desc_value[0].isupper(), f"Expected PascalCase after 'desc-', got: {desc_value}"

    def test_regional_damage_uses_bids_style_keys(self):
        """RegionalDamage should generate keys like 'atlas-{name}_desc-{source}'."""
        # Create simple test mask
        data = np.zeros((10, 10, 10))
        data[4:6, 4:6, 4:6] = 1
        mask_img = nib.Nifti1Image(data.astype(np.uint8), np.eye(4))

        mask_data = MaskData(
            mask_img=mask_img,
            metadata={"space": "MNI152NLin6Asym", "resolution": 2.0}
        )

        # Run analysis
        analysis = RegionalDamage(
            parcel_names=["Schaefer2018_100Parcels7Networks"],
            threshold=0.5
        )

        result = analysis.run(mask_data)

        # Check BIDS-style naming
        assert "RegionalDamage" in result.results
        damage_results = result.results["RegionalDamage"]

        # Check key format
        for key in damage_results.keys():
            assert "atlas-" in key, f"Expected 'atlas-' in key, got: {key}"
            assert "_desc-" in key or "_source-" in key, f"Expected '_desc-' or '_source-' in key, got: {key}"
            assert "_from_" not in key, f"Old format '_from_' found in key: {key}"


class TestPascalCaseResultValues:
    """Test that result values use PascalCase naming."""

    def test_functional_network_mapping_uses_pascal_case(self):
        """FunctionalNetworkMapping should use PascalCase (e.g., 'CorrelationMap')."""
        # Test will verify expected result keys after implementation
        # Expected keys: CorrelationMap, ZMap, TMap (not correlation_map, z_map, t_map)
        pytest.skip("Implementation pending - T147: Need to convert FunctionalNetworkMapping result keys to PascalCase")

    def test_structural_network_mapping_uses_pascal_case(self):
        """StructuralNetworkMapping should use PascalCase (e.g., 'DisconnectionMap')."""
        # Test will verify expected result keys after implementation
        # Expected keys: DisconnectionMap, LesionTractogram (not disconnection_map, lesion_tractogram)
        pytest.skip("Implementation pending - T147: Need to convert StructuralNetworkMapping result keys to PascalCase")

    def test_no_snake_case_in_result_values(self):
        """Verify no snake_case result values like 'correlation_map'."""
        # Create simple test mask
        data = np.zeros((10, 10, 10))
        data[4:6, 4:6, 4:6] = 1
        mask_img = nib.Nifti1Image(data.astype(np.uint8), np.eye(4))

        mask_data = MaskData(
            mask_img=mask_img,
            metadata={"space": "MNI152NLin6Asym", "resolution": 2.0}
        )

        # Test ParcelAggregation (already implemented with PascalCase in BIDS keys)
        analysis = ParcelAggregation(
            source="mask_img",
            parcel_names=["Schaefer2018_100Parcels7Networks"]
        )
        result = analysis.run(mask_data)

        # Check that result keys use PascalCase in desc- portion
        parcel_results = result.results["ParcelAggregation"]
        for key in parcel_results.keys():
            # Keys should not contain raw snake_case patterns
            assert "_from_" not in key, f"Found old snake_case pattern '_from_' in: {key}"
            # After desc-, should be PascalCase
            if "_desc-" in key:
                desc_part = key.split("_desc-")[1].split("_")[0]
                # Should start with capital (PascalCase)
                assert desc_part[0].isupper(), f"Expected PascalCase, got: {desc_part} in {key}"
                # Should not be all lowercase (snake_case indicator)
                assert not desc_part.islower(), f"Found lowercase (snake_case) in desc: {desc_part}"

