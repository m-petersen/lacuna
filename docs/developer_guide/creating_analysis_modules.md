# Creating Custom Analysis Modules

This guide shows you how to create custom analysis modules that plug into the LDK framework.

## Overview

All analysis modules in LDK inherit from `BaseAnalysis`, which provides a standardized interface and automatic result management. By implementing two simple methods, your analysis becomes a first-class citizen in the LDK ecosystem.

## The BaseAnalysis Contract

Every analysis module must:

1. **Inherit from `BaseAnalysis`**
2. **Implement `_validate_inputs()`** - Check that input data meets requirements
3. **Implement `_run_analysis()`** - Perform the analysis computation

The `run()` method (which users call) is already implemented and handles:
- Input validation
- Analysis execution
- Result namespacing
- Provenance tracking
- Immutability

## Template: Minimal Analysis

Here's the simplest possible analysis module:

```python
# File: my_custom_analysis.py

from ldk.analysis.base import BaseAnalysis
from ldk.core import LesionData

class MyCustomAnalysis(BaseAnalysis):
    """
    Brief description of what this analysis does.
    
    Parameters
    ----------
    param1 : type
        Description of parameter 1.
        
    Examples
    --------
    >>> analysis = MyCustomAnalysis(param1='value')
    >>> result = analysis.run(lesion_data)
    >>> print(result.results['MyCustomAnalysis'])
    """
    
    def __init__(self, param1: str = "default"):
        """Initialize with analysis-specific parameters."""
        super().__init__()
        self.param1 = param1
        
    def _validate_inputs(self, lesion_data: LesionData) -> None:
        """
        Validate that lesion_data meets requirements.
        
        Raises ValueError if validation fails.
        """
        # Example: Check coordinate space
        space = lesion_data.get_coordinate_space()
        if space not in ["MNI152", "native"]:
            raise ValueError(f"Unsupported coordinate space: {space}")
            
    def _run_analysis(self, lesion_data: LesionData) -> dict:
        """
        Perform the analysis computation.
        
        Returns dictionary of results (must be JSON-serializable).
        """
        # Extract data
        lesion_mask = lesion_data.lesion_img.get_fdata()
        
        # Perform computation
        volume_mm3 = lesion_data.get_volume_mm3()
        
        # Return results
        return {
            'volume_mm3': float(volume_mm3),
            'param1_used': self.param1
        }
```

## Usage Example

```python
from ldk import LesionData
from my_custom_analysis import MyCustomAnalysis

# Load data
lesion_data = LesionData.from_nifti("lesion.nii.gz")

# Run analysis
analysis = MyCustomAnalysis(param1="custom_value")
result = analysis.run(lesion_data)

# Access results (automatically namespaced)
print(result.results['MyCustomAnalysis']['volume_mm3'])
```

## Advanced Example: Parameterized Analysis

```python
from typing import Literal
import numpy as np
from ldk.analysis.base import BaseAnalysis
from ldk.core import LesionData

class LesionVolumeAnalysis(BaseAnalysis):
    """
    Calculate lesion volume with various metrics.
    
    Parameters
    ----------
    method : {'voxel_count', 'surface_area', 'both'}
        Calculation method to use.
    threshold : float, optional
        Threshold for continuous-valued lesions (default: 0.5).
    unit : {'mm3', 'ml', 'cc'}
        Output volume unit (default: 'mm3').
        
    Examples
    --------
    >>> analysis = LesionVolumeAnalysis(method='both', unit='ml')
    >>> result = analysis.run(lesion_data)
    >>> print(f"Volume: {result.results['LesionVolumeAnalysis']['volume']} ml")
    """
    
    def __init__(
        self,
        method: Literal['voxel_count', 'surface_area', 'both'] = 'voxel_count',
        threshold: float = 0.5,
        unit: Literal['mm3', 'ml', 'cc'] = 'mm3'
    ):
        super().__init__()
        self.method = method
        self.threshold = threshold
        self.unit = unit
        
        # Unit conversion factors
        self._unit_factors = {'mm3': 1.0, 'ml': 0.001, 'cc': 0.001}
        
    def _validate_inputs(self, lesion_data: LesionData) -> None:
        """Validate lesion data for volume calculation."""
        # Check that lesion mask exists
        if lesion_data.lesion_img is None:
            raise ValueError("Lesion image is required")
            
        # Check for reasonable voxel sizes
        voxel_sizes = np.abs(np.diag(lesion_data.affine)[:3])
        if np.any(voxel_sizes > 10):  # More than 10mm
            import warnings
            warnings.warn(
                f"Large voxel sizes detected: {voxel_sizes}. "
                "Consider resampling for accurate volume calculation."
            )
            
    def _run_analysis(self, lesion_data: LesionData) -> dict:
        """Calculate lesion volume."""
        data = lesion_data.lesion_img.get_fdata()
        
        results = {}
        
        # Voxel count method
        if self.method in ['voxel_count', 'both']:
            binary_mask = data >= self.threshold
            volume_mm3 = lesion_data.get_volume_mm3()
            volume = volume_mm3 * self._unit_factors[self.unit]
            
            results['volume'] = float(volume)
            results['n_voxels'] = int(np.sum(binary_mask))
            
        # Surface area method
        if self.method in ['surface_area', 'both']:
            binary_mask = data >= self.threshold
            surface_area = self._calculate_surface_area(binary_mask, lesion_data.affine)
            results['surface_area_mm2'] = float(surface_area)
            
        # Add metadata
        results['method'] = self.method
        results['unit'] = self.unit
        results['threshold'] = self.threshold
        
        return results
        
    def _calculate_surface_area(self, mask, affine):
        """Calculate surface area using marching cubes (simplified)."""
        # This is a placeholder - real implementation would use skimage.measure.marching_cubes
        voxel_dims = np.abs(np.diag(affine)[:3])
        # Rough approximation: surface voxels * average face area
        return float(np.sum(mask) * np.mean(voxel_dims) ** 2 * 6)
        
    def _get_parameters(self) -> dict:
        """Return parameters for provenance tracking."""
        return {
            'method': self.method,
            'threshold': self.threshold,
            'unit': self.unit
        }
        
    def _get_version(self) -> str:
        """Return analysis version."""
        return "1.0.0"
```

## Chaining Multiple Analyses

Analyses can be chained together, with later analyses accessing earlier results:

```python
class NetworkAnalysis(BaseAnalysis):
    """Analysis that depends on VolumeAnalysis results."""
    
    def _validate_inputs(self, lesion_data: LesionData) -> None:
        # Check that prerequisite analysis was run
        if 'LesionVolumeAnalysis' not in lesion_data.results:
            raise ValueError(
                "NetworkAnalysis requires LesionVolumeAnalysis to be run first"
            )
            
    def _run_analysis(self, lesion_data: LesionData) -> dict:
        # Access previous results
        volume = lesion_data.results['LesionVolumeAnalysis']['volume']
        
        # Use in computation
        network_score = self._compute_score(volume)
        
        return {'network_score': network_score}

# Usage
result = LesionVolumeAnalysis().run(lesion_data)
result = NetworkAnalysis().run(result)  # Chains on previous result
```

## Best Practices

### 1. **Clear Documentation**
```python
class MyAnalysis(BaseAnalysis):
    """
    One-line summary.
    
    Longer description explaining what the analysis does,
    what scientific methods it uses, and when to use it.
    
    Parameters
    ----------
    param : type
        Clear description with units and constraints.
        
    Raises
    ------
    ValueError
        When inputs don't meet requirements.
        
    See Also
    --------
    OtherAnalysis : Related analysis module.
    
    References
    ----------
    .. [1] Smith et al. (2023) "Method description" Journal Name.
    
    Examples
    --------
    >>> analysis = MyAnalysis(param=value)
    >>> result = analysis.run(lesion_data)
    """
```

### 2. **Helpful Error Messages**
```python
def _validate_inputs(self, lesion_data: LesionData) -> None:
    space = lesion_data.get_coordinate_space()
    if space != "MNI152":
        raise ValueError(
            f"Analysis requires MNI152 space, got '{space}'. "
            f"Use ldk.preprocess.normalize_to_mni(lesion_data) first."
        )
```

### 3. **JSON-Serializable Results**
```python
def _run_analysis(self, lesion_data: LesionData) -> dict:
    # Convert numpy types to Python types
    results = {
        'score': float(np.mean(scores)),  # numpy.float64 → float
        'count': int(n_voxels),            # numpy.int64 → int
        'values': scores.tolist(),         # numpy.array → list
    }
    return results
```

### 4. **Provenance Tracking**
```python
def _get_parameters(self) -> dict:
    """Record all parameters that affect results."""
    return {
        'threshold': self.threshold,
        'method': self.method,
        'connectome_version': self.connectome_version
    }
    
def _get_version(self) -> str:
    """Version your analysis for reproducibility."""
    return "1.2.0"  # Update when algorithm changes
```

### 5. **Test Your Analysis**
```python
# tests/test_my_analysis.py

def test_my_analysis_basic(synthetic_lesion_img):
    """Test basic functionality."""
    from ldk import LesionData
    from my_analysis import MyAnalysis
    
    lesion_data = LesionData(lesion_img=synthetic_lesion_img)
    result = MyAnalysis().run(lesion_data)
    
    assert 'MyAnalysis' in result.results
    assert result.results['MyAnalysis']['volume_mm3'] > 0
    
def test_my_analysis_validation():
    """Test input validation."""
    # ... test that validation catches bad inputs
```

## Common Patterns

### Pattern 1: Coordinate Space Requirement
```python
def _validate_inputs(self, lesion_data: LesionData) -> None:
    required_space = "MNI152"
    actual_space = lesion_data.get_coordinate_space()
    
    if actual_space != required_space:
        raise ValueError(
            f"Analysis requires {required_space} coordinate space, "
            f"got {actual_space}. Use spatial normalization first."
        )
```

### Pattern 2: Optional Anatomical Image
```python
def _validate_inputs(self, lesion_data: LesionData) -> None:
    if self.require_anatomical and lesion_data.anatomical_img is None:
        raise ValueError(
            "Anatomical image required. "
            "Load with: LesionData.from_nifti(lesion, anatomical)"
        )
```

### Pattern 3: Metadata Requirements
```python
def _validate_inputs(self, lesion_data: LesionData) -> None:
    required_fields = ['subject_id', 'age', 'sex']
    missing = [f for f in required_fields if f not in lesion_data.metadata]
    
    if missing:
        raise ValueError(
            f"Missing required metadata fields: {missing}. "
            f"Add when loading: LesionData.from_nifti(path, metadata={{...}})"
        )
```

### Pattern 4: Progressive Results
```python
def _run_analysis(self, lesion_data: LesionData) -> dict:
    """Return results with increasing detail levels."""
    results = {
        'summary': {
            'total_volume': volume,
            'mean_score': mean_score
        },
        'detailed': {
            'regional_volumes': regional_data.tolist(),
            'voxelwise_scores': scores.tolist()
        },
        'metadata': {
            'n_regions': len(regions),
            'processing_time_sec': time_elapsed
        }
    }
    return results
```

## Contributing Your Analysis

To contribute your analysis to LDK:

1. **Create the module** in `src/ldk/analysis/your_analysis.py`
2. **Write tests** in `tests/unit/test_your_analysis.py`
3. **Add documentation** to the module docstring
4. **Export** from `src/ldk/analysis/__init__.py`
5. **Submit a pull request** with description and examples

## Questions?

- See the API documentation: `help(BaseAnalysis)`
- Check existing analyses: `src/ldk/analysis/`
- Ask on GitHub Discussions

## Summary

Creating a custom analysis requires:

```python
from ldk.analysis.base import BaseAnalysis

class YourAnalysis(BaseAnalysis):
    def __init__(self, ...):
        super().__init__()
        # Store parameters
        
    def _validate_inputs(self, lesion_data):
        # Check requirements
        
    def _run_analysis(self, lesion_data):
        # Perform computation
        return results_dict
```

That's it! The BaseAnalysis framework handles everything else automatically.
