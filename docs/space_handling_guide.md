# Space Handling Quick Reference

## When to Use Space Helper Functions

### Import Statement
```python
from lacuna.core.spaces import (
    normalize_space_identifier,
    spaces_are_equivalent,
    validate_space_compatibility,
    validate_space_and_resolution
)
```

## Common Scenarios

### 1. Comparing Space Identifiers

❌ **Don't do this:**
```python
if atlas_space == input_space:  # Fails for aliases!
    return atlas_img
```

✅ **Do this:**
```python
from lacuna.core.spaces import spaces_are_equivalent

if spaces_are_equivalent(atlas_space, input_space):
    return atlas_img
```

**Why**: Handles space aliases (aAsym/bAsym/cAsym are equivalent)

---

### 2. Validating Analysis Inputs

❌ **Don't do this:**
```python
if lesion_space != "MNI152NLin2009cAsym":
    raise ValueError(f"Expected MNI152NLin2009cAsym, got {lesion_space}")
```

✅ **Do this:**
```python
from lacuna.core.spaces import validate_space_compatibility

validate_space_compatibility(
    actual_space=lesion_space,
    expected_space="MNI152NLin2009cAsym",
    context="FunctionalNetworkMapping",
    suggest_transform=True  # Adds transformation suggestion to error
)
```

**Why**: 
- Accepts equivalent aliases automatically
- Provides helpful error messages
- Suggests transformation if spaces incompatible

---

### 3. Validating Metadata Consistency

❌ **Don't do this:**
```python
# Just check if space exists
if "space" not in metadata:
    raise ValueError("Missing space")
# Resolution could be None - silent bug!
```

✅ **Do this:**
```python
from lacuna.core.spaces import validate_space_and_resolution

validate_space_and_resolution(
    space=metadata.get("space"),
    resolution=metadata.get("resolution"),
    strict=True  # Requires resolution when space is specified
)
```

**Why**:
- Ensures resolution provided when space specified
- Validates resolution is positive number
- Catches issues before transformation attempts

---

### 4. Normalizing Space for Lookups

❌ **Don't do this:**
```python
# Dictionary keys might not match aliases
if space in cache_dict:
    return cache_dict[space]
```

✅ **Do this:**
```python
from lacuna.core.spaces import normalize_space_identifier

canonical_space = normalize_space_identifier(space)
if canonical_space in cache_dict:
    return cache_dict[canonical_space]
```

**Why**: Ensures consistent lookup regardless of alias variant

---

## Space Aliases Reference

These space identifiers are **anatomically equivalent**:

| Variant | Normalized To |
|---------|---------------|
| `MNI152NLin2009aAsym` | `MNI152NLin2009cAsym` |
| `MNI152NLin2009bAsym` | `MNI152NLin2009cAsym` |
| `MNI152NLin2009cAsym` | `MNI152NLin2009cAsym` (canonical) |

**Note**: All other space identifiers remain unchanged during normalization.

---

## Error Message Examples

### Good: Using Helper Functions
```
ValueError: Space mismatch in FunctionalNetworkMapping: 
got 'native', expected 'MNI152NLin2009cAsym'
  → Consider transforming to 'MNI152NLin2009cAsym'
```

### Good: Resolution Validation
```
ValueError: RegionalDamage requires lesion data with 'resolution' metadata.
Resolution is required when space is specified.
Got space='MNI152NLin6Asym' but resolution=None
```

### Bad: Without Helper Functions
```
KeyError: 'MNI152NLin2009aAsym'  # No context!
AttributeError: 'NoneType' object has no attribute 'affine'  # Late failure!
```

---

## Migration Guide

### For Analysis Classes

**Before:**
```python
class MyAnalysis(BaseAnalysis):
    def _validate_inputs(self, lesion_data: LesionData) -> None:
        space = lesion_data.metadata.get("space")
        if space != "MNI152NLin2009cAsym":
            raise ValueError(f"Wrong space: {space}")
```

**After:**
```python
from lacuna.core.spaces import validate_space_compatibility

class MyAnalysis(BaseAnalysis):
    def _validate_inputs(self, lesion_data: LesionData) -> None:
        space = lesion_data.get_coordinate_space().name
        validate_space_compatibility(
            actual_space=space,
            expected_space="MNI152NLin2009cAsym",
            context=self.__class__.__name__,
            suggest_transform=True
        )
```

### For Atlas/Template Loading

**Before:**
```python
if atlas_space == lesion_space:  # Misses aliases
    use_atlas_directly()
else:
    transform_atlas()
```

**After:**
```python
from lacuna.core.spaces import spaces_are_equivalent

if spaces_are_equivalent(atlas_space, lesion_space):
    use_atlas_directly()  # Works for aAsym/cAsym too!
else:
    transform_atlas()
```

---

## Testing Examples

```python
from lacuna.core.spaces import spaces_are_equivalent

def test_space_handling():
    # These should all be True
    assert spaces_are_equivalent("MNI152NLin2009aAsym", "MNI152NLin2009cAsym")
    assert spaces_are_equivalent("MNI152NLin2009bAsym", "MNI152NLin2009cAsym")
    assert spaces_are_equivalent("MNI152NLin6Asym", "MNI152NLin6Asym")
    
    # These should be False
    assert not spaces_are_equivalent("native", "MNI152NLin6Asym")
    assert not spaces_are_equivalent("MNI152NLin6Asym", "MNI152NLin2009cAsym")
```

---

## Performance Tips

1. **Avoid unnecessary transformations**: Use `spaces_are_equivalent()` before calling transformation functions
2. **Normalize once**: If checking multiple times, normalize space identifier once and reuse
3. **Validate early**: Use validation functions in `_validate_inputs()` to fail fast

---

## See Also

- **Implementation Details**: `specs/002-neuroimaging-space-handling/IMPLEMENTATION_SUMMARY.md`
- **API Reference**: `src/lacuna/core/spaces.py`
- **Test Examples**: `tests/unit/test_space_normalization.py`
