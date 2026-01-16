# Configure TemplateFlow

This guide shows how to configure TemplateFlow for brain template access in Lacuna.

## Goal

Set up TemplateFlow to provide brain templates and spatial transforms for coordinate space conversions.

## What is TemplateFlow?

TemplateFlow is a repository of brain templates and transforms. Lacuna uses it for:

- Accessing standard MNI templates
- Transforming masks between coordinate spaces
- Providing reference images for spatial operations

## Installation

TemplateFlow is included with Lacuna. Verify it's available:

```python
import templateflow
print(f"TemplateFlow version: {templateflow.__version__}")
```

## Configuring the cache

### Default location

Templates are cached in:

| OS | Path |
|----|------|
| Linux | `~/.cache/templateflow/` |
| macOS | `~/Library/Caches/templateflow/` |

### Custom cache directory

Set before importing templateflow:

```bash
export TEMPLATEFLOW_HOME=/path/to/custom/cache
```

Or in Python:

```python
import os
os.environ["TEMPLATEFLOW_HOME"] = "/path/to/custom/cache"

# Now import templateflow
import templateflow
```

## Pre-downloading templates

For offline use, download templates in advance:

### Common templates for Lacuna

```python
from templateflow import api as tflow

# MNI152NLin6Asym (used by most functional connectomes)
tflow.get("MNI152NLin6Asym", resolution=2, suffix="T1w")
tflow.get("MNI152NLin6Asym", resolution=2, suffix="mask")

# MNI152NLin2009cAsym (used by structural connectomes)
tflow.get("MNI152NLin2009cAsym", resolution=2, suffix="T1w")
tflow.get("MNI152NLin2009cAsym", resolution=2, suffix="mask")
```

### Download all required templates

```python
from lacuna.assets import download_required_templates

download_required_templates()
print("All required templates downloaded!")
```

## Available templates

Common templates used with Lacuna:

| Template | Resolution | Use case |
|----------|------------|----------|
| MNI152NLin6Asym | 1mm, 2mm | Functional LNM |
| MNI152NLin2009cAsym | 1mm, 2mm | Structural LNM |
| MNI152NLin2009bAsym | 1mm, 2mm | Alternative |

## Verifying templates

```python
from templateflow import api as tflow

# Check if template is available
template = tflow.get(
    "MNI152NLin6Asym",
    resolution=2,
    suffix="T1w",
    raise_empty=False
)

if template:
    print(f"Template found: {template}")
else:
    print("Template not cached, will download on first use")
```

## Using with Lacuna

Lacuna automatically uses TemplateFlow for spatial operations:

```python
from lacuna.spatial.transform import transform_mask_data
from lacuna.core.spaces import CoordinateSpace, REFERENCE_AFFINES

# Transform between spaces (uses TemplateFlow)
target = CoordinateSpace(
    "MNI152NLin2009cAsym",
    2,
    REFERENCE_AFFINES[("MNI152NLin2009cAsym", 2)]
)
transformed = transform_mask_data(subject, target)
```

## Offline environments

For systems without internet access:

1. Download templates on a connected machine
2. Copy the cache directory to the target system
3. Set `TEMPLATEFLOW_HOME` to point to the copied directory

```bash
# On connected machine
python -c "from lacuna.assets import download_required_templates; download_required_templates()"
tar -czf templateflow_cache.tar.gz ~/.cache/templateflow/

# On offline machine
tar -xzf templateflow_cache.tar.gz -C ~/
export TEMPLATEFLOW_HOME=~/.cache/templateflow
```

## Troubleshooting

??? question "Template download fails"
    
    Check your internet connection and try again. If behind a proxy:
    
    ```bash
    export HTTP_PROXY=http://proxy:port
    export HTTPS_PROXY=http://proxy:port
    ```

??? question "Template not found"
    
    Verify the template exists in TemplateFlow:
    
    ```python
    from templateflow import api as tflow
    print(tflow.templates())  # List all available templates
    ```

??? question "Disk space issues"
    
    Templates can use several GB. Set a custom cache on a larger drive:
    
    ```bash
    export TEMPLATEFLOW_HOME=/external/drive/templateflow
    ```

## Resources

- [TemplateFlow Documentation](https://www.templateflow.org/)
- [TemplateFlow GitHub](https://github.com/templateflow/templateflow)
- [Available Templates](https://www.templateflow.org/browse/)
