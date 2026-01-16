# The Registry Pattern

Understanding how Lacuna uses registries for extensibility and plugin support.

## Overview

Lacuna uses the **registry pattern** to manage atlases, connectomes, and
analysis methods. This design enables extensibility without modifying core
code. You can register custom components and use them seamlessly with the
standard API.

## What Is a Registry?

A registry is a central catalog that maps names to implementations:

```python
# Conceptual structure
registry = {
    "HCP_S1200": <FunctionalConnectome>,
    "AAL": <AtlasLoader>,
    "flnm": <FunctionalLNMAnalysis>,
}
```

When you request a component by name, the registry returns the appropriate
implementation:

```python
# User code
result = analyze(subject, analysis="flnm", connectome="HCP_S1200")

# Under the hood
connectome = connectome_registry.get("HCP_S1200")
analysis = analysis_registry.get("flnm")
```

## Benefits of Registries

### 1. Decoupled Code

Components don't need to know about each other:

```python
# Analysis code doesn't import specific connectomes
def run_flnm(mask, connectome_name):
    connectome = connectome_registry.get(connectome_name)
    return compute_correlation(mask, connectome)
```

### 2. Easy Extension

Add new components without modifying existing code:

```python
# Your custom connectome
class MyConnectome:
    def load(self):
        return my_custom_data

# Register it
connectome_registry.register("my_connectome", MyConnectome)

# Use it immediately
result = analyze(subject, connectome="my_connectome")
```

### 3. Discoverability

Users can query what's available:

```python
# List all registered connectomes
print(connectome_registry.list())
# ['HCP_S1200', 'HCP_Tractogram', 'my_connectome']
```

### 4. Configuration over Code

Swap implementations via configuration:

```yaml
# config.yaml
analysis:
  connectome: my_custom_connectome
```

```python
# Code stays the same
result = analyze(subject, connectome=config.connectome)
```

## Registries in Lacuna

### Connectome Registry

Manages normative connectomes for network mapping:

```python
from lacuna.connectome import registry as connectome_registry

# List available connectomes
print(connectome_registry.list())

# Get specific connectome
hcp = connectome_registry.get("HCP_S1200")
```

Built-in connectomes:

| Name | Type | Description |
|------|------|-------------|
| `HCP_S1200` | Functional | HCP 1000-subject functional connectome |
| `HCP_Tractogram` | Structural | HCP tractography-based connectome |

### Atlas Registry

Manages brain parcellation atlases:

```python
from lacuna.atlas import registry as atlas_registry

# List available atlases
print(atlas_registry.list())

# Get specific atlas
aal = atlas_registry.get("AAL")
```

Built-in atlases:

| Name | Regions | Description |
|------|---------|-------------|
| `AAL` | 116 | Automated Anatomical Labeling |
| `Schaefer100` | 100 | Schaefer 100-parcel atlas |
| `Schaefer200` | 200 | Schaefer 200-parcel atlas |

### Analysis Registry

Manages analysis methods:

```python
from lacuna.analyze import registry as analysis_registry

# List available analyses
print(analysis_registry.list())
```

Built-in analyses:

| Name | Description |
|------|-------------|
| `flnm` | Functional lesion network mapping |
| `slnm` | Structural lesion network mapping |
| `regional` | Regional damage quantification |

## Registering Custom Components

### Custom Atlas

```python
from lacuna.atlas import Atlas, registry

class MyCustomAtlas(Atlas):
    """Custom atlas with 50 regions."""
    
    name = "my_atlas"
    n_regions = 50
    
    def load(self):
        """Load atlas data from your source."""
        import nibabel as nib
        return nib.load("/path/to/my_atlas.nii.gz")
    
    def get_labels(self):
        """Return region labels."""
        return [f"Region_{i}" for i in range(1, 51)]

# Register the atlas
registry.register(MyCustomAtlas)

# Now use it
result = analyze(subject, analysis="regional", atlas="my_atlas")
```

### Custom Connectome

```python
from lacuna.connectome import Connectome, registry

class MyConnectome(Connectome):
    """Custom local connectome."""
    
    name = "local_connectome"
    modality = "functional"
    
    def load(self):
        """Load connectome data."""
        import h5py
        with h5py.File("/path/to/connectome.h5", "r") as f:
            return f["connectivity"][:]

# Register it
registry.register(MyConnectome)
```

## How Registries Work Internally

### Registration

When you call `registry.register()`:

1. The component is validated (correct interface)
2. Its name is extracted (from `name` attribute or argument)
3. It's stored in an internal dictionary

```python
class Registry:
    def __init__(self):
        self._components = {}
    
    def register(self, component, name=None):
        name = name or component.name
        self._validate(component)
        self._components[name] = component
```

### Retrieval

When you call `registry.get(name)`:

1. The name is looked up in the dictionary
2. If found, the component is instantiated (if needed)
3. The instance is returned

```python
def get(self, name):
    if name not in self._components:
        raise KeyError(f"Unknown component: {name}")
    component = self._components[name]
    return component() if callable(component) else component
```

### Lazy Loading

Registries support lazy loading for efficiency:

```python
# Component isn't loaded until first use
registry.register("heavy_atlas", HeavyAtlas)

# Only loaded when requested
atlas = registry.get("heavy_atlas")  # Now it loads
```

## Plugin System

Registries enable a plugin architecture:

```python
# In your plugin package: lacuna_myatlas/__init__.py
from lacuna.atlas import registry
from .atlas import MyCustomAtlas

def register_plugin():
    registry.register(MyCustomAtlas)

# Entry point in setup.py or pyproject.toml
[project.entry-points."lacuna.plugins"]
myatlas = "lacuna_myatlas:register_plugin"
```

Users install your plugin and it's automatically available:

```bash
pip install lacuna-myatlas
```

```python
# Your atlas is now available
result = analyze(subject, atlas="my_atlas")
```

## Best Practices

### 1. Use Meaningful Names

```python
# Good: descriptive names
registry.register("schaefer_7networks_100", atlas)

# Bad: cryptic names
registry.register("s7n100", atlas)
```

### 2. Document Your Components

```python
class MyAtlas(Atlas):
    """Schaefer 100-parcel atlas with 7-network parcellation.
    
    Reference: Schaefer et al., 2018, Cerebral Cortex.
    """
```

### 3. Validate Inputs

```python
class MyConnectome(Connectome):
    def load(self):
        data = load_my_data()
        if data.shape != expected_shape:
            raise ValueError(f"Invalid shape: {data.shape}")
        return data
```

### 4. Handle Missing Dependencies

```python
class OptionalAtlas(Atlas):
    def load(self):
        try:
            import optional_package
        except ImportError:
            raise ImportError(
                "This atlas requires 'optional_package'. "
                "Install with: pip install optional_package"
            )
        return optional_package.load_atlas()
```

## See Also

- [The analyze() Function](analyze.md) — How registries integrate with analysis
- API Reference: `lacuna.registry` — Registry implementation details
- How-to: [Batch Processing](../how-to/batch-processing.md) — Using registries in workflows
