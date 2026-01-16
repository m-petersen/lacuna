# Installation

This guide covers all installation options for Lacuna.

## Quick install

```bash
pip install lacuna
```

## Requirements

- **Python**: 3.10 or higher
- **Operating System**: Linux, macOS, Windows
- **Memory**: 8GB RAM minimum (16GB recommended for large connectomes)

## Installation options

### From PyPI

The simplest way to install Lacuna:

```bash
pip install lacuna
```

### From source

For development or to get the latest features:

```bash
git clone https://github.com/lacuna/lacuna.git
cd lacuna
pip install -e .
```

### With optional dependencies

Install with visualization support:

```bash
pip install "lacuna[viz]"
```

Install with all development tools:

```bash
pip install "lacuna[dev]"
```

## Virtual environment (recommended)

We recommend using a virtual environment to avoid dependency conflicts:

=== "venv"

    ```bash
    # Create virtual environment
    python -m venv lacuna-env
    
    # Activate it
    source lacuna-env/bin/activate  # Linux/macOS
    # or
    lacuna-env\Scripts\activate     # Windows
    
    # Install Lacuna
    pip install lacuna
    ```

=== "conda"

    ```bash
    # Create environment
    conda create -n lacuna python=3.10
    
    # Activate it
    conda activate lacuna
    
    # Install Lacuna
    pip install lacuna
    ```

## Verify installation

Check that Lacuna is installed correctly:

```bash
python -c "import lacuna; print(f'Lacuna {lacuna.__version__}')"
```

Test the import of core modules:

```python
from lacuna import SubjectData
from lacuna.analysis import FunctionalNetworkMapping, StructuralNetworkMapping
print("All core modules imported successfully!")
```

## External dependencies

Some analyses require external tools:

| Analysis | Requirement | Installation Guide |
|----------|-------------|-------------------|
| Structural LNM | MRtrix3 | [Setup MRtrix3](setup-mrtrix3.md) |
| Functional LNM | Normative connectome | [Fetch Connectomes](fetch-connectomes.md) |
| Spatial transforms | TemplateFlow templates | [Configure TemplateFlow](templateflow.md) |

## Upgrading

To upgrade to the latest version:

```bash
pip install --upgrade lacuna
```

## Troubleshooting

### Common issues

??? question "pip install fails with version conflict"
    
    Try creating a fresh virtual environment:
    
    ```bash
    python -m venv fresh-env
    source fresh-env/bin/activate
    pip install lacuna
    ```

??? question "Import error: missing module"
    
    Ensure you're using the correct Python environment:
    
    ```bash
    which python  # Linux/macOS
    where python  # Windows
    ```

??? question "Permission denied during install"
    
    Use `--user` flag or install in a virtual environment:
    
    ```bash
    pip install --user lacuna
    ```

### Getting help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/lacuna/lacuna/issues)
2. Search existing discussions
3. Open a new issue with your error message and Python version
