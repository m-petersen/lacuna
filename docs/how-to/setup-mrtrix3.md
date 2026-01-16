# Setup MRtrix3

This guide shows how to install and configure MRtrix3 for structural lesion network mapping.

## Goal

Install MRtrix3 tools required for tractography-based structural connectivity analysis.

## Why MRtrix3?

Lacuna uses MRtrix3 for:

- Processing tractography data (.tck files)
- Computing streamline-lesion intersections
- Generating disconnection maps

## Installation

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get install git g++ python3 libeigen3-dev zlib1g-dev libqt5opengl5-dev libqt5svg5-dev libgl1-mesa-dev libfftw3-dev libtiff5-dev libpng-dev

# Clone and build MRtrix3
git clone https://github.com/MRtrix3/mrtrix3.git
cd mrtrix3
./configure
./build
```

Add to your PATH:

```bash
export PATH=/path/to/mrtrix3/bin:$PATH
```

Add this line to your `~/.bashrc` for persistence.

### macOS

Using Homebrew:

```bash
brew install mrtrix3
```

Or build from source:

```bash
brew install pkg-config eigen qt@5 fftw libtiff libpng
git clone https://github.com/MRtrix3/mrtrix3.git
cd mrtrix3
./configure
./build
```

### Conda

```bash
conda install -c mrtrix3 mrtrix3
```

### Docker

If you're using Lacuna's Docker container, MRtrix3 is pre-installed.

## Verify installation

```bash
# Check version
mrinfo --version

# Test a command
tckinfo --version
```

Expected output:

```
== mrinfo 3.x.x ==
...
```

## Test with Lacuna

```python
from lacuna.analysis import StructuralNetworkMapping

# This should not raise an error
snm = StructuralNetworkMapping(
    connectome_name="test",
    parcellation_name="Schaefer2018_100Parcels7Networks"
)
print("MRtrix3 integration working!")
```

## Troubleshooting

??? question "Command not found: mrinfo"
    
    Ensure MRtrix3 is in your PATH:
    
    ```bash
    which mrinfo
    # Should print: /path/to/mrtrix3/bin/mrinfo
    ```

??? question "Build fails on Linux"
    
    Install all dependencies:
    
    ```bash
    sudo apt-get update
    sudo apt-get install git g++ python3 libeigen3-dev zlib1g-dev \
        libqt5opengl5-dev libqt5svg5-dev libgl1-mesa-dev \
        libfftw3-dev libtiff5-dev libpng-dev
    ```

??? question "Qt errors on macOS"
    
    If Qt-related errors occur:
    
    ```bash
    export QMAKE=/opt/homebrew/opt/qt@5/bin/qmake
    ./configure
    ./build
    ```

## Resources

- [MRtrix3 Documentation](https://mrtrix.readthedocs.io/)
- [MRtrix3 GitHub](https://github.com/MRtrix3/mrtrix3)
- [Community Forum](https://community.mrtrix.org/)
