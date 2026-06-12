# Info command

The `lacuna info` command displays information about available resources.

## Synopsis

```
lacuna info <topic>
```

## Description

Displays detailed information about available resources such as brain parcellation atlases, registered connectomes, and the licenses of the datasets Lacuna downloads.

## Arguments

| Argument | Choices | Description |
|----------|---------|-------------|
| `topic` | `atlases`, `connectomes`, `licenses` | Topic to display information about |

The `licenses` topic prints the project `NOTICE`, which lists every bundled and downloaded dataset together with its license and citation. Some datasets — notably the FSL-derived `MNI152NLin6Asym` template data — are restricted to non-commercial use; Lacuna's MIT code license does not grant rights to the data.

## Examples

```bash
# List available brain parcellation atlases
lacuna info atlases

# List registered connectomes
lacuna info connectomes

# Show the licenses of the bundled and downloaded datasets
lacuna info licenses
```
