# Info Command

The `lacuna info` command displays information about available resources.

## Synopsis

```
lacuna info <topic>
```

## Description

Displays detailed information about available resources such as brain parcellation atlases and registered connectomes.

## Arguments

| Argument | Choices | Description |
|----------|---------|-------------|
| `topic` | `atlases`, `connectomes` | Topic to display information about |

## Examples

```bash
# List available brain parcellation atlases
lacuna info atlases

# List registered connectomes
lacuna info connectomes
```

## See Also

- [Custom Parcellation](../../how-to/custom-parcellation.md) — Using custom brain parcellations
- [Use Your Own Connectome](../../how-to/use-own-connectome.md) — Using custom connectomes
