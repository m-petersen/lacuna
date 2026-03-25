# Tutorial command

The `lacuna tutorial` command sets up a tutorial dataset for learning Lacuna.

## Synopsis

```
lacuna tutorial [output_dir] [options]
```

## Description

Copies the bundled tutorial dataset to a directory. The dataset includes:

- 3 synthetic subjects (`sub-01`, `sub-02`, `sub-03`)
- Binary lesion masks in MNI152NLin6Asym space
- BIDS-style directory structure ready for analysis

## Arguments

| Argument | Description |
|----------|-------------|
| `output_dir` | Output directory for tutorial data (default: `./lacuna_tutorial`) |

### Options

| Option | Description |
|--------|-------------|
| `--force`, `-f` | Overwrite existing directory if it exists |

## Examples

```bash
# Setup tutorial in default directory
lacuna tutorial

# Setup in custom directory
lacuna tutorial ./my_tutorial

# Overwrite existing tutorial data
lacuna tutorial ./my_tutorial --force
```

## See Also

- [Getting Started](../../tutorials/getting-started.ipynb) — Tutorial notebook
