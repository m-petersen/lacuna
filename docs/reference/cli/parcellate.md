# Parcellate command

The `lacuna parcellate` command reduces a whole-brain connectome to a parcel-level N×N connectivity matrix. The resulting TSV + JSON sidecar is the `--matrix-path` input for [`lacuna run afnm`](run.md#lacuna-run-afnm-accelerated-functional-network-mapping).

## Synopsis

```
lacuna parcellate --connectome-path <PATH> --modality <functional|structural> \
                  (--parcel-atlases <ATLAS> [...] | --custom-parcellation ...) \
                  --output <DIR> [options]
```

## Description

Given a whole-brain connectome (voxel-level functional HDF5 or a structural `.tck` tractogram) and a parcellation, `lacuna parcellate` produces a BIDS-style ConnectivityMatrix — a TSV with parcel labels as row/column index plus a JSON sidecar describing provenance.

The `--modality` flag is **required** and determines which connectome type is expected; modality is never inferred from the file path or extension.

## Required arguments

| Argument | Description |
|----------|-------------|
| `--connectome-path PATH` | Voxelwise HDF5 file or directory (`--modality functional`), or tractogram `.tck` (`--modality structural`). |
| `--modality {functional,structural}` | Connectome modality. Required. |
| `--output DIR` | Output directory for the ConnectivityMatrix TSV + JSON sidecar. |

## Parcellation selection

At least one of these must be supplied. Each atlas produces its own output file.

| Option | Description |
|--------|-------------|
| `--parcel-atlases ATLAS [...]` | Bundled atlas name(s). Use `lacuna info atlases` to list. |
| `--custom-parcellation NAME NIFTI LABELS SPACE` | Custom parcellation: short name for output labelling, NIfTI path, labels file, coordinate space. Repeat for multiple. |

## Examples

```bash
# Functional: reduce a voxel-level connectome to a parcel-level FC matrix
lacuna parcellate \
    --connectome-path ~/.cache/lacuna/connectomes/gsp1000/ \
    --modality functional \
    --parcel-atlases schaefer2018parcels400networks17 \
    --output /data/parcellated/

# Structural: reduce a tractogram to a parcel-level connectivity matrix
lacuna parcellate \
    --connectome-path /data/dtor985.tck \
    --modality structural \
    --parcel-atlases schaefer2018parcels400networks17 \
    --output /data/parcellated/
```

## Output

A BIDS-style ConnectivityMatrix per atlas, e.g.

```
method-parcellate_atlas-schaefer2018parcels400networks17_desc-groupconnectivity_connmatrix.tsv
method-parcellate_atlas-schaefer2018parcels400networks17_desc-groupconnectivity_connmatrix.json
```

The TSV uses parcel labels as both row index and column headers; the JSON sidecar records the source connectome, modality, atlas, and shape.

## See Also

- [`lacuna run afnm`](run.md#lacuna-run-afnm-accelerated-functional-network-mapping) — consumes the functional matrix
- [`lacuna fetch`](fetch.md) — download connectomes before parcellating
- [`lacuna info atlases`](info.md) — list available atlases
