# Prepare command

The `lacuna prepare` command precomputes the non-subject-specific data product a
given analysis consumes. Targets are named after the analysis they prepare for
(mirroring `lacuna run`). Currently the only target is `afnm`, which reduces a
whole-brain functional connectome to a parcel-level N×N connectivity matrix — the
`--matrix-path` input for
[`lacuna run afnm`](run.md#lacuna-run-afnm-accelerated-functional-network-mapping).

## Synopsis

```
lacuna prepare afnm --connectome-path <PATH> \
                    (--parcel-atlases <ATLAS> [...] | --custom-parcellation ...) \
                    --output <DIR> [options]
```

## `lacuna prepare afnm`

Given a voxelwise functional connectome (HDF5, the same format as
[`lacuna run fnm`](run.md)) and a parcellation, `lacuna prepare afnm` produces a
BIDS-style ConnectivityMatrix — a TSV with parcel labels as row/column index plus
a JSON sidecar describing provenance. Build it once per parcellation and reuse it
for all subjects in `lacuna run afnm`.

### Required arguments

| Argument | Description |
|----------|-------------|
| `--connectome-path PATH` | Voxelwise functional connectome: HDF5 file or directory. |
| `--output DIR` | Output directory for the ConnectivityMatrix TSV + JSON sidecar. |

### Parcellation selection

At least one of these must be supplied. Each atlas produces its own output file.

| Option | Description |
|--------|-------------|
| `--parcel-atlases ATLAS [...]` | Bundled atlas name(s). Use `lacuna info atlases` to list. |
| `--custom-parcellation NAME NIFTI LABELS SPACE` | Custom parcellation: short name for output labelling, NIfTI path, labels file, coordinate space. Repeat for multiple. |

### Example

```bash
lacuna prepare afnm \
    --connectome-path ~/.cache/lacuna/connectomes/gsp1000/ \
    --parcel-atlases schaefer2018parcels400networks17 \
    --output /data/parcellated/
```

### Output

A BIDS-style ConnectivityMatrix per atlas, e.g.

```
method-parcellate_atlas-schaefer2018parcels400networks17_desc-groupconnectivity_connmatrix.tsv
method-parcellate_atlas-schaefer2018parcels400networks17_desc-groupconnectivity_connmatrix.json
```

The TSV uses parcel labels as both row index and column headers; the JSON sidecar
records the source connectome, atlas, and shape.

## See also

- [`lacuna run afnm`](run.md#lacuna-run-afnm-accelerated-functional-network-mapping) — consumes the matrix via `--matrix-path`
- [`lacuna fetch`](fetch.md) — download connectomes before preparing
- [`lacuna info atlases`](info.md) — list available atlases
