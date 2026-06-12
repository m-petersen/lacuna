# System requirements & FAQ

## System requirements

| | Requirement |
|---|---|
| **Python** | 3.10 or newer |
| **Operating system** | Linux and macOS are supported and tested. On Windows, use [WSL2](https://learn.microsoft.com/windows/wsl/) — MRtrix3 and some dependencies are awkward to install natively. |
| **Structural network mapping** | [MRtrix3](https://www.mrtrix.org/) on your `PATH` |
| **Native-space masks** | Register to MNI first — see the [spatial normalization guide](../how-to/spatial-normalization.ipynb) (uses ANTsPy, installed separately) |
| **Disk space** | Depends on the normative connectome you fetch (see below) |

### Disk space for normative connectomes

| Connectome | Used by | Approx. size |
|---|---|---|
| GSP1000 | Functional network mapping | ~200 GB |
| HCP1065 | Structural network mapping | ~1.5 GB |
| dTOR985 full | Structural network mapping | ~11 GB |
| dTOR985 10% | Structural network mapping | ~1 GB |
| dTOR985 25% | Structural network mapping | ~3 GB |

## Frequently asked questions

### My lesion masks aren't in MNI space. What do I do?

Lacuna requires masks in `MNI152NLin6Asym` or `MNI152NLin2009cAsym`. If yours are
in native space, register them first — see the
[spatial normalization guide](../how-to/spatial-normalization.ipynb). Lacuna
handles alignment between supported MNI spaces automatically, but it will not
guess the space of a mask that carries no spatial information.

### Do I need to download a connectome to try Lacuna?

No. Run **focal damage** first — it uses the bundled atlases and needs no
download. Use `lacuna tutorial my_dataset` to generate a small synthetic dataset
to experiment with.

### Does fetching a connectome require an account or API key?

It depends on the source:

- **GSP1000** (functional) — requires a free Harvard Dataverse API key.
- **HCP1065** (structural) — no key required.
- **dTOR985 full** (structural) — requires a Figshare API key.
- **dTOR985 10%** (structural) — no key required.
- **dTOR985 25%** (structural) — no key required.

Provide keys via the `--api-key` flag, an environment variable
(`DATAVERSE_API_KEY` / `FIGSHARE_API_KEY`), or `~/.config/lacuna/config.yaml`.
See the [fetch CLI reference](../reference/cli/fetch.md) for details.

### Should I use FNM or AFNM?

Both compute the functional network of a lesion. Use the standard
[functional network mapping](lesion-network-mapping.md) when you need a
voxel-resolution map; use the accelerated version when you are processing many
subjects and a parcel-resolution result is sufficient.

### How do I process a whole cohort?

Point `lacuna run` at a BIDS dataset root rather than a single mask, or use
`batch_process` from the Python API. See the
[getting started tutorial](../tutorials/01-getting-started.ipynb).

### How do I cite Lacuna?

See [`CITATION.cff`](https://github.com/m-petersen/lacuna/blob/main/CITATION.cff)
in the repository.
