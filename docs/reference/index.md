# Reference

Technical reference documentation for Lacuna.

## Command-line interface

Lacuna is driven through its command-line interface. See the
[**CLI reference**](cli/index.md) for the full list of commands, options, and
examples:

- `lacuna tutorial` — set up the bundled tutorial dataset
- `lacuna bidsify` — convert a directory of masks into a BIDS layout
- `lacuna fetch` — download normative connectomes
- `lacuna prepare` — precompute analysis inputs (e.g. the AFNM matrix)
- `lacuna run` — run analyses (`fd`, `fnm`, `snm`, `afnm`)
- `lacuna check` — validate inputs and check output completeness
- `lacuna collect` — aggregate results across subjects
- `lacuna info` — list available atlases and connectomes

## Python API

Lacuna also exposes a Python API (`SubjectData`, `analyze`, `batch_process`, the
analysis classes, and the asset registries) used in the tutorials. Its usage is
shown throughout the [tutorials](../tutorials/index.md) and
[how-to guides](../how-to/index.md); run `help(...)` on any object for its
docstring.
