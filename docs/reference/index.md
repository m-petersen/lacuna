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

The command-line interface is built on a Python API (`SubjectData`, `Pipeline`,
`analyze`, `batch_process`, the analysis classes, and the asset registries) that
Lacuna uses under the hood. The tutorials and how-to guides drive Lacuna through
the CLI, not this API. If you want to call Lacuna programmatically, see the
[**API reference**](api/), generated from the source docstrings, or run
`help(...)` on any object for the same information at the interpreter.
