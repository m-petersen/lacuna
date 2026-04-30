# Asset envelopes

Every lacuna-managed asset directory — an NT atlas, an SNTF cache, an
ACE cache — carries a `lacuna_asset.json` sidecar that records:

- the asset's content fingerprint (`identity`)
- the upstream assets it was built from (`requires`), each pinned by
  its own fingerprint
- provenance (lacuna version, command, parameters)
- the per-asset-type payload (`data`)

The envelope is the contract a loader needs to verify *"do my inputs
still match what this cache was built from?"* without reaching into
asset-specific code.

## What this catches

When you run `lacuna run sntf`, the analysis reads the cache's envelope
and re-fingerprints every entry in the `requires` list against the
runtime `--connectome-path` and the runtime `--ntatlas-dir`. If anything
changed — a different tractogram passed by mistake, an NT atlas that was
re-fetched with different content, a renamed file — the run aborts with
an `AssetMismatchError` instead of silently producing wrong scores.

Two failure modes specifically:

```text
Required asset 'tractogram' (structural_connectome) at /tmp/hcp1065.tck
does not match the one this cache was built from.
  cache identity:   {'kind': 'sha256_first_mib+size', 'fields': {...}}
  runtime identity: {'kind': 'sha256_first_mib+size', 'fields': {...}}
Re-run the prepare step against the runtime asset, or point the runtime
path at the original asset.
```

```text
Required asset 'ntatlas' (ntatlas) at /data/ntatlas
does not match the one this cache was built from.
```

The first appears when `lacuna prepare sntf` was run against tractogram
A and `lacuna run sntf --connectome-path tractogram_B.tck` is then
pointed at the same `--precomputed-weights-dir`. The second appears
after re-running `lacuna fetch ntatlas` produces a different atlas than
the one a cache was originally built against.

## Schema

The envelope is a JSON file with five top-level keys:

```json
{
  "lacuna_schema_version": 1,
  "asset_type": "sntf_cache",
  "identity": {
    "kind": "sha256_concat",
    "fields": {"sha256": "…"}
  },
  "requires": [
    {
      "role": "tractogram",
      "asset_type": "structural_connectome",
      "identity": {
        "kind": "sha256_first_mib+size",
        "fields": {"sha256_first_mib": "…", "size_bytes": 0}
      },
      "path_hint": "/data/tractogram.tck"
    }
  ],
  "provenance": {
    "command": "lacuna prepare sntf",
    "n_streamlines": 985000
  },
  "data": {
    "n_streamlines": 985000,
    "targets": ["D1", "5HT1a"]
  }
}
```

`identity` is a content-based fingerprint of the asset's payload. The
hashing strategy depends on the asset type — large tractograms use
sha256 of the first 1 MiB plus file size (cheap to compute, specific
enough to detect any realistic mismatch); NT atlases hash the sorted
concatenation of `maps/<name>.nii.gz` filenames and contents.

`requires` lists every other asset this one was built from. Each entry
records the upstream asset's `identity` at build time. The
`path_hint` is informational — it is surfaced in error messages but
never used as a fallback path; runtime callers must supply the path
explicitly.

## Provenance sidecars

Subject-output provenance is a separate concern. By default,
`lacuna run` does not write per-subject `*_desc-provenance.json`
sidecars next to the result files — they add noise to a typical batch
run. Pass `--export-provenance` to opt in for reproducibility-critical
runs.
