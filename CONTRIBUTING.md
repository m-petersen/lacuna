# Contributing to Lacuna

Thanks for your interest in improving Lacuna! Bug reports, feature requests,
documentation fixes, and pull requests are all welcome.

Lacuna is in active alpha development, so expect APIs to move. If you are
planning a larger change, please open an issue first so we can discuss the
approach before you invest time in it.

## Reporting bugs and requesting features

Use the [issue tracker](https://github.com/m-petersen/lacuna/issues). For bugs,
please include:

- what you ran (the command or a minimal code snippet),
- what you expected and what happened instead (including the full traceback),
- your OS, Python version, and Lacuna version (`lacuna --version`).

## Development setup

```bash
git clone https://github.com/m-petersen/lacuna
cd lacuna
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"     # editable install with dev + docs + viz extras
```

Structural network mapping additionally requires [MRtrix3](https://www.mrtrix.org/)
on your `PATH`.

## Workflow

The [`Makefile`](Makefile) wraps the common tasks (`make help` lists them all):

```bash
make test-fast      # unit + contract tests (~1 min)
make test           # full suite
make lint           # ruff checks
make format         # apply formatting
make typecheck      # mypy
make ci-native      # run the full local CI gate before committing
```

Please run `make ci-native` before opening a pull request, and add or update
tests for any behavior you change. Lacuna uses a three-tier test layout —
`tests/unit/`, `tests/contract/`, and `tests/integration/`.

## Pull requests

1. Branch off `main`.
2. Keep changes focused; one logical change per PR.
3. Update [`CHANGELOG.md`](CHANGELOG.md) under the `Unreleased` heading.
4. Make sure CI passes.

## Releases

Releases are versioned from git tags via `setuptools-scm`. To cut a release,
move the `Unreleased` changelog entries under a new version heading, then:

```bash
make tag VERSION=0.1.0     # creates and pushes the v0.1.0 tag
```

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
