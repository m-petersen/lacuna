#!/usr/bin/env python3
"""Generate codemeta.json from CITATION.cff (single source of truth).

CITATION.cff holds the citation metadata that changes between releases
(version, release date, authors, DOI, keywords). This script maps those fields
into a CodeMeta 2.0 ``codemeta.json`` and fills in the codemeta-specific static
fields (programming language, runtime platform, issue tracker, …) that
CITATION.cff has no place for. Run it after editing CITATION.cff:

    python tools/gen_codemeta.py            # write codemeta.json
    python tools/gen_codemeta.py --check    # verify it is in sync (CI)

Keeping CITATION.cff as the source avoids the two files drifting apart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CITATION = ROOT / "CITATION.cff"
CODEMETA = ROOT / "codemeta.json"

# Static, codemeta-specific fields that CITATION.cff cannot express.
STATIC = {
    "programmingLanguage": "Python",
    "runtimePlatform": "Python 3",
    "operatingSystem": "Linux, macOS",
    "developmentStatus": "active",
}


def _authors(cff_authors: list[dict]) -> list[dict]:
    people = []
    for a in cff_authors:
        person = {"@type": "Person"}
        if "given-names" in a:
            person["givenName"] = a["given-names"]
        if "family-names" in a:
            person["familyName"] = a["family-names"]
        if a.get("orcid"):
            person["@id"] = a["orcid"]
        if a.get("affiliation"):
            person["affiliation"] = {"@type": "Organization", "name": a["affiliation"]}
        people.append(person)
    return people


def _doi(cff: dict) -> str | None:
    for ident in cff.get("identifiers", []):
        if ident.get("type") == "doi":
            return f"https://doi.org/{ident['value']}"
    return None


def build_codemeta(cff: dict) -> dict:
    # The CITATION title is descriptive ("Lacuna: …"); codemeta `name` wants the
    # short software name — take the part before the first colon.
    name = cff["title"].split(":", 1)[0].strip()
    abstract = " ".join(cff["abstract"].split())
    repo = cff["repository-code"]
    date = str(cff["date-released"])
    version = str(cff["version"])

    codemeta = {
        "@context": "https://doi.org/10.5063/schema/codemeta-2.0",
        "@type": "SoftwareSourceCode",
        "identifier": _doi(cff) or name.lower(),
        "name": name,
        "description": abstract,
        "codeRepository": repo,
        "url": cff.get("url", repo),
        "issueTracker": f"{repo}/issues",
        "license": f"https://spdx.org/licenses/{cff['license']}.html",
        **STATIC,
        "version": version,
        "softwareVersion": version,
        "datePublished": date,
        "dateModified": date,
        "keywords": list(cff.get("keywords", [])),
        "author": _authors(cff.get("authors", [])),
    }
    return codemeta


def render(cff: dict) -> str:
    return json.dumps(build_codemeta(cff), indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if codemeta.json differs from the generated output.",
    )
    args = parser.parse_args()

    cff = yaml.safe_load(CITATION.read_text())
    generated = render(cff)

    if args.check:
        current = CODEMETA.read_text() if CODEMETA.exists() else ""
        if current != generated:
            print(
                "codemeta.json is out of sync with CITATION.cff.\n"
                "Run `make codemeta` (or `python tools/gen_codemeta.py`) and commit.",
                file=sys.stderr,
            )
            return 1
        print("codemeta.json is in sync with CITATION.cff.")
        return 0

    CODEMETA.write_text(generated)
    print(f"Wrote {CODEMETA.relative_to(ROOT)} from {CITATION.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
