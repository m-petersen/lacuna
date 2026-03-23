"""Generate API reference pages for mkdocstrings."""
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
src = Path("src")

# Generate API reference pages for all Python modules
for path in sorted(src.rglob("*.py")):
    # Skip private modules (except __init__.py)
    if path.name.startswith("_") and path.name != "__init__.py":
        continue

    # Skip test files if any exist in src
    if "test" in path.parts:
        continue

    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference/api", doc_path)

    parts = tuple(module_path.parts)

    # Handle __init__.py files
    if parts[-1] == "__init__":
        parts = parts[:-1]
        if not parts:
            continue  # Skip root __init__.py
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        fd.write(f"::: {identifier}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

# Generate the navigation summary file
with mkdocs_gen_files.open("reference/api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
