#!/usr/bin/env python3
"""
Replace old MNI152_2mm/MNI152_1mm format with new format throughout codebase.

Old: {"space": "MNI152NLin6Asym", "resolution": 2}
New: {"space": "MNI152NLin6Asym", "resolution": 2}
"""

import re
from pathlib import Path


def replace_in_file(filepath: Path):
    """Replace old space format in a single file."""
    try:
        content = filepath.read_text()
        original = content

        # Pattern 1: {"space": "MNI152NLin6Asym", "resolution": 2}
        content = re.sub(
            r'\{"space":\s*"MNI152_2mm"\}', '{"space": "MNI152NLin6Asym", "resolution": 2}', content
        )

        # Pattern 2: metadata={"space": "MNI152NLin6Asym", "resolution": 2, ...}
        content = re.sub(
            r'(\{"space":\s*)"MNI152_2mm"', r'\1"MNI152NLin6Asym", "resolution": 2', content
        )

        # Pattern 3: "space": "MNI152_2mm" (with other fields before)
        content = re.sub(
            r'(,\s*)"space":\s*"MNI152_2mm"(\s*[,}])',
            r'\1"space": "MNI152NLin6Asym", "resolution": 2\2',
            content,
        )

        # Pattern 4: == "MNI152NLin6Asym" in assertions
        content = re.sub(r'==\s*"MNI152_2mm"', '== "MNI152NLin6Asym"', content)

        # Pattern 5: space='MNI152NLin6Asym' in docstrings/comments
        content = re.sub(r"'MNI152NLin6Asym'", "'MNI152NLin6Asym'", content)

        # Pattern 6: MNI152_1mm similarly
        content = re.sub(
            r'\{"space":\s*"MNI152_1mm"\}', '{"space": "MNI152NLin6Asym", "resolution": 1}', content
        )
        content = re.sub(
            r'(\{"space":\s*)"MNI152_1mm"', r'\1"MNI152NLin6Asym", "resolution": 1', content
        )
        content = re.sub(
            r'(,\s*)"space":\s*"MNI152_1mm"(\s*[,}])',
            r'\1"space": "MNI152NLin6Asym", "resolution": 1\2',
            content,
        )

        if content != original:
            filepath.write_text(content)
            print(f"Updated: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Process all Python files in the project."""
    root = Path("/home/marvin/mount/hdd8tb/CSI_MVCI/lacuna")

    # Find all Python files
    py_files = list(root.rglob("*.py"))
    md_files = list(root.rglob("*.md"))

    updated = 0
    for filepath in py_files + md_files:
        # Skip generated/vendored files
        if ".git" in str(filepath) or "htmlcov" in str(filepath):
            continue
        if replace_in_file(filepath):
            updated += 1

    print(f"\nTotal files updated: {updated}")


if __name__ == "__main__":
    main()
