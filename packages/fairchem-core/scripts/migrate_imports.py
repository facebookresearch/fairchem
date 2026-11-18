"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Standalone utility: rewrite old fairchem module paths to their
current locations across a file or source tree, using the rename
table in fairchem.core.common.checkpoint_migration. Run from a
repository checkout; not shipped in the fairchem-core wheel.
"""

from __future__ import annotations

import argparse
import os
import pathlib

from fairchem.core.common.checkpoint_migration import mapping

extensions = [".yaml", ".py"]


def replace_strings_in_file(file_path, replacements, dry_run):
    """
    Replaces input strings with output strings in a given file.

    Args:
        file_path (str): Path to the file to process.
        replacements (dict): Dictionary of input strings to output strings.
        dry_run (bool): Whether to perform a dry run (print changes without making them).
    """
    try:
        with open(file_path) as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    changes_made = False
    for i, line in enumerate(lines):
        for key, value in replacements.items():
            if key in line:
                changes_made = True
                if dry_run:
                    print(
                        f"Dry run: would replace '{key}' with '{value}' in {file_path} at line {i+1}:"
                    )
                    # print(f"  Original line: {line.strip()}")
                    # print(f"  New line: {line.strip().replace(key, value)}")
                else:
                    lines[i] = line.replace(key, value)

    if changes_made and not dry_run:
        with open(file_path, "w") as file:
            file.writelines(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Replace input strings with output strings in files"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Only executes if true otherwise perform a dryrun",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="file or Directory to recursively search for files",
        required=True,
    )
    args = parser.parse_args()

    if os.path.isfile(args.input):
        replace_strings_in_file(args.input, mapping, not args.execute)
    elif os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for file in files:
                file_path = os.path.join(root, file)
                if pathlib.Path(file).suffix in extensions:
                    replace_strings_in_file(file_path, mapping, not args.execute)
    else:
        raise ValueError("unknown input type")


if __name__ == "__main__":
    main()
