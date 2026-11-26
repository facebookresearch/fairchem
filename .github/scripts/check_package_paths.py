"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ".github/workflows/*.yml",
    ".github/release-drafter-*.yml",
    ".devcontainer/*",
    "postBuild",
)
PACKAGE_PATH = re.compile(r"(?<![\w./-])packages/[\w.*/-]+")
TEMPLATED = set("(|{$\\")


def main() -> int:
    """
    Fail if a CI, release, or environment file references a missing package path.
    """
    missing = []
    for pattern in SOURCES:
        for source in sorted(ROOT.glob(pattern)):
            if not source.is_file():
                continue
            for number, line in enumerate(source.read_text().splitlines(), 1):
                for match in PACKAGE_PATH.finditer(line):
                    # Skip regexes and templated paths.
                    if line[match.end() : match.end() + 1] in TEMPLATED:
                        continue
                    path = match.group().split("[")[0].removesuffix("/**").rstrip("/")
                    if not any(ROOT.glob(path)):
                        relative = source.relative_to(ROOT)
                        missing.append(f"{relative}:{number}: {path}")
    for entry in missing:
        print(entry)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
