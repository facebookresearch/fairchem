"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

DOCS_DIR = Path(__file__).resolve().parent
SITE_URL = "https://facebookresearch.github.io/fairchem"

CURATED_LINKS = {
    "Start here": [
        ("Introduction", "introduction"),
        ("Installation", "install"),
        ("Hello World", "quickstart"),
        ("Explore UMA's capabilities", "uma-capabilities"),
    ],
    "Models and workflows": [
        ("UMA model guide", "uma"),
        ("ASE calculator guide", "ase-calculator"),
        ("FAIR Chemistry papers", "fair-chemistry-papers"),
        (
            "FAIR Chemistry leaderboard",
            "https://huggingface.co/spaces/facebook/fairchem_leaderboard",
        ),
    ],
    "Help and learning": [
        ("Frequently asked questions", "faq"),
        ("Learning resources", "learning-resources"),
        ("No-code UMA playground", "https://aidemos.atmeta.com/uma?view=playground"),
    ],
}


def iter_toc_files(items: list[dict]) -> list[Path]:
    """
    Return Markdown files from a MyST table of contents in display order.

    Args:
        items: MyST table-of-contents entries.

    Returns:
        Paths relative to the documentation directory.
    """
    files: list[Path] = []
    for item in items:
        if file_name := item.get("file"):
            path = Path(file_name)
            if path.suffix == ".md":
                files.append(path)
        files.extend(iter_toc_files(item.get("children", [])))
    return files


def strip_frontmatter(text: str) -> str:
    """
    Remove YAML frontmatter from a Markdown document.

    Args:
        text: Markdown source text.

    Returns:
        Markdown without its leading YAML metadata block.
    """
    if text.startswith("---\n"):
        _, separator, body = text[4:].partition("\n---\n")
        text = body if separator else text
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def render_llms_index() -> str:
    """
    Render the concise LLM documentation index.

    Returns:
        The contents of ``llms.txt``.
    """
    lines = [
        "# FAIR Chemistry",
        "",
        "> Open datasets, pretrained models, and simulation tools for materials",
        "> science and quantum chemistry from Meta FAIR.",
        "",
        "FAIR Chemistry provides the UMA family of universal interatomic",
        "potentials and the `fairchem` Python packages used to run them.",
    ]
    for heading, links in CURATED_LINKS.items():
        lines.extend(("", f"## {heading}", ""))
        for title, target in links:
            url = target if target.startswith("https://") else f"{SITE_URL}/{target}"
            lines.append(f"- [{title}]({url})")
    lines.extend(
        (
            "",
            "## Complete documentation",
            "",
            f"- [All documentation in one file]({SITE_URL}/llms-full.txt)",
            "",
        )
    )
    return "\n".join(lines)


def render_llms_full() -> str:
    """
    Render all published Markdown sources in table-of-contents order.

    Returns:
        The contents of ``llms-full.txt``.
    """
    config = yaml.safe_load((DOCS_DIR / "myst.yml").read_text())
    paths = iter_toc_files(config["project"]["toc"])
    sections = [
        "# FAIR Chemistry: Complete Documentation",
        "",
        f"Canonical site: {SITE_URL}/",
    ]
    for path in dict.fromkeys(paths):
        source = DOCS_DIR / path
        sections.extend(
            (
                "",
                "---",
                "",
                f"Source: `docs/{path.as_posix()}`",
                "",
                strip_frontmatter(source.read_text()),
            )
        )
    return "\n".join(sections).rstrip() + "\n"


def update_file(path: Path, expected: str, check: bool) -> bool:
    """
    Write a generated file or report whether it is current.

    Args:
        path: Output file path.
        expected: Expected file contents.
        check: If true, do not write the file.

    Returns:
        Whether the existing file already contains the expected text.
    """
    current = path.read_text() if path.exists() else None
    if current == expected:
        return True
    if not check:
        path.write_text(expected)
    return False


def main() -> int:
    """
    Generate or check the two LLM-facing documentation files.

    Returns:
        A process exit status.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if generated files do not match their sources",
    )
    args = parser.parse_args()

    outputs = {
        DOCS_DIR / "llms.txt": render_llms_index(),
        DOCS_DIR / "llms-full.txt": render_llms_full(),
    }
    stale = [
        path
        for path, contents in outputs.items()
        if not update_file(path, contents, args.check)
    ]
    if args.check and stale:
        for path in stale:
            print(f"out of date: {path.relative_to(DOCS_DIR.parent)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
