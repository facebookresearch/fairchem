"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Import docs/check_links.py directly by path (it is a standalone script, not a
# package module) so this test needs no fairchem/torch imports and stays fast.
_CL_PATH = Path(__file__).resolve().parents[1] / "check_links.py"
_spec = importlib.util.spec_from_file_location("check_links", _CL_PATH)
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)


# --- link extraction -------------------------------------------------------


def test_inline_link_and_image():
    assert cl.extract_links("[text](foo.md) and ![alt](img/bar.png)") == [
        "foo.md",
        "img/bar.png",
    ]


def test_balanced_parens_not_truncated():
    # URL containing balanced parentheses must be captured whole.
    url = "https://en.wikipedia.org/wiki/Bravais_lattice_(crystallography)"
    assert cl.extract_links(f"see [lattice]({url})") == [url]


def test_link_with_title_stops_at_whitespace():
    assert cl.extract_links('[x](foo.md "a title")') == ["foo.md"]


def test_autolink():
    assert cl.extract_links("<https://example.com/page>") == [
        "https://example.com/page"
    ]


def test_myst_link_directive():
    text = (
        ":::{grid-item-card} General\n"
        ":link: catalysts/datasets/summary.md\n\n"
        "body\n:::\n"
    )
    assert "catalysts/datasets/summary.md" in cl.extract_links(text)


def test_reference_style_definition():
    text = "See [the guide][g] below.\n\n[g]: ./core/uma.md\n"
    assert "./core/uma.md" in cl.extract_links(text)


def test_reference_style_ignores_prose_target():
    # A non-path reference target must not be treated as a checkable link.
    assert cl.extract_links("[note]: see above\n") == []


def test_html_href_and_src():
    text = '<a href="other.md">x</a> <img src="pics/p.png">'
    links = cl.extract_links(text)
    assert "other.md" in links and "pics/p.png" in links


def test_code_blocks_are_ignored():
    text = (
        "real [a](a.md)\n"
        "```python\n"
        "url = '[nope](does_not_exist.md)'\n"
        "```\n"
        "inline `[also-nope](missing.md)` done\n"
    )
    links = cl.extract_links(text)
    assert links == ["a.md"]


# --- internal resolution ---------------------------------------------------


@pytest.fixture
def docs_tree(tmp_path: Path) -> Path:
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "uma.md").write_text("# uma")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "logo.png").write_bytes(b"x")
    (tmp_path / "index.md").write_text("# index")
    return tmp_path


def test_internal_existing_relative(docs_tree: Path):
    src = docs_tree / "index.md"
    assert cl.check_internal(docs_tree, src, "core/uma.md") is None


def test_internal_extensionless_page(docs_tree: Path):
    src = docs_tree / "index.md"
    # extension-less link resolves to the .md page
    assert cl.check_internal(docs_tree, src, "core/uma") is None


def test_internal_root_relative(docs_tree: Path):
    src = docs_tree / "core" / "uma.md"
    assert cl.check_internal(docs_tree, src, "/assets/logo.png") is None


def test_internal_anchor_and_query_stripped(docs_tree: Path):
    src = docs_tree / "index.md"
    assert cl.check_internal(docs_tree, src, "core/uma.md#section") is None
    assert cl.check_internal(docs_tree, src, "core/uma.md?x=1") is None


def test_internal_missing_is_error(docs_tree: Path):
    src = docs_tree / "index.md"
    assert cl.check_internal(docs_tree, src, "core/missing.md") is not None


def test_is_checkable_internal_filters():
    assert not cl.is_checkable_internal("https://x.com")
    assert not cl.is_checkable_internal("#anchor")
    assert not cl.is_checkable_internal("mailto:a@b.c")
    assert not cl.is_checkable_internal("{role}`x`")
    assert cl.is_checkable_internal("foo/bar.md")
