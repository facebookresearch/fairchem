"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import ast
from pathlib import Path

import fairchem.core.regression_suite as package

PACKAGE = package.__name__
# Each subpackage may import only from itself and the subpackages before it.
LAYERS = ("spec", "suites", "artifacts", "evaluation", "reporting", "execution", "cli")


def _layer(module: str) -> int | None:
    first = module.split(".")[0]
    if first in ("cli", "__main__"):
        first = "cli"
    return LAYERS.index(first) if first in LAYERS else None


def _imported(node: ast.AST, module: str) -> list[str]:
    """
    Return the package-relative names an import statement may load.
    """
    if isinstance(node, ast.Import):
        names = [alias.name for alias in node.names]
    elif isinstance(node, ast.ImportFrom):
        parent = ""
        if node.level:
            parent = ".".join([PACKAGE, *module.split(".")[: -node.level]])
        source = ".".join(part for part in (parent, node.module) if part)
        names = [source, *(f"{source}.{alias.name}" for alias in node.names)]
    else:
        return []
    prefix = f"{PACKAGE}."
    return [name.removeprefix(prefix) for name in names if name.startswith(prefix)]


def test_subpackages_import_only_from_lower_layers():
    root = Path(package.__file__).parent
    upward = []
    for path in sorted(root.rglob("*.py")):
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        source = _layer(module)
        if source is None:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            for target in _imported(node, module):
                layer = _layer(target)
                if layer is not None and layer > source:
                    upward.append(f"{module} imports {target}")
    assert upward == []
