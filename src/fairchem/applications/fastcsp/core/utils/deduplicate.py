"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Structure Deduplication Utilities for FastCSP

Core deduplication functions are imported from
fairchem.core.components.calculate.recipes.csp.
"""

from __future__ import annotations

from fairchem.core.components.calculate.recipes.csp import (
    deduplicate_structures,
    process_structure_group,
)

__all__ = [
    "deduplicate_structures",
    "process_structure_group",
]
