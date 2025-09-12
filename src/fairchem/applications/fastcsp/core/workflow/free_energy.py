"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Free energy calculations for FastCSP.

This module will provide free energy calculation capabilities for crystal structures.

TODO: Implementation in progress - will be available soon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def calculate_free_energies(
    structures_path: Path,
    output_path: Path,
    config: dict,
) -> None:
    """
    Calculate free energies for crystal structures.

    Args:
        structures_path: Path to directory containing crystal structures
        output_path: Path to output directory for free energy results
        config: Configuration dictionary containing free energy parameters

    Returns:
        None

    TODO: Implementation coming soon. This will include:
        - Integration with existing structure ranking pipeline
        - Support for different free energy methods
    """
    raise NotImplementedError(
        "Free energy calculations are not yet implemented. "
        "This feature is under development and will be available soon. "
        "Please check future releases or contact the developers for updates."
    )
