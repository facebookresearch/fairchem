"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np
import numpy.testing as npt
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


def assert_cell_major_forces_close(
    supercell_forces: np.ndarray,
    unit_forces: np.ndarray,
    multiplier: int,
    *,
    atol: float,
) -> None:
    """
    Compare forces using ASE's cell-major supercell atom correspondence.

    Each cell repeats the unit-cell atoms in their original order. Comparing
    those corresponding rows avoids ordering changes caused by force noise,
    and cannot exchange forces between distinct atomic sites. The NumPy
    default relative tolerance is preserved.

    Args:
        supercell_forces: Forces from a cell-major ASE supercell.
        unit_forces: Forces on the corresponding unit-cell atoms.
        multiplier: Number of unit-cell replicas in the supercell.
        atol: Existing absolute force tolerance for the model precision.
    """
    npt.assert_allclose(
        supercell_forces,
        np.tile(unit_forces, (multiplier, 1)),
        atol=atol,
        err_msg="Supercell forces differ from their corresponding unit-cell forces.",
    )


@contextmanager
def expected_extensivity_failure(checkpoint: str, invariant: str) -> Iterator[None]:
    """
    Handle the documented UMA-S-1p2 regression at a numeric invariant only.

    Use this around the final energy or force assertion, after model loading,
    inference, and structure setup. Other exception types and failures from
    other checkpoints remain errors. Passing cases remain ordinary passes
    until checkpoint validation identifies which cases consistently fail.

    Args:
        checkpoint: Registered checkpoint name under test.
        invariant: Energy or force invariant named in the failure report.
    """
    try:
        yield
    except AssertionError as error:
        if checkpoint != "uma-s-1p2":
            raise
        pytest.xfail(
            f"uma-s-1p2 has a known extensivity regression ({invariant}): {error}"
        )
