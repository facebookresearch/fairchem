"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Shared helpers for the local test_*.py scripts.

Switch graph generation backend with the GRAPH_GEN env var:

    GRAPH_GEN=gpu  (default) — model uses on-the-fly graph gen
    GRAPH_GEN=cpu           — pymatgen builds edges client-side,
                              model trusts external_graph_gen=True

Default radius and max_neigh match UMA-S inference. Adjust here once
to flip them across every script.
"""

from __future__ import annotations

import os

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

GRAPH_GEN = os.environ.get("GRAPH_GEN", "gpu").lower()
assert GRAPH_GEN in ("cpu", "gpu"), (
    f"GRAPH_GEN must be 'cpu' or 'gpu', got {GRAPH_GEN!r}"
)
EXTERNAL = GRAPH_GEN == "cpu"

RADIUS = 6.0
MAX_NEIGH = 300


def make_settings(
    *,
    compile: bool,
    execution_mode: str = "umas_fast_gpu_mixed",
    tf32: bool = True,
    **extra,
) -> InferenceSettings:
    """Build InferenceSettings with graph-gen backend resolved from env."""
    return InferenceSettings(
        tf32=tf32,
        activation_checkpointing=False,
        merge_mole=False,
        compile=compile,
        external_graph_gen=EXTERNAL,
        execution_mode=execution_mode,
        **extra,
    )


def make_atomic_data(atoms, *, task_name: str) -> AtomicData:
    """
    Build an AtomicData. Edges are always populated client-side so
    that ``batch.edge_index.shape[1]`` is meaningful; with GRAPH_GEN=gpu
    the model regenerates them, with GRAPH_GEN=cpu the model trusts them.
    """
    return AtomicData.from_ase(
        atoms,
        task_name=task_name,
        r_edges=True,
        radius=RADIUS,
        max_neigh=MAX_NEIGH,
    )
