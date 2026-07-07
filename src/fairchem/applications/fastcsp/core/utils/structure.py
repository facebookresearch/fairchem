"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Structure Conversion, Manipulation, and Validation Utilities for FastCSP
"""

from __future__ import annotations

import hashlib
from collections import deque
from typing import TYPE_CHECKING

import ase.io
import networkx as nx
import numpy as np
from ase import Atoms
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.sparse import csgraph

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
def cif_to_structure(cif: str) -> Structure | None:
    """Parse a CIF string to a pymatgen ``Structure`` (``None`` if empty/falsy)."""
    return Structure.from_str(cif, fmt="cif") if cif else None


def cif_to_atoms(cif: str) -> Atoms | None:
    """Parse a CIF string to an ASE ``Atoms`` (``None`` if empty/falsy)."""
    return AseAtomsAdaptor.get_atoms(cif_to_structure(cif)) if cif else None


def _to_structure(
    structure_or_atoms: Structure | Atoms | None,
) -> Structure | None:
    """Coerce ``None`` / ``Structure`` / ``Atoms`` -> ``Structure`` / ``None``."""
    if structure_or_atoms is None:
        return None
    if isinstance(structure_or_atoms, Structure):
        return structure_or_atoms
    return AseAtomsAdaptor.get_structure(structure_or_atoms)


# ---------------------------------------------------------------------------
# Partitioning / grouping keys
# ---------------------------------------------------------------------------
def get_partition_id(key: str, npartitions: int = 1000) -> int:
    """Return a deterministic ``key -> [0, npartitions)`` bucket (MD5-based)."""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % npartitions


def get_structure_group(
    mol_id: str,
    conf_id: str | None = None,
    z: int | None = None,
    spg: int | None = None,
    density: float | None = None,
    density_bin_size: float | None = None,
    energy: float | None = None,
    energy_bin_size: float | None = None,
) -> str:
    """Build a blocker-key string for deduplication grouping.

    Key always starts with ``mol_id`` and includes ``z``. Each other optional
    argument adds a segment when set, in order:

    - ``conf_id``  -> ``conf={id}``
    - ``spg``      -> ``spgN`` (generated space group number)
    - ``density`` + ``density_bin_size``  -> ``d{bin:g}``
    - ``energy`` + ``energy_bin_size``    -> ``e{bin:g}``

    Example: ``"ACBNZA02_conf=0_z4_spg14_d1.5_e0.01"``.
    """
    parts = [str(mol_id)]
    if conf_id is not None:
        parts.append(f"conf={conf_id}")
    parts.append(f"z{z}")
    if spg is not None:
        parts.append(f"spg{int(spg)}")
    if density is not None and density_bin_size is not None:
        parts.append(f"d{round(density / density_bin_size) * density_bin_size:g}")
    if energy is not None and energy_bin_size is not None:
        parts.append(f"e{round(energy / energy_bin_size) * energy_bin_size:g}")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# JmolNN adjacency + graph primitives
# ---------------------------------------------------------------------------
def _adjacency_from_nn_info(nn_info: list[list[dict]]) -> np.ndarray:
    """Build the 0/1 JmolNN adjacency matrix from a precomputed ``nn_info``.

    Split off so callers that also need the raw ``nn_info`` (e.g. the periodic
    image vectors in :func:`extract_molecules`) don't pay for a second
    ``JmolNN().get_all_nn_info(structure)`` call.
    """
    n = len(nn_info)
    adj = np.zeros((n, n), dtype=int)
    for i, neighbours in enumerate(nn_info):
        for nb in neighbours:
            adj[i, nb["site_index"]] = 1
    return adj


def _labeled_graph(nn_matrix: np.ndarray, structure: Structure) -> nx.Graph:
    """``nx.Graph`` from an adjacency matrix with an ``atomic_num`` per node.

    ``atomic_num`` is what the categorical node match in the isomorphism test
    keys off (see :func:`check_molecule_matches_reference`).
    """
    graph = nx.from_numpy_array(nn_matrix)
    for i in range(nn_matrix.shape[0]):
        graph.nodes[i]["atomic_num"] = structure[i].specie.number
    return graph


def jmolnn_adjacency(
    structure_or_atoms: Structure | Atoms,
) -> np.ndarray:
    """Return the 0/1 JmolNN adjacency matrix. Accepts ``Structure`` or ``Atoms``."""
    return _adjacency_from_nn_info(
        JmolNN().get_all_nn_info(_to_structure(structure_or_atoms))
    )


def extract_molecules(structure: Structure) -> list[Atoms]:
    """One PBC-unwrapped ASE ``Atoms`` per connected molecular fragment.

    Same JmolNN bond definition as :func:`jmolnn_adjacency`, plus a BFS that
    undoes periodic wrapping.
    """
    nn_info = JmolNN().get_all_nn_info(structure)
    _, labels = csgraph.connected_components(
        _adjacency_from_nn_info(nn_info), directed=False
    )
    lattice = structure.lattice.matrix
    cart_coords = structure.cart_coords
    species = [s.specie.symbol for s in structure.species]

    atoms_list: list[Atoms] = []
    for comp_id in range(int(labels.max()) + 1):
        comp = np.where(labels == comp_id)[0]
        anchor = int(comp[0])
        offsets = {anchor: np.zeros(3)}
        queue, visited = deque([anchor]), {anchor}
        while queue:
            u = queue.popleft()
            for nb in nn_info[u]:
                v = nb["site_index"]
                if v in visited or labels[v] != comp_id:
                    continue
                # nn_info[u] gives v's periodic image relative to u, so adding
                # is unconditionally correct (sign falls out naturally).
                offsets[v] = offsets[u] + np.array(nb["image"], dtype=float)
                visited.add(v)
                queue.append(v)
        positions = np.stack([cart_coords[i] + offsets[i] @ lattice for i in comp])
        atoms_list.append(
            Atoms(
                symbols=[species[i] for i in comp],
                positions=positions,
                pbc=False,
            )
        )
    return atoms_list


# ---------------------------------------------------------------------------
# Reference-molecule graph (Genarris seed anchor)
# ---------------------------------------------------------------------------
def reference_graph_from_atoms(
    reference_atoms: Atoms | None,
) -> nx.Graph | None:
    """Build an ``nx.Graph`` for a single-molecule reference conformer.

    Nodes carry ``atomic_num``; edges are JmolNN-derived (bond order dropped).
    Returns ``None`` on failure.
    """
    if reference_atoms is None:
        return None
    try:
        # XYZ-loaded molecules have no unit cell (cell rank < 3), which makes
        # AseAtomsAdaptor.get_structure raise LinAlgError on the singular
        # lattice. Pad with a large cubic box so pymatgen can build a periodic
        # Structure for JmolNN.
        if np.linalg.matrix_rank(np.array(reference_atoms.cell)) < 3:
            reference_atoms = reference_atoms.copy()
            reference_atoms.cell = np.eye(3) * 30.0
            reference_atoms.center()
            reference_atoms.pbc = True
        structure = AseAtomsAdaptor.get_structure(reference_atoms)
        nn_matrix = jmolnn_adjacency(structure)
        if nn_matrix.shape[0] < 1:
            return None
        return _labeled_graph(nn_matrix, structure)
    except Exception as e:
        get_central_logger().warning(f"Failed to build reference graph: {e}")
        return None


def load_reference_graph(
    conf_dir: Path | None,
    conf_id: str,
) -> nx.Graph | None:
    """Load ``<conf_dir>/<conf_id>.{xyz,sdf,mol}`` and return its reference graph.

    Returns ``None`` (and logs) if the directory / file is missing or unreadable.
    """
    logger = get_central_logger()
    if conf_dir is None or not conf_dir.is_dir():
        logger.warning(
            f"No reference geometry directory for conf_id={conf_id} "
            f"(conf_dir={conf_dir}); reference graph will be None."
        )
        return None
    for ext in (".xyz", ".sdf", ".mol"):
        candidate = conf_dir / f"{conf_id}{ext}"
        if candidate.is_file():
            try:
                return reference_graph_from_atoms(ase.io.read(candidate))
            except Exception as e:
                logger.warning(f"Failed to read reference geometry {candidate}: {e}")
                return None
    logger.warning(
        f"No reference geometry (.xyz/.sdf/.mol) for conf_id={conf_id} "
        f"in {conf_dir}; reference graph will be None."
    )
    return None


# ---------------------------------------------------------------------------
# Validity checks
# ---------------------------------------------------------------------------
def check_correct_z(
    structure_or_atoms: Structure | Atoms | None,
    requested_z: int,
) -> bool:
    """True iff the JmolNN connected-component count equals ``requested_z``.

    ``None`` inputs return ``False``.
    """
    structure = _to_structure(structure_or_atoms)
    if structure is None:
        return False
    return csgraph.connected_components(jmolnn_adjacency(structure))[0] == requested_z


def check_molecule_matches_reference(
    structure: Structure | Atoms | None,
    reference_graph: nx.Graph | None,
) -> bool:
    """True iff every connected fragment is isomorphic to ``reference_graph``.

    Each connected component of the full-cell JmolNN graph is compared to the
    reference via ``nx.is_isomorphic`` with a categorical node match on
    ``atomic_num``. Catches topology errors (tautomers, rearranged rings,
    wrong functional groups) that :func:`check_correct_z` cannot.

    ``False`` if either input is ``None`` or on exception.
    """
    structure = _to_structure(structure)
    if structure is None or reference_graph is None:
        return False
    try:
        graph = _labeled_graph(jmolnn_adjacency(structure), structure)
        node_match = nx.algorithms.isomorphism.categorical_node_match("atomic_num", 0)
        for comp_nodes in nx.connected_components(graph):
            if not nx.is_isomorphic(
                graph.subgraph(comp_nodes),
                reference_graph,
                node_match=node_match,
            ):
                return False
        return True
    except Exception as e:
        get_central_logger().warning(f"Failed molecule-matches-reference check: {e}")
        return False


def check_connectivity_unchanged(
    initial_structure_or_atoms: Structure | Atoms | None,
    final_structure_or_atoms: Structure | Atoms | None,
) -> bool:
    """True iff the JmolNN adjacency is element-wise equal between two cells.

    Used to compare pre- vs post-relax topology. ``False`` if either input is
    ``None``, if atom counts differ, or on exception.
    """
    if initial_structure_or_atoms is None or final_structure_or_atoms is None:
        return False
    try:
        initial_adj = jmolnn_adjacency(initial_structure_or_atoms)
        final_adj = jmolnn_adjacency(final_structure_or_atoms)
        if initial_adj.shape != final_adj.shape:
            return False
        return bool(np.array_equal(initial_adj, final_adj))
    except Exception as e:
        get_central_logger().warning(f"Failed connectivity-unchanged check: {e}")
        return False
