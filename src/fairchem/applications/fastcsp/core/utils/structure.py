"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Structure Conversion, Manipulation, and Validation Utilities for FastCSP

This module provides essential utilities for handling crystal structures throughout
the FastCSP workflow. It implements efficient conversions between different structure
representations, validation algorithms for structural integrity, and functions
for high-throughput crystal structure processing.

Key Features:
- Structure hashing for efficient comparison and caching
- Distributed processing support with consistent partitioning
- Chemical composition validation and bonding analysis
- Quality control checks for structural integrity

Structure Validation:
- Atomic composition conservation (Z-number preservation via ``check_correct_z``)
- Reference-anchored molecular topology check via
  ``check_molecule_matches_reference``

The module is designed for both individual structure operations and batch processing
of large crystal structure datasets common in high-throughput materials discovery.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.sparse import csgraph

if TYPE_CHECKING:
    from pathlib import Path

    from ase import Atoms


def cif_to_structure(cif: str) -> Structure | None:
    """
    Convert CIF (Crystallographic Information File) string to pymatgen Structure object.

    Args:
        cif: CIF format string containing crystal structure data

    Returns:
        Structure object if conversion successful, None if cif is empty/invalid
    """
    return Structure.from_str(cif, fmt="cif") if cif else None


def cif_to_atoms(cif: str) -> Atoms | None:
    """
    Convert CIF string to ASE Atoms object.

    Args:
        cif: CIF format string containing crystal structure data

    Returns:
        ASE Atoms object if conversion successful, None if cif is empty/invalid
    """
    return AseAtomsAdaptor.get_atoms(cif_to_structure(cif)) if cif else None


def get_partition_id(key: str, npartitions: int = 1000) -> int:
    """
    Generate consistent partition ID from key using MD5 hash.
    """
    key_encoded = key.encode("utf-8")
    md5_hash = hashlib.md5()
    md5_hash.update(key_encoded)
    consistent_hash_hex = md5_hash.hexdigest()
    consistent_hash_int = int(consistent_hash_hex, 16)
    return consistent_hash_int % npartitions


def get_structure_hash(
    structure: Structure,
    z: int,
    use_density: bool = True,
    use_volume: bool = True,
    density_bin_size: float = 0.1,
    vol_bin_size: float = 0.2,
) -> str:
    """
    Generate hash string for structure grouping based on formula and binned properties.

    Args:
        structure: Pymatgen Structure object to hash
        z: Number of formula units per unit cell
        use_density: Include density in hash for geometric grouping
        use_volume: Include volume in hash for size-based grouping
        density_bin_size: Bin size for density discretization (g/cm³)
        vol_bin_size: Bin size for volume discretization (Ų)

    Returns:
        Hash string like "C6H4O4_z4_d1.5_v125.2".
    """
    # Start with chemical composition
    formula = structure.composition.reduced_formula
    hash_components = [formula, "z" + str(z)]

    # Add density-based grouping if requested
    if use_density:
        density = structure.density
        # Bin density to group structures with similar packing
        density_bin = round(density / density_bin_size) * density_bin_size
        hash_components.append(f"d{density_bin:.1f}")

    # Add volume-based grouping if requested
    if use_volume:
        volume = structure.volume
        # Bin volume to group structures with similar cell sizes
        vol_bin = round(volume / vol_bin_size**3) * vol_bin_size**3
        hash_components.append(f"v{vol_bin:.1f}")

    # Combine all components into single hash string
    return "_".join(hash_components)


def check_correct_z(
    structure_or_atoms: Structure | Atoms | None,
    requested_z: int,
) -> bool:
    """
    Check whether the number of connected molecular fragments in the cell
    matches the requested number of formula units (Z).

    Args:
        structure_or_atoms: Pymatgen ``Structure`` or ASE ``Atoms`` for the
            generated unit cell. Returns ``False`` if ``None``.
        requested_z: Integer Z value the generator was asked for.

    Returns:
        True if the JmolNN connected-component count equals ``requested_z``,
        False otherwise (or if the input is ``None``).
    """
    # Handle error cases where the structure couldn't be processed
    if structure_or_atoms is None:
        return False

    # Accept either a pymatgen Structure or an ASE Atoms; convert if needed
    if isinstance(structure_or_atoms, Structure):
        structure = structure_or_atoms
    else:
        structure = AseAtomsAdaptor.get_structure(structure_or_atoms)

    # Build adjacency matrix using Jmol bonding radii
    nn_info = JmolNN().get_all_nn_info(structure)
    n = len(nn_info)
    nn_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for entry in nn_info[i]:
            nn_matrix[i, entry["site_index"]] = 1

    # Connected-component count == observed number of molecular fragments
    observed_z = csgraph.connected_components(nn_matrix)[0]
    return observed_z == requested_z


def reference_graph_from_atoms(
    reference_atoms: Atoms | None,
) -> nx.Graph | None:
    """
    Build a NetworkX graph from a reference molecular conformer.

    Nodes carry an ``atomic_num`` attribute; edges are JmolNN-derived covalent
    bonds (bond order discarded). The reference ``.xyz`` is assumed to be a
    single-molecule conformer (as produced by the Genarris seed pipeline).

    Args:
        reference_atoms: ASE ``Atoms`` for the reference single-molecule
            conformer (typically read from the .xyz that seeded Genarris).

    Returns:
        ``nx.Graph`` for the reference molecule, or ``None`` if the input is
        ``None``/empty or the adjacency cannot be built.
    """
    if reference_atoms is None:
        return None

    try:
        # XYZ-loaded molecules have no unit cell (cell rank < 3), which makes
        # AseAtomsAdaptor.get_structure raise LinAlgError on the singular
        # lattice. Pad with a generously large cubic box so the molecule sits
        # well inside and pymatgen can build a periodic Structure for JmolNN.
        if np.linalg.matrix_rank(np.array(reference_atoms.cell)) < 3:
            reference_atoms = reference_atoms.copy()
            reference_atoms.cell = np.eye(3) * 30.0
            reference_atoms.center()
            reference_atoms.pbc = True

        structure = AseAtomsAdaptor.get_structure(reference_atoms)

        # Build adjacency matrix using Jmol bonding radii (same pattern as
        # check_correct_z).
        nn_info = JmolNN().get_all_nn_info(structure)
        n = len(nn_info)
        if n < 1:
            return None

        nn_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for entry in nn_info[i]:
                nn_matrix[i, entry["site_index"]] = 1

        # Build undirected nx.Graph from the adjacency matrix and attach
        # atomic_num as a per-node attribute (used by the categorical node
        # match in check_molecule_matches_reference).
        graph = nx.from_numpy_array(nn_matrix)
        for i in range(n):
            graph.nodes[i]["atomic_num"] = structure[i].specie.number
        return graph
    except Exception as e:
        logger = get_central_logger()
        logger.warning(f"Failed to build reference graph: {e}")
        return None


def load_reference_graph(
    conf_dir: Path | None,
    conf_id: str,
) -> nx.Graph | None:
    """
    Locate and load the per-conformer reference-molecule graph.

    Searches ``conf_dir`` for ``<conf_id>.xyz``, ``.sdf``, or ``.mol`` (in that
    order), reads the first match with ASE, and converts it to a NetworkX
    graph via :func:`reference_graph_from_atoms`.

    Args:
        conf_dir: Directory containing ``<conf_id>.{xyz,sdf,mol}``. May be
            ``None`` (e.g., when a caller failed to derive a path); in that
            case the function logs and returns ``None``.
        conf_id: Conformer identifier. Used as the filename stem.

    Returns:
        ``nx.Graph`` for the reference molecule, or ``None`` if the file
        could not be located / parsed / converted to a graph.
    """
    import ase.io

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
                reference_atoms = ase.io.read(candidate)
                return reference_graph_from_atoms(reference_atoms)
            except Exception as e:
                logger.warning(f"Failed to read reference geometry {candidate}: {e}")
                return None

    logger.warning(
        f"No reference geometry (.xyz/.sdf/.mol) for conf_id={conf_id} "
        f"in {conf_dir}; reference graph will be None."
    )
    return None


def check_molecule_matches_reference(
    structure: Structure | None,
    reference_graph: nx.Graph | None,
) -> bool:
    """
    Check whether every connected molecular fragment in the cell is
    isomorphic to the reference molecule.

    For each connected component of the JmolNN adjacency, a subgraph view
    is taken from the full-cell ``nx.Graph`` (which carries ``atomic_num``
    per node). Each fragment is then compared to the reference via
    ``nx.is_isomorphic`` with a categorical match on atomic number. This
    catches topology errors (tautomers, rearranged rings, wrong functional
    groups) that the count-only ``check_correct_z`` cannot detect.

    Args:
        structure: Pymatgen ``Structure`` for the generated unit cell.
        reference_graph: Reference molecular graph built via
            ``reference_graph_from_atoms``.

    Returns:
        ``True`` iff every fragment in the cell is isomorphic to the
        reference. ``False`` if any fragment mismatches, if either input is
        ``None``, or on exception.
    """
    if structure is None or reference_graph is None:
        return False

    try:
        nn_info = JmolNN().get_all_nn_info(structure)
        n = len(nn_info)
        nn_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for entry in nn_info[i]:
                nn_matrix[i, entry["site_index"]] = 1

        # Build the full-cell graph once with atomic_num node attributes;
        # take per-fragment subgraph views for the isomorphism comparison.
        graph = nx.from_numpy_array(nn_matrix)
        for i in range(n):
            graph.nodes[i]["atomic_num"] = structure[i].specie.number
        node_match = nx.algorithms.isomorphism.categorical_node_match("atomic_num", 0)

        for comp_nodes in nx.connected_components(graph):
            fragment_graph = graph.subgraph(comp_nodes)
            if not nx.is_isomorphic(
                fragment_graph, reference_graph, node_match=node_match
            ):
                return False
        return True
    except Exception as e:
        logger = get_central_logger()
        logger.warning(f"Failed molecule-matches-reference check: {e}")
        return False
