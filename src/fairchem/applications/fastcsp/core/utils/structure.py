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
- Atomic composition conservation (Z-number preservation)
- Covalent bonding network analysis using coordination environments

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
    nn_matrix = np.zeros((len(nn_info), len(nn_info)))
    for i in range(len(nn_info)):
        for j in range(len(nn_info[i])):
            nn_matrix[i, nn_info[i][j]["site_index"]] = 1

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
        # check_correct_z and check_connectivity_changes).
        nn_info = JmolNN().get_all_nn_info(structure)
        n = len(nn_info)
        if n < 1:
            return None

        # Build the nx.Graph (atomic_num node attr; undirected edges
        graph = nx.Graph()
        for i in range(n):
            graph.add_node(i, atomic_num=structure[i].specie.number)
        for i in range(n):
            for entry in nn_info[i]:
                j = entry["site_index"]
                if i < j:
                    graph.add_edge(i, j)
        return graph
    except Exception as e:
        logger = get_central_logger()
        logger.warning(f"Failed to build reference graph: {e}")
        return None


def check_molecule_matches_reference(
    structure: Structure | None,
    reference_graph: nx.Graph | None,
) -> bool:
    """
    Check whether every connected molecular fragment in the cell is
    isomorphic to the reference molecule.

    For each connected component of the JmolNN adjacency, a small
    ``nx.Graph`` is built with ``atomic_num`` node attributes and the
    induced covalent edges. Each fragment graph is then compared to the
    reference via ``nx.is_isomorphic`` with a categorical match on atomic
    number. This catches topology errors (tautomers, rearranged rings,
    wrong functional groups) that the count-only ``check_correct_z`` and
    bond-matrix ``check_connectivity_changes`` cannot detect.

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
        nn_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(len(nn_info[i])):
                nn_matrix[i, nn_info[i][j]["site_index"]] = 1

        n_components, labels = csgraph.connected_components(nn_matrix)
        node_match = nx.algorithms.isomorphism.categorical_node_match("atomic_num", 0)

        for comp in range(n_components):
            indices = [i for i, lbl in enumerate(labels) if lbl == comp]
            fragment_graph = nx.Graph()
            for i in indices:
                fragment_graph.add_node(i, atomic_num=structure[i].specie.number)
            for i in indices:
                for entry in nn_info[i]:
                    j = entry["site_index"]
                    if j in indices and i < j:
                        fragment_graph.add_edge(i, j)
            if not nx.is_isomorphic(
                fragment_graph, reference_graph, node_match=node_match
            ):
                return False
        return True
    except Exception as e:
        logger = get_central_logger()
        logger.warning(f"Failed molecule-matches-reference check: {e}")
        return False


def check_connectivity_changes(
    initial_atoms: Atoms | None,
    final_atoms: Atoms | None,
    check_exact_bonds: bool = True,
    check_molecule_count: bool = True,
) -> dict:
    """
    Compare connectivity between two structures with one bond-matrix build per
    structure (no StructureMatcher / site reordering — sites are kept in their
    natural order).

    The JmolNN adjacency matrix is built once per input, then the requested
    checks are derived from those matrices: (a) exact-bond preservation via
    matrix equality and (b) molecule-count preservation via connected-component
    counts.

    Args:
        initial_atoms: Structure before relaxation (ASE ``Atoms``).
        final_atoms: Structure after relaxation (ASE ``Atoms``).
        check_exact_bonds: Whether to compare the full adjacency matrices.
        check_molecule_count: Whether to compare the connected-component counts.

    Returns:
        Dict with keys:
            - ``no_changes``: True iff every requested check passed.
            - ``exact_bonds_preserved``: True iff bond matrices are equal
              (or check disabled, in which case True by default).
            - ``molecule_count_preserved``: True iff component counts match
              (or check disabled, in which case True by default).
            - ``initial_molecule_count``: Component count in initial (0 if not checked).
            - ``final_molecule_count``: Component count in final (0 if not checked).
            - ``bonds_changed``: True iff exact-bond check ran and disagreed.
            - ``error``: Optional error message if something failed.
    """
    result = {
        "no_changes": True,
        "exact_bonds_preserved": True,
        "molecule_count_preserved": True,
        "initial_molecule_count": 0,
        "final_molecule_count": 0,
        "bonds_changed": False,
    }

    if initial_atoms is None or final_atoms is None:
        result.update(
            no_changes=False,
            exact_bonds_preserved=False,
            molecule_count_preserved=False,
            error="One or both input structures are None",
        )
        return result

    try:
        # Convert ASE Atoms to pymatgen Structures for neighbor analysis
        initial_structure = AseAtomsAdaptor.get_structure(initial_atoms)
        final_structure = AseAtomsAdaptor.get_structure(final_atoms)

        # Build adjacency matrix for initial structure using Jmol bonding radii
        initial_nn_info = JmolNN().get_all_nn_info(initial_structure)
        initial_nn_matrix = np.zeros((len(initial_nn_info), len(initial_nn_info)))
        for i in range(len(initial_nn_info)):
            for j in range(len(initial_nn_info[i])):
                initial_nn_matrix[i, initial_nn_info[i][j]["site_index"]] = 1

        # Build adjacency matrix for final (relaxed) structure
        final_nn_info = JmolNN().get_all_nn_info(final_structure)
        final_nn_matrix = np.zeros((len(final_nn_info), len(final_nn_info)))
        for i in range(len(final_nn_info)):
            for j in range(len(final_nn_info[i])):
                final_nn_matrix[i, final_nn_info[i][j]["site_index"]] = 1

        if check_exact_bonds:
            # Direct comparison in natural site order (no permutation)
            exact_bonds_preserved = bool(
                np.array_equal(initial_nn_matrix, final_nn_matrix)
            )
            result["exact_bonds_preserved"] = exact_bonds_preserved
            result["bonds_changed"] = not exact_bonds_preserved
            if not exact_bonds_preserved:
                result["no_changes"] = False

        if check_molecule_count:
            initial_molecule_count = int(
                csgraph.connected_components(initial_nn_matrix)[0]
            )
            final_molecule_count = int(csgraph.connected_components(final_nn_matrix)[0])
            result["initial_molecule_count"] = initial_molecule_count
            result["final_molecule_count"] = final_molecule_count
            molecule_count_preserved = initial_molecule_count == final_molecule_count
            result["molecule_count_preserved"] = molecule_count_preserved
            if not molecule_count_preserved:
                result["no_changes"] = False

        return result

    except Exception as e:
        result.update(
            no_changes=False,
            exact_bonds_preserved=False,
            molecule_count_preserved=False,
            error=str(e),
        )
        return result


def check_no_changes_in_covalent_matrix(
    initial_atoms: Atoms, final_atoms: Atoms
) -> bool:
    """
    Check if covalent bonding network is preserved after relaxation.

    Thin wrapper around ``check_connectivity_changes``; kept for backwards
    compatibility of the public API.
    """
    return check_connectivity_changes(
        initial_atoms,
        final_atoms,
        check_exact_bonds=True,
        check_molecule_count=False,
    )["exact_bonds_preserved"]


def check_no_changes_in_Z(
    initial_atoms: Atoms | None, final_atoms: Atoms | None
) -> bool:
    """
    Check if the number of connected molecular fragments (Z) is preserved
    after relaxation.

    Thin wrapper around ``check_connectivity_changes``.
    """
    return check_connectivity_changes(
        initial_atoms,
        final_atoms,
        check_exact_bonds=False,
        check_molecule_count=True,
    )["molecule_count_preserved"]
