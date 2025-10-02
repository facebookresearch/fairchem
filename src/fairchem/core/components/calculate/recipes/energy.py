"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Original source: https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/energy.py

MIT License

Copyright (c) 2022 Janosh Riebesell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

The software is provided "as is", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose and noninfringement. In no event shall the
authors or copyright holders be liable for any claim, damages or other
liability, whether in an action of contract, tort or otherwise, arising from,
out of or in connection with the software or the use or other dealings in the
software.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monty.dev import requires

try:
    from pymatgen.analysis.phase_diagram import Entry
    from pymatgen.core import Composition, Structure

    pmg_installed = True
except ImportError:
    Entry, Composition, Structure = None, None, None
    pmg_installed = False


if TYPE_CHECKING:
    from pymatgen.util.typing import EntryLike


@requires(pmg_installed, message="Requires `pymatgen` to be installed")
def calc_energy_from_e_refs(
    struct_or_entry: EntryLike | Structure | Composition | str,
    ref_energies: dict[str, float],
    total_energy: float | None = None,
) -> float:
    """Calculate energy per atom relative to reference states (e.g., for formation or
    cohesive energy calculations).

    Args:
        struct_or_entry (EntryLike | Structure | Composition | str): Either:
            - A pymatgen Entry (PDEntry, ComputedEntry, etc.) or entry dict containing
              'energy' and 'composition' keys
            - A Structure or Composition object or formula string (must also provide
              total_energy)
        ref_energies (dict[str, float]): Dictionary of reference energies per atom.
            For formation energy: elemental reference energies (e.g.
            mp_elemental_ref_energies).
            For cohesive energy: isolated atom reference energies
        total_energy (float | None): Total energy of the structure/composition. Required
            if struct_or_entry is not an Entry or entry dict. Ignored otherwise.

    Returns:
        float: Energy per atom relative to references (e.g., formation or cohesive
        energy) in the same units as input energies.

    Raises:
        TypeError: If input types are invalid
        ValueError: If missing reference energies for some elements
    """
    if isinstance(struct_or_entry, dict):  # entry dict case
        energy = struct_or_entry["energy"]
        comp = Composition(struct_or_entry["composition"])
    # Entry/ComputedEntry/ComputedStructureEntry instance case
    elif isinstance(struct_or_entry, Entry):
        energy = struct_or_entry.energy
        comp = struct_or_entry.composition
    else:  # Structure/Composition/formula case
        if total_energy is None:
            raise ValueError("total_energy can't be None when 1st arg is not an Entry")
        energy = total_energy

        if isinstance(struct_or_entry, str):
            comp = Composition(struct_or_entry)
        elif isinstance(struct_or_entry, Structure):
            comp = struct_or_entry.composition
        elif isinstance(struct_or_entry, Composition):
            comp = struct_or_entry
        else:
            cls_name = type(struct_or_entry).__name__
            raise TypeError(
                "Expected Entry, Structure, Composition or formula string, "
                f"got {cls_name}"
            )

    # Check that we have all needed reference energies
    if missing_refs := set(map(str, comp)) - set(ref_energies):
        raise ValueError(f"Missing reference energies for elements: {missing_refs}")

    # Calculate reference energy
    e_ref = sum(ref_energies[str(el)] * amt for el, amt in comp.items())

    return (energy - e_ref) / comp.num_atoms
