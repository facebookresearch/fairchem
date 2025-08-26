core.components.calculate.recipes.energy
========================================

.. py:module:: core.components.calculate.recipes.energy

.. autoapi-nested-parse::

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



Attributes
----------

.. autoapisummary::

   core.components.calculate.recipes.energy.pmg_installed


Functions
---------

.. autoapisummary::

   core.components.calculate.recipes.energy.calc_energy_from_e_refs


Module Contents
---------------

.. py:data:: pmg_installed
   :value: True


.. py:function:: calc_energy_from_e_refs(struct_or_entry: pymatgen.util.typing.EntryLike | pymatgen.core.Structure | pymatgen.core.Composition | str, ref_energies: dict[str, float], total_energy: float | None = None) -> float

   Calculate energy per atom relative to reference states (e.g., for formation or
   cohesive energy calculations).

   :param struct_or_entry: Either:
                           - A pymatgen Entry (PDEntry, ComputedEntry, etc.) or entry dict containing
                             'energy' and 'composition' keys
                           - A Structure or Composition object or formula string (must also provide
                             total_energy)
   :type struct_or_entry: EntryLike | Structure | Composition | str
   :param ref_energies: Dictionary of reference energies per atom.
                        For formation energy: elemental reference energies (e.g.
                        mp_elemental_ref_energies).
                        For cohesive energy: isolated atom reference energies
   :type ref_energies: dict[str, float]
   :param total_energy: Total energy of the structure/composition. Required
                        if struct_or_entry is not an Entry or entry dict. Ignored otherwise.
   :type total_energy: float | None

   :returns: Energy per atom relative to references (e.g., formation or cohesive
             energy) in the same units as input energies.
   :rtype: float

   :raises TypeError: If input types are invalid
   :raises ValueError: If missing reference energies for some elements


