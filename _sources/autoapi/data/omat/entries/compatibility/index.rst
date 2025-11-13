data.omat.entries.compatibility
===============================

.. py:module:: data.omat.entries.compatibility

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   data.omat.entries.compatibility.OMAT24_CONFIG_FILE


Classes
-------

.. autoapisummary::

   data.omat.entries.compatibility.OMat24Compatibility


Functions
---------

.. autoapisummary::

   data.omat.entries.compatibility.generate_cse_parameters
   data.omat.entries.compatibility.generate_computed_structure_entry
   data.omat.entries.compatibility.apply_mp_style_corrections


Module Contents
---------------

.. py:data:: OMAT24_CONFIG_FILE

.. py:class:: OMat24Compatibility(compat_type: Literal['GGA', 'Advanced'] = 'Advanced', correct_peroxide: bool = True, strict_anions: Literal['require_exact', 'require_bound', 'no_check'] = 'require_bound', check_potcar: bool = True, check_potcar_hash: bool = False, config_file: str | None = None)

   Bases: :py:obj:`pymatgen.entries.compatibility.MaterialsProject2020Compatibility`


   Exact same as MaterialsProject2020Compatibility but with different defaults.

   See documentation of MaterialsProject2020Compatibility for more details:
       https://pymatgen.org/pymatgen.entries.html


.. py:function:: generate_cse_parameters(input_set: pymatgen.io.vasp.sets.VaspInputSet) -> dict

   Generate parameters for a ComputedStructureEntry from a VASP input set in order


.. py:function:: generate_computed_structure_entry(structure: Structure, total_energy: float, correction_type: Literal['MP2020', 'OMat24'] = 'OMat24', check_potcar: bool = True) -> pymatgen.entries.computed_entries.ComputedStructureEntry

.. py:function:: apply_mp_style_corrections(energy: float, atoms: Atoms, correction_type: Literal['MP2020', 'OMat24'] = 'OMat24', check_potcar: bool = False) -> float

   Applies Materials Project style energy corrections to an ASE Atoms object

   :param energy: The uncorrected energy to be corrected.
   :param atoms: ASE Atoms object for which to apply the corrections.
   :param correction_type: Type of corrections to apply: MP2020 or OMat24.
   :param check_potcar: Whether to check POTCAR consistency when applying corrections.

   :returns: Corrected energy.


