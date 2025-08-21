data.omol.orca.calc
===================

.. py:module:: data.omol.orca.calc

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Attributes
----------

.. autoapisummary::

   data.omol.orca.calc.ECP_SIZE
   data.omol.orca.calc.BASIS_DICT
   data.omol.orca.calc.ORCA_FUNCTIONAL
   data.omol.orca.calc.ORCA_BASIS
   data.omol.orca.calc.ORCA_SIMPLE_INPUT
   data.omol.orca.calc.ORCA_BLOCKS
   data.omol.orca.calc.NBO_FLAGS
   data.omol.orca.calc.ORCA_ASE_SIMPLE_INPUT
   data.omol.orca.calc.LOOSE_OPT_PARAMETERS
   data.omol.orca.calc.OPT_PARAMETERS
   data.omol.orca.calc.TIGHT_OPT_PARAMETERS


Classes
-------

.. autoapisummary::

   data.omol.orca.calc.Vertical


Functions
---------

.. autoapisummary::

   data.omol.orca.calc.get_symm_break_block
   data.omol.orca.calc.get_n_basis
   data.omol.orca.calc.get_mem_estimate
   data.omol.orca.calc.write_orca_inputs


Module Contents
---------------

.. py:data:: ECP_SIZE

.. py:data:: BASIS_DICT

.. py:data:: ORCA_FUNCTIONAL
   :value: 'wB97M-V'


.. py:data:: ORCA_BASIS
   :value: 'def2-TZVPD'


.. py:data:: ORCA_SIMPLE_INPUT
   :value: ['EnGrad', 'RIJCOSX', 'def2/J', 'NoUseSym', 'DIIS', 'NOSOSCF', 'NormalConv', 'DEFGRID3', 'ALLPOP']


.. py:data:: ORCA_BLOCKS
   :value: ['%scf Convergence Tight maxiter 300 end', '%elprop Dipole true Quadrupole true end', '%output...


.. py:data:: NBO_FLAGS
   :value: '%nbo NBOKEYLIST = "$NBO NPA NBO E2PERT 0.1 $END" end'


.. py:data:: ORCA_ASE_SIMPLE_INPUT

.. py:data:: LOOSE_OPT_PARAMETERS

.. py:data:: OPT_PARAMETERS

.. py:data:: TIGHT_OPT_PARAMETERS

.. py:class:: Vertical(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   Create a collection of name/value pairs.

   Example enumeration:

   >>> class Color(Enum):
   ...     RED = 1
   ...     BLUE = 2
   ...     GREEN = 3

   Access them by:

   - attribute access:

     >>> Color.RED
     <Color.RED: 1>

   - value lookup:

     >>> Color(1)
     <Color.RED: 1>

   - name lookup:

     >>> Color['RED']
     <Color.RED: 1>

   Enumerations can be iterated over, and know how many members they have:

   >>> len(Color)
   3

   >>> list(Color)
   [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]

   Methods can be added to enumerations, and members can have their own
   attributes -- see the documentation for details.


   .. py:attribute:: Default
      :value: 'default'



   .. py:attribute:: MetalOrganics
      :value: 'metal-organics'



   .. py:attribute:: Oss
      :value: 'open-shell-singlet'



.. py:function:: get_symm_break_block(atoms: ase.Atoms, charge: int) -> str

   Determine the ORCA Rotate block needed to break symmetry in a singlet

   This is determined by taking the sum of atomic numbers less any charge (because
   electrons are negatively charged) and removing any electrons that are in an ECP
   and dividing by 2. This gives the number of occupied orbitals, but since ORCA is
   0-indexed, it gives the index of the LUMO.

   We use a rotation angle of 20 degrees or about a 12% mixture of LUMO into HOMO.
   This is somewhat arbitrary but similar to the default setting in Q-Chem, and seemed
   to perform well in tests of open-shell singlets.


.. py:function:: get_n_basis(atoms: ase.Atoms) -> int

   Get the number of basis functions that will be used for the given input.

   We assume our basis is def2-tzvpd. The number of basis functions is used
   to estimate the memory requirments of a given job.

   :param atoms: atoms to compute the number of basis functions of
   :return: number of basis functions as printed by Orca


.. py:function:: get_mem_estimate(atoms: ase.Atoms, vertical: enum.Enum = Vertical.Default, mult: int = 1) -> int

   Get an estimate of the memory requirement for given input in MB.

   If the estimate is less than 1000MB, we return 1000MB.

   :param atoms: atoms to compute the number of basis functions of
   :param vertical: Which vertical this is for (all metal-organics are
                    UKS, as are all regular open-shell calcs)
   :param mult: spin multiplicity of input
   :return: estimated (upper-bound) to the memory requirement of this Orca job


.. py:function:: write_orca_inputs(atoms: ase.Atoms, output_directory, charge: int = 0, mult: int = 1, nbo: bool = True, orcasimpleinput: str = ORCA_ASE_SIMPLE_INPUT, orcablocks: str = ' '.join(ORCA_BLOCKS), vertical: enum.Enum = Vertical.Default, scf_MaxIter: int = None)

   One-off method to be used if you wanted to write inputs for an arbitrary
   system. Primarily used for debugging.


