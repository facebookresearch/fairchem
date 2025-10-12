data.omat.vasp.sets
===================

.. py:module:: data.omat.vasp.sets

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   data.omat.vasp.sets.OMat24StaticSet
   data.omat.vasp.sets.OMat24RelaxSet
   data.omat.vasp.sets.OMat24AIMDSet


Module Contents
---------------

.. py:class:: OMat24StaticSet

   Bases: :py:obj:`pymatgen.io.vasp.sets.VaspInputSet`


   Create input files for a OMat24 PBE static calculation.
   The default POTCAR versions used are PBE_54

   :param structure: The Structure to create inputs for. If None, the input
                     set is initialized without a Structure but one must be set separately before
                     the inputs are generated.
   :type structure: Structure
   :param \*\*kwargs: Keywords supported by VaspInputSet.


   .. py:attribute:: CONFIG


.. py:class:: OMat24RelaxSet

   Bases: :py:obj:`OMat24StaticSet`


   Create input files for a OMat24 PBE relaxation calculation.

   :param structure: The Structure to create inputs for. If None, the input
                     set is initialized without a Structure but one must be set separately before
                     the inputs are generated.
   :type structure: Structure
   :param \*\*kwargs: Keywords supported by VaspInputSet.


   .. py:property:: incar_updates
      :type: dict[str, str | int]


      Updates to the INCAR config for this calculation type.


.. py:class:: OMat24AIMDSet

   Bases: :py:obj:`pymatgen.io.vasp.sets.VaspInputSet`


   Create input files for a OMat24 PBE static calculation.
   The default POTCAR versions used are PBE_54

   :param structure: The Structure to create inputs for. If None, the input
                     set is initialized without a Structure but one must be set separately before
                     the inputs are generated.
   :type structure: Structure
   :param \*\*kwargs: Keywords supported by VaspInputSet.


   .. py:attribute:: start_temperature
      :type:  float
      :value: 1000



   .. py:attribute:: end_temperature
      :type:  float
      :value: 1000



   .. py:attribute:: ensemble
      :type:  Literal['nvt', 'npt']
      :value: 'nvt'



   .. py:attribute:: thermostat
      :type:  Literal['nose', 'langevin']
      :value: 'nose'



   .. py:attribute:: steps
      :type:  int
      :value: 100



   .. py:attribute:: time_step
      :type:  float
      :value: 2.0



   .. py:attribute:: pressure
      :type:  float | None
      :value: None



   .. py:property:: incar_updates
      :type: dict[str, Any]


      Updates to the INCAR config for this calculation type.


   .. py:property:: kpoints_updates
      :type: pymatgen.io.vasp.Kpoints


      Updates to the kpoints configuration for this calculation type.


