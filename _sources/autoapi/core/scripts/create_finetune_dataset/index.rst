core.scripts.create_finetune_dataset
====================================

.. py:module:: core.scripts.create_finetune_dataset


Attributes
----------

.. autoapisummary::

   core.scripts.create_finetune_dataset.parser


Functions
---------

.. autoapisummary::

   core.scripts.create_finetune_dataset.compute_normalizer_and_linear_reference
   core.scripts.create_finetune_dataset.extract_energy_and_forces
   core.scripts.create_finetune_dataset.compute_lin_ref
   core.scripts.create_finetune_dataset.write_ase_db
   core.scripts.create_finetune_dataset.launch_processing


Module Contents
---------------

.. py:function:: compute_normalizer_and_linear_reference(train_path, num_workers)

   Given a path to an ASE database file, compute the normalizer value and linear
   reference coefficients. These are used to normalize energies and forces during
   training. For large datasets, compute this for only a subset of the data.


.. py:function:: extract_energy_and_forces(idx)

   Extract energy and forces from an ASE atoms object at a given index in the dataset.


.. py:function:: compute_lin_ref(atomic_numbers, energies)

   Compute linear reference coefficients given atomic numbers and energies.


.. py:function:: write_ase_db(mp_arg)

   Write ASE atoms objects to an ASE database file. This function is designed to be
   run in parallel using multiprocessing.


.. py:function:: launch_processing(data_dir, output_dir, num_workers)

   Driver script to launch processing of ASE atoms files into an ASE database.


.. py:data:: parser

