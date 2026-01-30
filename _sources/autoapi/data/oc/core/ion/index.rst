data.oc.core.ion
================

.. py:module:: data.oc.core.ion

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   data.oc.core.ion.Ion


Module Contents
---------------

.. py:class:: Ion(ion_atoms: ase.Atoms = None, ion_id_from_db: int | None = None, ion_db_path: str = ION_PKL_PATH)

   Initializes an ion object in one of 2 ways:
   - Directly pass in an ase.Atoms object.
   - Pass in index of ion to select from ion database.

   :param ion_atoms: ion structure.
   :type ion_atoms: ase.Atoms
   :param ion_id_from_db: Index of ion to select.
   :type ion_id_from_db: int
   :param ion_db_path: Path to ion database.
   :type ion_db_path: str


   .. py:attribute:: ion_id_from_db


   .. py:attribute:: ion_db_path


   .. py:method:: __len__()


   .. py:method:: __str__()


   .. py:method:: _load_ion(ion: dict) -> None

      Saves the fields from an ion stored in a database. Fields added
      after the first revision are conditionally added for backwards
      compatibility with older database files.



   .. py:method:: get_ion_concentration(volume)

      Compute the ion concentration units of M, given a volume in units of
      Angstrom^3.



