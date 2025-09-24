core.datasets.atoms_sequence
============================

.. py:module:: core.datasets.atoms_sequence

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.datasets.atoms_sequence.AtomsSequence
   core.datasets.atoms_sequence.AtomsDatasetSequence


Module Contents
---------------

.. py:class:: AtomsSequence

   Bases: :py:obj:`Protocol`


   Base class for protocol classes.

   Protocol classes are defined as::

       class Proto(Protocol):
           def meth(self) -> int:
               ...

   Such classes are primarily used with static type checkers that recognize
   structural subtyping (static duck-typing).

   For example::

       class C:
           def meth(self) -> int:
               return 0

       def func(x: Proto) -> int:
           return x.meth()

       func(C())  # Passes static type check

   See PEP 544 for details. Protocol classes decorated with
   @typing.runtime_checkable act as simple-minded runtime protocols that check
   only the presence of given attributes, ignoring their type signatures.
   Protocol classes can be generic, they are defined as::

       class GenProto[T](Protocol):
           def meth(self) -> T:
               ...


   .. py:method:: __getitem__(index: int) -> ase.Atoms
                  __getitem__(index: slice) -> AtomsSequence


   .. py:method:: __len__() -> int


.. py:class:: AtomsDatasetSequence(dataset: fairchem.core.datasets.ase_datasets.AseAtomsDataset)

   Turn an AseAtomsDataset into an AtomsSequence that iterates over atoms objects.


   .. py:attribute:: dataset


   .. py:method:: __getitem__(index: int | slice) -> ase.Atoms | AtomsSequence


   .. py:method:: __len__() -> int


