core.components.calculate.recipes.utils
=======================================

.. py:module:: core.components.calculate.recipes.utils

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Functions
---------

.. autoapisummary::

   core.components.calculate.recipes.utils.get_property_from_atoms
   core.components.calculate.recipes.utils.normalize_property
   core.components.calculate.recipes.utils.get_property_dict_from_atoms


Module Contents
---------------

.. py:function:: get_property_from_atoms(atoms: ase.Atoms, property_name: str) -> int | float | numpy.typing.ArrayLike

   Retrieve a property from an Atoms object, either from its properties or info dictionary.

   :param atoms: The ASE Atoms object to extract properties from
   :param property_name: Name of the property to retrieve

   :returns: The property value as an integer, float, or array-like object

   :raises ValueError: If the property is not found in either the properties or info dictionary


.. py:function:: normalize_property(property_value: float | numpy.typing.ArrayLike, atoms: ase.Atoms, normalize_by: str)

   Normalize a property value by either the number of atoms or another property.

   :param property_value: The property value to normalize
   :param atoms: The ASE Atoms object containing the normalization information
   :param normalize_by: Normalization method, either "natoms" to divide by number of atoms
                        or a property name to divide by that property's value

   Returns: The normalized property value


.. py:function:: get_property_dict_from_atoms(properties: collections.abc.Sequence[str], atoms: ase.Atoms, normalize_by: dict[str, str] | None = None) -> dict[str, float | numpy.typing.ArrayLike]

   Get a sequence of properties from an atoms object and return a dict.

   :param properties: Sequence of property names to retrieve from the atoms object
   :param atoms: The ASE Atoms object to extract properties from
   :param normalize_by: Dictionary mapping property names to normalization methods

   :returns:

             Dictionary containing the requested properties as keys and
                 normalized properties if specified in normalize_by


