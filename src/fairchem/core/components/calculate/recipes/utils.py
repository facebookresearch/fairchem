"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ase.calculators.calculator import PropertyNotPresent

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase import Atoms
    from numpy.typing import ArrayLike


def get_property_from_atoms(
    atoms: Atoms, property_name: str
) -> int | float | ArrayLike:
    """Retrieve a property from an Atoms object, either from its properties or info dictionary.

    Args:
        atoms: The ASE Atoms object to extract properties from
        property_name: Name of the property to retrieve

    Returns:
        The property value as an integer, float, or array-like object

    Raises:
        ValueError: If the property is not found in either the properties or info dictionary
    """
    try:
        # get_properties returns a Properties dict-like object, so we index again for the property requested
        prop = atoms.get_properties([property_name])[property_name]
    except (PropertyNotPresent, ValueError, RuntimeError):
        try:
            prop = atoms.info[property_name]
        except KeyError as err:
            raise ValueError(
                f"The listed property {property_name} in `save_target_properties` is not available from"
                f" the atoms object or its info dictionary"
            ) from err
    return prop


def normalize_property(
    property_value: float | ArrayLike, atoms: Atoms, normalize_by: str
):
    """Normalize a property value by either the number of atoms or another property.

    Args:
        property_value: The property value to normalize
        atoms: The ASE Atoms object containing the normalization information
        normalize_by: Normalization method, either "natoms" to divide by number of atoms
            or a property name to divide by that property's value

    Returns: The normalized property value
    """
    if normalize_by == "natoms":
        return property_value / len(atoms)
    else:
        norm_prop = get_property_from_atoms(atoms, normalize_by)
        return property_value / norm_prop


def get_property_dict_from_atoms(
    properties: Sequence[str], atoms: Atoms, normalize_by: dict[str, str] | None = None
) -> dict[str, float | ArrayLike]:
    """Get a sequence of properties from an atoms object and return a dict.

    Args:
        properties: Sequence of property names to retrieve from the atoms object
        atoms: The ASE Atoms object to extract properties from
        normalize_by: Dictionary mapping property names to normalization methods

    Returns:
        Dictionary containing the requested properties as keys and
            normalized properties if specified in normalize_by
    """
    normalize_by = normalize_by or {}
    results = {}
    for property_name in properties:
        results[property_name] = get_property_from_atoms(atoms, property_name)
        if property_name in normalize_by:
            norm_by = normalize_by[property_name]
            key = (
                f"{property_name}_per_atom"
                if norm_by == "natoms"
                else f"{property_name}_per_{norm_by}"
            )
            results[key] = normalize_property(results[property_name], atoms, norm_by)
    return results
