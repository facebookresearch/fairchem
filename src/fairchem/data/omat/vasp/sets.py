"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from pymatgen.io.vasp import Kpoints, sets
from pymatgen.io.vasp.sets import VaspInputSet

sets.MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class OMat24StaticSet(VaspInputSet):
    """Create input files for a OMat24 PBE static calculation.
    The default POTCAR versions used are PBE_54

    Args:
        structure (Structure): The Structure to create inputs for. If None, the input
            set is initialized without a Structure but one must be set separately before
            the inputs are generated.
        **kwargs: Keywords supported by VaspInputSet.
    """

    CONFIG = sets._load_yaml_config("OMat24StaticSet")


@dataclass
class OMat24RelaxSet(OMat24StaticSet):
    """Create input files for a OMat24 PBE relaxation calculation.

    Args:
        structure (Structure): The Structure to create inputs for. If None, the input
            set is initialized without a Structure but one must be set separately before
            the inputs are generated.
        **kwargs: Keywords supported by VaspInputSet.
    """

    @property
    def incar_updates(self) -> dict[str, str | int]:
        updates = {
            "IBRION": 2,
            "NSW": 99,
            "ISIF": 3,
        }
        return updates


@dataclass
class OMat24AIMDSet(VaspInputSet):
    """Create input files for a OMat24 PBE static calculation.
    The default POTCAR versions used are PBE_54

    Args:
        structure (Structure): The Structure to create inputs for. If None, the input
            set is initialized without a Structure but one must be set separately before
            the inputs are generated.
        **kwargs: Keywords supported by VaspInputSet.
    """

    start_temperature: float = 1000
    end_temperature: float = 1000
    ensemble: Literal["nvt", "npt"] = "nvt"
    thermostat: Literal["nose", "langevin"] = "nose"
    steps: int = 100
    time_step: float = 2.0
    pressure: float | None = None  # pressure in kB

    @property
    def incar_updates(self) -> dict[str, Any]:
        """Updates to the INCAR config for this calculation type."""

        updates = {
            "TEBEG": self.start_temperature,
            "TEEND": self.end_temperature,
            "NSW": self.steps,
            "POTIM": self.time_step,
            "IBRION": 0,
            "LREAL": True,
            "ISYM": 0,
        }

        if self.thermostat == "langevin":
            updates |= {
                "MDALGO": 3,
                "LANGEVIN_GAMMA": [10] * self.structure.n_elems,
            }
        elif self.thermostat == "nose":
            updates |= {"MDALGO": 2, "SMASS": 0}
        else:
            raise ValueError(f"{self.thermostat} not a valid thermostat.")

        if self.ensemble == "nvt":
            updates |= {"ISIF": 0}
        elif self.ensemble == "npt":
            if self.thermostat != "langevin":
                raise ValueError(
                    "langevin thermostat needs to be used for an npt ensemble."
                )
            updates |= {"ISIF": 3}
            if self.pressure is not None:
                updates |= {"PSTRESS": self.pressure}
        else:
            raise ValueError(f"{self.ensemble} is not a valid ensemble choice!")

        return updates

    @property
    def kpoints_updates(self) -> Kpoints:
        """Updates to the kpoints configuration for this calculation type."""
        return Kpoints.gamma_automatic()
