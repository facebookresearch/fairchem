"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress
from ase.units import Bohr, Hartree

if TYPE_CHECKING:
    from typing import Literal

    from ase import Atoms

    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

logger = logging.getLogger(__name__)


# wB97M-D4 BJ-EEQ-ATM damping parameters from the dftd4 parameter database.
# Reference: Caldeweyher et al., J. Comput. Chem. 2020, DOI: 10.1002/jcc.26411
WB97M_D4_PARAMS: dict[str, float] = {
    "s6": 1.0,
    "s8": 0.7761,
    "a1": 0.7514,
    "a2": 2.7099,
}


class D4CorrectedCalculator(Calculator):
    """
    Calculator wrapper that adds DFT-D4 dispersion correction to a base calculator.

    By default, adds only the three-body Axilrod-Teller-Muto (ATM) term
    using ωB97M-D4 damping parameters. This is intended for correcting UMA
    ``omol`` predictions, which are trained on ωB97M-V reference data that
    captures two-body dispersion via VV10 non-local correlation but lacks
    the three-body dispersion contribution.

    The ATM three-body energy uses C9 coefficients derived from the D4
    charge-dependent C6 coefficients (C9_ABC ≈ -√(C6_AB·C6_BC·C6_AC)).
    The ``a1`` and ``a2`` BJ-damping parameters enter the ATM damping
    function through the van der Waals radii R0_ij = a1·√(C8/C6) + a2,
    so they are preserved even when the two-body terms are zeroed out.

    Setting ``three_body_only=False`` adds the full D4 correction
    (two-body + three-body).

    Requires the ``dftd4`` Python package::

        pip install dftd4

    Usage notes:
        The default ``three_body_only=True`` mode is designed for UMA
        ``omol`` predictions. The ωB97M-V functional used to generate
        the omol training data includes VV10 non-local correlation,
        which captures two-body dispersion well but lacks three-body
        contributions. The D4 ATM term fills that gap. By zeroing
        ``s6`` and ``s8``, we avoid double-counting the two-body
        dispersion the model already learned.

        For other UMA tasks (``omat``, ``oc20``) whose reference DFT
        does not include dispersion, you may want
        ``three_body_only=False`` to add the full D4 correction, or
        use a different ``method`` / ``damping_params`` matching the
        target level of theory. For tasks whose training data already
        includes D3 (``omc``, ``odac``, ``oc25``), adding D4 on top
        would double-count and is not recommended without first
        understanding the interaction between D3 and D4 parameters.

    Example::

        from fairchem.core import FAIRChemCalculator
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        base_calc = FAIRChemCalculator.from_model_checkpoint(
            "uma-s-1p1", task_name="omol"
        )
        calc = D4CorrectedCalculator(base_calc)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
    """

    def __init__(
        self,
        calculator: Calculator,
        method: str | None = None,
        damping_params: dict[str, float] | None = None,
        three_body_only: bool = True,
    ):
        """
        Initialize the D4-corrected calculator.

        Args:
            calculator: Base ASE calculator to wrap.
            method: DFT-D4 method name (e.g., ``"wb97m"``, ``"pbe"``).
                If provided, damping parameters are loaded from the dftd4
                library. Mutually exclusive with ``damping_params``.
            damping_params: Custom damping parameter dict with keys
                ``s6``, ``s8``, ``s9``, ``a1``, ``a2``, ``alp``.
                Defaults to ωB97M-D4 parameters if neither ``method``
                nor ``damping_params`` is provided.
            three_body_only: If True (default), only the three-body ATM
                term is added by zeroing ``s6`` and ``s8``. The ``a1``
                and ``a2`` parameters are preserved since they enter the
                ATM damping function. If False, the full D4 correction
                (two-body + three-body) is applied.
        """
        super().__init__()

        try:
            from dftd4.interface import DampingParam
        except ImportError as err:
            raise ImportError(
                "The dftd4 package is required for D4 dispersion corrections. "
                "Install it with: pip install dftd4"
            ) from err

        if method is not None and damping_params is not None:
            raise ValueError(
                "method and damping_params are mutually exclusive. "
                "Provide one or the other, not both."
            )

        self.calculator = calculator
        self.three_body_only = three_body_only

        if method is not None:
            if three_body_only:
                from dftd4.parameters import get_damping_param

                params = get_damping_param(method)
                self._dpar = DampingParam(
                    s6=0.0,
                    s8=0.0,
                    s9=params.get("s9", 1.0),
                    a1=params["a1"],
                    a2=params["a2"],
                )
            else:
                self._dpar = DampingParam(method=method)
        else:
            params = (
                dict(damping_params)
                if damping_params is not None
                else dict(WB97M_D4_PARAMS)
            )
            if three_body_only:
                params["s6"] = 0.0
                params["s8"] = 0.0
                params.setdefault("s9", 1.0)
            self._dpar = DampingParam(**params)

        if hasattr(calculator, "implemented_properties"):
            self.implemented_properties = list(calculator.implemented_properties)

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list:
        """
        Check for system changes, delegating to the base calculator.

        This ensures that changes detected by the base calculator (e.g.,
        ``atoms.info`` changes for charge/spin in FAIRChemCalculator)
        also trigger recalculation of the D4 correction.

        Args:
            atoms: The atomic structure to check.
            tol: Tolerance for detecting changes.

        Returns:
            A list of detected changes.
        """
        if hasattr(self.calculator, "check_state"):
            return self.calculator.check_state(atoms, tol=tol)
        return super().check_state(atoms, tol=tol)

    def calculate(
        self,
        atoms: Atoms,
        properties: list[str],
        system_changes: list[str],
    ) -> None:
        """
        Calculate properties with D4 dispersion correction applied.

        The base calculator is run first, then the D4 three-body ATM
        correction (or full D4 if ``three_body_only=False``) is added
        to the energy, forces, and stress.

        Args:
            atoms: The atomic structure to calculate properties for.
            properties: The list of properties to calculate.
            system_changes: The list of changes in the system.
        """
        from dftd4.interface import DispersionModel

        Calculator.calculate(self, atoms, properties, system_changes)

        # Run the base calculator
        self.calculator.calculate(atoms, properties, system_changes)
        self.results = dict(self.calculator.results)

        # Build D4 dispersion model (dftd4 uses atomic units / Bohr)
        disp_kwargs: dict = {
            "numbers": atoms.numbers,
            "positions": atoms.positions / Bohr,
        }
        if atoms.pbc.any():
            disp_kwargs["lattice"] = atoms.cell.array / Bohr
            disp_kwargs["periodic"] = atoms.pbc
        if hasattr(atoms, "info") and "charge" in atoms.info:
            disp_kwargs["charge"] = float(atoms.info["charge"])

        disp = DispersionModel(**disp_kwargs)
        res = disp.get_dispersion(param=self._dpar, grad=True)

        # Add energy correction (Hartree -> eV)
        if "energy" in self.results:
            energy_corr = float(res["energy"]) * Hartree
            self.results["energy"] += energy_corr
            if "free_energy" in self.results:
                self.results["free_energy"] += energy_corr

        # Add force correction (negative gradient; Hartree/Bohr -> eV/Å)
        if "forces" in self.results:
            force_corr = -np.asarray(res["gradient"]) * Hartree / Bohr
            self.results["forces"] = self.results["forces"] + force_corr

        # Add stress correction for periodic systems (virial -> Voigt stress)
        if "stress" in self.results and atoms.pbc.any():
            virial = np.asarray(res["virial"]) * Hartree
            stress_corr_3x3 = virial / atoms.get_volume()
            stress_corr_voigt = full_3x3_to_voigt_6_stress(stress_corr_3x3)
            self.results["stress"] = self.results["stress"] + stress_corr_voigt

    @classmethod
    def from_model_checkpoint(
        cls,
        name_or_path: str,
        task_name: str | None = "omol",
        method: str | None = None,
        damping_params: dict[str, float] | None = None,
        three_body_only: bool = True,
        inference_settings: InferenceSettings | str = "default",
        device: Literal["cuda", "cpu"] | None = None,
    ) -> D4CorrectedCalculator:
        """
        Create a D4CorrectedCalculator from a FAIRChem model checkpoint.

        Convenience classmethod that instantiates the base
        ``FAIRChemCalculator`` and wraps it with D4 dispersion corrections.

        Args:
            name_or_path: Model name or path (passed to
                ``FAIRChemCalculator.from_model_checkpoint``).
            task_name: Task name for the UMA model. Defaults to ``"omol"``.
            method: DFT-D4 method name for damping parameters.
            damping_params: Custom damping parameters dict.
            three_body_only: If True (default), only add three-body ATM term.
            inference_settings: Inference settings for the base calculator.
            device: Device for the base calculator.

        Returns:
            A D4CorrectedCalculator wrapping a FAIRChemCalculator.
        """
        from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

        base_calc = FAIRChemCalculator.from_model_checkpoint(
            name_or_path,
            task_name=task_name,
            inference_settings=inference_settings,
            device=device,
        )
        return cls(
            calculator=base_calc,
            method=method,
            damping_params=damping_params,
            three_body_only=three_body_only,
        )
