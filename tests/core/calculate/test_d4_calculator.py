"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from ase.build import bulk, molecule
from ase.calculators.emt import EMT


def _dftd4_available() -> bool:
    """
    Check if the dftd4 package is importable.
    """
    try:
        import dftd4  # noqa: F401

        return True
    except ImportError:
        return False


requires_dftd4 = pytest.mark.skipif(
    not _dftd4_available(), reason="dftd4 not installed"
)


@pytest.fixture
def h2o_atoms():
    """
    Water molecule for testing.
    """
    atoms = molecule("H2O")
    atoms.center(vacuum=5.0)
    return atoms


@pytest.fixture
def cu_bulk():
    """
    Cu bulk with PBC for periodic testing.
    """
    return bulk("Cu", "fcc", a=3.6)


@requires_dftd4
class TestD4CorrectedCalculatorInit:
    """
    Tests for D4CorrectedCalculator initialization.
    """

    def test_default_params_three_body_only(self):
        """
        Default initialization uses wB97M-D4 params with three_body_only.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        base = EMT()
        calc = D4CorrectedCalculator(base)
        assert calc.three_body_only is True
        assert calc.calculator is base

    def test_method_and_params_mutually_exclusive(self):
        """
        Providing both method and damping_params raises ValueError.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        with pytest.raises(ValueError, match="mutually exclusive"):
            D4CorrectedCalculator(
                EMT(),
                method="pbe",
                damping_params={"s8": 1.0, "a1": 0.4, "a2": 5.0},
            )

    def test_inherited_properties(self):
        """
        Implemented properties are inherited from the base calculator.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        base = EMT()
        calc = D4CorrectedCalculator(base)
        assert calc.implemented_properties == list(
            base.implemented_properties
        )

    def test_method_param_loads(self):
        """
        Passing a method name successfully creates damping parameters.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        calc = D4CorrectedCalculator(EMT(), method="pbe")
        assert calc._dpar is not None

    def test_custom_damping_params(self):
        """
        Custom damping parameters are accepted.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        params = {"s6": 1.0, "s8": 0.5, "a1": 0.4, "a2": 5.0}
        calc = D4CorrectedCalculator(
            EMT(), damping_params=params, three_body_only=False
        )
        assert calc.three_body_only is False
        assert calc._dpar is not None

    def test_import_error_without_dftd4(self):
        """
        ImportError is raised with a helpful message when dftd4 is missing.

        Note: This test only runs when dftd4 IS available since the class
        is already importable. The import check is in __init__, so we'd
        need to test it differently. Keeping as a placeholder.
        """


@requires_dftd4
class TestD4CorrectedCalculatorMolecular:
    """
    Tests for D4 correction on molecular (non-periodic) systems.
    """

    def test_energy_correction(self, h2o_atoms):
        """
        D4 ATM correction modifies the energy.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        base = EMT()
        h2o_atoms.calc = base
        base_energy = h2o_atoms.get_potential_energy()

        calc = D4CorrectedCalculator(base)
        h2o_atoms.calc = calc
        corrected_energy = h2o_atoms.get_potential_energy()

        assert corrected_energy != base_energy

    def test_forces_correction(self, h2o_atoms):
        """
        D4 ATM correction modifies forces.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        base = EMT()
        h2o_atoms.calc = base
        base_forces = h2o_atoms.get_forces().copy()

        calc = D4CorrectedCalculator(base)
        h2o_atoms.calc = calc
        corrected_forces = h2o_atoms.get_forces()

        assert corrected_forces.shape == base_forces.shape
        assert not np.allclose(corrected_forces, base_forces)

    def test_free_energy_matches_energy(self, h2o_atoms):
        """
        free_energy should equal energy after correction.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        calc = D4CorrectedCalculator(EMT())
        h2o_atoms.calc = calc
        energy = h2o_atoms.get_potential_energy()
        free_energy = calc.results["free_energy"]
        npt.assert_almost_equal(energy, free_energy)

    def test_no_stress_for_molecules(self, h2o_atoms):
        """
        Stress should not appear for non-periodic molecules.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        calc = D4CorrectedCalculator(EMT())
        h2o_atoms.calc = calc
        h2o_atoms.get_potential_energy()
        assert "stress" not in calc.results

    def test_charge_forwarded(self, h2o_atoms):
        """
        Charge from atoms.info is forwarded to the D4 model.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        h2o_atoms.info["charge"] = -1

        calc = D4CorrectedCalculator(EMT())
        h2o_atoms.calc = calc
        # Should not raise — charge is passed through
        h2o_atoms.get_potential_energy()


@requires_dftd4
class TestD4CorrectedCalculatorPeriodic:
    """
    Tests for D4 correction on periodic systems.
    """

    def test_energy_correction(self, cu_bulk):
        """
        D4 correction modifies energy in periodic systems.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        base = EMT()
        cu_bulk.calc = base
        base_energy = cu_bulk.get_potential_energy()

        calc = D4CorrectedCalculator(base, three_body_only=False)
        cu_bulk.calc = calc
        corrected_energy = cu_bulk.get_potential_energy()

        assert corrected_energy != base_energy

    def test_forces_correction(self, cu_bulk):
        """
        D4 correction modifies forces in periodic systems.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        base = EMT()
        cu_bulk.calc = base
        base_forces = cu_bulk.get_forces().copy()

        calc = D4CorrectedCalculator(base, three_body_only=False)
        cu_bulk.calc = calc
        corrected_forces = cu_bulk.get_forces()

        assert not np.allclose(corrected_forces, base_forces)

    def test_stress_correction(self, cu_bulk):
        """
        D4 correction modifies stress in periodic systems.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        base = EMT()
        cu_bulk.calc = base
        base_stress = cu_bulk.get_stress().copy()

        calc = D4CorrectedCalculator(base, three_body_only=False)
        cu_bulk.calc = calc
        corrected_stress = cu_bulk.get_stress()

        assert corrected_stress.shape == (6,)
        assert not np.allclose(corrected_stress, base_stress)


@requires_dftd4
class TestD4CorrectedCalculatorPhysics:
    """
    Tests for physical correctness of D4 corrections.
    """

    def test_three_body_smaller_than_full(self, h2o_atoms):
        """
        Three-body-only correction magnitude should be smaller than full D4.
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        base = EMT()
        h2o_atoms.calc = base
        base_energy = h2o_atoms.get_potential_energy()

        full_calc = D4CorrectedCalculator(base, three_body_only=False)
        h2o_atoms.calc = full_calc
        full_energy = h2o_atoms.get_potential_energy()

        atm_calc = D4CorrectedCalculator(base, three_body_only=True)
        h2o_atoms.calc = atm_calc
        atm_energy = h2o_atoms.get_potential_energy()

        full_corr = abs(full_energy - base_energy)
        atm_corr = abs(atm_energy - base_energy)
        assert atm_corr < full_corr

    def test_unit_conversion_consistency(self, h2o_atoms):
        """
        Verify D4 corrections use correct unit conversions.

        The D4 ASE calculator should give the same result as our wrapper
        when both use the full D4 correction.
        """
        from dftd4.ase import DFTD4

        from fairchem.core.calculate.d4_calculator import (
            WB97M_D4_PARAMS,
            D4CorrectedCalculator,
        )

        # Pure D4 via dftd4's own ASE calculator
        d4_ase = DFTD4(
            method="wb97m",
        )
        h2o_atoms.calc = d4_ase
        d4_energy = h2o_atoms.get_potential_energy()
        d4_forces = h2o_atoms.get_forces().copy()

        # D4 via our wrapper with a zero-energy base calculator
        zero_calc = _ZeroCalculator()
        d4_wrapped = D4CorrectedCalculator(
            zero_calc,
            damping_params=WB97M_D4_PARAMS,
            three_body_only=False,
        )
        h2o_atoms.calc = d4_wrapped
        wrapped_energy = h2o_atoms.get_potential_energy()
        wrapped_forces = h2o_atoms.get_forces()

        npt.assert_almost_equal(wrapped_energy, d4_energy, decimal=10)
        npt.assert_array_almost_equal(wrapped_forces, d4_forces, decimal=10)

    def test_larger_system_has_atm_contribution(self):
        """
        ATM three-body dispersion should be non-negligible for systems
        with many close neighbors (e.g., benzene).
        """
        from fairchem.core.calculate.d4_calculator import D4CorrectedCalculator

        benzene = molecule("C6H6")
        benzene.center(vacuum=5.0)

        base = EMT()
        benzene.calc = base
        base_energy = benzene.get_potential_energy()

        calc = D4CorrectedCalculator(base, three_body_only=True)
        benzene.calc = calc
        atm_energy = benzene.get_potential_energy()

        # ATM correction should be non-zero for benzene
        assert atm_energy != base_energy


class _ZeroCalculator(EMT):
    """
    Calculator that returns zero energy and forces.

    Used for unit conversion consistency tests.
    """

    def calculate(self, atoms, properties, system_changes):
        """
        Return zero energy, forces, and stress.
        """
        from ase.calculators.calculator import Calculator

        Calculator.calculate(self, atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["free_energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))
        if atoms.pbc.any():
            self.results["stress"] = np.zeros(6)
