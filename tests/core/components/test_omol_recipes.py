"""
Unit tests for the OMol evaluation recipes.

These tests verify the functionality of molecular property evaluation functions
used for assessing machine learning interatomic potentials against DFT reference data.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ase import Atoms
from ase.optimize import BFGS

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.components.calculate.recipes.omol import (
    conformers,
    distance_scaling,
    ieea,
    ligand_pocket,
    ligand_strain,
    protonation,
    relax_job,
    single_point_job,
    singlepoint,
    spin_gap,
)


class TestOmolRecipes(unittest.TestCase):
    """Test suite for OMol calculation recipes."""

    def setUp(self):
        """Set up common test fixtures."""
        # Create real ASE Atoms objects for testing
        # Simple water molecule
        self.water_atoms = Atoms(
            symbols=["O", "H", "H"],
            positions=[[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]],
        )

        # Simple methane molecule
        self.methane_atoms = Atoms(
            symbols=["C", "H", "H", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [-0.36, 1.03, 0.0],
                [-0.36, -0.51, 0.89],
                [-0.36, -0.51, -0.89],
            ],
        )

        # Use water as the main test atoms object
        self.test_atoms = self.water_atoms.copy()

        # Real ASE Calculator using FAIRChem
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
        self.calculator = FAIRChemCalculator(predictor, task_name="omol")

        # Mock optimization flags
        self.opt_flags = {
            "optimizer": BFGS,
            "optimizer_kwargs": {},
            "fmax": 0.05,
            "max_steps": 10,
        }

    @patch("fairchem.core.components.calculate.recipes.omol.MSONAtoms")
    def test_relax_job_success(self, mock_mson_atoms):
        """Test successful geometry optimization."""
        # Setup
        mock_mson_atoms.return_value.as_dict.return_value = {"test": "atoms"}

        # Execute with real calculator and optimizer
        result = relax_job(self.water_atoms.copy(), self.calculator, self.opt_flags)

        # Verify structure of result
        assert "initial" in result
        assert "final" in result
        assert "energy" in result["initial"]
        assert "energy" in result["final"]
        assert "forces" in result["initial"]
        assert "forces" in result["final"]
        assert "atoms" in result["initial"]
        assert "atoms" in result["final"]

        assert isinstance(result["initial"]["energy"], (int, float))
        assert isinstance(result["final"]["energy"], (int, float))
        assert isinstance(result["initial"]["forces"], list)
        assert isinstance(result["final"]["forces"], list)

        assert len(result["initial"]["forces"]) == 3  # water has 3 atoms
        assert len(result["final"]["forces"]) == 3
        assert (
            len(result["initial"]["forces"][0]) == 3
        )  # each force has x,y,z components
        assert len(result["final"]["forces"][0]) == 3

        # The forces should be different (unless already perfectly optimized)
        # At minimum, we should verify they are separate objects
        assert (
            result["initial"]["forces"] != result["final"]["forces"]
        ), "Forces should change during optimization "

    @patch("fairchem.core.components.calculate.recipes.omol.MSONAtoms")
    def test_single_point_job(self, mock_mson_atoms):
        """Test single-point energy and force calculation."""
        # Setup
        mock_mson_atoms.return_value.as_dict.return_value = {"test": "atoms"}

        # Execute with real calculator
        result = single_point_job(self.water_atoms.copy(), self.calculator)

        # Verify structure of result
        assert "atoms" in result
        assert "energy" in result
        assert "forces" in result
        assert result["atoms"] == {"test": "atoms"}

        # Verify that we get realistic values
        assert isinstance(result["energy"], (int, float))
        assert isinstance(result["forces"], list)
        assert len(result["forces"]) == 3  # water has 3 atoms
        assert len(result["forces"][0]) == 3  # each force has x,y,z components

    def test_single_point_integration(self):
        """Integration test with real calculator."""
        # Test that we can actually run a single point calculation
        water = self.water_atoms.copy()

        # Execute real calculation
        result = single_point_job(water, self.calculator)

        # Verify we get real results
        assert "atoms" in result
        assert "energy" in result
        assert "forces" in result
        assert isinstance(result["atoms"], dict)
        assert result["atoms"]["@class"] == "MSONAtoms"
        assert isinstance(result["energy"], (int, float))
        assert isinstance(result["forces"], list)
        assert len(result["forces"]) == 3  # water has 3 atoms

        # Check that calculator was properly cleaned up
        assert water.calc is None

    def test_relax_job_integration(self):
        """Integration test for geometry optimization with real calculator."""
        distorted_water = self.water_atoms.copy()
        distorted_water.positions[1] += [0.2, 0.0, 0.0]
        distorted_water.positions[2] += [0.0, 0.2, 0.0]

        result = relax_job(distorted_water, self.calculator, self.opt_flags)

        assert "initial" in result
        assert "final" in result
        assert isinstance(result["initial"]["atoms"], dict)
        assert isinstance(result["final"]["atoms"], dict)
        assert isinstance(result["initial"]["atoms"], dict)
        assert result["initial"]["atoms"]["@class"] == "MSONAtoms"
        assert isinstance(result["final"]["atoms"], dict)
        assert result["final"]["atoms"]["@class"] == "MSONAtoms"
        assert isinstance(result["initial"]["energy"], (int, float))
        assert isinstance(result["final"]["energy"], (int, float))
        assert isinstance(result["initial"]["forces"], list)
        assert isinstance(result["final"]["forces"], list)
        assert len(result["initial"]["forces"]) == 3
        assert len(result["final"]["forces"]) == 3

        # Verify that optimization actually changed the structure
        # Energy should decrease during optimization
        assert (
            result["final"]["energy"] <= result["initial"]["energy"]
        ), "Energy should not increase during optimization"

        # Forces should be reduced during optimization
        initial_force_magnitude = sum(
            sum(f**2 for f in force_vec) ** 0.5
            for force_vec in result["initial"]["forces"]
        )
        final_force_magnitude = sum(
            sum(f**2 for f in force_vec) ** 0.5
            for force_vec in result["final"]["forces"]
        )

        assert (
            final_force_magnitude <= initial_force_magnitude
        ), "Force magnitude should decrease during optimization"

    @patch("fairchem.core.components.calculate.recipes.omol.relax_job")
    @patch("fairchem.core.components.calculate.recipes.omol.tqdm")
    def test_conformers(self, mock_tqdm, mock_relax_job):
        """Test conformer geometry optimization."""
        mock_tqdm.side_effect = lambda x: x  # Pass through without progress bar
        mock_relax_job.return_value = {"test": "result"}

        input_data = {
            "molecule_family_1": [
                {"sid": "conf1", "initial_atoms": self.water_atoms.copy()},
                {"sid": "conf2", "initial_atoms": self.methane_atoms.copy()},
            ]
        }

        result = conformers(input_data, self.calculator)

        assert "molecule_family_1" in result
        assert "conf1" in result["molecule_family_1"]
        assert "conf2" in result["molecule_family_1"]
        assert result["molecule_family_1"]["conf1"] == {"test": "result"}
        assert mock_relax_job.call_count == 2

    @patch("fairchem.core.components.calculate.recipes.omol.relax_job")
    @patch("fairchem.core.components.calculate.recipes.omol.tqdm")
    def test_protonation(self, mock_tqdm, mock_relax_job):
        """Test protonation state calculations."""
        mock_tqdm.side_effect = lambda x: x
        mock_relax_job.return_value = {"test": "protonation_result"}

        input_data = {
            "molecule_family_1": {
                "0_1_1": {"initial_atoms": self.water_atoms.copy()},
                "1_0_1": {"initial_atoms": self.water_atoms.copy()},
            }
        }

        result = protonation(input_data, self.calculator)

        assert "molecule_family_1" in result
        assert "0_1_1" in result["molecule_family_1"]
        assert "1_0_1" in result["molecule_family_1"]
        assert result["molecule_family_1"]["0_1_1"] == {"test": "protonation_result"}
        assert mock_relax_job.call_count == 2

    @patch("fairchem.core.components.calculate.recipes.omol.single_point_job")
    @patch("fairchem.core.components.calculate.recipes.omol.tqdm")
    def test_ieea(self, mock_tqdm, mock_single_point):
        """Test unoptimized ionization energy and electron affinity calculations."""
        mock_tqdm.side_effect = lambda x: x
        mock_single_point.return_value = {"test": "ieea_result"}

        input_data = {
            "water_mol": {
                "1": {"1": {"atoms": self.water_atoms.copy()}},  # neutral, singlet
                "2": {"2": {"atoms": self.water_atoms.copy()}},  # cation, doublet
                "0": {"2": {"atoms": self.water_atoms.copy()}},  # anion, doublet
            }
        }

        result = ieea(input_data, self.calculator)

        assert "water_mol" in result
        assert "1" in result["water_mol"]
        assert "2" in result["water_mol"]
        assert "0" in result["water_mol"]
        assert "1" in result["water_mol"]["1"]
        assert "2" in result["water_mol"]["2"]
        assert "2" in result["water_mol"]["0"]
        assert mock_single_point.call_count == 3

    @patch("fairchem.core.components.calculate.recipes.omol.single_point_job")
    @patch("fairchem.core.components.calculate.recipes.omol.tqdm")
    def test_spin_gap(self, mock_tqdm, mock_single_point):
        """Test unoptimized spin gap calculations."""
        mock_tqdm.side_effect = lambda x: x
        mock_single_point.return_value = {"test": "spin_gap_result"}

        input_data = {
            "mcc_idx_61150_0": {
                "1": {"atoms": self.methane_atoms.copy()},  # singlet
                "3": {"atoms": self.methane_atoms.copy()},  # triplet
            }
        }

        result = spin_gap(input_data, self.calculator)

        assert "mcc_idx_61150_0" in result
        assert "1" in result["mcc_idx_61150_0"]
        assert "3" in result["mcc_idx_61150_0"]
        assert result["mcc_idx_61150_0"]["1"] == {"test": "spin_gap_result"}
        assert result["mcc_idx_61150_0"]["3"] == {"test": "spin_gap_result"}
        assert mock_single_point.call_count == 2

    @patch("fairchem.core.components.calculate.recipes.omol.single_point_job")
    @patch("fairchem.core.components.calculate.recipes.omol.tqdm")
    def test_ligand_pocket(self, mock_tqdm, mock_single_point):
        """Test protein-ligand interaction calculations."""
        mock_tqdm.side_effect = lambda x, **kwargs: x
        mock_single_point.return_value = {"test": "pocket_result"}

        input_data = {
            "protein_complex": {
                "ligand": self.water_atoms.copy(),
                "pocket": self.methane_atoms.copy(),
                "ligand_pocket": self.water_atoms.copy(),
            }
        }

        result = ligand_pocket(input_data, self.calculator)

        assert "protein_complex" in result
        assert "ligand" in result["protein_complex"]
        assert "pocket" in result["protein_complex"]
        assert "ligand_pocket" in result["protein_complex"]
        assert result["protein_complex"]["ligand"] == {"test": "pocket_result"}
        assert mock_single_point.call_count == 3

    @patch("fairchem.core.components.calculate.recipes.omol.single_point_job")
    @patch("fairchem.core.components.calculate.recipes.omol.relax_job")
    @patch("fairchem.core.components.calculate.recipes.omol.tqdm")
    def test_ligand_strain(self, mock_tqdm, mock_relax_job, mock_single_point):
        """Test ligand strain energy calculations."""
        mock_tqdm.side_effect = lambda x: x
        mock_single_point.return_value = {"test": "bioactive_result"}
        mock_relax_job.return_value = {"test": "conformer_result"}

        input_data = {
            "drug_ligand": {
                "bioactive_conf": self.water_atoms.copy(),
                "conformers": [self.methane_atoms.copy(), self.water_atoms.copy()],
            }
        }

        result = ligand_strain(input_data, self.calculator)

        assert "drug_ligand" in result
        assert "bioactive" in result["drug_ligand"]
        assert "gas_phase" in result["drug_ligand"]
        assert result["drug_ligand"]["bioactive"] == {"test": "bioactive_result"}
        assert 0 in result["drug_ligand"]["gas_phase"]
        assert 1 in result["drug_ligand"]["gas_phase"]
        assert mock_single_point.call_count == 1
        assert mock_relax_job.call_count == 2

    @patch("fairchem.core.components.calculate.recipes.omol.single_point_job")
    @patch("fairchem.core.components.calculate.recipes.omol.tqdm")
    def test_distance_scaling(self, mock_tqdm, mock_single_point):
        """Test distance scaling calculations."""
        mock_tqdm.side_effect = lambda x: x
        mock_single_point.return_value = {"test": "distance_result"}

        input_data = {
            "biomolecules": {
                "dimer_1": {
                    "scale_0.8": self.water_atoms.copy(),
                    "scale_1.0": self.water_atoms.copy(),
                    "scale_1.2": self.water_atoms.copy(),
                },
                "dimer_2": {
                    "scale_0.9": self.methane_atoms.copy(),
                    "scale_1.1": self.methane_atoms.copy(),
                },
            }
        }

        result = distance_scaling(input_data, self.calculator)

        assert "biomolecules" in result
        assert "dimer_1" in result["biomolecules"]
        assert "dimer_2" in result["biomolecules"]
        assert "scale_0.8" in result["biomolecules"]["dimer_1"]
        assert "scale_1.0" in result["biomolecules"]["dimer_1"]
        assert "scale_1.2" in result["biomolecules"]["dimer_1"]
        assert "scale_0.9" in result["biomolecules"]["dimer_2"]
        assert "scale_1.1" in result["biomolecules"]["dimer_2"]
        assert mock_single_point.call_count == 5

    @patch("fairchem.core.components.calculate.recipes.omol.single_point_job")
    @patch("fairchem.core.components.calculate.recipes.omol.tqdm")
    def test_singlepoint(self, mock_tqdm, mock_single_point):
        """Test general single-point calculations."""
        mock_tqdm.side_effect = lambda x, **kwargs: x
        mock_single_point.return_value = {"test": "singlepoint_prediction"}

        input_data = {
            "system1": self.water_atoms.copy(),
            "system2": self.methane_atoms.copy(),
        }

        result = singlepoint(input_data, self.calculator)

        assert "system1" in result
        assert "system2" in result
        assert result["system1"] == {"test": "singlepoint_prediction"}
        assert result["system2"] == {"test": "singlepoint_prediction"}
        assert mock_single_point.call_count == 2


if __name__ == "__main__":
    unittest.main()
