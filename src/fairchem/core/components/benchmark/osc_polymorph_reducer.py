"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

import ase.units
import numpy as np
import pandas as pd
from monty.dev import requires
from pymatgen.analysis.local_env import JmolNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor, MSONAtoms
from tqdm import tqdm

from fairchem.core.components.benchmark.benchmark_reducer import JsonDFReducer
from fairchem.core.components.calculate.recipes.local_env import construct_bond_matrix
from fairchem.core.components.calculate.relaxation_runner import RelaxationRunner
from fairchem.core.components.calculate.singlepoint_runner import SinglePointRunner

try:
    from scipy.stats import kendalltau, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    sklearn_scipy_installed = True
except ImportError:
    sklearn_scipy_installed = False


ev2kJ = ase.units.eV * ase.units.mol / ase.units.kJ


@requires(
    sklearn_scipy_installed, "Requires `scipy` and `scikit-learn` to be installed"
)
class OMCPolymorphReducer(JsonDFReducer):
    def __init__(
        self,
        benchmark_name: str,
        target_data_key: str,
        molecule_id_key: str,
        calculate_structural_metrics: bool = False,
        index_name: str | None = None,
    ):
        """
        Args:
            benchmark_name: Name of the benchmark, used for file naming
            target_data_key: Name of the normalized target energy
            molecule_id_key: Key name of molecule to identify polymorphs
            calculate_structural_metrics: Whether to calculate structural metrics: match rate, and root mean squared
                distance of atomic positions for matched structures.
                A structure is considered a match if atomic positions match within given tolerance and no
                and matrices of covelent bond networks computed using JmolNN between target and relaxed structure
                are identical. RMSD is only computed if a structures is a match.
            index_name: Optional name of the column to use as index
        """
        self._molecule_id_key = molecule_id_key
        self._calc_structural_metrics = calculate_structural_metrics

        if self._calc_structural_metrics:
            self._structure_matcher = StructureMatcher(
                primitive_cell=False
            )  # ltol=0.5, stol=0.5, angle_tol=10,
            self._jmolnn = JmolNN()
        else:
            self._structure_matcher = None
            self._jmolnn = None

        super().__init__(
            benchmark_name=benchmark_name,
            index_name=index_name,
            target_data_keys=[target_data_key],
        )

    @property
    def runner_type(self) -> type[SinglePointRunner | RelaxationRunner]:
        """The runner type this reducer is associated with."""
        return RelaxationRunner if self._calc_structural_metrics else SinglePointRunner

    def compute_metrics(self, results: pd.DataFrame, run_name: str) -> pd.DataFrame:
        """Compute OMC polymorph metrics for single point or relaxed energy and structure predictions.

        Args:
            results: DataFrame containing prediction results with energy values
            run_name: Identifier for the current evaluation run

        Returns:
            DataFrame containing computed metrics for different material subsets
        """
        metrics = {}
        energy_key = self.target_data_keys[0]
        for molecule_id in results[self._molecule_id_key].unique():
            logging.info(f"Computing metrics for {molecule_id=}")
            polymorph_results = results[results[self._molecule_id_key] == molecule_id]

            ref_structure_id = polymorph_results[f"{energy_key}_target"].argmin()
            ref_energy = polymorph_results[energy_key].iloc[ref_structure_id]
            ref_energy_target = polymorph_results[f"{energy_key}_target"].iloc[
                ref_structure_id
            ]

            energy_latt = polymorph_results[energy_key] - ref_energy
            energy_latt_target = (
                polymorph_results[f"{energy_key}_target"] - ref_energy_target
            )

            rankings = pd.Series(energy_latt).rank(method="min")
            rankings_target = pd.Series(energy_latt_target).rank(method="min")

            metrics.update(
                {
                    f"{molecule_id},mae": ev2kJ
                    * mean_absolute_error(energy_latt_target, energy_latt),
                    f"{molecule_id},rmse": ev2kJ
                    * np.sqrt(mean_squared_error(energy_latt_target, energy_latt)),
                    f"{molecule_id},r2": r2_score(energy_latt_target, energy_latt),
                    f"{molecule_id},kendall": kendalltau(
                        rankings_target, rankings
                    ).statistic,
                    f"{molecule_id},spearmanr": spearmanr(
                        rankings_target, rankings
                    ).statistic,
                }
            )

            if self._calc_structural_metrics:  # TODO this will need to be parallelized
                _rmsds = []
                for index in tqdm(
                    polymorph_results.index,
                    desc=f"Matching polymorphs for {molecule_id=}",
                ):
                    entry = polymorph_results.loc[index]
                    relaxed_structure = AseAtomsAdaptor.get_structure(
                        MSONAtoms.from_dict(entry["atoms"])
                    )
                    reference_structure = AseAtomsAdaptor.get_structure(
                        MSONAtoms.from_dict(entry["atoms_relaxed_target"])
                    )

                    # not clean but call this directly to avoid rematching (rms, max_dist, mask, cost, mapping)
                    reference_structure, relaxed_structure, _, _ = (
                        self._structure_matcher._preprocess(
                            reference_structure, relaxed_structure, niggli=True
                        )
                    )
                    match = self._structure_matcher._match(
                        reference_structure, relaxed_structure, fu=1, use_rms=True
                    )
                    if match is not None:
                        reference_matrix = construct_bond_matrix(
                            reference_structure, nn_finder=self._jmolnn
                        )
                        relaxed_matrix = construct_bond_matrix(
                            relaxed_structure,
                            nn_finder=self._jmolnn,
                            site_permutations=match[4],
                        )
                        if np.array_equal(relaxed_matrix, reference_matrix) is True:
                            # rmsd from pmg structure matcher is normalized by (V/nsites)^(1/3)
                            avg_vol = (
                                relaxed_structure.volume + reference_structure.volume
                            ) / 2
                            rmsd = match[0] * (avg_vol / len(reference_structure)) ** (
                                1 / 3
                            )
                            _rmsds.append(rmsd)

                metrics.update(
                    {
                        f"{molecule_id},rmsd": np.mean(_rmsds) if _rmsds else 0.0,
                        f"{molecule_id},rmsd_std": np.std(_rmsds) if _rmsds else 0.0,
                        f"{molecule_id},match_rate": len(_rmsds)
                        / len(polymorph_results),
                    }
                )

        return pd.DataFrame([metrics], index=[run_name])

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True

    def load_state(self, checkpoint_location: str | None) -> None:
        pass
