"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
from glob import glob
from typing import TYPE_CHECKING, Any

import pandas as pd
from monty.dev import requires
from tqdm import tqdm

from fairchem.core.components.benchmark import JsonDFReducer
from fairchem.core.components.calculate import (
    RelaxationRunner,
)
from fairchem.core.components.calculate.recipes.energy import calc_energy_from_e_refs

try:
    from matbench_discovery.enums import MbdKey
    from matbench_discovery.metrics.discovery import stable_metrics
    from matbench_discovery.metrics.geo_opt import calc_geo_opt_metrics
    from matbench_discovery.structure import symmetry
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
    from pymatgen.io.ase import AseAtomsAdaptor, MSONAtoms
    from pymatviz.enums import Key

    mbd_installed = True
except ImportError:
    mbd_installed = False

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.entries.compatibility import Compatibility


MP2020Compatibility = MaterialsProject2020Compatibility()


def as_dict_handler(obj: Any) -> dict[str, Any] | None:
    """Pass this to json.dump(default=) or as pandas.to_json(default_handler=) to
    serialize Python classes with as_dict(). Warning: Objects without a as_dict() method
    are replaced with None in the serialized data.

    From matbench_discovery: https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/data.py
    """
    try:
        return obj.as_dict()  # all MSONable objects implement as_dict()
    except AttributeError:
        return None


@requires(mbd_installed, message="Requires `matbench_discovery` to be installed")
class MaterialsDiscoveryReducer(JsonDFReducer):
    def __init__(
        self,
        benchmark_name: str,
        target_data_path: str,
        cse_data_path: str | None = None,
        elemental_references_path: str | None = None,
        index_name: str | None = None,
        corrections: Compatibility | None = MP2020Compatibility,
        max_error_threshold: float = 5.0,
        analyze_geo_opt: bool = True,
        geo_symprec: float = 1e-5,
    ):
        """
        Args:
            benchmark_name: Name of the benchmark, used for file naming
            target_data_path: Path to the target data JSON file
            cse_data_path: Path to the WBM computed structure entries JSON file
            elemental_references_path: Path to elemental energy references JSON file
            index_name: Optional name of the column to use as index
            corrections: Optional correction class to apply to all entries
            max_error_threshold: Maximum allowed mean absolute formation energy per atom error threshold
            analyze_geo_opt: Whether to analyze geometry of relaxed structures and compute RMSD
            geo_symprec: Symmetry precision of moyopy.
        """
        index_name = index_name or str(Key.mat_id)
        self._corrections = corrections
        self._max_error_threshold = max_error_threshold
        self._elemental_references_path = elemental_references_path
        self._cse_data_path = cse_data_path
        self._analyze_geo_opt = analyze_geo_opt
        self._geo_symprec = geo_symprec
        super().__init__(
            benchmark_name=benchmark_name,
            target_data_path=target_data_path,
            index_name=index_name,
        )

    @property
    def runner_type(self) -> type[RelaxationRunner]:
        """The runner type this reducer is associated with."""
        return RelaxationRunner

    @staticmethod
    def load_targets(path: str, index_name: str | None) -> pd.DataFrame:
        """Load target data from a JSON file into a pandas DataFrame.

        Args:
            path: Path to the target JSON file
            index_name: Optional name of the column to use as index

        Returns:
            DataFrame containing the target data, sorted by index
        """
        df_wbm = pd.read_csv(path)
        if index_name is not None:
            df_wbm = df_wbm.set_index(index_name)
        return df_wbm.sort_index()

    @staticmethod
    def _load_elemental_ref_energies(
        elemental_references_path: str,
    ) -> dict[str, float]:
        elem_ref_entries = (
            pd.read_json(elemental_references_path, typ="series")
            .map(ComputedEntry.from_dict)
            .to_dict()
        )
        elemental_ref_energies = {
            elem: entry.energy_per_atom for elem, entry in elem_ref_entries.items()
        }
        return elemental_ref_energies

    @staticmethod
    def _load_computed_structure_entries(
        cse_data_path: str, results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert prediction results to computed structure entries with updated energies and structures.

        Returns:
            DataFrame of computed structure entries indexed by material IDs
        """
        # WBM DF used to obtain CSEs for corrections and phase diagram input
        df_wbm_cse = pd.DataFrame(pd.read_json(cse_data_path, lines=True)).set_index(
            Key.mat_id
        )
        df_wbm_cse = df_wbm_cse.sort_index()

        df_wbm_cse[Key.computed_structure_entry] = [
            ComputedStructureEntry.from_dict(dct)
            for dct in tqdm(
                df_wbm_cse[Key.computed_structure_entry], desc="Creating pmg CSEs"
            )
        ]
        # transfer energies and relaxed structures to ComputedStructureEntries in order to apply corrections.
        # corrections applied below are structure-dependent (for oxides and sulfides)
        df_result_cse: list[dict[str, str | ComputedStructureEntry]] = []
        for mat_id in tqdm(
            results.index,
            desc="Converting predicted structures and energies to computed structure entries",
        ):
            cse = df_wbm_cse.loc[mat_id, Key.computed_structure_entry].copy()
            atoms = MSONAtoms.from_dict(results.loc[mat_id, "atoms"])
            structure = AseAtomsAdaptor.get_structure(atoms)
            energy = results.loc[mat_id, "energy"]
            cse._energy = energy
            cse._structure = structure
            df_result_cse.append(
                {Key.mat_id: mat_id, Key.computed_structure_entry: cse}
            )
        df_result_cse = pd.DataFrame(df_result_cse).set_index(Key.mat_id)
        return df_result_cse, df_wbm_cse

    def _apply_corrections(
        self, computed_structure_entries: list[ComputedStructureEntry]
    ) -> None:
        """Apply compatibility corrections to computed structure entries.

        Args:
            computed_structure_entries: List of ComputedStructureEntry objects to apply corrections to

        Raises:
            ValueError: If not all entries were successfully processed after applying corrections
        """
        if self._corrections is not None:
            processed = self._corrections.process_entries(
                computed_structure_entries, verbose=True, clean=True
            )
            if len(processed) != len(computed_structure_entries):
                raise ValueError(
                    f"not all entries processed: {len(processed)=} {len(computed_structure_entries)=}"
                )

    def _analyze_relaxed_geometry(
        self,
        pred_structures: dict[str, Structure],
        target_structures: dict[str, Structure],
    ) -> dict[str, float]:
        """Analyze geometry of relaxed structures and calculate RMSD wrt to the target structures.

        Args:
            pred_structures: Dictionary mapping material IDs to predicted Structure objects
            target_structures: Dictionary mapping material IDs to target Structure objects

        Returns:
            Dictionary containing geometric analysis metrics
        """
        df_symm_pred = symmetry.get_sym_info_from_structs(
            pred_structures,
            symprec=self._geo_symprec,
        )
        df_symm_target = symmetry.get_sym_info_from_structs(
            target_structures,
            symprec=self._geo_symprec,
        )
        df_geo_analysis = symmetry.pred_vs_ref_struct_symmetry(
            df_symm_pred,
            df_symm_target,
            pred_structures,
            target_structures,
        )
        return df_geo_analysis

    def join_results(self, results_dir: str, glob_pattern: str) -> pd.DataFrame:
        """Join results from multiple relaxation JSON files into a single DataFrame.

        Joins results for relaxed energy, applies compatibility corrections, and computes formation energy
        w.r.t to MP reference structures in MatBench Discovery

        Args:
            results_dir: Directory containing result files
            glob_pattern: Pattern to match result files

        Returns:
            Combined DataFrame containing all results
        """
        results = pd.concat(
            [
                pd.read_json(f).set_index("sid")
                for f in tqdm(
                    sorted(glob(os.path.join(results_dir, glob_pattern))),
                    desc=f"Loading results from {results_dir}",
                )
            ]
        ).sort_index()
        results.index.name = Key.mat_id

        df_cse_pred, df_cse_target = self._load_computed_structure_entries(
            self._cse_data_path, results
        )
        self._apply_corrections(df_cse_pred[Key.computed_structure_entry].tolist())

        # compute formation energy per atom
        elemental_ref_energies = self._load_elemental_ref_energies(
            self._elemental_references_path
        )
        results["e_form_per_atom"] = [
            calc_energy_from_e_refs(cse, ref_energies=elemental_ref_energies)
            for cse in tqdm(
                df_cse_pred[Key.computed_structure_entry],
                total=len(results),
                desc="Computing formation energies",
            )
        ]

        if self._analyze_geo_opt:
            pred_structures = {
                mat_id: cse.structure
                for mat_id, cse in df_cse_pred[Key.computed_structure_entry].items()
            }
            target_structures = {
                mat_id: cse.structure
                for mat_id, cse in df_cse_target[Key.computed_structure_entry].items()
            }
            df_geo_analysis = self._analyze_relaxed_geometry(
                pred_structures, target_structures
            )
            results = results.join(df_geo_analysis)

        return results

    def save_results(self, results: pd.DataFrame, results_dir: str) -> None:
        """Save joined results to a single file

        Saves the results in two formats:
        1. CSV file containing only numerical data
        2. JSON file containing all data including relaxed structures

        Args:
            results: DataFrame containing the prediction results
            results_dir: Directory path where result files will be saved
        """
        # save only numerical results
        results.select_dtypes("number").to_csv(
            os.path.join(results_dir, f"{self.benchmark_name}_results.csv.gz")
        )

        # save results including relaxed structures
        results.reset_index().to_json(
            os.path.join(results_dir, f"{self.benchmark_name}_results.json.gz"),
            default_handler=as_dict_handler,
            orient="records",
            lines=True,
        )

    def compute_metrics(self, results: pd.DataFrame, run_name: str) -> pd.DataFrame:
        """Compute Matbench discovery metrics for relaxed energy and structure predictions.

        Args:
            results: DataFrame containing prediction results with energy values
            run_name: Identifier for the current evaluation run

        Returns:
            DataFrame containing computed metrics for different material subsets
        """
        df_wbm = self.target_data
        df_wbm[[*results]] = results.round(4)

        # remove bad outlier predictions
        bad_mask = (
            abs(df_wbm["e_form_per_atom"] - df_wbm[MbdKey.e_form_dft])
            > self._max_error_threshold
        )
        df_wbm.loc[bad_mask, "e_form_per_atom"] = pd.NA
        n_preds, n_bad = len(df_wbm.dropna()), sum(bad_mask)
        logging.info(
            f"{n_bad:,} of {n_preds:,} unrealistic predictions with formation energy error "
            f"> {self._max_error_threshold} eV/atom"
        )

        # compute the predicted energy above hull
        e_above_hull_pred = (
            df_wbm[MbdKey.each_true]
            + df_wbm["e_form_per_atom"]
            - df_wbm[MbdKey.e_form_dft]
        )

        df_uniq_proto = df_wbm[df_wbm[MbdKey.uniq_proto]]
        e_above_hull_pred_uniq_proto = (
            df_uniq_proto[MbdKey.each_true]
            + df_uniq_proto["e_form_per_atom"]
            - df_uniq_proto[MbdKey.e_form_dft]
        )

        # now compute metrics
        metrics: dict[str, float] = stable_metrics(
            df_wbm[MbdKey.each_true], e_above_hull_pred, fillna=True
        )

        metrics_uniq_proto: dict[str, float] = stable_metrics(
            df_uniq_proto[MbdKey.each_true], e_above_hull_pred_uniq_proto, fillna=True
        )

        # Get the 10,000 most stable materials based on predicted energy above hull
        most_stable_10k = e_above_hull_pred_uniq_proto.nsmallest(10_000)
        metrics_most_stable_10k = stable_metrics(
            df_wbm[MbdKey.each_true].loc[most_stable_10k.index],
            most_stable_10k,
            fillna=True,
        )

        if self._analyze_geo_opt:
            metrics.update(calc_geo_opt_metrics(results))
            metrics_uniq_proto.update(
                calc_geo_opt_metrics(results[df_wbm[MbdKey.uniq_proto]])
            )
            metrics_most_stable_10k.update(
                calc_geo_opt_metrics(results.loc[most_stable_10k.index])
            )

        all_metrics = pd.DataFrame(
            [metrics, metrics_uniq_proto, metrics_most_stable_10k],
            index=["full", "uniq-proto", "most-stable-10k"],
        )

        return all_metrics

    def log_metrics(self, metrics: pd.DataFrame, run_name: str) -> None:
        """Log metrics to the configured logger if available.

        Args:
            metrics: DataFrame containing the computed metrics
            run_name: Name of the current run
        """
        # drop these columns for cleaner logging
        metrics = metrics.drop(columns=["TP", "FP", "TN", "FN"])
        if self.logger is not None:
            for split in metrics.index:  # log each MBD split into a different table
                split_metrics = pd.DataFrame(
                    data=[metrics.loc[split]],
                    index=[run_name],
                )
                self.logger.log_dataframe(
                    name=f"{self.benchmark_name}-{split}",
                    dataframe=split_metrics.reset_index(),
                )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True

    def load_state(self, checkpoint_location: str | None) -> None:
        pass
