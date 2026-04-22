from __future__ import annotations

import os
import pickle

from aiida.tools.dbimporters.plugins.cod import (
    CodDbImporter,  # https://aiida.readthedocs.io/projects/aiida-core/en/v1.0.1/import_export/dbimporters/cod.html
)
from tqdm import tqdm
from xrdwiz.utils import ChemFormula


class CIF_DB:
    def __init__(self):
        self.data = {}
        self.cif_folder_tree_format = "/cif"

    def create_db(
        self, cif_folder_dir, dbname="", folder_tree_format="formula/cif", output_dir=""
    ):
        self.cif_folder_tree_format = folder_tree_format
        if folder_tree_format == "formula/cif":
            # creating db {comp:[cif list using src id]}

            bulks = {}
            formula_list = glob.glob(f"{cif_folder_dir}/*")

            for formula in formula_list:
                formula = os.path.basename(formula)
                cif_list = [
                    os.path.basename(file)[:-4]
                    for file in glob.glob(f"{cif_folder_dir}/{formula}/*.cif")
                ]

                formated_formula = ChemFormula.sorted_formula(
                    ChemFormula.strip_zeros(formula)
                )
                if formula != formated_formula:
                    os.rename(
                        f"{cif_folder_dir}/{formula}",
                        f"{cif_folder_dir}/{formated_formula}",
                    )
                    formula = formated_formula

                bulks[formula] = cif_list

        elif folder_tree_format == "/cif":
            bulks = {}
            cif_list = glob.glob(f"{cif_folder_dir}/*")

            for cifdir in cif_list:
                cifid = os.path.basename(cifdir)[:-4]
                try:
                    with open(cifdir) as f:
                        cif = f.read()
                    formula = ChemFormula.sorted_formula(
                        ChemFormula.convert_string_to_formula(
                            ChemFormula.extract_formula(cif)
                        )
                    )

                    if formula not in bulks.keys():
                        bulks[formula] = [cifid]
                    else:
                        bulks[formula].append(cifid)
                except:
                    print(f"{cifid} was not imported")
                    continue
        else:
            raise ValueError("Could not create database")

        with open(f"{output_dir}/{dbname}.pickle", "wb") as f:
            pickle.dump(bulks, f)

        self.load_db(db_file_dir=f"{output_dir}/{dbname}.pickle")

    def load_db(self, db_file_dir=None):
        if db_file_dir:
            self.db_file_dir = os.path.abspath(db_file_dir)
            self.cif_folder_dir = f"{os.path.dirname(db_file_dir)}/cif"
            self.data = pickle.load(open(self.db_file_dir, "rb"))
            self.get_existing_chemical_space_groups()
        else:
            raise ValueError(
                "You should input a db file directory and type for the database"
            )

    def get_existing_chemical_space_groups(self):
        self.chemical_spaces = []
        for key in self.data:
            chemical_space = ChemFormula.get_chemical_space(key)
            if chemical_space in self.chemical_spaces:
                continue
            else:
                self.chemical_spaces.append(sorted(chemical_space))

    def query_by_chemical_space(self, chemical_space, limit_number_of_elements=True):
        formula_list = []

        for formula in self.data.keys():
            # add formulas with the same number of elements only
            if limit_number_of_elements:
                if sorted(chemical_space) == sorted(
                    ChemFormula.get_chemical_space(formula)
                ):
                    formula_list.append(formula)
            else:
                # add all materials that contain the elements in the chemical space given
                for combo in ChemFormula.formula_to_combos(formula):
                    chemical_space2 = list(combo)
                    if sorted(chemical_space) == sorted(chemical_space2):
                        formula_list.append(formula)
                        break
        return formula_list

    def lookup_oxide_phases(self, composition):
        oxide_combo_list = ChemFormula.gen_oxide_combo_list(composition)
        oxide_phases = []
        for oxide in oxide_combo_list:
            formula_list = self.query_by_chemical_space(oxide)
            if len(formula_list) > 0:
                oxide_phases.extend(formula_list)

        return oxide_phases

    def lookup_phases_in_composition_chemical_space(self, composition, num_elements):
        formula_list = []

        chemical_space = ChemFormula.get_chemical_space(composition)
        combos = ChemFormula.chemical_space_to_combos(chemical_space)
        for combo in combos:
            if len(combo) == num_elements:
                found_formulas = self.query_by_chemical_space(combo)
                formula_list.extend(found_formulas)

        if composition in formula_list:
            formula_list.remove(composition)

        return formula_list

    def get_cifids(self, formula_list, prefix=None):
        cifid_dict = {}
        for formula in formula_list:
            for cifid in self.data[formula]:
                if formula in cifid_dict:
                    cifid_dict[formula].append((prefix, cifid))
                else:
                    cifid_dict[formula] = [(prefix, cifid)]

        return cifid_dict

    def download_from_COD(
        self,
        combo,
        exclusion_list=[["O"], ["C"]],
        printout=True,
        update_local_CODdb=False,
    ):
        self.get_existing_chemical_space_groups()
        if "not_in_cod" not in list(self.data.keys()):
            self.data["not_in_cod"] = []

        add_new_cifs = True

        if printout:
            print(f"{'>'*20}")
        if combo in exclusion_list:
            # Don't import combos that are explicitly excluded
            add_new_cifs = False
        elif combo in self.data["not_in_cod"]:
            add_new_cifs = False
            if printout:
                print(f"{combo} does not exist in COD")
        if combo in self.chemical_spaces:
            add_new_cifs = False

        skip_count = 0
        if add_new_cifs:
            if printout:
                print(
                    f"Looking for structures in {combo} chemical space from COD DB online"
                )
            # query strcutures from COD
            importer = CodDbImporter()
            results = importer.query(element=combo, number_of_elements=len(combo))

            if len(results) == 0:
                if printout:
                    print(f"{combo} was not found in COD")
                self.data["not_in_cod"].append(combo)
            else:
                for entry in tqdm(results, total=len(results)):
                    if printout:
                        print(
                            f"{combo}: found {len(results)} structures, downloading..."
                        )
                    # try importing entry as cif
                    try:
                        cif = entry.get_raw_cif()
                        atoms = entry.get_ase_structure()
                        ase_formula = atoms.get_chemical_formula()
                        cifid = ChemFormula.extract_cifid(cif)[1]
                    except:
                        if printout:
                            print("Failed to parse {entry}")
                        skip_count += 1
                        continue

                    # try extracting  formula
                    try:
                        comp = ChemFormula.sorted_formula(
                            ChemFormula.convert_string_to_formula(
                                format_form.extract_formula_from_cif(cif)
                            )
                        )
                    except:
                        comp = ChemFormula.sorted_formula(
                            ChemFormula.convert_string_to_formula(ase_formula)
                        )

                    # add composition key and update cif list
                    if comp not in self.data.keys():
                        self.data[comp] = [cifid]
                    else:
                        self.data[comp].append(cifid)

                    # export cif file if not existing
                    file_path = f"{self.cif_folder_dir}/{cifid}.cif"
                    if not os.path.exists(file_path):
                        with open(file_path, "w") as f:
                            f.write(cif)

                if printout:
                    print(f"number of failed ase imports: {skip_count}")
                if printout:
                    print(f"{'-'*20}")

            if update_local_CODdb:
                with open(self.db_file_dir, "wb") as file:
                    pickle.dump(self.data, file)


def cifdict2list(cifdict):
    ciflist = []
    formulalist = []
    sourcelist = []
    for formula, cifitems in cifdict.items():
        for cif in cifitems:
            ciflist.append(cif[1])
            formulalist.append(formula)
            sourcelist.append(cif[0])
    return ciflist, formulalist, sourcelist


def ciflist2dict(ciflist, formulalist, sourcelist):
    cifdict = {}
    for i, formula in enumerate(formulalist):
        if formula in cifdict:
            cifdict[formula].append((sourcelist[i], ciflist[i]))
        else:
            cifdict[formula] = [(sourcelist[i], ciflist[i])]

    return cifdict
