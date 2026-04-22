from __future__ import annotations

import os
import pickle
import re
from itertools import chain, combinations

from aiida.tools.dbimporters.plugins.cod import (
    CodDbImporter,  # https://aiida.readthedocs.io/projects/aiida-core/en/v1.0.1/import_export/dbimporters/cod.html
)
from tqdm import tqdm

importer = CodDbImporter()
# importer.get_supported_keywords()


def extract_cifid(cif):
    uri = cif.splitlines()[3]
    cifid = uri.rsplit("/", 1)[1].rsplit(".cif")[0]
    return uri, cifid


def extract_formula(cif):
    search_pattern = r"^.*_chemical_formula_sum.*$"
    matching_lines = re.findall(search_pattern, cif, re.MULTILINE)
    txt = matching_lines[0]
    search_pattern = r"'(.*?)'"
    matching_txt = re.search(search_pattern, txt)
    formula = matching_txt.group(1)
    formula = formula.replace(" ", "")
    return formula


def format_formula_for_sorting(l):
    l = l.split("-")
    new_list = []
    start = 0
    idx = 1
    while idx < len(l):
        new_list.append(l[start : idx + 1])
        start = idx + 1
        idx += 2
    return new_list


def sorted_formula(l):
    comp = sorted(format_formula_for_sorting(l))
    comp = "-".join(list(chain.from_iterable(comp)))
    return comp


def get_chemical_space(comp):
    elm_list = []
    for elm, frac in format_formula_for_sorting(comp):
        elm_list.append(elm)
    elm_list = sorted(elm_list)
    return elm_list


def formula_to_combos(comp):
    chemical_space = get_chemical_space(comp)
    combos = []
    chemical_space = get_chemical_space(comp)
    for n in range(1, len(chemical_space) + 1):
        combos.append(list(combinations(chemical_space, n)))
    combos = list(chain.from_iterable(combos))
    return combos


def check_shuffled_formula(str1, str2):
    if len(str1) == len(str2):
        list1 = format_formula_for_sorting(str1)
        list1 = sorted(list1)
        list2 = format_formula_for_sorting(str2)
        list2 = sorted(list2)
    else:
        return False

    return list1 == list2


def convert_string_to_formula(txt):
    new_comp = ""

    elements = re.findall("[A-Z][a-z]*", txt)
    if "." in txt:
        elements_w_num = re.findall(r"([A-Z][a-z]*)\d+\.\d+", txt)
        numbers = re.findall(r"\d+\.\d+", txt)
    else:
        elements_w_num = re.findall(r"([A-Z][a-z]*)\d+", txt)
        numbers = re.findall(r"\d+", txt)

    if len(elements) != len(numbers):
        for idx, element in enumerate(elements):
            if element not in elements_w_num:
                numbers.insert(idx, "1")

    numbers = [float(x) for x in numbers]
    sum_num = sum(numbers)
    numbers = [round(x / sum_num, 3) for x in numbers]

    for i, element in enumerate(elements):
        new_comp += element + "-"
        new_comp += str(numbers[i])
        if i < len(elements) - 1:
            new_comp += "-"

    return new_comp


def chemical_space_to_combos(chemical_space, combo_list=[1, 2, 3]):
    combos = []
    for n in combo_list:
        combos.append(list(combinations(chemical_space, n)))
    combos = list(chain.from_iterable(combos))
    return combos


def query_by_chemical_space(chemical_space, bulks_db, limit_number_of_elements=True):
    formula_list = []
    for formula in bulks_db.keys():
        # add formulas with the same number of elements only
        if limit_number_of_elements:
            if sorted(chemical_space) == sorted(get_chemical_space(formula)):
                formula_list.append(formula)
                continue
        else:
            # add all materials that contain the elements in the chemical space given
            for combo in formula_to_combos(formula):
                chemical_space2 = list(combo)
                if sorted(chemical_space) == sorted(chemical_space2):
                    formula_list.append(formula)
                    break
    return formula_list


def import_cif_from_COD(
    chemical_space=[],
    combo_list=[1, 2, 3],
    printout=True,
    exclusion_list=[["O"], ["C"]],
    folder_dir="",
    db_dir="",
):
    failed_cif = []
    cod_dict = {}

    combos = chemical_space_to_combos(chemical_space)
    if printout:
        print(f"{'>'*20}")
    if printout:
        print(f"Importing {len(combos)} unique chemical space(s) from COD")

    # creating a list of all chemical spaces in COD
    if db_dir:
        with open(db_dir, "rb") as file:
            COD_DB = pickle.load(file)

        chemical_spaces_in_COD = []
        for key in COD_DB.keys():
            chemical_space = get_chemical_space(key)
            if chemical_space in chemical_spaces_in_COD:
                continue
            else:
                chemical_spaces_in_COD.append(sorted(chemical_space))

        if "not_in_cod" not in list(COD_DB.keys()):
            COD_DB["not_in_cod"] = []

    count = 0
    for combo in combos:
        skip_count = 0
        add_new_cifs = True
        combo = sorted(list(combo))
        # Don't import combos that are explicitly excluded
        if combo in exclusion_list:
            add_new_cifs = False

        if db_dir:
            # Don't import combos that were already imported before
            if combo in chemical_spaces_in_COD:
                add_new_cifs = False
                if printout:
                    print(f"{combo} was downloaded before")

            if combo in COD_DB["not_in_cod"]:
                add_new_cifs = False
                if printout:
                    print(f"{combo} was not found previously in COD")

        if add_new_cifs:
            # query strcutures from COD
            results = importer.query(element=combo, number_of_elements=len(combo))

            if len(results) == 0:
                if printout:
                    print(f"{combo} is not found in COD")
                if db_dir:
                    COD_DB["not_in_cod"].append(combo)
            else:
                for entry in tqdm(results, total=len(results)):
                    if printout:
                        print(f"{combo}: found {len(results)} structures")
                    # try importing entry as cif
                    try:
                        cif = entry.get_raw_cif()
                        atoms = entry.get_ase_structure()
                        ase_formula = atoms.get_chemical_formula()
                        cifid = extract_cifid(cif)[1]
                    except:
                        failed_cif.append(entry)
                        skip_count += 1
                        continue
                    # try extracting composition formula
                    try:
                        comp = sorted_formula(
                            convert_string_to_formula(extract_formula(cif))
                        )
                    except:
                        comp = sorted_formula(convert_string_to_formula(ase_formula))

                    # add composition key and update cif list
                    if comp not in cod_dict:
                        cod_dict[comp] = [cifid]
                    else:
                        cod_dict[comp].append(cifid)

                    # export cif file if not existing
                    file_path = f"{folder_dir}/cif/{cifid}.cif"
                    if not os.path.exists(file_path):
                        with open(file_path, "w") as f:
                            f.write(cif)
                    count += 1

                if printout:
                    print(f"number of failed ase imports: {skip_count}")
                if printout:
                    print(f"{'-'*20}")

    if printout:
        print(f"total number of succesful ase imports: {count}")
    if printout:
        print(f"total number of failed ase imports: {len(failed_cif)}")

    if db_dir:
        COD_DB.update(cod_dict)
        with open(db_dir, "wb") as file:
            pickle.dump(COD_DB, file)
        return COD_DB
    else:
        return cod_dict, failed_cif
