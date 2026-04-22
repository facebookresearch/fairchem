from __future__ import annotations

import re
from itertools import chain, combinations

import numpy as np
from scipy.spatial.distance import euclidean


def strip_zeros(comp):
    """
    converts 'Au-0.200-Cu-0.800' to 'Au-0.2-Cu-0.8'
    """
    new_comp = ""
    sp = comp.split("-")
    for idx in range(len(sp)):
        if idx % 2 == 0:
            new_comp += sp[idx]
        else:
            new_comp += str(float(sp[idx]))

        if idx < len(sp) - 1:
            new_comp += "-"
    return new_comp


def rev_strip_zeros(comp):
    """
    converts 'Au-0.2-Cu-0.8' to 'Au-0.200-Cu-0.800'
    """
    new_comp = ""
    sp = comp.split("-")
    for idx in range(len(sp)):
        if idx % 2 == 0:
            new_comp += sp[idx]
        else:
            while len(sp[idx]) < 5:
                sp[idx] += "0"
            new_comp += sp[idx]

        if idx < len(sp) - 1:
            new_comp += "-"
    return new_comp


def count_elements(comp):
    splits = comp.split("-")
    num_elements = len(splits) // 2
    return int(num_elements)


def chemical_space_to_combos(chemical_space, combo_list=[1, 2, 3]):
    combos = []
    for n in combo_list:
        combos.append(list(combinations(chemical_space, n)))
    combos = list(chain.from_iterable(combos))
    return combos


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


def get_chemical_space(comp):
    elm_list = []
    for elm, frac in format_formula_for_sorting(comp):
        elm_list.append(elm)
    elm_list = sorted(elm_list)
    return elm_list


def extract_cifid(cif):
    uri = cif.splitlines()[3]
    cifid = uri.rsplit("/", 1)[1].rsplit(".cif")[0]
    return uri, cifid


def extract_formula(cif):
    search_keyword = "_chemical_formula_sum"
    search_pattern = rf"^.*{search_keyword}.*$"
    matching_lines = re.findall(search_pattern, cif, re.MULTILINE)
    txt = matching_lines[0]
    txt = txt.replace(search_keyword, "").strip()

    search_pattern = r"'(.*?)'"
    matching_txt = re.search(search_pattern, txt)

    if matching_txt == None:
        search_pattern = r'"(.*?)"'
        matching_txt = re.search(search_pattern, txt)

    if matching_txt:
        formula = matching_txt.group(1)
    else:
        formula = txt

    formula = formula.replace(" ", "")
    return formula


def sorted_formula(comp):
    comp = sorted(format_formula_for_sorting(comp))
    comp = "-".join(list(chain.from_iterable(comp)))
    return comp


def extract_formula_from_cif(cif):
    search_pattern = r"^.*_chemical_formula_sum.*$"
    matching_lines = re.findall(search_pattern, cif, re.MULTILINE)
    txt = matching_lines[0]
    search_pattern = r"'(.*?)'"
    matching_txt = re.search(search_pattern, txt)
    formula = matching_txt.group(1)
    formula = formula.replace(" ", "")
    return formula


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


def gen_elm_list(comp):
    elm_list = []
    chemical_space = get_chemical_space(comp)
    if len(chemical_space) > 1:
        for elm in chemical_space:
            elm_frac = elm + "-1.0"
            elm_list.append(elm_frac)
    return elm_list


def gen_oxide_combo_list(comp):
    chemical_spaces_list = []
    combos = formula_to_combos(comp)
    for combo in combos:
        chemical_space = list(combo) + ["O"]
        chemical_spaces_list.append(chemical_space)

    return chemical_spaces_list


def broadcast_formula(comp, comp2):
    """
    convert formula of a given formula to broadcast and include all elements of a formula 2
    example:
    comp1= Ag-1.0
    comp2= Au-0.2-Ag-0.1-Zn-0.9
    output_comp = Ag-1.0-Au-0.0-Zn-0.0
    """
    comp = sorted_formula(comp)
    comp2 = sorted_formula(comp2)
    out_comp = f"{comp}-"
    if count_elements(comp) < count_elements(comp2):
        for n in np.arange(0, count_elements(comp), 2):
            if comp.split("-")[n] != comp2.split("-")[n]:
                print("not the same")
        for n in np.arange(count_elements(comp) * 2, count_elements(comp2) * 2, 2):
            out_comp += f"{comp2.split('-')[n]}-{0.0}-"
        out_comp = out_comp[:-1]

    return out_comp


def check_formula_within_stdev(comp1, comp2, stdev):
    stoich1 = tuple(map(float, re.findall(r"\d+.\d+", comp1)))
    stoich2 = tuple(map(float, re.findall(r"\d+.\d+", comp2)))
    diffs = np.abs(np.array(stoich1) - np.array(stoich2))
    if np.all(diffs <= np.array(stdev) / 100 * 2):
        return True


def get_distance_between_formulas(comp1, comp2, distance_type="euclidean"):
    stoich1 = tuple(map(float, re.findall(r"\d+.\d+", comp1)))
    stoich2 = tuple(map(float, re.findall(r"\d+.\d+", comp2)))
    return euclidean(stoich1, stoich2)


def get_diff_between_formulas(comp1, comp2):
    stoich1 = tuple(map(float, re.findall(r"\d+.\d+", comp1)))
    stoich2 = tuple(map(float, re.findall(r"\d+.\d+", comp2)))
    return np.array(stoich1) - np.array(stoich2)
