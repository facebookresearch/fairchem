from __future__ import annotations

import glob
import io
import multiprocessing
import os
import re
import sys
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pdf lib
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from scipy.stats import wasserstein_distance

from xrdwiz.utils import ChemFormula, dbtools, xrdtools


def create_xrd_df(
    xrd_file_dir,
    outfolder,
    tmin=20,
    tmax=60,
    plot=True,
    save=True,
    skip_rows=None,
    xrd_idkey=None,
    pp_idkey_dir=None,
    naming_format=None,
):
    """
    xrd filename format: {synthesis date, sample pos in rack, batch-no, sample-pos-sonicator, xrdid, noelements, sonication-temp, sonication-time}
    """

    # create xrd df
    xrd_df = pd.DataFrame()
    xrd_files_list = [f for f in glob.glob(xrd_file_dir) if os.path.isfile(f)]

    start_cutoff = tmin
    end_cutoff = tmax

    for idx, xrd_dir in enumerate(xrd_files_list):
        if naming_format == 1:
            # extracting filename
            basename = os.path.basename(xrd_dir)
            filename = re.search(r"id(\d+)", basename).group(1)
            # extracting composition formula using xrdid2bulkid
            comp = ChemFormula.sorted_formula(xrd_idkey[int(filename)])
            xrd_df.loc[idx, "filename"] = basename.split(".")[0]
            xrd_df.loc[idx, "xrdid"] = int(filename)
            xrd_df.loc[idx, "comp"] = comp
            xrd_df.loc[idx, "xrd_dir"] = xrd_dir

            # reading xrd
            _df = pd.read_csv(xrd_dir, delimiter="\t", header=None)
            _df.columns = ["2theta", "intensity"]

            # label for plot
            label = f"{comp}"

        elif naming_format == 2:
            annealing_temp = ""
            annealing_time = ""
            annealing_env = ""
            basename = os.path.basename(xrd_dir)
            filename = re.search(r"^(.*)\.\w+$", basename).group(1)
            file_data = filename.split("-")
            date = file_data[0]
            splitid = file_data[1]
            rep = file_data[2]
            if len(file_data) > 3:
                annealing_temp = file_data[3][:-1]
                annealing_time = file_data[4][:-1]
                annealing_env = file_data[5][:]
            comp = ChemFormula.sorted_formula(xrd_idkey[int(splitid)])
            xrd_df.loc[idx, "xrd_dir"] = xrd_dir
            xrd_df.loc[idx, "filename"] = filename
            xrd_df.loc[idx, "xrdid"] = int(splitid)
            xrd_df.loc[idx, "comp"] = comp
            xrd_df.loc[idx, "date"] = date
            xrd_df.loc[idx, "rep"] = rep
            xrd_df.loc[idx, "annealing temp"] = annealing_temp
            xrd_df.loc[idx, "annealing time"] = annealing_time
            xrd_df.loc[idx, "annealing env"] = annealing_env
        else:
            pp_idkey = pd.read_csv(pp_idkey_dir)
            basename = os.path.basename(xrd_dir)
            filename = re.search(r"^(.*)\.\w+$", basename).group(1)
            file_data = filename.split("_")
            sample_source = re.sub(r"\d+", "", file_data[0])
            batch_number = re.findall(r"\d+", file_data[0])[0]
            date = file_data[1]
            comp = file_data[2]
            pp_id = file_data[3]
            if pp_id != "rt":
                row = pp_idkey[pp_idkey["pp_id"] == pp_id]
                annealing_temp = row["annealing_temp"][0]
                annealing_time = row["annealing_time"][0]
                annealing_env = row["annealing_env"][0]
            else:
                annealing_temp = ""
                annealing_time = ""
                annealing_env = ""
            rep = re.findall(r"\d+", file_data[4])[0]

            xrd_df.loc[idx, "sample_id"] = filename
            xrd_df.loc[idx, "xrd_dir"] = xrd_dir
            xrd_df.loc[idx, "date"] = date
            xrd_df.loc[idx, "rep"] = rep
            xrd_df.loc[idx, "target_comp"] = comp
            xrd_df.loc[idx, "annealing temp"] = annealing_temp
            xrd_df.loc[idx, "annealing time"] = annealing_time
            xrd_df.loc[idx, "annealing env"] = annealing_env

        # reading xrd
        xrd = xrdtools.XRD()
        xrd.read_xy_file(
            xrd_dir, delimiter=" ", skip_rows=skip_rows
        )  # pd.read_csv(xrd_dir, delimiter=' ')
        # _df = _df.rename(columns={'count':'intensity'})

        # label for plot
        label = f"{comp}: {annealing_temp}C-{annealing_time}h"

        # plotting xrd
        indx = np.where((start_cutoff < xrd.theta) & (xrd.theta < end_cutoff))
        X = xrd.theta[indx]
        Y = xrd.intensity[indx]

        plt.plot(X, Y, label=label)

    xrd_df = xrd_df.sort_values(by=["date", "target_comp"])
    xrd_df = xrd_df.reset_index(drop=True)

    plt.xlabel("2theta")
    plt.ylabel("intensity, a.u.")

    if plot:
        plt.show()

    plt.close()

    if save:
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        xrd_df.to_csv(f"{outfolder}/xrd.csv")

    return xrd_df


def create_xrd_dict(xrd_idx, xrd_df):
    """
    xrd filename format: {synthesis date, sample pos in rack, batch-no, sample-pos-sonicator, xrdid, noelements, sonication-temp, sonication-time}
    """
    # selecting xrd file for analysis
    comp = xrd_df.loc[xrd_idx, "target_comp"]
    xrd_dir = xrd_df.loc[xrd_idx, "xrd_dir"]
    sample_id = xrd_df.loc[xrd_idx, "sample_id"]

    return {
        "xrd_idx": xrd_idx,
        "sample_id": sample_id,
        "target_comp": comp,
        "xrd_dir": xrd_dir,
    }


def import_xrd(xrd_dict, tmin, tmax, mask=None, outfolder=None, skip_rows=None):
    start_cutoff = tmin
    end_cutoff = tmax
    xrd = xrdtools.XRD()
    xrd_dir = xrd_dict["xrd_dir"]
    xrd_id = xrd_dict["xrd_idx"]

    # importing xy data for XRD
    try:
        xrd.read_xy_file(xrd_dir, delimiter="\t")
    except:
        xrd.read_xy_file(xrd_dir, delimiter=" ", skip_rows=skip_rows)

    fig, axs = plt.subplots(3 if mask else 2, 1, figsize=(8, 12))
    axs[0].plot(xrd.theta, xrd.intensity, label="raw", color="black")

    # applying edits to xrd
    xrd.apply_theta_cutoffs(start_cutoff, end_cutoff)
    xrd_back = xrdtools.XRD()
    xrd_back.load_theta_intensity(xrd.theta, xrd.intensity)
    xrd.remove_bckg(technique=None, poly_degree=3)
    axs[0].plot(
        xrd.theta, xrd_back.intensity - xrd.intensity, label="background", color="red"
    )
    xrd.normalize()

    axs[1].plot(
        xrd.theta,
        xrd.intensity,
        label="background removed and normalized",
        color="black",
    )
    if mask:
        xrd.remove_substrate_peaks("Si111")

    # extracting expereimental DP from XRD
    DP = xrdtools.DiffractionPattern()
    DP.extract_pattern_from_xrd(xrd, peak_distance=5, intensity_threshold=0.02, width=5)
    DP.resize_array(tmin=xrd.tmin, tmax=xrd.tmax, npoints=len(xrd.theta))
    xrd.set_DP(DP)
    xrd.to_csv()

    if mask:
        axs[2].plot(xrd.theta, xrd.intensity, label="peaks removed", color="black")
        axs[2].bar(
            xrd.DP.theta,
            xrd.DP.intensity / 5,
            label="diffraction pattern",
            color="red",
            width=0.2,
        )
    else:
        axs[1].bar(
            xrd.DP.theta,
            xrd.DP.intensity / 5,
            label="diffraction pattern",
            color="red",
            width=0.2,
        )

    for ax in axs:
        ax.set_xlabel("2θ (degrees)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.legend(loc="upper right")
    plt.tight_layout()
    plt.close(fig)
    fig.savefig(f"{outfolder}/{xrd_dict['sample_id']}.png")

    return xrd


def populate_phases(xrd_dict, DB_list, phase_source, tmin, tmax):
    # given a composition populate all phases in the database that are relevant for the analysis i.e., create a Dict with formula as a key and cifid as a value

    comp = xrd_dict["target_comp"]
    cifdict = {}

    print(f"Populating phases and cif strcutures\n{'*'*80}")

    for i, DB in enumerate(DB_list):
        phases = []
        _cifdict = {}
        if comp in DB.data.keys():
            phases.extend([comp])

        oxides = DB.lookup_oxide_phases(comp)
        phases.extend(oxides)
        for num_elements in range(1, ChemFormula.count_elements(comp) + 1):
            phases.extend(
                DB.lookup_phases_in_composition_chemical_space(
                    comp, num_elements=num_elements
                )
            )

        _cifdict = DB.get_cifids(phases, prefix=list(phase_source.keys())[i])
        cifdict = {
            key: sorted(list(set(_cifdict.get(key, []) + cifdict.get(key, []))))[::-1]
            for key in set(_cifdict) | set(cifdict)
        }

    # filtering out similar cif structures
    cifdict1 = {}
    for formula, ciflist in cifdict.items():
        ciflist = xrdtools.filter_structures(
            formula, ciflist, method="pymatgen_matcher", phase_source=phase_source
        )
        cifdict1[formula] = ciflist

    return cifdict1


def import_xrf(xrd_dict, cifdict, grabtop, outfolder=None):
    ciflist, formulas, sourcelist = dbtools.cifdict2list(cifdict)
    ref_formulas = list(set(formulas))
    xrf_df_dir = f"{outfolder}/xrf.csv"
    xrd_df_dir = f"{outfolder}/xrd.csv"
    xrd_idx = xrd_dict["xrd_idx"]
    xrd_df = pd.read_csv(xrd_df_dir)

    # read xrf data and extract mean, stdev, and elements
    xrf_df = pd.read_csv(xrf_df_dir)
    mask = xrf_df.columns.str.endswith("_mean")
    xrf_mean = xrf_df.loc[:, mask]
    mask = xrf_df.columns.str.endswith("_stdev")
    xrf_stdev = xrf_df.loc[:, mask]
    cols = sorted([col.split("_")[0] for col in xrf_mean.columns])

    # find sample id in XRF
    sample_id = xrd_dict["sample_id"]
    target_comp = xrd_dict["target_comp"]
    row = xrf_df[xrf_df["sample_id"] == sample_id]
    elements = re.findall("[A-Z][a-z]*", target_comp)

    if len(row) < 1:
        xrd_df.loc[xrd_idx, "xrf_comp"] = None
        xrd_df.loc[xrd_idx, "matched_comp"] = target_comp
        xrd_df.to_csv(xrd_df_dir, index=False)
        return target_comp

    # make formula using xrf data
    comp = ""
    stdev = []
    for i in range(len(elements)):
        stoich = float(row[f"{cols[i]}_mean"]) / 100
        if (
            stoich < 0
        ):  # use 0 for elements that have stoichemtry below 0 and skip adding element to formula
            continue
        elif stoich > 1:  # use 1 for stoichemtries above 1
            comp = f"{elements[i]}-1.0"
            stdev.append(float(row[f"{cols[i]}_stdev"]))
            break
        comp += f"{elements[i]}-{np.round(stoich, 3)}-"
        stdev.append(float(row[f"{cols[i]}_stdev"]))

        if i == len(cols) - 1:
            comp = comp[:-1]
    xrf_comp = ChemFormula.strip_zeros(comp)

    # measure distance between formula and reference formulas
    matched_comp = ""
    distance_array = np.ones(len(ref_formulas))
    formulas_within_stdev = []
    for i, comp2 in enumerate(ref_formulas):
        # make sure xrf_comp is a subset of comp2; have the same elements
        for n in np.arange(0, ChemFormula.count_elements(xrf_comp), 2):
            if comp.split("-")[n] != comp2.split("-")[n]:
                continue

        # add all formulas within stdev
        if len(stdev) < ChemFormula.count_elements(comp2):
            stdev = stdev + [0] * (ChemFormula.count_elements(comp2) - len(stdev))
        if ChemFormula.check_formula_within_stdev(xrf_comp, comp2, stdev):
            formulas_within_stdev.append(comp2)

        # make sure that both vectors have the same size
        if ChemFormula.count_elements(xrf_comp) < ChemFormula.count_elements(comp2):
            temp_comp = ChemFormula.broadcast_formula(xrf_comp, comp2)
            temp_comp2 = comp2
        elif ChemFormula.count_elements(xrf_comp) > ChemFormula.count_elements(comp2):
            temp_comp2 = ChemFormula.broadcast_formula(comp2, xrf_comp)
            temp_comp = xrf_comp
        else:
            temp_comp = xrf_comp
            temp_comp2 = comp2

        # get stichiometries to calculate distance
        distance = ChemFormula.get_distance_between_formulas(temp_comp, temp_comp2)
        distance_array[i] = distance

    # measure offset between target comp and xrd_comp
    offset = ChemFormula.get_diff_between_formulas(target_comp, xrf_comp)

    matched_comp = ref_formulas[np.argmin(distance_array)]
    if matched_comp in formulas_within_stdev:
        formulas_within_stdev.remove(matched_comp)
    extra_phases = [
        i for i, formula in enumerate(formulas) if formula in formulas_within_stdev
    ]
    extra_phases_array = [[i] + [-1] * (grabtop - 1) for i in extra_phases]

    xrd_df.loc[xrd_idx, "xrf_comp"] = xrf_comp
    xrd_df.loc[xrd_idx, "matched_comp"] = matched_comp
    xrd_df.to_csv(xrd_df_dir, index=False)
    # print(f"{target_comp}-->{xrf_comp}-->{matched_comp}")
    # print(f"offset:  {offset}")
    return matched_comp, np.array(extra_phases_array)


def simulate_phases(cifdict1, phase_source, xrd, instprm, GSAS_outfolder):
    cell_output = f"\nSimualting XRD and DP data\n{'*'*80}"
    print(cell_output)
    # creating lists from cifdict (*using lists allow considering multiple cif structures for every phase)
    ciflist, formulas, sourcelist = dbtools.cifdict2list(cifdict1)
    simXRDs, simDPs, output = xrdtools.sim_phases(
        ciflist,
        sourcelist,
        formulas,
        phase_source,
        xrd.tmin,
        xrd.tmax,
        len(xrd.theta),
        instprm,
        outfolder=GSAS_outfolder,
        method="MP",
    )
    cell_output += output
    return simXRDs, simDPs, output


def get_corr_array(Icalc_array, Iobs, metric="emd"):
    """
    metric (str): distance metric used for similarity analysis . Must be one of {'euclidean', 'cityblock',
                    'cosine', 'hamming', 'jaccard', 'chebyshev', 'minkowski'}
    """

    scores = np.zeros(len(Icalc_array))
    for i, Icalc in enumerate(Icalc_array):
        if metric == "emd":
            scores[i] = wasserstein_distance(Icalc, Iobs)
        elif metric == "chi square":
            scores[i] = 0.5 * np.sum(((Icalc - Iobs) ** 2) / (Icalc + Iobs + 1e-6))
        else:
            scores[i] = distance.cdist([Icalc], [Iobs], metric)[0, 0]

    return scores


def similarity_analysis(cifdict1, xrd, simDPs, max_trials, grabtop, base_phases_array):
    # inputs
    ciflist, formulas, sourcelist = dbtools.cifdict2list(cifdict1)
    theta = xrd.theta

    # setting master arrays to store result of every iteration/trial
    grabtop_idx_array = np.ones((max_trials, grabtop, len(ciflist)), dtype=object) * -1
    CorrArray = np.ones((max_trials, grabtop, len(ciflist)), dtype=object) * 9999
    sel_phases_array = np.ones((max_trials, grabtop), dtype=object) * -1

    j = 0
    simDPs_dup = simDPs.copy()
    skip_idx = []
    print_nxt_it = True
    count = 0

    # iterate max_trails times
    while j < max_trials and count < len(simDPs_dup):
        # start with the original experimental xrd DP every iteration
        Iobs = xrd.DP.intensity

        # iniate sel_phases array of size graptob and -1 default value
        sel_phases = np.ones(grabtop) * -1

        i = 0
        # avoid using the same starting phases for every iteration again: add the first phase from sel_phases from previous iteration to a skip list
        if j > 0:
            skip_idx.append(sel_phases_array[j - 1][0])

        while i < grabtop:
            # start with the original DP from experimental XRD: Iobs_current = Iobs
            if i == 0:
                Iobs_current = Iobs
            # plt.figure()
            # plt.bar(theta, Iobs_current, label='Iobs')

            # create similarity score matrix between simulated DPs and Iobs_current
            scores = get_corr_array(simDPs_dup, Iobs_current, metric="emd")
            # sort scores and store in grabtop_idx array
            grabtop_idx = np.argsort(scores)

            # store in master arrays
            grabtop_idx_array[j, i] = grabtop_idx
            CorrArray[j, i] = scores

            # iterate through grabtop_idx and pick the idx that hasn't been used before
            for n in grabtop_idx:
                if n in sel_phases or (i == 0 and n in skip_idx):
                    continue
                else:
                    idx = n
                    break

            # get the simulated DP for that index
            Icalc = simDPs_dup[idx]
            # plt.bar(theta, Icalc, label='Icalc')

            # remove the simulated DP from the experimental DP
            Iobs_current = xrdtools.remove_overlapped_DP(
                theta=theta, Icalc=Icalc, Iobs=Iobs_current, window=1.2
            )
            sel_phases[i] = idx

            if len(np.where(Iobs_current > 0)[0]) == 0:
                break
            # plt.legend()
            i += 1

        # check if sel_phases array exists in master array: avoid adding the same combo again
        if not any(
            np.all(np.sort(sel_phases) == np.sort(row)) for row in sel_phases_array
        ) and not any(
            np.all(np.sort(sel_phases) == np.sort(row)) for row in base_phases_array
        ):
            sel_phases_array[j] = sel_phases
            j += 1
        else:
            skip_idx.append(sel_phases[0])

        count += 1
    mask = np.all(sel_phases_array == -1, axis=1)
    sel_phases_array = sel_phases_array[~mask]
    CorrArray = CorrArray[~mask]
    grabtop_idx_array = grabtop_idx_array[~mask]

    return grabtop_idx_array, CorrArray, sel_phases_array


def baseline_analysis(
    comp, cifdict, xrd, simDPs, grabtop, filter_method="energy", phase_source=None
):
    simDPs_dup = simDPs.copy()
    ciflist, formulas, sourcelist = dbtools.cifdict2list(cifdict)
    chemical_space = ChemFormula.get_chemical_space(comp)
    elements = [elm + "-1.0" for elm in chemical_space]

    Iobs = xrd.DP.intensity

    # filter out single elements from cifdict and choose one cif file per element using similarity analysis; get index list of elements that could be used in the master ciflist
    base_cifdict = {key: cifdict[key] for key in elements if key in cifdict}
    idx_list = []

    for key, item in base_cifdict.items():
        elm_ciflist, elm_formulas, elm_sourcelist = dbtools.cifdict2list({key: item})
        elm_simDPs = []
        elm_idx_list = []
        # get index number of all strcutures for that element in the simDP array
        for idx, formula in enumerate(formulas):
            if formula == key:
                elm_idx_list.append(idx)
                elm_simDPs.append(simDPs_dup[idx])

        if len(item) > 1:
            if filter_method == "xrd similarity":
                scores = get_corr_array(elm_simDPs, Iobs, metric="emd")
                sorted_idx = np.argsort(scores)
            else:
                try:
                    energy = xrdtools.sort_by_energy_cif(
                        elm_formulas, elm_ciflist, elm_sourcelist, phase_source
                    )
                    sorted_idx = np.argsort(energy)
                except:
                    print(
                        "Sorting by calculating energy failed. Switching to xrd similarity"
                    )
                    scores = get_corr_array(elm_simDPs, Iobs, metric="emd")
                    sorted_idx = np.argsort(scores)
            idx_list.append(elm_idx_list[sorted_idx[0]])
            base_cifdict[key] = [
                (elm_sourcelist[sorted_idx[0]], elm_ciflist[sorted_idx[0]])
            ]
        else:
            idx_list.append(elm_idx_list[0])
            base_cifdict[key] = [item]

    # create a combo list of all indicies
    base_phases = []
    if len(elements) > 1:
        combos = []
        for r in range(1, len(idx_list) + 1):
            combos.extend(combinations(idx_list, r))

        for combo in combos:
            combo = list(combo)
            if len(combo) < grabtop + 1:
                combo.extend([-1] * (grabtop - len(combo)))
            base_phases.append(np.array(combo))

    # add index of the target structure to list
    target_idx = []
    for idx, formula in enumerate(formulas):
        if formula == comp:
            target_idx.append([idx] + [-1] * (grabtop - 1))
    base_phases.extend(np.array(target_idx))

    base_phases_array = np.array(base_phases, dtype=object)

    return base_phases_array


def print_analysis_result(sel_phases_array, cifdict):
    output = (
        f"\nRunning similarity analysis to identify top correlated phases\n{'*'*80}"
    )
    print(output)
    ciflist, formulas, sourcelist = dbtools.cifdict2list(cifdict)
    for i, sel_phases in enumerate(sel_phases_array):
        if np.all(sel_phases == -1):
            continue
        print(f"iteration number: {i+1}")
        output += f"iteration number: {i+1}\n"
        print(f"number of phases matched: {np.count_nonzero(sel_phases !=-1)}")
        output += f"number of phases matched: {np.count_nonzero(sel_phases !=-1)}\n"
        print(
            f"phases matched: {[(formulas[int(i)], sourcelist[int(i)], ciflist[int(i)]) for i in sel_phases if i !=-1]}"
        )
        output += f"phases matched: {[(formulas[int(i)], sourcelist[int(i)], ciflist[int(i)]) for i in sel_phases if i !=-1]}\n"
        print("-" * 80)
        output += "-" * 80 + "\n"

    return output


def qRietRefine(
    xrd,
    cifdict,
    phase_source,
    sel_phases_array,
    max_trials,
    grabtop,
    instprm,
    timeout=None,
    outfolder=".",
):
    ciflist, formulas, sourcelist = dbtools.cifdict2list(cifdict)

    # Initializing score matrices
    rwp_matrix = np.ones(max_trials) * 999
    wt_matrix = np.zeros(max_trials, dtype=object)
    resd_matrix = np.zeros(max_trials)
    cell_output = ""
    refine_list = []
    with multiprocessing.Manager() as manager:
        for i in range(max_trials):
            sel_phase_idx = sel_phases_array[i]
            if np.all(sel_phase_idx == -1):
                continue
            queue = manager.Queue()
            process = multiprocessing.Process(
                target=xrdtools.GSASrefine_ciflist_multiprocessing,
                args=(
                    xrd,
                    ciflist,
                    sourcelist,
                    formulas,
                    phase_source,
                    sel_phase_idx,
                    instprm,
                    queue,
                    outfolder,
                ),
            )
            process.start()
            process.join(timeout=timeout)

            try:
                output, result = queue.get_nowait()
            except multiprocessing.queues.Empty:
                output, result = "Timeout or no output", None

            if process.is_alive():
                process.terminate()
                process.join()

            if result is None:
                print(f"Refining phases from iteration #{i+1} failed, skipping it")
                cell_output += (
                    f"Refining phases from iteration #{i+1} failed, skipping it\n"
                )
                rwp_matrix[i] = 9999
                wt_matrix[i] = np.ones(grabtop) * -1
                resd_matrix[i] = 9999
                refine_list.append(None)
                process.terminate()
                process.join()
            else:
                print(
                    f"Refining phases from iteration #{i+1} is completed succesfully!"
                )
                cell_output += (
                    f"Refining phases from iteration #{i+1} is completed succesfully!\n"
                )
                rwp_matrix[i] = result[0]
                wt_matrix[i] = result[1]
                resd_matrix[i] = result[2]
                refine_list.append(result[3])
                print(f"Rwp: {result[0]:.2f}")
                cell_output += f"Rwp: {result[0]:.2f}\n"

    # Remove all files used for GSAS refinement
    for f in glob.glob(f"{outfolder}/temp*"):
        if os.path.isfile(f):
            os.remove(f)

    return refine_list, rwp_matrix, wt_matrix, resd_matrix, cell_output


def get_major_phases(wt_t, filter_ind, wt_matrix, sel_phases_array):
    major_phases_idx = []
    major_phases_wt = []
    for i, idx in enumerate(filter_ind):
        if np.all(sel_phases_array[idx] == -1):
            continue

        if len(wt_matrix[idx]) > 1:
            if wt_t:
                ind_list = np.where(wt_matrix[idx] >= wt_t)[0]
            else:
                ind_list = np.where(wt_matrix[idx] == max(wt_matrix[idx]))[0]

            if len(ind_list) > 0:
                major_phases_idx.extend(sel_phases_array[idx][ind_list])
                major_phases_wt.extend(wt_matrix[idx][ind_list])
            else:
                major_phases_idx.extend([None])
                major_phases_wt.extend([None])
        else:
            major_phases_idx.extend([sel_phases_array[idx][0]])
            major_phases_wt.extend([1.0])

    return major_phases_idx, major_phases_wt


def produce_pdf(
    comp,
    refine_list,
    rwp_matrix,
    wt_matrix,
    resd_matrix,
    xrd,
    xrd_dict,
    cifdict,
    sel_phases_array,
    simDPs,
    outfolder,
    max_trials,
    metric_threshold=30,
    wt_threshold=-1,
    major_phase_wt=0.7,
):
    output = ""
    ciflist, formulas, sourcelist = dbtools.cifdict2list(cifdict)

    sample_id = xrd_dict["sample_id"]
    xrd_idx = xrd_dict["xrd_idx"]

    # load excel file
    xrd_df_dir = f"{outfolder}/xrd.csv"
    xrd_df = pd.read_csv(xrd_df_dir)

    # start a pdf document object
    pdf = canvas.Canvas(f"{outfolder}/{sample_id}.pdf", pagesize=A4)

    # filter by metric
    metric = rwp_matrix.copy()
    metric_mask = np.where(metric < metric_threshold)[0]
    filter_ind = metric_mask[np.argsort(metric[metric_mask])]

    # plotting experimental xrd
    print(f"Rietveld refinement for {comp}")
    output += f"Rietveld refinement for {comp}\n"
    print("*" * 80)
    output += "*" * 80 + "\n"

    text = pdf.beginText(50, 820)
    text.setFont("Helvetica", 10)
    text.textLines(f"Rietveld refinement for {comp}, xrd_id:{xrd_idx}\n{'*'*80}\n")
    pdf.drawText(text)
    pdf.drawImage(f"{outfolder}/{sample_id}.png", 75, 110, width=450, height=670)
    pdf.drawString(500, 20, f"Page {1}")
    pdf.showPage()

    # check if we have no solutions
    if len(filter_ind) < 1:
        xrd_df.loc[xrd_idx, "matching result"] = "no fits"
        xrd_df.to_csv(xrd_df_dir, index=False)
        pdf.drawString(250, 200, "No fits below threshold were found!")
        print(f"no fits were found below {metric_threshold:.3f}")
        output += f"no fits were found below {metric_threshold:.3f}\n"
        pdf.drawString(500, 20, f"Page {2}")
        pdf.showPage()
        pdf.save()
        return output

    # get the index for all target phase strcutures from ciflist
    combos = ChemFormula.formula_to_combos(comp)
    a_ind = np.arange(sel_phases_array.shape[0])
    base_ind = a_ind[a_ind > (max_trials + len(combos) - 1)]
    target_phase_idx = sel_phases_array[base_ind][:, 0].astype(int)

    # get the index, cif for all major phases in every solution; return None if there isn't a major phase > major_phase_wt
    major_phases_idx, major_phases_wt = get_major_phases(
        major_phase_wt, filter_ind, wt_matrix, sel_phases_array
    )
    major_phases = []
    for idx in major_phases_idx:
        if idx:
            phaseid = ciflist[int(idx)]
        else:
            phaseid = None
        major_phases.extend([phaseid])

    # find the intersection betweem target_phase_idx and major_phase_idx
    target_rankings = []
    target_rwp_scores = []
    target_matched_phases = []
    target_matched_formulas = []
    for i, idx in enumerate(major_phases_idx):
        if idx and ((int(idx) in target_phase_idx) or (formulas[int(idx)] == comp)):
            target_matched_phases.extend([ciflist[int(idx)]])
            target_matched_formulas.extend([formulas[int(idx)]])
            target_rankings.extend([i])
            target_rwp_scores.extend([metric[filter_ind[i]]])

    # get all matched phases; full list of phases
    matched_phases = []
    rankings = np.arange(1, len(filter_ind) + 1)
    rwp_scores = metric[filter_ind]
    wts_list = []
    for n, i in enumerate(filter_ind):
        # redirecting cell output
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        print(f"#ranked fit number: {n+1}")
        output += f"#ranked fit number: {n+1}\n"
        print("Phases chosen for refinement:")
        output += "Phases chosen for refinement:\n"
        sel_ciflist = []
        for j, phase_idx in enumerate(sel_phases_array[i]):
            if phase_idx != -1 and (
                wt_matrix[i][j] > wt_threshold or len(wt_matrix[i]) == 1
            ):
                phase_idx = int(phase_idx)
                sel_ciflist.append(ciflist[phase_idx])
                print(
                    f"    * {formulas[phase_idx]}_{sourcelist[phase_idx]}-{ciflist[phase_idx]}"
                )
                output += f"    * {formulas[phase_idx]}_{sourcelist[phase_idx]}-{ciflist[phase_idx]}\n"

        mask = np.where(wt_matrix[i] > wt_threshold)[0]

        wts = (
            wt_matrix[i][mask] / np.sum(wt_matrix[i][mask])
            if len(wt_matrix[i][mask]) > 1
            else [1.0]
        )
        print(
            f"rwp_score: {rwp_matrix[i]:.1f}, resd_score: {resd_matrix[i]:.3f}, weights: [{', '.join(f'{wt:.2f}' for wt in wts)}]"
        )
        output += f"rwp_score: {rwp_matrix[i]:.1f}, resd_score: {resd_matrix[i]:.3f}, weights: [{', '.join(f'{wt:.2f}' for wt in wts)}]\n"
        print("-" * 80)
        output += "-" * 80 + "\n"

        # tabulating all solution in excel
        wts_list.append([round(wt, 2) for wt in wts])
        matched_phases.append(sel_ciflist)

        # redirecting cell output back
        sys.stdout = old_stdout

        output_string = new_stdout.getvalue()
        print(output_string)

        # printing text and image in pdf
        text = pdf.beginText(50, 820)
        text.setFont("Helvetica", 10)
        text.textLines(f"Rietveld refinement for {comp}, xrd_id:{xrd_idx}\n{'*'*80}\n")
        text.textLines(output_string)
        pdf.drawText(text)

        # plotting if refinement exsit
        if refine_list[i]:
            # plotting refine fitting plots
            fig1 = xrdtools.plot_GSASrefine(
                refine_list[i],
                i,
                f"{n+1}: score={resd_matrix[i]:.3f}, rwp={rwp_matrix[i]:.3f}",
            )
            fig1.savefig(f"{outfolder}/temp{n}.png")
            plt.show()
            plt.close(fig1)
            pdf.drawImage(f"{outfolder}/temp{n}.png", 100, 410, width=400, height=300)
            if os.path.exists(f"{outfolder}/temp{n}.png"):
                os.remove(f"{outfolder}/temp{n}.png")
            # plotting refine map
            if len(wt_matrix[i]) < 2:
                sel_phases = sel_phases_array[i]
            else:
                sel_phases = sel_phases_array[i][
                    np.where(wt_matrix[i] > wt_threshold)[0]
                ]

            # adding diffraction map
            try:
                fig2 = xrdtools.plot_pattern_mapv2(
                    xrd=xrd,
                    Iarray=simDPs,
                    ciflist=ciflist,
                    sourcelist=sourcelist,
                    formulas=formulas,
                    sel_phase_idx=sel_phases,
                    samplename=comp,
                    tmin=xrd.tmin,
                    tmax=xrd.tmax,
                )
                fig2.savefig(f"{outfolder}/temp{n}_2.png")
                plt.show()
                plt.close(fig2)
                pdf.drawImage(
                    f"{outfolder}/temp{n}_2.png", 0, 100, width=550, height=275
                )
                if os.path.exists(f"{outfolder}/temp{n}_2.png"):
                    os.remove(f"{outfolder}/temp{n}_2.png")
            except:
                print("DP map was not generated due to error")

        # add pdf pages
        pdf.drawString(500, 20, f"Page {n+2}")
        pdf.showPage()

        # tabulate matching results in xrd_df
        xrd_df.loc[xrd_idx, "matching result"] = (
            "matched" if len(target_matched_phases) > 0 else "not matched"
        )
        xrd_df.loc[xrd_idx, "total no of solutions"] = int(len(filter_ind))
        xrd_df.loc[xrd_idx, "matched phases_target"] = str(
            [[phase] for phase in target_matched_phases]
        )
        xrd_df.loc[xrd_idx, "matched formulas_target"] = str(
            [[formula] for formula in target_matched_formulas]
        )
        xrd_df.loc[xrd_idx, "rankings_target"] = str(target_rankings)
        xrd_df.loc[xrd_idx, "rwp scores_target"] = str(
            [round(score, 1) for score in target_rwp_scores]
        )
        xrd_df.loc[xrd_idx, "major phases"] = str(major_phases)
        xrd_df.loc[xrd_idx, "major phases weights"] = str(major_phases_wt)
        xrd_df.loc[xrd_idx, "matched phases"] = str(matched_phases)
        xrd_df.loc[xrd_idx, "weights"] = str(wts_list)
        xrd_df.loc[xrd_idx, "rwp scores"] = str(
            [round(score, 1) for score in rwp_scores]
        )
        xrd_df.to_csv(xrd_df_dir, index=False)

    pdf.save()

    return output
