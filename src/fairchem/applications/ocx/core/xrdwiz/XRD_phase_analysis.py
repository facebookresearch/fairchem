from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import xrdwiz
from xrdwiz.utils import ChemFormula, dbtools

# ============================================================
# USER-DEFINED PATHS — set these before running
# ============================================================

# Data directories
DATA_DIR = ""  # e.g. "/checkpoint/abedj/notebooks/co2rr_analysis/data"
XRD_FOLDERNAME = ""  # e.g. "vsp-batch1-240510"

# XRD ID spreadsheets
XRDID_EXCEL = ""  # e.g. "<DATA_DIR>/jk_xrdid.xlsx"
MOO_EXCEL = ""  # e.g. "<DATA_DIR>/MOO_CO2R_prediction_list-231218.xlsx"

# GSAS instrument parameter file
INSTPRM_FILE = ""  # e.g. "<DATA_DIR>/XRD/GSAS/tut.instprm"

# Database files
COD_DB_FILE = ""  # e.g. "<DATA_DIR>/XRD/COD/COD_DB.pickle"
PBE_DB_FILE = ""  # e.g. "<DATA_DIR>/PBE_bulks.pickle"
PBE_CIF_DIR = ""  # e.g. "<DATA_DIR>/pbe_bulk_cifs"

# ============================================================


# loading XRD_id df; Jiheon's xrdid sheet
_df = pd.read_excel(XRDID_EXCEL, header=None)
xrdid2bulkid = list(_df[0].apply(ChemFormula.strip_zeros))

# loading XRD_id df; Brook's MOO pipeline
split_df = pd.read_excel(MOO_EXCEL, sheet_name="230901-cat", skiprows=1).iloc[:, 1:]


def prepare_inputs():
    main_dir = DATA_DIR
    foldername = XRD_FOLDERNAME

    # step 1
    xrd_file_dir = f"{main_dir}/XRD/{foldername}/*"
    # xrd_idkey= list(split_df['samples']) #xrdid2bulkid
    pp_idkey_dir = f"{main_dir}/materials_post_processing_id.csv"
    skip_rows = [2]
    tmin = 20
    tmax = 60
    wafer_mask = True

    # step 2
    GSAS_outfolder = f"{main_dir}/XRD/{foldername}/"
    instprm = INSTPRM_FILE

    # steps 4 and 5
    grabtop = 3
    max_trials = 50

    # step 6
    GSAS_timeout = 30

    # results
    outfolder = f"{main_dir}/XRD/{foldername}/analysis"
    metric_threshold = 50
    wt_threshold = -1

    return [
        tmin,
        tmax,
        xrd_file_dir,
        pp_idkey_dir,
        grabtop,
        max_trials,
        GSAS_timeout,
        GSAS_outfolder,
        instprm,
        outfolder,
        metric_threshold,
        wt_threshold,
        wafer_mask,
        skip_rows,
    ]


def logging(
    output,
    outfolder,
):
    with open(outfolder + "/log.txt", "a") as file:
        file.write("\n" + output + "\n")


def prepare_DB():
    # step 0 --> setup new dbs object (then preparing a phase_source dict with mixed dbs, more than 1 db source)
    COD = dbtools.CIF_DB()
    COD.load_db(COD_DB_FILE)
    PBE = dbtools.CIF_DB()
    PBE.load_db(PBE_DB_FILE)
    PBE.cif_folder_dir = PBE_CIF_DIR
    PBE.cif_folder_tree_format = "formula/cif"
    DB_list = [PBE, COD]
    # create dictionary providing a description of the DB objects in the DB_list
    phase_source = {
        "PBE": {
            "filetype": "cif",
            "dir": PBE.cif_folder_dir,
            "db_structure": PBE.cif_folder_tree_format,
        },
        "COD": {
            "filetype": "cif",
            "dir": COD.cif_folder_dir,
            "db_structure": COD.cif_folder_tree_format,
        },
    }

    return (
        DB_list,
        phase_source,
    )


def run(xrd_df, xrd_idx):
    extra_phases_array = None

    # step 0
    inputs = prepare_inputs()
    DB_list, phase_source = prepare_DB()
    logging(f"{datetime.now()}\n", outfolder=inputs[9])
    # step 1
    xrd_dict = xrdwiz.create_xrd_dict(xrd_idx=xrd_idx, xrd_df=xrd_df)
    xrd = xrdwiz.import_xrd(
        xrd_dict,
        tmin=inputs[0],
        tmax=inputs[1],
        mask=inputs[12],
        outfolder=inputs[9],
        skip_rows=inputs[13],
    )
    comp = xrd_dict["target_comp"]
    # step 2
    cifdict = xrdwiz.populate_phases(
        xrd_dict, DB_list, phase_source, tmin=inputs[0], tmax=inputs[1]
    )
    print(f"{xrd_idx}: Populating phases for {comp}\n{'|'*80}+\n")
    logging(
        f"{xrd_idx}: Populating phases for {comp}\n{'|'*80}+\n", outfolder=inputs[9]
    )
    # step 3 (optional)
    comp, extra_phases_array = xrdwiz.import_xrf(
        xrd_dict, cifdict, outfolder=inputs[9], grabtop=inputs[4]
    )
    print(f"{xrd_idx}: Running XRD analysis for {comp}\n{'|'*80}+\n")
    logging(
        f"{xrd_idx}: Running XRD analysis for {comp}\n{'|'*80}+\n", outfolder=inputs[9]
    )
    # step 3
    _simXRDs, simDPs, output = xrdwiz.simulate_phases(
        cifdict, phase_source, xrd, instprm=inputs[8], GSAS_outfolder=inputs[7]
    )
    logging(output, outfolder=inputs[9])
    # step 4a
    base_phases_array = xrdwiz.baseline_analysis(
        comp,
        cifdict,
        xrd,
        simDPs,
        grabtop=inputs[4],
        phase_source=phase_source,
        filter_method="energy",
    )
    # step 4b
    _grabtop_idx_array, _CorrArray, sel_phases_array = xrdwiz.similarity_analysis(
        cifdict,
        xrd,
        simDPs,
        max_trials=inputs[5],
        grabtop=inputs[4],
        base_phases_array=base_phases_array,
    )
    # step 4c
    if extra_phases_array is not None and len(extra_phases_array) > 0:
        sel_phases_array = np.vstack(
            (sel_phases_array, base_phases_array, extra_phases_array)
        )
    else:
        sel_phases_array = np.vstack((sel_phases_array, base_phases_array))
    output = xrdwiz.print_analysis_result(sel_phases_array, cifdict)
    logging(output, outfolder=inputs[9])
    # step 5
    refine_list, rwp_matrix, wt_matrix, resd_matrix, output = xrdwiz.qRietRefine(
        xrd,
        cifdict,
        phase_source,
        sel_phases_array,
        max_trials=len(sel_phases_array),
        grabtop=inputs[4],
        instprm=inputs[8],
        timeout=inputs[6],
        outfolder=inputs[9],
    )
    logging(output, outfolder=inputs[9])
    # step 6
    output = xrdwiz.produce_pdf(
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
        outfolder=inputs[9],
        max_trials=inputs[5],
        metric_threshold=inputs[10],
        wt_threshold=inputs[11],
    )
    logging(output, outfolder=inputs[9])


# main
inputs = prepare_inputs()
xrd_df = xrdwiz.create_xrd_df(
    xrd_file_dir=inputs[2],
    pp_idkey_dir=inputs[3],
    outfolder=inputs[9],
    tmin=inputs[0],
    tmax=inputs[1],
    plot=True,
    save=True,
    skip_rows=inputs[13],
)
if os.path.exists(inputs[9] + "/log.txt"):
    os.remove(inputs[9] + "/log.txt")

for xrd_idx in range(len(xrd_df)):
    run(xrd_df, xrd_idx)
