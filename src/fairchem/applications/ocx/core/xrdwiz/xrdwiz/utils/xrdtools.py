from __future__ import annotations

import csv
import glob
import os
import sys

import chardet
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from BaselineRemoval import BaselineRemoval
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.io.cif import CifParser
from scipy.signal import find_peaks, savgol_filter
from tqdm import tqdm

GSASII_DIR = ""  # e.g. "/private/home/abedj/g2full/GSASII"
if GSASII_DIR:
    sys.path.insert(0, GSASII_DIR)
import io
from contextlib import redirect_stdout
from copy import deepcopy

import ase
import ase.io
import GSASIIscriptable as G2sc
import seaborn as sns
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pymatgen.analysis.structure_matcher import ElementComparator, StructureMatcher


class XRD:
    def __init__(self):
        self.theta = []
        self.intensity = []

    def load_theta_intensity(self, theta=None, intensity=None):
        self.theta = np.array(theta)
        self.intensity = np.array(intensity)
        if len(theta) > 0:
            self.tmin = min(self.theta)
            self.tmax = max(self.theta)
            self.imin = min(self.intensity)
            self.imax = max(self.intensity)
            self.raw_intensity = self.intensity
            self.raw_theta = self.theta
            self.update_res()

    def read_xy_file(self, filepath=None, delimiter=None, skip_rows=0):
        self.filepath = filepath

        with open(filepath, "rb") as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result["encoding"]

        with open(filepath, encoding=encoding) as f:
            reader = csv.reader(f, delimiter=delimiter)

            for n, row in enumerate(reader):
                if skip_rows:
                    if n < skip_rows[0]:
                        continue
                    if len(skip_rows) > 1 and n > skip_rows[1]:
                        break

                if len(row) > 1:
                    self.theta.append(float(row[0]))
                    self.intensity.append(float(row[1]))
                else:
                    raise ValueError(
                        "The XRD file was not correctly read. Check the delimiter chosen and make sure there are only two columns in the file."
                    )

            self.load_theta_intensity(theta=self.theta, intensity=self.intensity)

    def remove_bckg(self, technique="ZhangFit", poly_degree=None):
        baseObj = BaselineRemoval(self.intensity)
        if technique == "ZhangFit":
            self.intensity = baseObj.ZhangFit()
        else:
            poly_degree = poly_degree
            if technique == "Modpoly":
                self.intensity = baseObj.ModPoly(poly_degree)
            else:
                self.intensity = baseObj.IModPoly(poly_degree)
        self.update_res()

    def normalize(self):
        a = np.max(self.intensity)
        b = np.min(self.intensity)
        self.intensity = (self.intensity - b) / (a - b)
        self.update_res()

    def remove_substrate_peaks(self, substrate=None):
        subdict = {
            "Si111": {
                "peak positions": [[28.6, 29.2], [33, 33.7], [54.8, 55.3], [56.54, 57]],
                "prominence": 0.002,
                "peak width": 1.6,
                "noise sample len": 0.4,
            }
        }
        tmin, tmax = min(self.theta), max(self.theta)
        res = len(self.theta) / (tmax - tmin)

        # find peaks bounded by ranges from dict
        target_2theta_range = subdict[substrate]["peak positions"]
        peaks_list = []
        s_intensity = savgol_filter(
            self.intensity, window_length=int(res / 2), polyorder=3
        )

        for r in target_2theta_range:
            peaks = find_peaks(
                s_intensity[int((r[0] - tmin) * res) : int((r[1] - tmin) * res)],
                prominence=subdict[substrate]["prominence"],
            )[0]
            peaks_list.extend([peak + int((r[0] - tmin) * res) for peak in peaks])
        peaks_array = np.array(peaks_list)

        for peak in peaks_array:
            if peak > 0 and peak < len(self.intensity) - 1:
                # get edges of the peak using a predefined delta
                peak_edges = [
                    max(0, peak - int(subdict[substrate]["peak width"] / 2 * res)),
                    min(
                        len(self.intensity) - 1,
                        peak + int(subdict[substrate]["peak width"] / 2 * res),
                    ),
                ]

                # get average value for background level
                background_level = (
                    self.intensity[peak_edges[0]] + self.intensity[peak_edges[1]]
                ) / 2

                # mask the peak
                masked_intensity = self.intensity[peak_edges[0] : peak_edges[1]]

                # determine length of noise sample
                noise_sample_length = int(
                    subdict[substrate]["noise sample len"] / 2 * res
                )

                # select region before the peak edge
                noise_sample_edges = [
                    max(0, peak_edges[0] - noise_sample_length),
                    min(len(self.intensity) - 1, peak_edges[0] + noise_sample_length),
                ]
                noise_sample_mask1 = list(
                    range(noise_sample_edges[0], noise_sample_edges[1])
                )

                # select region after the peak edge
                noise_sample_edges = [
                    max(0, peak_edges[1] - noise_sample_length),
                    min(len(self.intensity) - 1, peak_edges[1] + noise_sample_length),
                ]
                noise_sample_mask2 = list(
                    range(noise_sample_edges[0], noise_sample_edges[1])
                )

                # concatentate both masks and get std the two selected regions
                noise_sample_mask = noise_sample_mask1 + noise_sample_mask2
                noise_level = np.std(self.intensity[noise_sample_mask]) / 2

                # apply background with noise to remove the peaks
                masked_intensity = background_level
                masked_intensity += np.random.normal(
                    0, noise_level, size=int(subdict[substrate]["peak width"] * res)
                )

                self.intensity[peak_edges[0] : peak_edges[1]] = masked_intensity

    def update_res(self):
        if len(self.theta) > 0:
            self.res = (self.tmax - self.tmin) / len(self.theta)

    def reset_raw_data(self):
        self.intensity = self.raw_intensity
        self.theta = self.raw_theta
        self.tmin = min(self.theta)
        self.tmax = max(self.theta)
        self.imin = min(self.intensity)
        self.imax = max(self.intensity)
        self.update_res()

    def apply_theta_cutoffs(self, tmin=None, tmax=None):
        if tmin and tmax:
            self.tmin = tmin
            self.tmax = tmax
            ind_list = np.where((self.tmin < self.theta) & (self.theta < self.tmax))
            self.theta = self.theta[ind_list]
            self.intensity = self.intensity[ind_list]
            self.update_res()
            self.filter_indices(ind_list)

    def apply_intensity_cutoffs(self, imin=None, imax=None):
        if imin and imax:
            self.imin = imin
            self.imax = imax
            ind_list = np.where(
                (self.intensity >= self.imin) & (self.intensity <= self.imax)
            )
            self.theta = self.theta[ind_list]
            self.intensity = self.intensity[ind_list]
            self.update_res()
            self.filter_indices(ind_list)

    def set_DP(self, DP):
        self.DP = DP

    def filter_indices(self, indices):
        self.filtered_indices = indices

    def to_csv(self, outdir=None, filename=None):
        if not outdir:
            outdir = os.path.join(os.path.dirname(self.filepath), "processed")
            os.makedirs(outdir, exist_ok=True)

        if not filename:
            filename = os.path.basename(self.filepath).split(".")[0]

        data = np.column_stack((self.theta, self.intensity))
        np.savetxt(f"{outdir}/{filename}.csv", data, delimiter=",")

        self.processed_filepath = f"{outdir}/{filename}.csv"


class DiffractionPattern(XRD):
    def set_calc(self):
        self.xrd_calc = XRDCalculator(wavelength="CuKa", symprec=0.0)

    def gen_pattern_from_cif(self, cifpath):
        self.cifpath = cifpath
        self.set_calc()
        try:
            cif = CifParser(self.cifpath)
            structure = cif.parse_structures()[0]
            pattern = self.xrd_calc.get_pattern(structure)
            super().load_theta_intensity(pattern.x, pattern.y)
        except:
            raise ValueError("Loading cif structure failed!")

    def extract_pattern_from_xrd(
        self, xrd, peak_distance=5, intensity_threshold=0.02, width=5
    ):
        peak_pos, _ = find_peaks(
            xrd.intensity,
            height=intensity_threshold,
            distance=peak_distance,
            width=width,
        )
        super().load_theta_intensity(xrd.theta[peak_pos], xrd.intensity[peak_pos])

    def resize_array(self, tmin=None, tmax=None, npoints=None):
        x_new = np.linspace(tmin, tmax, num=int(npoints))
        y_new = np.zeros_like(x_new)
        res = (tmax - tmin) / npoints
        if len(x_new) != len(self.theta):
            indices = ((self.theta - tmin) / res).astype(int)
            y_new[indices] = self.intensity

            self.theta = x_new
            self.intensity = y_new
            self.filter_indices(indices)

    def select_top_reflections(self, grabtop=4):
        indices = np.argsort(self.intensity)
        self.intensity = self.intensity[indices[-grabtop:]]
        self.theta = self.theta[indices[-grabtop:]]
        self.filter_indices(indices[-grabtop:])

    def get_unique_elements(self, tolerance=0.5):
        unique_elements, indices = np.unique(self.theta, return_index=True)
        self.intensity = self.intensity[indices]
        self.theta = self.theta[indices]
        self.filter_indices(indices)

    def remove_overlapping_elements(self, tolerance):
        theta = self.theta
        I = self.intensity
        differences = np.abs(theta[:, None] - theta)
        indices = np.where(differences <= tolerance)
        indices = list(zip(*indices))

        # remove indices when i==j and (i,j)==(j,i)
        indices_set = []
        remove_indices = []
        for idx in indices:
            if idx[0] != idx[1] and (idx[1], idx[0]) not in indices_set:
                indices_set.append(idx)
                if I[idx[0]] < I[idx[1]]:
                    remove_idx = idx[0]
                else:
                    remove_idx = idx[1]
                remove_indices.append(remove_idx)

        I[remove_indices] = 0
        new_indices = np.where(I > 0)
        I = I[new_indices]
        theta = theta[new_indices]

        self.intensity = I
        self.theta = theta
        self.filter_indices(new_indices)


def remove_overlapped_DP(theta, Icalc, Iobs, window=0.5):
    npoints = len(Iobs)
    res = npoints / (max(theta) - min(theta))
    window = (window / 2) * res

    Iobs_new = Iobs.copy()
    pos_calc = np.where(Icalc > 0)[0]
    pos_obs = np.where(Iobs > 0)[0]
    indices = np.where(np.any(np.abs(pos_obs[:, None] - pos_calc) <= window, axis=1))[0]
    Iobs_new[pos_obs[indices]] = 0

    return Iobs_new


class GSAS_XRD(XRD):
    def set_gpx_project(self, project_name=None, outfolder=".", instprm=None):
        self.project_name = project_name
        os.makedirs(outfolder, exist_ok=True)
        self.project_path = f"{outfolder}/{project_name}.gpx"
        self.gpx = G2sc.G2Project(newgpx=self.project_path)
        self.instprm = instprm

    def simulate_phase(self, cifpath, phasename, tmin, tmax, npoints, n_cycle=20):
        self.cifpath = cifpath
        self.phasename = phasename

        phase = self.gpx.add_phase(
            self.cifpath, phasename=self.phasename, fmthint="CIF"
        )
        hist = self.gpx.add_simulated_powder_histogram(
            self.phasename,
            iparams=self.instprm,
            Tmin=tmin,
            Tmax=tmax,
            Npoints=npoints,
            phases=self.gpx.phases(),
            scale=1.0,
        )
        self.gpx.set_Controls("cycles", n_cycle)
        self.gpx.do_refinements([{}])
        x = self.gpx.histogram(0).getdata("x")
        y = self.gpx.histogram(0).getdata("ycalc")
        super().load_theta_intensity(x, y)

    def get_miller_labels(self):
        pwdr = self.gpx.histograms()[0]
        phase = list(pwdr.reflections().keys())[0]
        reflection_dict = pwdr.reflections()[phase]["RefList"]
        self.DP = DiffractionPattern()
        miller_indices = []
        dspaces = []
        thetas = []
        intensity = []
        for array in reflection_dict:
            (h, k, l), dspace, theta, icalc = self.parse_reflections(array)
            miller_indices.append((h, k, l))
            dspaces.append(dspace)
            thetas.append(theta)
            intensity.append(icalc)

        intensity = intensity / max(intensity)
        self.DP.dspaces = np.array(dspaces)
        self.DP.miller_indices = np.array(miller_indices)
        self.DP.load_theta_intensity(thetas, intensity)

    def parse_reflections(self, array):
        """
        0,1,2	(float) h,k,l
        3	(int) multiplicity
        4	(float) d-space, Å
        5	(float) pos, two-theta
        6	(float) sig, Gaussian width
        7	(float) gam, Lorenzian width
        8	(float) F2obs
        9	(float) F2calc
        10	(float) reflection phase, in degrees
        11	(float) intensity correction for reflection, this times F2obs or F2calc gives Iobs or Icalc
        12	(float) Preferred orientation correction
        13	(float) Transmission (absorption correction)
        14	(float) Extinction correction
        """

        h = array[0]
        k = array[1]
        l = array[2]
        dspace = array[4]
        theta = array[5]
        icalc = array[9] * array[11]
        return ((h, k, l), dspace, theta, icalc)

    def refine(
        self,
        xrd_csv,
        cifdict,
        phase_source=None,
        n_cycles=20,
        selected_phases=None,
        timeout=None,
    ):
        self.gpx.data["Controls"]["data"]["max cyc"] = n_cycles
        hist = self.gpx.add_powder_histogram(datafile=xrd_csv, iparams=self.instprm)
        hist.data["Instrument Parameters"][0]["I(L2)/I(L1)"] = [0.5, 0.5, 0]

        phases = []
        for formula in cifdict.keys():
            if selected_phases and (formula not in selected_phases):
                continue
            cifitems = cifdict[formula]
            for cifitem in cifitems:
                source = cifitem[0]
                cifid = cifitem[1]
                cifpath = get_cifpath(formula, source, cifid, phase_source)
                phases.append(
                    self.gpx.add_phase(
                        cifpath, phasename=f"{formula}_{cifid}", histograms=[hist]
                    )
                )

        # enabling Cell=True is causing problem. Not sure why?
        refdict0 = {
            "set": {
                "Background": {"no. coeffs": 6, "refine": True},
                "Scale": True,
                "Instrument Parameters": ["Zero"],
                "Sample Parameters": ["DisplaceX"],
            }
        }

        refdict_ori = {"set": {"Pref.Ori.": True}, "phases": phases[0]}
        refdict1 = {"set": {"Instrument Parameters": ["U", "V", "W"]}}
        refList = [refdict0, refdict1, refdict_ori]

        self.gpx.save(self.project_path)
        self.gpx.do_refinements(refList)
        self.get_wfracs(no_phases=len(phases))
        self.get_rwp()
        res = np.array(self.gpx.histogram(0).getdata("residual"))
        self.residual_distance = np.linalg.norm(res)

    def refine_one_phase(self, cifpath, xrd_csv, phasename="dummy", n_cycles=10):
        self.gpx.data["Controls"]["data"]["max cyc"] = n_cycles
        hist = self.gpx.add_powder_histogram(datafile=xrd_csv, iparams=self.instprm)
        hist.data["Instrument Parameters"][0]["I(L2)/I(L1)"] = [0.5, 0.5, 0]
        phases = []
        phases.append(
            self.gpx.add_phase(cifpath, phasename=phasename, histograms=[hist])
        )

        refdict0 = {
            "set": {
                "Background": {"no. coeffs": 6, "refine": True},
                "Scale": True,
                "Instrument Parameters": ["Zero"],
                "Sample Parameters": ["DisplaceX"],
            }
        }
        refdict_ori = {"set": {"Pref.Ori.": True}, "phases": phases[0]}
        refdict1 = {"set": {"Instrument Parameters": ["U", "V", "W"]}}
        refList = [refdict0, refdict1, refdict_ori]

        self.gpx.save(self.project_path)
        self.gpx.do_refinements(refList)
        self.get_wfracs(no_phases=len(phases))
        self.get_rwp()
        res = np.array(self.gpx.histogram(0).getdata("residual"))
        self.residual_distance = np.linalg.norm(res)

    def refine_ciflist(
        self,
        xrd_csv,
        ciflist,
        sourcelist,
        formulas,
        phase_source,
        sel_phase_idx,
        n_cycles=10,
    ):
        self.gpx.data["Controls"]["data"]["max cyc"] = n_cycles
        hist = self.gpx.add_powder_histogram(datafile=xrd_csv, iparams=self.instprm)
        hist.data["Instrument Parameters"][0]["I(L2)/I(L1)"] = [0.5, 0.5, 0]

        phases = []
        for idx in sel_phase_idx:
            if idx != -1:
                idx = int(idx)
                source = sourcelist[idx]
                cifid = ciflist[idx]
                formula = formulas[idx]
                cifpath = get_cifpath(formula, source, cifid, phase_source)
                phases.append(
                    self.gpx.add_phase(
                        cifpath, phasename=f"{formula}_{cifid}", histograms=[hist]
                    )
                )

        # enabling Cell=True is causing problem. Not sure why?
        refdict0 = {
            "set": {
                "Background": {"no. coeffs": 6, "refine": True},
                "Scale": True,
                "Instrument Parameters": ["Zero"],
                "Sample Parameters": ["DisplaceX"],
            }
        }

        refdict_ori = {"set": {"Pref.Ori.": True}, "phases": phases[0]}
        refdict1 = {"set": {"Instrument Parameters": ["U", "V", "W"]}}
        refList = [refdict0, refdict1, refdict_ori]

        self.gpx.save(self.project_path)
        self.gpx.do_refinements(refList)
        self.get_wfracs(no_phases=len(phases))
        self.get_rwp()
        res = np.array(self.gpx.histogram(0).getdata("residual"))
        self.residual_distance = np.linalg.norm(res)

    def get_wfracs(self, no_phases=None):
        wtfracs = []
        with open(self.project_path.replace(".gpx", ".lst")) as file:
            data = file.read()
            for i in range(1, no_phases + 1):
                try:
                    wt = eval(
                        data.split("Weight fraction")[i].split(",")[0].split(":")[-1]
                    )
                except:
                    wt = -1
                wtfracs.append(wt)
        self.wt = np.array(wtfracs)

    def get_rwp(self):
        rwp = -1
        for hist in self.gpx.histograms():
            self.rwp = hist.get_wR()


def sim_phases(
    ciflist,
    sourcelist,
    formulas,
    phase_source,
    tmin,
    tmax,
    npoints,
    instprm,
    outfolder,
    method="GSAS",
):
    """
    generating simXRDs and simDPs arrays for analysis
    input:
    ciflist: a list of cif ids
    sourcelist: a list of the parent directory of the cif files for each cif id
    formulas: a list of chemical formulas for the cif files
    phase_source: a dict containing description of the databases
    tmin, tmax, npoints: the minimum, maximum, and total number of points for the 2theta array/x-axis
    instprm: the instrument parameter used for XRD simulation
    outfolder: output folder directory
    method: the method used to simulate XRD data: GSAS or MP calculator
    """

    simXRDs = []
    simDPs = []
    output = ""
    for i, cifid in enumerate(tqdm(ciflist)):
        cifpath = get_cifpath(formulas[i], sourcelist[i], cifid, phase_source)
        print(cifpath)
        output += f"{cifpath}\n"
        if method == "GSAS":
            # get diffraction pattern using GSAS
            simxrd = GSAS_XRD()
            simxrd.set_gpx_project(
                instprm=instprm,
                project_name=f"{formulas[i]}_{sourcelist[i]}-{cifid}",
                outfolder=outfolder,
            )
            simxrd.simulate_phase(
                cifpath=cifpath,
                phasename=str(cifid),
                tmin=tmin,
                tmax=tmax,
                npoints=npoints,
            )
            simxrd.normalize()
            simxrd.get_miller_labels()
            simxrd.DP.normalize()
            simxrd.DP.apply_intensity_cutoffs(imin=0.01, imax=1)
            simxrd.DP.get_unique_elements()
            # simxrd.DP.remove_overlapping_elements(tolerance=1)
            # simxrd.DP.select_top_reflections(grabtop=5)
            simxrd.DP.resize_array(tmin=tmin, tmax=tmax, npoints=npoints)

            simDPs.append(simxrd.DP.intensity)
            simXRDs.append(simxrd.intensity)
        else:
            # simulate pattern using MP calculator
            simxrd = DiffractionPattern()
            simxrd.gen_pattern_from_cif(cifpath)
            simxrd.normalize()
            simxrd.apply_theta_cutoffs(tmin, tmax)
            simxrd.get_unique_elements()
            simxrd.resize_array(tmin=tmin, tmax=tmax, npoints=npoints)
            simDPs.append(simxrd.intensity)

    return np.array(simXRDs), np.array(simDPs), output


def get_cifpath(formula, source, cifid, phase_source):
    if phase_source[source]["db_structure"] == "/cif":
        cifpath = f"{phase_source[source]['dir']}/{cifid}.cif"
    else:
        cifpath = f"{phase_source[source]['dir']}/{formula}/{cifid}.cif"

    if os.path.exists(os.path.dirname(cifpath) + "/ase/" + os.path.basename(cifpath)):
        cifpath = os.path.dirname(cifpath) + "/ase/" + os.path.basename(cifpath)
    else:
        cifpath = correct_cif(cifpath)

    return cifpath


def correct_cif(cifpath):
    # using ase to read and export a corrected cif file
    atoms = ase.io.read(cifpath)
    ase_path = os.path.join(os.path.dirname(cifpath), "ase", os.path.basename(cifpath))

    if not os.path.exists(ase_path):
        os.makedirs(os.path.dirname(ase_path), exist_ok=True)
        print(f"created a corrected cif file at: {ase_path}")
        atoms.write(ase_path)

    return cifpath


def compare_with_tolerance(val1, val2, tol):
    return abs(val1 - val2) <= tol


def pymatgen_struc_matcher(formula, ciflist, phase_source=None, kwargs=None):
    """using pymatgen strcuture matcher tool -> match spacegroup, symmetry, and disregard atom position"""
    if not kwargs:
        kwargs = {
            "ltol": 0.2,
            "stol": 0.3,
            "angle_tol": 5,
            "primitive_cell": False,
            "scale": True,
            "attempt_supercell": True,
            "comparator": ElementComparator(),
        }

    sm = StructureMatcher(**kwargs)

    match_array = np.zeros((len(ciflist), len(ciflist)))
    match_dict = {}
    skip_idx = []
    match_factor = 0.8

    for i, cif in tqdm(enumerate(ciflist), total=match_array.shape[0]):
        if i in skip_idx:
            continue

        source1 = cif[0]
        cifid1 = cif[1]
        cifpath = get_cifpath(formula, source1, cifid1, phase_source)
        s1 = CifParser(cifpath).get_structures()[0]

        for j, cif2 in enumerate(ciflist):
            if i < j:
                source2 = cif2[0]
                cifid2 = cif2[1]
                cifpath = get_cifpath(formula, source2, cifid2, phase_source)
                s2 = CifParser(cifpath).get_structures()[0]
                match_array[i, j] = sm.fit(s1, s2)
                match_array[j, i] = match_array[i, j]

        indices = np.where(compare_with_tolerance(match_array[i], 1, 1 - match_factor))
        skip_idx.extend(list(indices[0]))

        index_array = np.array(np.meshgrid(indices, indices)).T.reshape(-1, 2)
        match_array[index_array[:, 0], index_array[:, 1]] = 1
        np.fill_diagonal(match_array, 1)
        match_dict[(source1, cifid1)] = [
            match for i, match in enumerate(ciflist) if i in list(indices[0])
        ]

    return list(match_dict.keys())


def sort_by_energy_cif(formulas, ciflist, sourcelist, phase_source):
    energy_array = np.ones(len(ciflist)) * 999
    for i, cif in tqdm(enumerate(ciflist)):
        source = sourcelist[i]
        formula = formulas[i]
        cifpath = get_cifpath(formula, source, cif, phase_source)
        atoms = ase.io.read(cifpath)
        atoms.set_calculator(EMT())
        opt = BFGS(atoms, maxstep=300)
        opt.run()
        energy = opt.atoms.get_potential_energy() / len(atoms)
        energy_array[i] = energy
    return energy_array


def highest_corr_to_exp(formula, cifdict, phase_source, xrd, instprm, outfolder="."):
    Iobs = np.array(xrd.intensity)
    scores = np.zeros(len(cifdict))

    for i, cif in tqdm(enumerate(cifdict), total=scores.shape[0]):
        source1 = cif[0]
        cifid1 = cif[1]
        cifpath = get_cifpath(formula, source1, cifid1, phase_source)
        s1 = CifParser(cifpath).get_structures()[0]
        xrd2 = GSAS_XRD()
        xrd2.set_gpx_project(project_name="temp", outfolder=outfolder, instprm=instprm)
        xrd2.simulate_phase(
            cifpath=cifpath,
            phasename=cifid1,
            tmin=xrd.tmin,
            tmax=xrd.tmax,
            npoints=len(xrd.theta),
        )
        Icalc = np.array(xrd2.intensity)
        for file in glob.glob("temp.*"):
            os.remove(file)
        score = np.corrcoef(Icalc, Iobs)[0, 1]
        scores[i] = score

    return [cifdict[np.argmax(scores)]]


def filter_structures(
    formula,
    ciflist,
    method="pymatgen_matcher",
    phase_source=None,
    xrd=None,
    instprm=None,
    outfolder=".",
):
    if len(ciflist) <= 1:
        return ciflist

    if method == "pymatgen_matcher":
        return pymatgen_struc_matcher(formula, ciflist, phase_source)
    elif method == "highest_corr_to_exp":
        return highest_corr_to_exp(
            formula, ciflist, phase_source, xrd, instprm, outfolder
        )


def resize_array(x, y, npoints=None):
    x_new = np.arange(0, npoints + 1, 1)
    y_new = np.zeros_like(x_new)
    res = (max(x) - min(x)) / npoints
    if len(x_new) != len(x):
        indices = ((x - min(x)) / res).astype(int)
        y_new[indices] = y

    return y_new


def get_DiffractionPattern_array(
    cifdict, phase_source, tmin, tmax, npoints, Imin=None, Imax=None
):
    Iarray = []

    for formula, ciflist in cifdict.items():
        for phase in ciflist:
            source = phase[0]
            cifid = phase[1]
            cifpath = get_cifpath(formula, source, cifid, phase_source)
            DP = DiffractionPattern()
            DP.gen_pattern_from_cif(cifpath)
            DP.intensity = DP.intensity / 100

            DP.apply_intensity_cutoffs(Imin, Imax)
            DP.apply_theta_cutoffs(tmin, tmax)
            DP.resize_array(tmin, tmax, npoints=npoints)
            Iarray.append(DP.intensity)

    return np.array(Iarray)


def GSASrefine_group(
    xrd,
    phase,
    cifdict,
    queue,
    instprm,
    phase_source=None,
    outfolder=None,
    project_name=None,
):
    prj_name = str(phase)
    for i, phasename in enumerate(phase):
        phasename += str(phasename) + "_"
    phasename = "[" + phasename[:-1] + "]"
    refine2 = GSAS_XRD()
    refine2.set_gpx_project(
        project_name=f"{project_name}-{phasename}", instprm=instprm, outfolder=outfolder
    )
    refine2.refine(
        xrd_csv=xrd.processed_filepath,
        n_cycles=20,
        cifdict=cifdict,
        phase_source=phase_source,
        selected_phases=phase,
    )
    queue.put((refine2.rwp, np.array(refine2.wt), refine2.residual_distance, refine2))


def GSASrefine_ciflist_multiprocessing(
    xrd,
    ciflist,
    sourcelist,
    formulas,
    phase_source,
    sel_phase_idx,
    instprm,
    queue,
    outfolder=".",
    prj_name="temp",
    phasename="temp",
    n_cycles=30,
):
    try:
        f = io.StringIO()
        with redirect_stdout(f):
            refineprj = GSAS_XRD()
            refineprj.set_gpx_project(
                project_name=f"{prj_name}-{phasename}",
                instprm=instprm,
                outfolder=outfolder,
            )
            refineprj.refine_ciflist(
                xrd_csv=xrd.processed_filepath,
                ciflist=ciflist,
                sourcelist=sourcelist,
                formulas=formulas,
                phase_source=phase_source,
                sel_phase_idx=sel_phase_idx,
                n_cycles=n_cycles,
            )
            result = (
                refineprj.rwp,
                np.array(refineprj.wt),
                refineprj.residual_distance,
                refineprj,
            )
        output = f.getvalue()
        queue.put((output, result))
    except Exception as e:
        queue.put((str(e), None))


def GSASrefine_ciflist(
    xrd,
    ciflist,
    sourcelist,
    formulas,
    phase_source,
    sel_phase_idx,
    instprm,
    outfolder=".",
    prj_name="temp",
    phasename="temp",
    n_cycles=30,
):
    refineprj = GSAS_XRD()
    refineprj.set_gpx_project(
        project_name=f"{prj_name}-{phasename}", instprm=instprm, outfolder=outfolder
    )
    refineprj.refine_ciflist(
        xrd_csv=xrd.processed_filepath,
        ciflist=ciflist,
        sourcelist=sourcelist,
        formulas=formulas,
        phase_source=phase_source,
        sel_phase_idx=sel_phase_idx,
        n_cycles=n_cycles,
    )
    return refineprj.rwp, np.array(refineprj.wt), refineprj.residual_distance, refineprj


def plot_GSASrefine(refine, no, title):
    fig = plt.figure(no)
    x = refine.gpx.histogram(0).getdata("x")
    yobs = refine.gpx.histogram(0).getdata("yobs")
    ycalc = refine.gpx.histogram(0).getdata("ycalc")
    bck = refine.gpx.histogram(0).getdata("background")
    res = refine.gpx.histogram(0).getdata("residual")
    plt.plot(x, yobs / max(yobs), label="exp")
    plt.plot(x, ycalc / max(ycalc), label="calc")
    plt.plot(x, bck / max(yobs), label="background")
    plt.plot(x, res - 0.3, label="residual", linestyle="--")
    plt.xlabel("2theta")
    plt.ylabel("Intensity, a.u.")
    plt.title(title)
    plt.legend()
    fig.show()

    return fig


def plot_xrd(xrd, filename=None):
    label = f"{filename}"
    fig = go.Figure()
    line_trace = go.Scatter(
        x=xrd.theta, y=xrd.intensity, mode="lines", name=filename, showlegend=True
    )
    fig.add_trace(line_trace)

    fig.update_layout(
        title={
            "x": 0.45,  # Center horizontally
            "y": 0.95,  # Center closer to the top
        },
        width=800,
        height=600,
        plot_bgcolor="white",
        xaxis=dict(
            title="2theta",
        ),
        yaxis=dict(
            title="Intensity",
        ),
    )
    offset = 5
    fig.update_xaxes(range=[min(xrd.theta), max(xrd.theta)])
    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=False)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=False)
    xlabel_step = 5
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)", dtick=5
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor="rgba(0, 0, 0, 0.1)", dtick=0.25
    )

    return fig


def plot_phases(
    fig,
    phases=None,
    phase_source=None,
    theta_cutoffs=(None, None),
    intensity_cutoffs=(None, None),
    resize=0.5,
):
    fig2 = deepcopy(fig)
    for phase, data in phases.items():
        if phases and phase_source:
            for file_info in data:
                source = file_info[0]
                filetype = phase_source[source]["filetype"]
                folder_dir = phase_source[source]["dir"]
                filename = file_info[1]
                file_dir = f"{folder_dir}/{filename}.{filetype}"
                if filetype == "cif":
                    xrd = DiffractionPattern()
                    xrd.gen_pattern_from_cif(file_dir)
                    xrd.intensity = xrd.intensity / 100 * resize
                else:
                    xrd = XRD()
                    xrd.read_xy_file(delimiter="\t")
                    xrd.remove_bckg()
                    xrd.normalize()

                xrd.apply_intensity_cutoffs(intensity_cutoffs[0], intensity_cutoffs[1])
                xrd.apply_theta_cutoffs(theta_cutoffs[0], theta_cutoffs[1])
                if filetype == "cif":
                    trace = go.Bar(
                        x=xrd.theta,
                        y=xrd.intensity,
                        name=f"{source}_{filename}",
                        visible="legendonly",
                        width=0.2,
                        legendgroup=phase,
                        legendgrouptitle_text=phase,
                    )
                else:
                    trace = go.Scatter(
                        x=xrd.theta,
                        y=xrd.intensity,
                        name=f"{filename}",
                        visible="legendonly",
                        legendgroup=phase,
                        legendgrouptitle_text=phase,
                    )

                fig2.add_trace(trace)

    return fig2


def plot_pattern_map(xrd, samplename, cifdict, phase_source, tmin, tmax, npoints):
    Iarray = get_DiffractionPattern_array(
        cifdict=cifdict,
        phase_source=phase_source,
        tmin=xrd.tmin,
        tmax=xrd.tmax,
        npoints=len(xrd.theta),
    )
    xstep = 5
    xticks = (tmax - tmin) / xstep
    phaseid_list = [
        f"{key}_{cifid[0]}-{cifid[1]}"
        for key, item in cifdict.items()
        for cifid in item
    ]
    x_range = np.arange(0, len(Iarray[0]))

    fig4 = plt.figure(figsize=(10, 6))
    ratio_plots = 0.4

    # XRD plot
    ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=10 - int(10 * ratio_plots))
    ax1.plot(xrd.intensity, color="b", label=samplename)

    ax1.bar(x_range, xrd.DP.intensity, width=3, color="r")
    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax1.legend()

    # matching pattern plot
    ax2 = plt.subplot2grid(
        (10, 1),
        (10 - int(10 * ratio_plots), 0),
        rowspan=int(10 * ratio_plots),
        sharex=ax1,
    )
    sns.heatmap(
        Iarray,
        cmap="Blues",
        cbar=False,
        cbar_kws={"shrink": 0.5, "location": "bottom"},
        ax=ax2,
    )
    ax2.set_xticks(np.linspace(0, len(Iarray[0]), int(xticks + 1)))
    ax2.set_xticklabels(np.linspace(tmin, tmax, int(xticks + 1)), rotation=0)
    ax2.set_yticks(range(0, len(phaseid_list), 1))
    ax2.set_yticklabels(phaseid_list, rotation=0, fontsize=6)
    ax2.set_xlabel("2theta", fontsize=12)
    ax2.set_ylabel("Phase", fontsize=12)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=10)
    # cbar.set_label('Normlaized Peak Intensity', size=10)
    ax2.invert_yaxis()

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)  # smaller pad value
    plt.colorbar(ax2.get_children()[0], cax=cax, orientation="horizontal")

    plt.tight_layout()
    fig4.show()


def plot_pattern_mapv2(
    xrd, Iarray, ciflist, sourcelist, formulas, sel_phase_idx, samplename, tmin, tmax
):
    xstep = 5
    xticks = (tmax - tmin) / xstep
    x_range = np.arange(0, len(Iarray[0]))
    phaseid_list = [
        f"{formulas[int(i)]}_{sourcelist[int(i)]}-{ciflist[int(i)]}"
        for i in sel_phase_idx
        if i != -1
    ]

    fig = plt.figure(figsize=(8, 4))
    ratio_plots = 0.4

    Iarray = Iarray[[int(i) for i in sel_phase_idx[sel_phase_idx != -1]]]
    Inorm = Iarray.copy()
    for i, I in enumerate(Inorm):
        Inorm[i] = I / (np.max(I))

        # print(np.max(I), int(np.where(I==1.0)[0])*0.04, I[np.where(I!=0)])
    # Iarray_norm = (Iarray - np.min(Iarray, axis=1, keepdims=True)) / (np.max(Iarray, axis=1, keepdims=True) - np.min(Iarray, axis=1, keepdims=True))

    # XRD plot
    ax1 = plt.subplot2grid((10, 1), (0, 0), rowspan=10 - int(10 * ratio_plots))
    ax1.plot(xrd.intensity, color="b", label=samplename)

    ax1.bar(x_range, xrd.DP.intensity, width=5, color="r")
    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax1.legend()

    # matching pattern plot
    ax2 = plt.subplot2grid(
        (10, 1),
        (10 - int(10 * ratio_plots), 0),
        rowspan=int(10 * ratio_plots),
        sharex=ax1,
    )
    sns.heatmap(
        Inorm,
        cmap="Blues",
        cbar=False,
        cbar_kws={"shrink": 0.5, "location": "bottom"},
        ax=ax2,
    )
    ax2.set_xticks(np.linspace(0, len(Iarray[0]), int(xticks + 1)))
    ax2.set_xticklabels(np.linspace(tmin, tmax, int(xticks + 1)), rotation=0)
    ax2.set_yticks(range(0, len(phaseid_list), 1))
    ax2.set_yticklabels(phaseid_list, rotation=0, fontsize=6)
    ax2.set_xlabel("2theta", fontsize=12)
    ax2.set_ylabel("Phase", fontsize=12)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=10)
    # cbar.set_label('Normlaized Peak Intensity', size=10)
    ax2.invert_yaxis()

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)  # smaller pad value
    plt.colorbar(ax2.get_children()[0], cax=cax, orientation="horizontal")

    plt.tight_layout()
    fig.show()

    return fig
