"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ase.io import read
from atomate2.vasp.files import write_vasp_input_set
from atomate2.vasp.sets.core import RelaxSetGenerator, StaticSetGenerator
from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

INCAR_DIR = Path("./incars").absolute()


def create_input_generator(args):
    incar_path = Path(args.incar_yml_dir).resolve()

    # load incar settings
    with open(incar_path / f"incar_{args.type}.yml") as f:
        INCAR_SETTINGS = yaml.full_load(f)

    if args.type == "relax":  # default to tight relaxation w/ EDIFFG=-0.001 eV/A
        input_generator = RelaxSetGenerator(
            user_incar_settings=INCAR_SETTINGS,
            user_potcar_functional="PBE_54_W_HASH",
            auto_kspacing=True,
        )
    elif args.type == "static":
        input_generator = StaticSetGenerator(
            user_incar_settings=INCAR_SETTINGS,
            user_potcar_functional="PBE_54_W_HASH",
            auto_kspacing=True,
        )
    else:
        raise ValueError("Only 'relax' and 'static' types are supported!")
    return input_generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="create-vasp-inputs", description="Write VASP inputs."
    )

    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["relax", "static"],
        default="relax",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Path to input cif files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        help="Path to output directory to write VASP input files.",
    )
    parser.add_argument(
        "-y",
        "--incar-yml-dir",
        type=str,
        help=f"Path to incar yml files. Default is {INCAR_DIR}.",
        default=INCAR_DIR,
    )

    args = parser.parse_args()

    # create output dir
    if args.output_dir is None:
        output_dir = args.input_dir
    else:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    # create VASP input generator
    input_generator = create_input_generator(args)

    # load ase atoms objects from cifs
    # and create VASP inputs
    cif_dir = Path(args.input_dir).resolve()
    sids = [sid.stem for sid in cif_dir.rglob("*.cif")]

    atoms_by_id = {}
    invalid_cif_errors = []
    for sid in tqdm(sids, desc="reading cif files"):
        cif_file = cif_dir / (sid + ".cif")
        try:
            atoms = read(cif_file)
            atoms_by_id[sid] = atoms.copy()
        except Exception as err:
            invalid_cif_errors.append(f"{sid}.cif: {err}")
            continue

    if len(invalid_cif_errors) > 0:
        print("The following cif files had errors:")
        for msg in invalid_cif_errors:
            print(msg)

    existing_vasp_input_paths = []
    for sid, atoms in tqdm(atoms_by_id.items(), desc="Writing VASP inputs"):
        if (output_dir / sid).exists():  # do not overwrite
            existing_vasp_input_paths.append(sid)
            continue

        structure = AseAtomsAdaptor.get_structure(atoms)

        write_vasp_input_set(structure, input_generator, directory=output_dir / sid)

    if len(existing_vasp_input_paths) > 0:
        print(
            "Writing VASP input files for the following structures was skipped because they exist in: \n"
            f"{output_dir}\n {existing_vasp_input_paths}"
        )
