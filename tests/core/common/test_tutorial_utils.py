import json
import os
import sys
from io import StringIO

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from fairchem.core.common import tutorial_utils as tu
from fairchem.core.scripts import download_large_files


def test_describe_fairchem():
    """
    Test fairchem description printing
    """
    print_buffer = StringIO()
    old_stdout = sys.stdout
    sys.stdout = print_buffer
    tu.describe_fairchem()
    sys.stdout = old_stdout
    print_msg = print_buffer.getvalue()
    assert "\nfairchem repo is at git commit:" in print_msg
    assert "\nnumpy:" in print_msg
    assert "\nase:" in print_msg
    assert "\ntorch:" in print_msg
    assert "\nPlatform:" in print_msg


def test_train_test_val_split():
    """
    Test that train, test, and val db are created with the appropriate number
    of systems.
    """
    download_large_files.download_file_group("docs")
    with open(
        tu.fairchem_main().parent
        / "docs"
        / "core"
        / "fine-tuning"
        / "supporting-information.json",
        "rb",
    ) as f:
        d = json.loads(f.read())

    polymorphs = list(d["TiO2"].keys())
    db_name = "oxides.db"
    ase_db = connect(db_name)

    for polymorph in polymorphs:
        for c in d["TiO2"][polymorph]["PBE"]["EOS"]["calculations"]:
            atoms = Atoms(
                symbols=c["atoms"]["symbols"],
                positions=c["atoms"]["positions"],
                cell=c["atoms"]["cell"],
                pbc=c["atoms"]["pbc"],
            )
            atoms.set_tags(np.ones(len(atoms)))
            calc = SinglePointCalculator(
                atoms, energy=c["data"]["total_energy"], forces=c["data"]["forces"]
            )
            atoms.calc = calc
            ase_db.write(atoms)

    sizes = {"test.db": 5, "train.db": 45, "val.db": 7, "oxides.db": 57}
    train, test, val = tu.train_test_val_split(db_name)
    for fname in (db_name, "test.db", "train.db", "val.db"):
        assert os.path.exists(fname)
        assert len(connect(fname)) == sizes[fname]
        # Clean up after ourselves
        os.remove(fname)
