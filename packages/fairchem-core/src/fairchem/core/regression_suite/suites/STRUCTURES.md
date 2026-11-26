# Regression suite structures

The checked-in structures are frozen, xz-compressed extended XYZ files
(read them with `ase.io.read` or `xzcat`). Each plan hashes its structures
as ASE reads them (numbers, positions, cell, periodicity, tags, fixed atoms,
charge, and spin), so any edit to these files changes the plan hash and
requires a new baseline. `python -m fairchem.core.regression_suite.suites.plan_digest`
reports whether every plan still reproduces its audited hash.

Unless noted, the files come from the structure snapshots of PR #82 at commit
`260aaf2f` (`src/fairchem/core/components/regression_suite/structures/`),
which were selected from upstream benchmark storage by that PR's
`build_structures.py`. The comparisons here use paired FP64 references rather
than DFT targets, so DFT labels were removed where noted.

| File | Contents | Source and changes | Used by |
| --- | --- | --- | --- |
| `prediction_fidelity/structures/{omat,omol,oc20,oc22,oc25,odac,omc}.extxyz.xz` | 10 validation structures per UMA task | All frames of PR #82 `accuracy/<task>.extxyz`; DFT energy, force, and stress labels removed | `prediction_fidelity` |
| `edge_cases/structures/omat.extxyz.xz` | 0.70x isotropically compressed `mp-17048` and its 2x2x1 supercell | The two derived dynamics frames of PR #82 `edge_cases/omat.extxyz` | `edge_cases` NPT expansion and NVT dense transient |
| `elastic/structures/mp_pbe_elasticity_10.extxyz.xz` | 10 crystals from the Materials Project PBE elasticity subset | Byte-identical to PR #82 `elastic/mp_pbe_elasticity_10.extxyz` | `elastic` |
| `phonon_crystal/structures/mdr_phonon_10.extxyz.xz` | 10 MDR phonon crystals with supercell and primitive matrices | Byte-identical to PR #82 `phonon/mdr_phonon_10.extxyz` | `phonon_crystal`, and the canonical `mp-17048` frame of `edge_cases` |
| `physical_constraints/structures/molecules.extxyz.xz` | 5 NCIAtlas dimers (5.0122, 6.0032, 7.0022, 5.0142, 9.0022) | First 5 frames of PR #82 `ncia/ncia_dimers_10.extxyz` | `physical_constraints`, `finite_difference_frequencies`, `autograd_frequencies` |
| `physical_constraints/structures/periodic.extxyz.xz` | 5 small OMat crystals (66415, 90738, 19633, 21897, 3946) | Frames of PR #82 `accuracy/omat.extxyz`; labels removed | `physical_constraints` |
| `nve_md/structures/md22/*.xyz.xz`, `nve_md/structures/tm23/*.xyz.xz` | Last frames of 7 MD22 trajectories and 3 TM23 melts | PR #82 `md22/` and `tm23/`; geometry unchanged, forces removed (MD22 files keep the source energy in their comment line) | `nve_md` |

Two suites generate their structures in code instead: the `edge_cases` static
probes (rattled FCC cells, a cutoff dimer, an isolated atom, the graph-churn
cells, and the neighbor-cap hydrogen clusters) and the `inference` FCC carbon
cells. See `source_frames()` in each
suite's `suite.py`.
