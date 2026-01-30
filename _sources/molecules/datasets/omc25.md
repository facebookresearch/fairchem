# OMC25

The Open Molecular Crystals 2025 (OMC25) dataset comprises >25 million structures of organic molecular crystals from relaxation trajectories of random packings of OE62 molecules into various 3D unit cells using Genarris 3.0 package. The dataset contains structures labeled with total energy (eV), forces (eV/A), and stress (ev/A^3) via VASP.

The training and validation splits of the OMC25 dataset are available for download from HuggingFace at https://huggingface.co/facebook/OMC25, under the CC BY 4.0 license, after applying for the repository access on HuggingFace.

## Dataset format

The dataset is provided in ASE DB compatible lmdb files (*.aselmdb).

## Level of theory

OMC25 was calculated at the PBE+D3 level via VASP. To reproduce the calculations, please use `fairchem.data.omc.scripts.create_vasp_inputs.py` to write compatible VASP inputs.

## Citing

We encourage users to cite this paper when using the OMC25 dataset or pretrained models for molecular crystals in their research.

```bibtex
@misc{gharakhanyan2025openmolecularcrystals2025omc25dataset,
      title={Open Molecular Crystals 2025 (OMC25) Dataset and Models},
      author={Vahe Gharakhanyan and Luis Barroso-Luque and Yi Yang and Muhammed Shuaibi and Kyle Michel and Daniel S. Levine and Misko Dzamba and Xiang Fu and Meng Gao and Xingyu Liu and Haoran Ni and Keian Noori and Brandon M. Wood and Matt Uyttendaele and Arman Boromand and C. Lawrence Zitnick and Noa Marom and Zachary W. Ulissi and Anuroop Sriram},
      year={2025},
      eprint={2508.02651},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2508.02651},
}
```
