# OMol25

The Open Molecules 2025 (OMol25) dataset contains over 100 million single point calculations of non-equilibrium structures and
structural relaxations across a wide swath of organic and inorganic molecular space, including things like transition metal complexes and electrolytes. The dataset contains structures labeled with total energy (eV) and forces (eV/A) via ORCA6. A much larger amount of electronic structure data were also stored during generation and we hope to make these available to the community (reach out via github issue). 

All information about the dataset is available at the [OMol25 HuggingFace site](https://huggingface.co/facebook/OMol25). If you have issues with the gated model request form, please reach out via a github issue on this repository.  

## Dataset format 

The dataset is provided in ASE DB compatible lmdb files (*.aselmdb). The dataset contains labels of the total charge and spin multiplicity, saved in the `atoms.info` dictionary because ASE does not support these as default properties. 

## Level of theory

OMol25 was calculated at the wB97M-V/def2-TZVPD level, including non-local dispersion, as defined in ORCA6. To reproduce the calculations, please `fairchem.data.om.omdata.orca.calc` to write compatible ORCA inputs. 

### Citing OMol25

The OMol25 dataset is licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode).

Please consider citing the following paper in any publications that uses this dataset:

```bib
@misc{levine2025openmolecules2025omol25,
      title={The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models}, 
      author={Daniel S. Levine and Muhammed Shuaibi and Evan Walter Clark Spotte-Smith and Michael G. Taylor and Muhammad R. Hasyim and Kyle Michel and Ilyes Batatia and Gábor Csányi and Misko Dzamba and Peter Eastman and Nathan C. Frey and Xiang Fu and Vahe Gharakhanyan and Aditi S. Krishnapriyan and Joshua A. Rackers and Sanjeev Raja and Ammar Rizvi and Andrew S. Rosen and Zachary Ulissi and Santiago Vargas and C. Lawrence Zitnick and Samuel M. Blau and Brandon M. Wood},
      year={2025},
      eprint={2505.08762},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2505.08762}, 
}
```
