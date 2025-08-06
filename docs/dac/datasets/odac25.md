# ODAC25

The Open DAC 2025 (ODAC25) dataset contains nearly 70 million DFT single-point 
calculations for CO₂, H₂O, N₂, and O₂ adsorption in nearly 15,000 metal-organic frameworks 
(MOFs). This dataset represents a significant expansion upon ODAC23, introducing 
greater chemical and configurational diversity through functionalized MOFs, 
high-energy GCMC-derived placements, and synthetically generated frameworks.

The dataset enables training of state-of-the-art machine-learned interatomic 
potentials for direct air capture applications. All structures are labeled 
with total energies (eV) and forces (eV/Å) computed using VASP with the 
PBE+D3 functional.

All information about the dataset is available at the 
[ODAC25 Huggingface site](https://huggingface.co/facebook/ODAC25). 
For questions or issues, please open a GitHub issue in this repository.

## Dataset format 

The dataset is provided in ASE DB compatible lmdb files (`*.aselmdb`).  

### Citing ODAC25

The ODAC25 dataset is licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode).

Please consider citing the following paper in any publications that uses this dataset:

```bib
@misc{sriram2025odac25,
    title={The Open DAC 2025 Dataset for Sorbent Discovery in Direct Air Capture}, 
    author={Anuroop Sriram and Logan M. Brabson and Xiaohan Yu and Sihoon Choi and Kareem Abdelmaqsoud and Elias Moubarak and Pim de Haan and Sindy Löwe and Johann Brehmer and John R. Kitchin and Max Welling and C. Lawrence Zitnick and Zachary Ulissi and Andrew J. Medford and David S. Sholl},
    year={2025},
    eprint={},
    archivePrefix={arXiv},
    primaryClass={},
    url={}, 
}
```
