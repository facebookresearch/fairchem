# OMol25 Electronic Structures

The Open Molecules 2025 (OMol25) dataset represents the largest dataset of its kind, with more than 100 million density functional theory (DFT) calculations at the ωB97M-V/def2-TZVPD level of theory, spanning several chemical domains including small molecules, biomolecules, metal complexes, and electrolytes.

At release, the OMol25 dataset provided structure energies, per-atom forces, and Lowdin/Mulliken charges and spins, where available. These properties were sufficient to train state-of-the-art machine learning interatomic potentials (MLIPs) and are already demonstrating incredible performance across a wide range of applications. However, to maximize the community benefit of these calculations, we have partnered with the [Department of Energy’s Argonne National Laboratory](https://www.anl.gov/) to provide access to the raw DFT outputs and additional files for the OMol25 dataset.

By releasing the [ORCA](https://www.faccts.de/docs/orca/6.0/manual/) output files, users will be able to parse NBO orbital/bonding information, reduced orbital populations, Fock matrices, and more. By releasing the ORCA GBW files, users will be able to run electronic structure post-processing in order to obtain higher quality partial charges and partial spins and a variety of more advanced electronic features that could be extremely valuable for physics-informed ML models. Finally, the release will provide critical high quality data for nascent ML models that train directly on electron densities.

## Data Description

The OMol25 dataset is broken into several training splits - All and 4M. The 4M split corresponds to a randomly sampled 4M subset of the full OMol25 dataset. Given the size of the full dataset, O(petabytes), we are first releasing all electronic structure and ORCA output data for the 4M split. Based on community interest, we will work to provide the full dataset.

For each calculation, the following data is available:

* **orca.tar.zst**: Bundle of the raw [ORCA](https://www.faccts.de/docs/orca/6.0/manual/) outputs - including (orca.out, orca.inp orca.engrad, orca_property.txt, orca.xyz). To open:

```
>> tar --zstd -xvf orca.tar.zst
orca.engrad
orca.inp
orca.inp.orig
orca.out
orca.xyz
orca_property.txt
orca_stderr
```

* **orca.gbw.zstd0**: Geometry-Basis-Wavefunction file - containing molecular orbitals and wavefunction information for the converged SCF.

```
>> zstd -d orca.gbw.zstd0 -o orca.gbw
orca.gbw.zstd0      : 9462880 bytes
```

* **density_mat.npz**: The upper-triangle of the density matrix ("orca.scfp") (two in the case of unrestricted systems with the addition of the spin density ("orca.scfr")). This vectorized form of the density can be inflated into a symmetric matrix with the following code:

```python
import numpy as np

# Load the NPZ file
with np.load('density_mat.npz') as loaded_data:
    dens_vector = loaded_data['orca.scfp']

# Re-inflate the symmetric matrix
n = (np.sqrt(8 * len(dens_vector) + 1) - 1) // 2
mat = np.zeros((n,n))
mat[np.triu_indices(n)] = dens_vector
mat = mat + mat.T - np.diag(mat.diagonal())
```

The dataset is organized on the Argonne cluster based on how we organized it internally for generation. The easiest way to find systems that you may be interested in is by using the ASE-DB format of the dataset that can be downloaded at [train_4M.tar.gz](https://huggingface.co/facebook/OMol25/blob/main/DATASET.md#dataset-splits):

```python
# pip install fairchem-core if not already installed
from fairchem.core.datasets import AseDBDataset

dataset = AseDBDataset({"src": "path/to/train_4M/"})
indices = range(len(dataset))

argonne_paths = []
for idx in indices:
    # ASE Atoms object that can be visualized/examined
    atoms = dataset.get_atoms(idx)
    # Check if this is a system you care about.
    is_relevant = is_atoms_object_relevant(atoms)
    if is_relevant:
	# Extract the relative path that matches the Argonne cluster
       relative_dir = os.path.dirname(atoms.info["source"])
	argonne_paths.append(relative_dir)
```

## How to Access the Data

The data are stored and accessible via storage on the Eagle cluster at Argonne National Laboratory. For free access to the data, you will need to do the following:
1. Fill in the [access form](https://forms.gle/RyGGmbMkDSQ57wS2A). Use institutional credentials where possible with Globus.
2. After acceptance (this step requires human validation, so may take some time), navigate to the collection with the following link: [Globus OMol25 Collection](https://app.globus.org/file-manager?origin_id=0b73865a-ff20-4f57-a1d7-573d86b54624&origin_path=%2F).
    * Acceptance will trigger an email to be sent to the address used in the form.
3. To download the data, you can access via HTTPS calls (slower) or by downloading the [Globus Connect Personal client](https://www.globus.org/globus-connect-personal) (preferred method) and creating a local endpoint.


## Contact Us

[General Issues](https://github.com/facebookresearch/fairchem)

Dataset questions?
* [Muhammed Shuaibi](mshuaibi@meta.com)
* [Daniel Levine](levineds@meta.com)

Cluster/Access questions?
* [Ben Blaiszik](blaiszik@uchicago.edu)
