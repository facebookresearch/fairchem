# Datasets in `fairchem`:
`fairchem` provides training and evaluation code for tasks and models that take arbitrary
chemical structures as input to predict energies / forces / positions / stresses,
and can be used as a base scaffold for research projects. For an overview of
tasks, data, and metrics, please read the documentations and respective papers:
 - [OC20](catalysts/datasets/oc20)
 - [OC22](catalysts/datasets/oc22)
 - [ODAC23](dac/datasets/odac)
 - [OC20Dense](catalysts/datasets/oc20dense)
 - [OC20NEB](catalysts/datasets/oc20neb)
 - [OMat24](inorganic_materials/datasets/omat24)
 - [OMol25](https://ai.meta.com/blog/meta-fair-science-new-open-source-releases/)

# Projects and models built on `fairchem` version v2:

- UMA (Universal Model for Atoms) [[`arXiv`](https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/)] [[`code`](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/core/models/uma)]

# Projects and models built on `fairchem` version v1:

You can still find these in the v1 version of fairchem github.
However, many of these implementations are no longer actively supported.

- GemNet-dT [[`arXiv`](https://arxiv.org/abs/2106.08903)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet)]
- PaiNN [[`arXiv`](https://arxiv.org/abs/2102.03150)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/painn)]
- Graph Parallelism [[`arXiv`](https://arxiv.org/abs/2203.09697)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/gemnet_gp)]
- GemNet-OC [[`arXiv`](https://arxiv.org/abs/2204.02782)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/gemnet_oc)]
- SCN [[`arXiv`](https://arxiv.org/abs/2206.14331)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/scn)]
- AdsorbML [[`arXiv`](https://arxiv.org/abs/2211.16486)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/applications/AdsorbML)]
- eSCN [[`arXiv`](https://arxiv.org/abs/2302.03655)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/escn)]
- EquiformerV2 [[`arXiv`](https://arxiv.org/abs/2306.12059)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/equiformer_v2)]
- SchNet [[`arXiv`](https://arxiv.org/abs/1706.08566)]
- DimeNet++ [[`arXiv`](https://arxiv.org/abs/2011.14115)] 
- CGCNN [[`arXiv`](https://arxiv.org/abs/1710.10324)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/cgcnn.py)]
- DimeNet [[`arXiv`](https://arxiv.org/abs/2003.03123)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/dimenet.py)]
- SpinConv [[`arXiv`](https://arxiv.org/abs/2106.09575)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/spinconv.py)]
- ForceNet [[`arXiv`](https://arxiv.org/abs/2103.01436)] [[`code`](https://github.com/FAIR-Chem/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/forcenet.py)]