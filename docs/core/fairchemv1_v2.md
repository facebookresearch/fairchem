# `fairchem>=2.0`
`fairchem>=2.0` is a major upgrade and we completely rewrote the trainer, fine-tuning, models and calculators.

We plan to bring back the following models compatible with Fairchem V2 soon:
* Gemnet-OC
* EquiformerV2
* eSEN

We will also be releasing more detailed documentation on how to use Fairchem V2, stay tuned!

The old OCPCalculator, trainer code will NOT be revived. We apologize for the inconvenience and please raise Issues if you need help!
In the meantime, you can still use models from fairchem version 1, by installing version 1,

```bash
pip install fairchem-core==1.10
```

And using the `OCPCalculator`
```python
from fairchem.core import OCPCalculator

calc = OCPCalculator(
    model_name="EquiformerV2-31M-S2EF-OC20-All+MD",
    local_cache="pretrained_models",
    cpu=False,
)
```

## Projects and models built on `fairchem` version v2:

- UMA (Universal Model for Atoms) [[`arXiv`](https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/)] [[`code`](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/core/models/uma)]

## Projects and models built on `fairchem` version v1:

You can still find these in the v1 version of fairchem github.
However, many of these implementations are no longer actively supported.

- GemNet-dT [[`arXiv`](https://arxiv.org/abs/2106.08903)] [[`code`](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/gemnet)]
- PaiNN [[`arXiv`](https://arxiv.org/abs/2102.03150)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/painn)]
- Graph Parallelism [[`arXiv`](https://arxiv.org/abs/2203.09697)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/gemnet_gp)]
- GemNet-OC [[`arXiv`](https://arxiv.org/abs/2204.02782)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/gemnet_oc)]
- SCN [[`arXiv`](https://arxiv.org/abs/2206.14331)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/scn)]
- AdsorbML [[`arXiv`](https://arxiv.org/abs/2211.16486)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/applications/AdsorbML)]
- eSCN [[`arXiv`](https://arxiv.org/abs/2302.03655)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/escn)]
- EquiformerV2 [[`arXiv`](https://arxiv.org/abs/2306.12059)] [[`code`](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/equiformer_v2)]
- SchNet [[`arXiv`](https://arxiv.org/abs/1706.08566)]
- DimeNet++ [[`arXiv`](https://arxiv.org/abs/2011.14115)] 
- CGCNN [[`arXiv`](https://arxiv.org/abs/1710.10324)] [[`code`](https://github.com/facebookresearch/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/cgcnn.py)]
- DimeNet [[`arXiv`](https://arxiv.org/abs/2003.03123)] [[`code`](https://github.com/facebookresearch/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/dimenet.py)]
- SpinConv [[`arXiv`](https://arxiv.org/abs/2106.09575)] [[`code`](https://github.com/facebookresearch/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/spinconv.py)]
- ForceNet [[`arXiv`](https://arxiv.org/abs/2103.01436)] [[`code`](https://github.com/facebookresearch/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/forcenet.py)]
