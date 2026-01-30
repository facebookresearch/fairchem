# `fairchem>=2.0`

:::{warning}
`fairchem>=2.0` is a major upgrade with completely rewritten trainer, fine-tuning, models, and calculators. The old `OCPCalculator` and trainer code will NOT be revived.
:::

We plan to bring back the following models compatible with Fairchem V2 soon:
* Gemnet-OC
* EquiformerV2
* eSEN

:::{tip}
We will be releasing more detailed documentation on how to use Fairchem V2. Stay tuned!
:::

## Using Fairchem V1

If you need to use models from fairchem version 1, you can still do so by installing version 1:

```bash
pip install fairchem-core==1.10
```

And using the `OCPCalculator`:

```python
from fairchem.core import OCPCalculator

calc = OCPCalculator(
    model_name="EquiformerV2-31M-S2EF-OC20-All+MD",
    local_cache="pretrained_models",
    cpu=False,
)
```

## Projects and models built on `fairchem` version v2

::::{grid} 1

:::{card} UMA (Universal Model for Atoms)
:link: https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/

The latest universal model for atomistic simulations.

[[arXiv]](https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/) | [[code]](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/core/models/uma)
:::

::::

## Projects and models built on `fairchem` version v1

:::{note}
You can still find these in the v1 version of fairchem github. However, many of these implementations are no longer actively supported.
:::

::::{grid} 1 2 2 3

:::{card} GemNet-dT
[[arXiv]](https://arxiv.org/abs/2106.08903) | [[code]](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/gemnet)
:::

:::{card} PaiNN
[[arXiv]](https://arxiv.org/abs/2102.03150) | [[code]](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/painn)
:::

:::{card} Graph Parallelism
[[arXiv]](https://arxiv.org/abs/2203.09697) | [[code]](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/gemnet_gp)
:::

:::{card} GemNet-OC
[[arXiv]](https://arxiv.org/abs/2204.02782) | [[code]](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/gemnet_oc)
:::

:::{card} SCN
[[arXiv]](https://arxiv.org/abs/2206.14331) | [[code]](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/scn)
:::

:::{card} AdsorbML
[[arXiv]](https://arxiv.org/abs/2211.16486) | [[code]](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/applications/AdsorbML)
:::

:::{card} eSCN
[[arXiv]](https://arxiv.org/abs/2302.03655) | [[code]](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/escn)
:::

:::{card} EquiformerV2
[[arXiv]](https://arxiv.org/abs/2306.12059) | [[code]](https://github.com/facebookresearch/fairchem/tree/fairchem_core-1.10.0/src/fairchem/core/models/equiformer_v2)
:::

:::{card} SchNet
[[arXiv]](https://arxiv.org/abs/1706.08566)
:::

:::{card} DimeNet++
[[arXiv]](https://arxiv.org/abs/2011.14115)
:::

:::{card} CGCNN
[[arXiv]](https://arxiv.org/abs/1710.10324) | [[code]](https://github.com/facebookresearch/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/cgcnn.py)
:::

:::{card} DimeNet
[[arXiv]](https://arxiv.org/abs/2003.03123) | [[code]](https://github.com/facebookresearch/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/dimenet.py)
:::

:::{card} SpinConv
[[arXiv]](https://arxiv.org/abs/2106.09575) | [[code]](https://github.com/facebookresearch/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/spinconv.py)
:::

:::{card} ForceNet
[[arXiv]](https://arxiv.org/abs/2103.01436) | [[code]](https://github.com/facebookresearch/fairchem/blob/e7a8745eb307e8a681a1aa9d30c36e8c41e9457e/ocpmodels/models/forcenet.py)
:::

::::
