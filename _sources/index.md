<p align="center">
  <img width="559" height="200" src="https://github.com/user-attachments/assets/25cd752c-3c56-469d-8524-4e493646f6b2"?
</p>


<h4 align="center">

![tests](https://github.com/facebookresearch/fairchem/actions/workflows/test.yml/badge.svg?branch=main)
![PyPI - Version](https://img.shields.io/pypi/v/fairchem-core)
![Static Badge](https://img.shields.io/badge/python-3.10%2B-blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15587498.svg)](https://doi.org/10.5281/zenodo.15587498)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new/facebookresearch/fairchem?quickstart=1)
</h4>

# `fairchem` by FAIR Chemistry

`fairchem` is the [FAIR](https://ai.meta.com/research/) Chemistry's centralized repository of all its data, models,
demos, and application efforts for materials science and quantum chemistry.

> :warning: **FAIRChem version 2 is a breaking change from version 1 and is not compatible with our previous pretrained models and code.**
> If you want to use an older model or code from version 1 you will need to install [version 1](https://pypi.org/project/fairchem-core/1.10.0/),
> as detailed [here](#looking-for-fairchem-v1-models-and-code).

> :warning: Some of the docs and new features in FAIRChem version 2 are still being updated so you may see some changes over the next few weeks. Check back here for the latest instructions. Thank you for your patience!

## Read our latest release post!
Read about the [UMA model and OMol25 dataset](https://ai.meta.com/blog/meta-fair-science-new-open-source-releases/) release.

[![Meta FAIR Science Release](https://github.com/user-attachments/assets/acddd09b-ed6f-4d05-9a4b-9ba5e2301150)](https://ai.meta.com/blog/meta-fair-science-new-open-source-releases/?ref=shareable)

## Try the demo!
If you want to explore model capabilities check out our
[educational demo](https://facebook-fairchem-uma-demo.hf.space/)

[![Educational Demo](https://github.com/user-attachments/assets/7005d1bb-4459-403d-b299-d41fdd8c48ec)](https://facebook-fairchem-uma-demo.hf.space/)


````{admonition} Need to install fairchem-core or get UMA access or getting permissions/401 errors?
:class: dropdown


1. Install the necessary packages using pip, uv etc
```{code-cell} ipython3
:tags: [skip-execution]

! pip install fairchem-core fairchem-data-oc fairchem-applications-cattsunami
```

2. Get access to any necessary huggingface gated models 
    * Get and login to your Huggingface account
    * Request access to https://huggingface.co/facebook/UMA
    * Create a Huggingface token at https://huggingface.co/settings/tokens/ with the permission "Permissions: Read access to contents of all public gated repos you can access"
    * Add the token as an environment variable using `huggingface-cli login` or by setting the HF_TOKEN environment variable. 

```{code-cell} ipython3
:tags: [skip-execution]

# Login using the huggingface-cli utility
! huggingface-cli login

# alternatively,
import os
os.environ['HF_TOKEN'] = 'MY_TOKEN'
```

````