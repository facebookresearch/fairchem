# Installation & License

## Installation

To install `fairchem-core` you will need to setup the `fairchem-core` environment. We support either pip or uv. Conda is no longer supported and has also been dropped by pytorch itself. Note you can still create environments with conda and use pip to install the packages.

**Note FairchemV2 is a major breaking changing from FairchemV1. If you are looking for old V1 code, you will need install v1 (`pip install fairchem-core==1.10`)**

We recommend installing fairchem inside a virtual enviornment instead of directly onto your system. For example, you can create one like so using your favorite venv tool:

```
virtualenv -p python3.12 fairchem
source fairchem/bin/activate
```

Then to install the fairchem package, you can simply use pip:

```
pip install fairchem-core
```

#### For developers that want to contribute to fairchem, clone the repo and install it in edit mode

```
git clone git@github.com:facebookresearch/fairchem.git

cd fairchem

pip install -e src/packages/fairchem-core[dev]
```


In V2, we removed all dependencies on 3rd party libraries such as torch-geometric, pyg, torch-scatter, torch-sparse etc that made installation difficult. So no additional steps are required!

## Subpackages

In addition to `fairchem-core`, there are related packages for specialized tasks or applications. Each can be installed with `pip` or `uv` just like `fairchem-core`:
* `fairchem-data-oc`
* `fairchem-applications-cattsunami`
* `fairchem-demo-ocpapi`

## Access to gated models on huggingface

To access gated models like UMA, you need to get a HuggingFace account and request access to the UMA models.

1. Get and login to your Huggingface account
2. Request access to https://huggingface.co/facebook/UMA
3. Create a Huggingface token at https://huggingface.co/settings/tokens/ with the permission "Permissions: Read access to contents of all public gated repos you can access"
4. Add the token as an environment variable (using `huggingface-cli login` or by setting the HF_TOKEN environment variable.

## License

### Repository software

The software in this repo is licensed under an MIT license unless otherwise specified.

```
MIT License

Copyright (c) Meta, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Model checkpoints and datasets

Please check each dataset and model for their own licenses.
