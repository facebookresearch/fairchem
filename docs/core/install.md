# Installation & License

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

In V2, we removed all dependencies on 3rd party libraries such as torch-geometric, pyg, torch-scatter, torch-sparse etc that made installation difficult. So no additional steps are required!


# License

## Repository software

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

## Model checkpoints and datasets
Please check each dataset and model for their own licenses.