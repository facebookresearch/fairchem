# Installation

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
