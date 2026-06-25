# Distributed Batch Inference with Ray Serve

````{admonition} Need to install fairchem-core or get UMA access or getting permissions/401 errors?
:class: dropdown

1. Install the necessary packages using pip, uv etc
```{code-cell} ipython3
:tags: [skip-execution]

! pip install fairchem-core[extras]
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

:::{warning}
The Ray Serve batch inference server is experimental and under active development. The API may change. Ray support requires the `extras` install option: `pip install fairchem-core[extras]`.
:::

When running many independent ASE calculations across a cluster, the Ray Serve inference server lets you centralize GPU inference behind a single deployment. Multiple workers — whether Ray remote tasks or external processes — submit requests concurrently, and the server automatically batches them together for efficient GPU utilization.

There are two server modes:

| Mode | Use case |
| --- | --- |
| **Single-model** (`BatchPredictServer`) | One model loaded upfront; all clients share it |
| **Multiplexed** (`MultiplexedBatchPredictServer`) | Models loaded on demand per request; up to 3 cached per replica |

In both modes, clients interact through `BatchServerPredictUnit`, which is a drop-in replacement for `MLIPPredictUnit` and is fully compatible with `FAIRChemCalculator`.

## Single-Model Server

### Starting the server

Load a predict unit and pass it to `setup_batch_predict_server`. This deploys a `BatchPredictServer` on your Ray cluster and starts accepting requests immediately.

```python
import ray
from fairchem.core import pretrained_mlip
from fairchem.core.units.mlip_unit.batch_server import (
    setup_batch_predict_server,
    wait_for_serve_ready,
)

ray.init()

predict_unit = pretrained_mlip.get_predict_unit("uma-s-1p2", device="cuda")

setup_batch_predict_server(
    predict_unit,
    deployment_name="fairchem-inference",
    ray_actor_options={"num_cpus": 4, "num_gpus": 1},
)
wait_for_serve_ready(app_name="fairchem-inference")
```

:::{tip}
`wait_for_serve_ready` blocks until the deployment is healthy and ready to serve. Always call it before submitting work to avoid dropped requests during startup.
:::

`setup_batch_predict_server` accepts several tuning parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `max_batch_size` | `512` | Maximum atoms per batch |
| `batch_wait_timeout_s` | `0.1` | How long to wait for a full batch before processing a partial one |
| `split_oom_batch` | `True` | Automatically halve the batch on GPU OOM and retry |
| `num_replicas` | `1` | Number of server replicas |
| `autoscaling_config` | `None` | Ray autoscaling configuration dict |

### Using the server from Ray remote tasks

The most common pattern is submitting work as Ray remote tasks. Each task connects to the running deployment by name and uses `BatchServerPredictUnit` exactly like a local predict unit.

```python
import ray
from ase import Atoms
from ase.build import bulk
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit


@ray.remote
def compute_energy_forces(deployment_name: str, atoms_dict: dict) -> dict:
    atoms = Atoms.fromdict(atoms_dict)
    unit = BatchServerPredictUnit.from_deployment_connection_info(
        deployment_name=deployment_name
    )
    atoms.calc = FAIRChemCalculator(unit, task_name="omat")
    return {
        "energy": atoms.get_potential_energy(),
        "forces": atoms.get_forces(),
        "stress": atoms.get_stress(voigt=False),
    }


systems = [bulk("Cu"), bulk("Al"), bulk("Fe"), bulk("Ni")]

futures = [
    compute_energy_forces.remote("fairchem-inference", atoms.todict())
    for atoms in systems
]
results = ray.get(futures)
```

Requests from all concurrent tasks are automatically collected and batched together on the server before each GPU forward pass.

### Using the server from outside Ray

You can also connect from a process that is not part of the Ray cluster — for example a Jupyter notebook or a script on the head node. The API is identical; `from_deployment_connection_info` handles the Ray client connection.

```python
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit
from fairchem.core.units.mlip_unit.batch_server import get_ray_connection_info

# If connecting to a remote Ray cluster, read its address from a head.json file
conn_info = get_ray_connection_info("/path/to/head.json")
unit = BatchServerPredictUnit.from_deployment_connection_info(
    deployment_name="fairchem-inference",
    ray_address=conn_info["ray_address"],
    namespace=conn_info["namespace_serve_fairchem"],
)

# Use exactly like a local MLIPPredictUnit
atoms = bulk("Cu")
atoms.calc = FAIRChemCalculator(unit, task_name="omat")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

For a local Ray cluster (same machine, same session), `ray_address` and `namespace` can be omitted:

```python
unit = BatchServerPredictUnit.from_deployment_connection_info(
    deployment_name="fairchem-inference"
)
```

## Multiplexed Server

The multiplexed server loads models on demand rather than requiring one to be loaded upfront. An LRU cache keeps up to 3 models resident per replica, so switching between a small number of models incurs no repeated loading cost.

### Starting the server

No predict unit is required at startup:

```python
import ray
from fairchem.core.units.mlip_unit.batch_server import (
    setup_multiplexed_batch_predict_server,
    wait_for_serve_ready,
)

ray.init()

setup_multiplexed_batch_predict_server(
    deployment_name="fairchem-multiplex",
    ray_actor_options={"num_cpus": 4, "num_gpus": 1},
)
wait_for_serve_ready(app_name="fairchem-multiplex")
```

### Connecting with a model ID

Clients specify which model to use via a `multiplexed_model_id` in the format `"model_name:settings"`, where `settings` is a recognized inference settings name (`"default"`, `"turbo"`) or an empty string for the default:

```python
from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit

unit = BatchServerPredictUnit.from_deployment_connection_info(
    deployment_name="fairchem-multiplex",
    multiplexed_model_id="uma-s-1p2:default",
)
```

The model is loaded on the first request and cached on the replica. Subsequent requests with the same ID are served from the cache.

### Switching between models

Different `BatchServerPredictUnit` instances with different model IDs can coexist and be used interchangeably. Each request is routed to the correct model automatically:

```python
unit_s = BatchServerPredictUnit.from_deployment_connection_info(
    deployment_name="fairchem-multiplex",
    multiplexed_model_id="uma-s-1p2:default",
)
unit_m = BatchServerPredictUnit.from_deployment_connection_info(
    deployment_name="fairchem-multiplex",
    multiplexed_model_id="uma-m-1p2:default",
)

atoms = bulk("Cu")

atoms.calc = FAIRChemCalculator(unit_s, task_name="omat")
energy_s = atoms.get_potential_energy()

atoms.calc = FAIRChemCalculator(unit_m, task_name="omat")
energy_m = atoms.get_potential_energy()
```

Concurrent requests for different models are grouped by model ID inside the server before each GPU forward pass, so batching still works efficiently across models.

### Multiplexed with Ray remote tasks

The pattern is identical to the single-model case — just pass `multiplexed_model_id` when constructing the unit inside the task:

```python
@ray.remote
def compute_energy(deployment_name: str, model_id: str, atoms_dict: dict) -> float:
    atoms = Atoms.fromdict(atoms_dict)
    unit = BatchServerPredictUnit.from_deployment_connection_info(
        deployment_name=deployment_name,
        multiplexed_model_id=model_id,
    )
    atoms.calc = FAIRChemCalculator(unit, task_name="omat")
    return atoms.get_potential_energy()


systems = [bulk("Cu"), bulk("Al"), bulk("Fe"), bulk("Ni")]
futures = [
    compute_energy.remote("fairchem-multiplex", "uma-s-1p2:default", a.todict())
    for a in systems
]
results = ray.get(futures)
```

## Accessing model metadata

`BatchServerPredictUnit` exposes the same metadata properties as a local predict unit, fetched lazily from the server on first access:

```python
unit = BatchServerPredictUnit.from_deployment_connection_info(
    deployment_name="fairchem-inference"
)

# Which datasets and tasks this model supports
print(unit.dataset_to_tasks)

# Element reference energies used for normalization
print(unit.atom_refs)

# Active inference settings on the server
print(unit.inference_settings)
```

## Choosing a mode

| | Single-model | Multiplexed |
| --- | --- | --- |
| **Model loaded** | At startup | On first request per model ID |
| **Number of models** | 1 per deployment | Up to 3 cached per replica |
| **Best for** | High-throughput single-model workloads | Workflows comparing multiple models or checkpoints |
| **Setup** | Requires a predict unit at deploy time | No predict unit needed at deploy time |

:::{note}
For high-throughput workloads over static structures where you control batching explicitly, see [Batch Inference with UMA Models](batch_inference.md). For running many independent simulations on a single node, see [Batched Atomic Simulations with InferenceBatcher](inference_batcher.md).
:::
