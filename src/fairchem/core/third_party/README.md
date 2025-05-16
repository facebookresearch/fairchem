
# Trimmed-down `pytorch_geometric`

The following code has been copied from [MACE](https://github.com/ACEsuit/mace/tree/main/mace/tools/torch_geometric) which is a modified subset of code originally used in pytorch_geometric. We are very grateful to the developers of both MACE and PyTorch Geomteric for their contributions!

MACE(FAIRChem) uses [`pytorch_geometric`](https://pytorch-geometric.readthedocs.io/en/latest/) [1, 2] framework. However as only use a very limited subset of that library: the most basic graph data structures.

We follow the same approach to NequIP (https://github.com/mir-group/nequip/tree/main/nequip) and copy their code here.

To avoid adding a large number of unnecessary second-degree dependencies, and to simplify installation, we include and modify here the small subset of `torch_geometric` that is necessary for our code.

We are grateful to the developers of PyTorch Geometric for their ongoing and very useful work on graph learning with PyTorch.

[1]  Fey, M., & Lenssen, J. E. (2019). Fast Graph Representation Learning with PyTorch Geometric (Version 2.0.1) [Computer software]. https://github.com/pyg-team/pytorch_geometric <br>
[2]  https://arxiv.org/abs/1903.02428
