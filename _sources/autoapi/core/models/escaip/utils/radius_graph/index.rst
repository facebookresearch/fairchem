core.models.escaip.utils.radius_graph
=====================================

.. py:module:: core.models.escaip.utils.radius_graph

.. autoapi-nested-parse::

   Modified from MinDScAIP: Minimally biased Differentiable Scaled Attention Interatomic Potential
   Credit: Ryan Liu



Functions
---------

.. autoapisummary::

   core.models.escaip.utils.radius_graph.safe_norm
   core.models.escaip.utils.radius_graph.safe_normalize
   core.models.escaip.utils.radius_graph.envelope_fn
   core.models.escaip.utils.radius_graph.shifted_sine
   core.models.escaip.utils.radius_graph.soft_rank
   core.models.escaip.utils.radius_graph.hard_rank
   core.models.escaip.utils.radius_graph.soft_rank_low_mem
   core.models.escaip.utils.radius_graph.build_radius_graph
   core.models.escaip.utils.radius_graph.batched_radius_graph
   core.models.escaip.utils.radius_graph.biknn_radius_graph


Module Contents
---------------

.. py:function:: safe_norm(x: torch.Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-12) -> torch.Tensor

   Computes the norm of a tensor with a small epsilon to avoid division by zero.
   :param x: The input tensor.
   :param dim: The dimension to reduce.
   :param keepdim: Whether to keep the reduced dimension.
   :param eps: The epsilon value.

   :returns: The norm of the input tensor.


.. py:function:: safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor

   Computes the normalized vector with a small epsilon to avoid division by zero.
   :param x: The input tensor.
   :param dim: The dimension to reduce.
   :param eps: The epsilon value.

   :returns: The L2 normalized tensor.


.. py:function:: envelope_fn(x: torch.Tensor, envelope: bool = True) -> torch.Tensor

   Computes the envelope function in log space that smoothly vanishes to -inf at x = 1.
   :param x: The input tensor.
   :param envelope: Whether to use the envelope function. Default: True

   :returns: The envelope function in log space.


.. py:function:: shifted_sine(x: torch.Tensor) -> torch.Tensor

   Shifted sine function for the low memory soft knn. Designed such that the behavior
   matches sigmoid for small x and the step function for large x.
   :param x: the input tensor

   :returns: the shifted sine function value
   :rtype: y


.. py:function:: soft_rank(dist: torch.Tensor, scale: float) -> torch.Tensor

   calculate the soft rankings for the soft knn
   :param dist: the pairwise distance tensor
   :param scale: the scale factor for the sigmoid function (Å).

   :returns: the soft rankings
   :rtype: ranks


.. py:function:: hard_rank(dist: torch.Tensor) -> torch.Tensor

   calculate the hard rankings for the hard knn
   :param dist: the pairwise distance tensor

   :returns: the hard rankings
   :rtype: ranks


.. py:function:: soft_rank_low_mem(dist: torch.Tensor, k: int, scale: float, delta: int = 20) -> torch.Tensor

   calculate the soft rankings for the soft knn. Approximate with low memory by
   truncating the distance matrix to be [0, k + delta]. This is not exact but is a good
   approximation. It is valid when the difference of distance at k+delta and k is
   larger than pi * scale.
   :param dist: the pairwise distance tensor
   :param k: the number of neighbors
   :param scale: the scale factor for the shifted sine function (Å).
   :param delta: the delta factor for the truncation

   :returns: the soft rankings
   :rtype: ranks


.. py:function:: build_radius_graph(pos: torch.Tensor, cell: torch.Tensor, image_id: torch.Tensor, cutoff: float, start_index: int, device: torch.device, k: int = 30, soft: bool = False, sigmoid_scale: float = 0.2, lse_scale: float = 0.1, use_low_mem: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

   construct the biknn radius graph for one system.
   :param pos: the atomic positions tensor
   :param cell: the cell tensor for the periodic boundary condition
   :param image_id: the image identifier for different PBC images
   :param cutoff: the cutoff distance in Angstrom
   :param start_index: the starting index of the system in the batch
   :param device: the device on which the tensors are allocated
   :param k: the number of neighbors
   :param soft: the flag for the soft knn
   :param sigmoid_scale: the scale factor for the sigmoid function
   :param lse_scale: the scale factor for the log-sum-exp function
   :param use_low_mem: the flag for the low memory soft knn

   :returns: the source index of the neighbors
             index2: the destination index of the neighbors
             index1_rank: the rank of the edge in source neighbors by envelope function
             index2_rank: the rank of the edge in destination neighbors by envelope function
             disp: the displacement vector of the neighbors
             env: the envelope vector of the neighbors
   :rtype: index1


.. py:function:: batched_radius_graph(pos_list: list[torch.Tensor], cell_list: list[torch.Tensor], image_id_list: list[torch.Tensor], N: int, natoms: torch.Tensor, knn_k: int, knn_soft: bool, knn_sigmoid_scale: float, knn_lse_scale: float, knn_use_low_mem: bool, knn_pad_size: int, cutoff: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

   calculate the biknn radius graph for the batch of systems
   :param pos_list: the list of atomic positions tensors
   :param anum_list: the list of atomic number tensors
   :param cell_list: the list of cell tensors
   :param image_id_list: the list of image identifier tensors
   :param N: the total number of atoms in the batch
   :param natoms: the number of atoms in each system
   :param knn_params: the parameters for the knn algorithm
   :param cutoff: the cutoff distance in Angstrom
   :param device: the device on which the tensors are allocated

   :returns: the padded displacement tensor
             src_env: the source envelope tensor
             dst_env: the destination envelope tensor
             src_index: the destination layout to source layout index tensor
             dst_index: the source layout to destination layout index tensor
             edge_index: the edge index tensor
   :rtype: padded_disp


.. py:function:: biknn_radius_graph(data: fairchem.core.datasets.atomic_data.AtomicData, cutoff: float, knn_k: int, knn_soft: bool, knn_sigmoid_scale: float, knn_lse_scale: float, knn_use_low_mem: bool, knn_pad_size: int, use_pbc: bool, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]

   function to construct the biknn radius graph for the batch of systems. This function
   calculates the number of images to be included in the PBC and constructs the
   image identifier list and call the batched_radius_graph function to perform the
   construction.
   :param data: the `torch_geometric.data.Data` object containing the atomic information
   :param cutoff: the cutoff distance in Angstrom
   :param knn_params: the parameters for the knn algorithm
   :param use_pbc: the flag for the periodic boundary condition
   :param device: the device on which the tensors are allocated

   :returns: the padded displacement tensor
             src_env: the source envelope tensor
             dst_env: the destination envelope tensor
             src_index: the destination layout to source layout index tensor
             dst_index: the source layout to destination layout index tensor
             edge_index: the edge index tensor
   :rtype: padded_disp


