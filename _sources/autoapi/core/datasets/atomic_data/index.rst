core.datasets.atomic_data
=========================

.. py:module:: core.datasets.atomic_data

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.

   modified from troch_geometric Data class



Attributes
----------

.. autoapisummary::

   core.datasets.atomic_data.IndexType
   core.datasets.atomic_data._REQUIRED_KEYS
   core.datasets.atomic_data._OPTIONAL_KEYS


Classes
-------

.. autoapisummary::

   core.datasets.atomic_data.AtomicData


Functions
---------

.. autoapisummary::

   core.datasets.atomic_data.size_repr
   core.datasets.atomic_data.get_neighbors_pymatgen
   core.datasets.atomic_data.reshape_features
   core.datasets.atomic_data.atomicdata_list_to_batch
   core.datasets.atomic_data.tensor_or_int_to_tensor


Module Contents
---------------

.. py:data:: IndexType

.. py:data:: _REQUIRED_KEYS
   :value: ['pos', 'atomic_numbers', 'cell', 'pbc', 'natoms', 'edge_index', 'cell_offsets', 'nedges',...


.. py:data:: _OPTIONAL_KEYS
   :value: ['energy', 'forces', 'stress', 'dataset']


.. py:function:: size_repr(key: str, item: torch.Tensor, indent=0) -> str

.. py:function:: get_neighbors_pymatgen(atoms: ase.Atoms, cutoff, max_neigh)

   Preforms nearest neighbor search and returns edge index, distances,
   and cell offsets


.. py:function:: reshape_features(c_index: numpy.ndarray, n_index: numpy.ndarray, n_distance: numpy.ndarray, offsets: numpy.ndarray)

   Stack center and neighbor index and reshapes distances,
   takes in np.arrays and returns torch tensors


.. py:class:: AtomicData(pos: torch.Tensor, atomic_numbers: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, natoms: torch.Tensor, edge_index: torch.Tensor, cell_offsets: torch.Tensor, nedges: torch.Tensor, charge: torch.Tensor, spin: torch.Tensor, fixed: torch.Tensor, tags: torch.Tensor, energy: torch.Tensor | None = None, forces: torch.Tensor | None = None, stress: torch.Tensor | None = None, batch: torch.Tensor | None = None, sid: list[str] | None = None, dataset: list[str] | str | None = None)

   .. py:attribute:: __keys__


   .. py:attribute:: pos


   .. py:attribute:: atomic_numbers


   .. py:attribute:: cell


   .. py:attribute:: pbc


   .. py:attribute:: natoms


   .. py:attribute:: edge_index


   .. py:attribute:: cell_offsets


   .. py:attribute:: nedges


   .. py:attribute:: charge


   .. py:attribute:: spin


   .. py:attribute:: fixed


   .. py:attribute:: tags


   .. py:attribute:: sid


   .. py:attribute:: __slices__
      :value: None



   .. py:attribute:: __cumsum__
      :value: None



   .. py:attribute:: __cat_dims__
      :value: None



   .. py:attribute:: __natoms_list__
      :value: None



   .. py:property:: task_name


   .. py:method:: assign_batch_stats(slices, cumsum, cat_dims, natoms_list)


   .. py:method:: get_batch_stats()


   .. py:method:: validate()


   .. py:method:: from_ase(input_atoms: ase.Atoms, r_edges: bool = False, radius: float = 6.0, max_neigh: int | None = None, sid: str | None = None, molecule_cell_size: float | None = None, r_energy: bool = True, r_forces: bool = True, r_stress: bool = True, r_data_keys: list[str] | None = None, task_name: str | None = None) -> AtomicData
      :classmethod:



   .. py:method:: to_ase_single() -> ase.Atoms


   .. py:method:: to_ase() -> list[ase.Atoms]


   .. py:method:: from_dict(dictionary)
      :classmethod:


      Creates a data object from a python dictionary.



   .. py:method:: to_dict()


   .. py:method:: values()


   .. py:property:: num_nodes
      :type: int


      Returns or sets the number of nodes in the graph.


   .. py:property:: num_edges
      :type: int


      Returns the number of edges in the graph.


   .. py:property:: num_graphs
      :type: int


      Returns the number of graphs in the batch.


   .. py:method:: __len__()


   .. py:method:: get(key, default)


   .. py:method:: __getitem__(idx)


   .. py:method:: __setitem__(key: str, value: torch.Tensor)

      Sets the attribute :obj:`key` to :obj:`value`.



   .. py:method:: __setattr__(key: str, value: torch.Tensor)


   .. py:method:: __delitem__(key: str)

      Deletes the attribute :obj:`key`.



   .. py:method:: keys()


   .. py:method:: __contains__(key)

      Returns :obj:`True`, if the attribute :obj:`key` is present in the
      data.



   .. py:method:: __iter__()

      Iterates over all present attributes in the data, yielding their
      attribute names and content.



   .. py:method:: __call__(*keys)

      Iterates over all attributes :obj:`*keys` in the data, yielding
      their attribute names and content.
      If :obj:`*keys` is not given this method will iterative over all
      present attributes.



   .. py:method:: __cat_dim__(key, value) -> int

      Returns the dimension for which :obj:`value` of attribute
      :obj:`key` will get concatenated when creating batches.

      .. note::

          This method is for internal use only, and should only be overridden
          if the batch concatenation process is corrupted for a specific data
          attribute.



   .. py:method:: __inc__(key, value) -> int

      Returns the incremental count to cumulatively increase the value
      of the next attribute of :obj:`key` when creating batches.

      .. note::

          This method is for internal use only, and should only be overridden
          if the batch concatenation process is corrupted for a specific data
          attribute.



   .. py:method:: __apply__(item, func)


   .. py:method:: apply(func)

      Applies the function :obj:`func` to all tensor attributes



   .. py:method:: contiguous()

      Ensures a contiguous memory layout for all tensor attributes



   .. py:method:: to(device, **kwargs)

      Performs tensor dtype and/or device conversion for all tensor attributes



   .. py:method:: cpu()

      Copies all tensor attributes to CPU memory.



   .. py:method:: cuda(device=None, non_blocking=False)

      Copies all tensor attributes to GPU memory.



   .. py:method:: clone()

      Performs a deep-copy of the data object.



   .. py:method:: __repr__()


   .. py:method:: get_example(idx: int) -> AtomicData

      Reconstructs the :class:`AtomicData` object at index
      :obj:`idx` from a batched AtomicData object.



   .. py:method:: index_select(idx: IndexType) -> list[AtomicData]


   .. py:method:: batch_to_atomicdata_list() -> list[AtomicData]

      Reconstructs the list of :class:`torch_geometric.data.Data` objects
      from the batch object.
      The batch object must have been created via :meth:`from_data_list` in
      order to be able to reconstruct the initial objects.



.. py:function:: atomicdata_list_to_batch(data_list: list[AtomicData], exclude_keys: Optional[list] = None) -> AtomicData

   all data points must be single graphs and have the same set of keys.
   TODO: exclude keys?


.. py:function:: tensor_or_int_to_tensor(x, dtype=torch.int)

