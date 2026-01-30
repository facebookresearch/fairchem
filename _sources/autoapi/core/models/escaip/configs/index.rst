core.models.escaip.configs
==========================

.. py:module:: core.models.escaip.configs


Classes
-------

.. autoapisummary::

   core.models.escaip.configs.GlobalConfigs
   core.models.escaip.configs.MolecularGraphConfigs
   core.models.escaip.configs.GraphNeuralNetworksConfigs
   core.models.escaip.configs.RegularizationConfigs
   core.models.escaip.configs.EScAIPConfigs


Functions
---------

.. autoapisummary::

   core.models.escaip.configs.resolve_type_hint
   core.models.escaip.configs.init_configs


Module Contents
---------------

.. py:class:: GlobalConfigs

   .. py:attribute:: regress_forces
      :type:  bool


   .. py:attribute:: direct_forces
      :type:  bool


   .. py:attribute:: hidden_size
      :type:  int


   .. py:attribute:: num_layers
      :type:  int


   .. py:attribute:: activation
      :type:  Literal['squared_relu', 'gelu', 'leaky_relu', 'relu', 'smelu', 'star_relu']
      :value: 'gelu'



   .. py:attribute:: regress_stress
      :type:  bool
      :value: False



   .. py:attribute:: use_compile
      :type:  bool
      :value: True



   .. py:attribute:: use_padding
      :type:  bool
      :value: True



   .. py:attribute:: use_fp16_backbone
      :type:  bool
      :value: False



   .. py:attribute:: dataset_list
      :type:  list


.. py:class:: MolecularGraphConfigs

   .. py:attribute:: use_pbc
      :type:  bool


   .. py:attribute:: max_num_elements
      :type:  int


   .. py:attribute:: max_atoms
      :type:  int


   .. py:attribute:: max_batch_size
      :type:  int


   .. py:attribute:: max_radius
      :type:  float


   .. py:attribute:: knn_k
      :type:  int


   .. py:attribute:: knn_soft
      :type:  bool


   .. py:attribute:: knn_sigmoid_scale
      :type:  float


   .. py:attribute:: knn_lse_scale
      :type:  float


   .. py:attribute:: knn_use_low_mem
      :type:  bool


   .. py:attribute:: knn_pad_size
      :type:  int


   .. py:attribute:: distance_function
      :type:  Literal['gaussian', 'sigmoid', 'linearsigmoid', 'silu']
      :value: 'gaussian'



   .. py:attribute:: use_envelope
      :type:  bool
      :value: True



.. py:class:: GraphNeuralNetworksConfigs

   .. py:attribute:: atten_name
      :type:  Literal['math', 'memory_efficient', 'flash']


   .. py:attribute:: atten_num_heads
      :type:  int


   .. py:attribute:: atom_embedding_size
      :type:  int
      :value: 128



   .. py:attribute:: node_direction_embedding_size
      :type:  int
      :value: 64



   .. py:attribute:: node_direction_expansion_size
      :type:  int
      :value: 10



   .. py:attribute:: edge_distance_expansion_size
      :type:  int
      :value: 600



   .. py:attribute:: edge_distance_embedding_size
      :type:  int
      :value: 512



   .. py:attribute:: readout_hidden_layer_multiplier
      :type:  int
      :value: 2



   .. py:attribute:: output_hidden_layer_multiplier
      :type:  int
      :value: 2



   .. py:attribute:: ffn_hidden_layer_multiplier
      :type:  int
      :value: 2



   .. py:attribute:: use_angle_embedding
      :type:  Literal['scalar', 'bias', 'none']
      :value: 'none'



   .. py:attribute:: angle_expansion_size
      :type:  int
      :value: 10



   .. py:attribute:: angle_embedding_size
      :type:  int
      :value: 8



   .. py:attribute:: use_graph_attention
      :type:  bool
      :value: False



   .. py:attribute:: use_message_gate
      :type:  bool
      :value: False



   .. py:attribute:: use_global_readout
      :type:  bool
      :value: False



   .. py:attribute:: use_frequency_embedding
      :type:  bool
      :value: True



   .. py:attribute:: freequency_list
      :type:  list


   .. py:attribute:: energy_reduce
      :type:  Literal['sum', 'mean']
      :value: 'sum'



.. py:class:: RegularizationConfigs

   .. py:attribute:: normalization
      :type:  Literal['layernorm', 'rmsnorm', 'skip']
      :value: 'rmsnorm'



   .. py:attribute:: mlp_dropout
      :type:  float
      :value: 0.0



   .. py:attribute:: atten_dropout
      :type:  float
      :value: 0.0



   .. py:attribute:: stochastic_depth_prob
      :type:  float
      :value: 0.0



   .. py:attribute:: node_ffn_dropout
      :type:  float
      :value: 0.0



   .. py:attribute:: edge_ffn_dropout
      :type:  float
      :value: 0.0



   .. py:attribute:: scalar_output_dropout
      :type:  float
      :value: 0.0



   .. py:attribute:: vector_output_dropout
      :type:  float
      :value: 0.0



.. py:class:: EScAIPConfigs

   .. py:attribute:: global_cfg
      :type:  GlobalConfigs


   .. py:attribute:: molecular_graph_cfg
      :type:  MolecularGraphConfigs


   .. py:attribute:: gnn_cfg
      :type:  GraphNeuralNetworksConfigs


   .. py:attribute:: reg_cfg
      :type:  RegularizationConfigs


.. py:function:: resolve_type_hint(cls, field)

   Resolves forward reference type hints from string to actual class objects.


.. py:function:: init_configs(cls, kwargs)

   Initialize a dataclass with the given kwargs.


