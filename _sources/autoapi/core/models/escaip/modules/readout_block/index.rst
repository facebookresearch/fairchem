core.models.escaip.modules.readout_block
========================================

.. py:module:: core.models.escaip.modules.readout_block


Classes
-------

.. autoapisummary::

   core.models.escaip.modules.readout_block.ReadoutBlock


Module Contents
---------------

.. py:class:: ReadoutBlock(global_cfg: fairchem.core.models.escaip.configs.GlobalConfigs, gnn_cfg: fairchem.core.models.escaip.configs.GraphNeuralNetworksConfigs, reg_cfg: fairchem.core.models.escaip.configs.RegularizationConfigs)

   Bases: :py:obj:`torch.nn.Module`


   Readout from each graph attention block for energy and force output


   .. py:attribute:: backbone_dtype


   .. py:attribute:: energy_reduce


   .. py:attribute:: use_edge_readout


   .. py:attribute:: use_global_readout


   .. py:attribute:: node_ffn


   .. py:attribute:: pre_node_norm


   .. py:method:: forward(data, node_features, edge_features)

      Output:
          Global Readout (G, H);
          Node Readout (N, H);
          Edge Readout (N, max_nei, H)



