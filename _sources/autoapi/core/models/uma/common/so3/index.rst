core.models.uma.common.so3
==========================

.. py:module:: core.models.uma.common.so3

.. autoapi-nested-parse::

   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.



Classes
-------

.. autoapisummary::

   core.models.uma.common.so3.CoefficientMapping
   core.models.uma.common.so3.SO3_Grid


Module Contents
---------------

.. py:class:: CoefficientMapping(lmax, mmax)

   Bases: :py:obj:`torch.nn.Module`


   Helper module for coefficients used to reshape l <--> m and to get coefficients of specific degree or order

   :param lmax_list (list: int):   List of maximum degree of the spherical harmonics
   :param mmax_list (list: int):   List of maximum order of the spherical harmonics
   :param use_rotate_inv_rescale: Whether to pre-compute inverse rotation rescale matrices
   :type use_rotate_inv_rescale: bool


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: res_size


   .. py:attribute:: m_size


   .. py:method:: complex_idx(m, lmax, m_complex, l_harmonic)

      Add `m_complex` and `l_harmonic` to the input arguments
      since we cannot use `self.m_complex`.



   .. py:method:: pre_compute_coefficient_idx()

      Pre-compute the results of `coefficient_idx()` and access them with `prepare_coefficient_idx()`



   .. py:method:: prepare_coefficient_idx()

      Construct a list of buffers



   .. py:method:: coefficient_idx(lmax: int, mmax: int)


   .. py:method:: pre_compute_rotate_inv_rescale()


   .. py:method:: __repr__()


.. py:class:: SO3_Grid(lmax: int, mmax: int, normalization: str = 'integral', resolution: int | None = None, rescale: bool = True)

   Bases: :py:obj:`torch.nn.Module`


   Helper functions for grid representation of the irreps

   :param lmax: Maximum degree of the spherical harmonics
   :type lmax: int
   :param mmax: Maximum order of the spherical harmonics
   :type mmax: int


   .. py:attribute:: lmax


   .. py:attribute:: mmax


   .. py:attribute:: lat_resolution


   .. py:attribute:: mapping


   .. py:attribute:: rescale


   .. py:method:: get_to_grid_mat(device=None)


   .. py:method:: get_from_grid_mat(device=None)


   .. py:method:: to_grid(embedding, lmax: int, mmax: int)


   .. py:method:: from_grid(grid, lmax: int, mmax: int)


