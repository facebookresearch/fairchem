from __future__ import annotations

import unittest

import torch

from fairchem.core.models.uma.common.so3 import CoefficientMapping
from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution, SO2_m_Conv


class TestSO2_m_Conv(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.edges = 16
        self.m = 1
        self.sphere_channels = 2 * 1
        self.m_output_channels = 2
        self.lmax = 2
        self.mmax = 2
        self.num_coefficents = self.lmax - self.m + 1
        self.num_channels = self.num_coefficents * self.sphere_channels

        self.so2mc = SO2_m_Conv(
            m=self.m,
            sphere_channels=self.sphere_channels,
            m_output_channels=self.m_output_channels,
            lmax=self.lmax,
            mmax=self.mmax,
        )

    def test_function_domain_and_codomain(self):
        x_m = torch.randn(self.edges, self.sphere_channels, self.num_channels)
        x_m_r, x_m_i = self.so2mc(x_m)
        assert isinstance(x_m_r, torch.Tensor)
        assert isinstance(x_m_i, torch.Tensor)

    def test_output_shape(self):
        x_m = torch.randn(self.edges, self.sphere_channels, self.num_channels)
        x_m_r, x_m_i = self.so2mc(x_m)
        assert x_m_r.shape == (self.edges, self.sphere_channels, self.m_output_channels)
        assert x_m_i.shape == (self.edges, self.sphere_channels, self.m_output_channels)


class TestSO2_Convolution(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.edges = 16
        self.sphere_channels = 16
        self.m_output_channels = 4
        self.lmax = 2
        self.mmax = 2
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)

        self.sum_ls = sum((2 * l + 1) for l in range(self.lmax + 1))
        self.cutoff = 12.0
        self.max_num_elements = 15

        self.edge_channels = 8
        self.distance_embedding = 7
        self.edge_channels_list = [
            self.distance_embedding + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]
        self.extra_m0_output_channels = 10

        self.so2_conv_1 = SO2_Convolution(
            sphere_channels=self.sphere_channels,
            m_output_channels=self.m_output_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mappingReduced=self.mappingReduced,
            internal_weights=False,
            edge_channels_list=self.edge_channels_list,
            extra_m0_output_channels=self.extra_m0_output_channels,
        )

        self.so2_conv_2 = SO2_Convolution(
            sphere_channels=self.m_output_channels,
            m_output_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mappingReduced=self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )

    def test_function_domain_and_codomain_1(self):
        x_message = torch.randn(self.edges, self.sum_ls, self.sphere_channels)
        x_edge = torch.randn(self.edges, self.edge_channels_list[0])
        x_message_p, x_0_gating = self.so2_conv_1(x_message, x_edge)
        assert isinstance(x_message_p, torch.Tensor)
        assert isinstance(x_0_gating, torch.Tensor)

    def test_function_domain_and_codomain_2(self):
        x_message = torch.randn(self.edges, self.sum_ls, self.m_output_channels)
        x_edge = torch.randn(self.edges, self.edge_channels_list[0])
        x_message_pp = self.so2_conv_2(x_message, x_edge)
        assert isinstance(x_message_pp, torch.Tensor)

    def test_output_shape_1(self):
        x_message = torch.randn(self.edges, self.sum_ls, self.sphere_channels)
        x_edge = torch.randn(self.edges, self.edge_channels_list[0])
        x_message_p, x_0_gating = self.so2_conv_1(x_message, x_edge)
        assert x_message_p.shape == (self.edges, self.sum_ls, self.m_output_channels)
        assert x_0_gating.shape == (self.edges, self.extra_m0_output_channels)

    def test_output_shape_2(self):
        x_message = torch.randn(self.edges, self.sum_ls, self.m_output_channels)
        x_edge = torch.randn(self.edges, self.edge_channels_list[0])
        x_message_pp = self.so2_conv_2(x_message, x_edge)
        assert x_message_pp.shape == (self.edges, self.sum_ls, self.sphere_channels)
