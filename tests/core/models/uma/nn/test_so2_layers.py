from __future__ import annotations

import unittest

import torch

from fairchem.core.models.uma.common.so3 import CoefficientMapping
from fairchem.core.models.uma.nn.so2_layers import (
    SO2_Conv1_WithRadialBlock,
    SO2_Conv2_InternalBlock,
    SO2_Convolution,
    SO2_m_Conv,
    SO2_m_Conv_Block,
    convert_so2_conv1,
    convert_so2_conv2,
)


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


class TestSO2_m_Conv_Block(unittest.TestCase):
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

        self.block = SO2_m_Conv_Block(
            m=self.m,
            sphere_channels=self.sphere_channels,
            m_output_channels=self.m_output_channels,
            lmax=self.lmax,
            mmax=self.mmax,
        )

    def test_output_shape(self):
        x_m = torch.randn(self.edges, 2, self.num_channels)
        x_m_r, x_m_i = self.block(x_m)
        num_l = self.num_coefficents
        assert x_m_r.shape == (self.edges, num_l, self.m_output_channels)
        assert x_m_i.shape == (self.edges, num_l, self.m_output_channels)

    def test_matches_so2_m_conv(self):
        """Block GEMM must produce identical output to standard GEMM."""
        ref = SO2_m_Conv(
            m=self.m,
            sphere_channels=self.sphere_channels,
            m_output_channels=self.m_output_channels,
            lmax=self.lmax,
            mmax=self.mmax,
        )
        # Copy weights from ref to block
        self.block.fc.load_state_dict(ref.fc.state_dict())
        self.block._w_block = None  # force rebuild

        x_m = torch.randn(self.edges, 2, self.num_channels)
        ref_r, ref_i = ref(x_m)
        blk_r, blk_i = self.block(x_m)
        torch.testing.assert_close(blk_r, ref_r)
        torch.testing.assert_close(blk_i, ref_i)

    def test_w_block_built_after_forward(self):
        assert self.block._w_block is None
        x_m = torch.randn(self.edges, 2, self.num_channels)
        self.block(x_m)
        assert self.block._w_block is not None

    def test_build_w_block_explicit(self):
        assert self.block._w_block is None
        self.block._build_w_block()
        assert self.block._w_block is not None
        expected_shape = (
            2 * self.block.out_channels_half,
            2 * self.num_channels,
        )
        assert self.block._w_block.shape == expected_shape


class TestSO2_Conv1_WithRadialBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.edges = 16
        self.sphere_channels = 16
        self.m_output_channels = 4
        self.lmax = 2
        self.mmax = 2
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)
        self.sum_ls = sum((2 * l + 1) for l in range(self.lmax + 1))

        self.edge_channels = 8
        self.distance_embedding = 7
        self.edge_channels_list = [
            self.distance_embedding + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]
        self.extra_m0_output_channels = self.lmax * self.m_output_channels

        self.conv1_block = SO2_Conv1_WithRadialBlock(
            sphere_channels=self.sphere_channels,
            m_output_channels=self.m_output_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mappingReduced=self.mappingReduced,
            extra_m0_output_channels=self.extra_m0_output_channels,
            edge_channels_list=self.edge_channels_list[:],
        )

    def test_output_is_always_tuple(self):
        """Conv1 always returns (output, gating) -- no conditional."""
        x = torch.randn(self.edges, self.sum_ls, self.sphere_channels)
        x_edge = torch.randn(self.edges, self.edge_channels_list[0])
        out, gating = self.conv1_block(x, x_edge)
        assert isinstance(out, torch.Tensor)
        assert isinstance(gating, torch.Tensor)

    def test_output_shapes(self):
        x = torch.randn(self.edges, self.sum_ls, self.sphere_channels)
        x_edge = torch.randn(self.edges, self.edge_channels_list[0])
        out, gating = self.conv1_block(x, x_edge)
        assert out.shape == (
            self.edges,
            self.sum_ls,
            self.m_output_channels,
        )
        assert gating.shape == (
            self.edges,
            self.extra_m0_output_channels,
        )

    def test_matches_so2_convolution(self):
        """Must produce identical output to SO2_Convolution."""
        ref = SO2_Convolution(
            sphere_channels=self.sphere_channels,
            m_output_channels=self.m_output_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mappingReduced=self.mappingReduced,
            internal_weights=False,
            edge_channels_list=self.edge_channels_list[:],
            extra_m0_output_channels=self.extra_m0_output_channels,
        )
        # Copy weights from ref to block variant
        self.conv1_block.load_state_dict(ref.state_dict())

        x = torch.randn(self.edges, self.sum_ls, self.sphere_channels)
        x_edge = torch.randn(self.edges, self.edge_channels_list[0])

        ref_out, ref_gating = ref(x, x_edge)
        blk_out, blk_gating = self.conv1_block(x, x_edge)
        torch.testing.assert_close(blk_out, ref_out)
        torch.testing.assert_close(blk_gating, ref_gating)

    def test_uses_so2_m_conv_block_internally(self):
        for m_conv in self.conv1_block.so2_m_conv:
            assert isinstance(m_conv, SO2_m_Conv_Block)


class TestSO2_Conv2_InternalBlock(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.edges = 16
        self.sphere_channels = 16
        self.m_output_channels = 4
        self.lmax = 2
        self.mmax = 2
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)
        self.sum_ls = sum((2 * l + 1) for l in range(self.lmax + 1))

        self.conv2_block = SO2_Conv2_InternalBlock(
            sphere_channels=self.m_output_channels,
            m_output_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mappingReduced=self.mappingReduced,
        )

    def test_output_is_single_tensor(self):
        x = torch.randn(self.edges, self.sum_ls, self.m_output_channels)
        out = self.conv2_block(x)
        assert isinstance(out, torch.Tensor)

    def test_output_shape(self):
        x = torch.randn(self.edges, self.sum_ls, self.m_output_channels)
        out = self.conv2_block(x)
        assert out.shape == (
            self.edges,
            self.sum_ls,
            self.sphere_channels,
        )

    def test_accepts_x_edge_kwarg(self):
        """Edgewise passes x_edge to conv2 -- must accept and ignore it."""
        x = torch.randn(self.edges, self.sum_ls, self.m_output_channels)
        x_edge = torch.randn(self.edges, 10)
        out = self.conv2_block(x, x_edge)
        assert isinstance(out, torch.Tensor)

    def test_matches_so2_convolution(self):
        """Must produce identical output to SO2_Convolution."""
        ref = SO2_Convolution(
            sphere_channels=self.m_output_channels,
            m_output_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mappingReduced=self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )
        self.conv2_block.load_state_dict(ref.state_dict())

        x = torch.randn(self.edges, self.sum_ls, self.m_output_channels)
        x_edge = torch.randn(self.edges, 10)

        ref_out = ref(x, x_edge)
        blk_out = self.conv2_block(x, x_edge)
        torch.testing.assert_close(blk_out, ref_out)

    def test_uses_so2_m_conv_block_internally(self):
        for m_conv in self.conv2_block.so2_m_conv:
            assert isinstance(m_conv, SO2_m_Conv_Block)


class TestConvertSO2Conv(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.edges = 16
        self.sphere_channels = 16
        self.hidden_channels = 4
        self.lmax = 2
        self.mmax = 2
        self.mappingReduced = CoefficientMapping(self.lmax, self.mmax)
        self.sum_ls = sum((2 * l + 1) for l in range(self.lmax + 1))

        self.edge_channels = 8
        self.distance_embedding = 7
        self.edge_channels_list = [
            self.distance_embedding + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]
        self.extra_m0_output_channels = self.lmax * self.hidden_channels

    def test_convert_so2_conv1_returns_correct_type(self):
        old = SO2_Convolution(
            sphere_channels=self.sphere_channels,
            m_output_channels=self.hidden_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mappingReduced=self.mappingReduced,
            internal_weights=False,
            edge_channels_list=self.edge_channels_list[:],
            extra_m0_output_channels=self.extra_m0_output_channels,
        )
        new = convert_so2_conv1(old)
        assert isinstance(new, SO2_Conv1_WithRadialBlock)

    def test_convert_so2_conv1_numerically_identical(self):
        old = SO2_Convolution(
            sphere_channels=self.sphere_channels,
            m_output_channels=self.hidden_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mappingReduced=self.mappingReduced,
            internal_weights=False,
            edge_channels_list=self.edge_channels_list[:],
            extra_m0_output_channels=self.extra_m0_output_channels,
        )
        new = convert_so2_conv1(old)

        x = torch.randn(self.edges, self.sum_ls, self.sphere_channels)
        x_edge = torch.randn(self.edges, self.edge_channels_list[0])

        old_out, old_gating = old(x, x_edge)
        new_out, new_gating = new(x, x_edge)
        torch.testing.assert_close(new_out, old_out)
        torch.testing.assert_close(new_gating, old_gating)

    def test_convert_so2_conv1_w_block_prebuilt(self):
        old = SO2_Convolution(
            sphere_channels=self.sphere_channels,
            m_output_channels=self.hidden_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mappingReduced=self.mappingReduced,
            internal_weights=False,
            edge_channels_list=self.edge_channels_list[:],
            extra_m0_output_channels=self.extra_m0_output_channels,
        )
        new = convert_so2_conv1(old)
        for m_conv in new.so2_m_conv:
            assert m_conv._w_block is not None

    def test_convert_so2_conv2_returns_correct_type(self):
        old = SO2_Convolution(
            sphere_channels=self.hidden_channels,
            m_output_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mappingReduced=self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )
        new = convert_so2_conv2(old)
        assert isinstance(new, SO2_Conv2_InternalBlock)

    def test_convert_so2_conv2_numerically_identical(self):
        old = SO2_Convolution(
            sphere_channels=self.hidden_channels,
            m_output_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mappingReduced=self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )
        new = convert_so2_conv2(old)

        x = torch.randn(self.edges, self.sum_ls, self.hidden_channels)
        x_edge = torch.randn(self.edges, 10)

        old_out = old(x, x_edge)
        new_out = new(x, x_edge)
        torch.testing.assert_close(new_out, old_out)

    def test_convert_so2_conv2_w_block_prebuilt(self):
        old = SO2_Convolution(
            sphere_channels=self.hidden_channels,
            m_output_channels=self.sphere_channels,
            lmax=self.lmax,
            mmax=self.mmax,
            mappingReduced=self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=None,
        )
        new = convert_so2_conv2(old)
        for m_conv in new.so2_m_conv:
            assert m_conv._w_block is not None


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
