"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.common.so3 import CoefficientMapping
from fairchem.core.models.uma.nn.embedding import DatasetEmbedding, EdgeDegreeEmbedding
from fairchem.core.models.uma.nn.execution_backends import get_execution_backend


class TestDatasetEmbedding:
    """Test the DatasetEmbedding class."""

    def test_embeddings_trainable_when_grad_true(self):
        """Test that embeddings have requires_grad=True when grad=True."""
        dataset_mapping = {"oc20": "oc20", "omat": "omat", "omol": "omol"}
        embedding_size = 64

        layer = DatasetEmbedding(
            embedding_size=embedding_size,
            enable_grad=True,
            dataset_mapping=dataset_mapping,
        )

        # Check all embedding parameters have requires_grad=True
        for dataset in dataset_mapping:
            for param in layer.dataset_emb_dict[dataset].parameters():
                assert (
                    param.requires_grad is True
                ), f"Expected requires_grad=True for dataset '{dataset}'"

    def test_embeddings_not_trainable_when_grad_false(self):
        """Test that embeddings have requires_grad=False when grad=False."""
        dataset_mapping = {"oc20": "oc20", "omat": "omat", "omol": "omol"}
        embedding_size = 64

        layer = DatasetEmbedding(
            embedding_size=embedding_size,
            enable_grad=False,
            dataset_mapping=dataset_mapping,
        )

        # Check all embedding parameters have requires_grad=False
        for dataset in dataset_mapping:
            for param in layer.dataset_emb_dict[dataset].parameters():
                assert (
                    param.requires_grad is False
                ), f"Expected requires_grad=False for dataset '{dataset}'"

    def test_dataset_mapping(self):
        """Test that dataset_mapping correctly maps one dataset to another's embedding."""
        dataset_mapping = {
            "oc20_subset": "oc20",
            "oc20": "oc20",
            "omat": "omat",
            "omol": "omol",
        }
        no_dataset_mapping = {
            "oc20_subset": "oc20_subset",
            "oc20": "oc20",
            "omat": "omat",
            "omol": "omol",
        }
        embedding_size = 64

        # Instance 1: no mapping
        torch.manual_seed(42)
        layer_no_mapping = DatasetEmbedding(
            embedding_size=embedding_size,
            enable_grad=False,
            dataset_mapping=no_dataset_mapping,
        )
        layer_no_mapping.eval()

        # Instance 2: with mapping
        torch.manual_seed(42)
        layer_with_mapping = DatasetEmbedding(
            embedding_size=embedding_size,
            enable_grad=False,
            dataset_mapping=dataset_mapping,
        )
        layer_with_mapping.eval()

        # Test 1: layer_with_mapping(["oc20_subset"]) == layer_with_mapping(["oc20"])
        # Both resolve to oc20's embedding when mapping is active
        assert torch.allclose(
            layer_with_mapping(["oc20_subset"]), layer_with_mapping(["oc20"])
        ), "With mapping, 'oc20_subset' should return same embedding as 'oc20'"

        # Test 2: layer_with_mapping(["oc20_subset"]) == layer_no_mapping(["oc20"])
        # With mapping, oc20_subset uses oc20's embedding
        assert torch.allclose(
            layer_with_mapping(["oc20_subset"]), layer_no_mapping(["oc20"])
        ), "With mapping, 'oc20_subset' should return same embedding as unmapped 'oc20'"

        # Test 3: layer_with_mapping(["oc20_subset"]) != layer_no_mapping(["oc20_subset"])
        # Without mapping, oc20_subset uses its own embedding
        assert not torch.allclose(
            layer_with_mapping(["oc20_subset"]), layer_no_mapping(["oc20_subset"])
        ), "With mapping, 'oc20_subset' should differ from unmapped 'oc20_subset'"

        # Test 4: oc20 should not equal omol or omat
        oc20_embedding = layer_with_mapping(["oc20"])
        omol_embedding = layer_with_mapping(["omol"])
        omat_embedding = layer_with_mapping(["omat"])

        assert not torch.allclose(
            oc20_embedding, omol_embedding
        ), "'oc20' should not equal 'omol' embedding"
        assert not torch.allclose(
            oc20_embedding, omat_embedding
        ), "'oc20' should not equal 'omat' embedding"


class TestEdgeDegreeEmbedding:
    """
    Test the EdgeDegreeEmbedding class.
    """

    num_nodes = 10
    num_edges = 40
    sphere_channels = 8
    edge_channels_list = (12, 16, 16)

    def _make_module(
        self,
        activation_checkpoint_chunk_size: int | None = None,
        execution_mode: str = "general",
    ) -> EdgeDegreeEmbedding:
        """
        Build a small lmax=mmax=2 EdgeDegreeEmbedding in float64.
        """
        torch.manual_seed(42)
        return EdgeDegreeEmbedding(
            sphere_channels=self.sphere_channels,
            lmax=2,
            mmax=2,
            edge_channels_list=list(self.edge_channels_list),
            rescale_factor=5.0,
            mappingReduced=CoefficientMapping(2, 2),
            activation_checkpoint_chunk_size=activation_checkpoint_chunk_size,
            backend=get_execution_backend(execution_mode),
        ).double()

    @pytest.fixture()
    def inputs(self):
        """
        Random node features, edge features, targets and inverse Wigner matrices.

        Targets repeat, and the last node receives no edges.
        """
        torch.manual_seed(0)
        x = torch.randn(self.num_nodes, 9, self.sphere_channels, dtype=torch.float64)
        x_edge = torch.randn(
            self.num_edges, self.edge_channels_list[0], dtype=torch.float64
        )
        scatter_target = torch.randint(0, self.num_nodes - 1, (self.num_edges,))
        wigner_inv = torch.randn(self.num_edges, 9, 9, dtype=torch.float64)
        return x, x_edge, scatter_target, wigner_inv

    def test_output_is_input_plus_per_edge_contributions(self, inputs):
        """
        Each edge adds its own contribution to its target node, and nodes with
        no incoming edges are unchanged.
        """
        x, x_edge, scatter_target, wigner_inv = inputs
        module = self._make_module()

        out = module(x, x_edge, scatter_target, wigner_inv)

        zeros = torch.zeros_like(x)
        per_edge = [
            module(
                zeros,
                x_edge[i : i + 1],
                scatter_target[i : i + 1],
                wigner_inv[i : i + 1],
            )
            for i in range(self.num_edges)
        ]
        torch.testing.assert_close(out, x + torch.stack(per_edge).sum(dim=0))
        torch.testing.assert_close(out[-1], x[-1], rtol=0, atol=0)

    def test_only_m0_wigner_columns_are_used(self, inputs):
        """
        The embedding only fills m=0 coefficients, so only the first
        m_0_num_coefficients columns of the M-ordered inverse Wigner matrices
        affect the output.
        """
        x, x_edge, scatter_target, wigner_inv = inputs
        module = self._make_module()
        m0 = module.m_0_num_coefficients
        # One m=0 coefficient per degree l = 0, 1, 2
        assert m0 == 3

        out = module(x, x_edge, scatter_target, wigner_inv)

        other_columns_changed = wigner_inv.clone()
        other_columns_changed[:, :, m0:] = torch.randn_like(wigner_inv[:, :, m0:])
        torch.testing.assert_close(
            module(x, x_edge, scatter_target, other_columns_changed), out
        )

        m0_columns_changed = wigner_inv.clone()
        m0_columns_changed[:, :, :m0] = torch.randn_like(wigner_inv[:, :, :m0])
        assert not torch.allclose(
            module(x, x_edge, scatter_target, m0_columns_changed), out
        )

    @pytest.mark.parametrize("chunk_size", [1, 7, 64])
    def test_chunked_matches_unchunked(self, inputs, chunk_size):
        """
        Activation-checkpointed chunks give the same outputs and gradients as a
        single pass. The chunk sizes cover one edge per chunk, a shorter final
        chunk, and a single chunk larger than the number of edges.
        """
        x, x_edge, scatter_target, wigner_inv = inputs
        reference = self._make_module()
        chunked = self._make_module(activation_checkpoint_chunk_size=chunk_size)
        chunked.load_state_dict(reference.state_dict())

        def run(module):
            grad_inputs = [t.clone().requires_grad_() for t in (x, x_edge, wigner_inv)]
            out = module(grad_inputs[0], grad_inputs[1], scatter_target, grad_inputs[2])
            grads = torch.autograd.grad(
                out.square().sum(), [*grad_inputs, *module.parameters()]
            )
            return out, grads

        ref_out, ref_grads = run(reference)
        out, grads = run(chunked)

        torch.testing.assert_close(out, ref_out)
        for grad, ref_grad in zip(grads, ref_grads, strict=True):
            torch.testing.assert_close(grad, ref_grad)

    def test_compact_wigner_backend_matches_general(self, inputs):
        """
        The umas_fast_gpu backend packs the Wigner blocks into a compact
        [E, 35] layout and selects m=0 entries by hard-coded index. Its
        edge-degree scatter is plain PyTorch, so it can be checked against the
        general backend on CPU.
        """
        x, x_edge, scatter_target, _ = inputs
        general = self._make_module(execution_mode="general")
        compact = self._make_module(execution_mode="umas_fast_gpu")
        compact.load_state_dict(general.state_dict())

        # Wigner matrices are block diagonal in l, with blocks of size 1, 3 and 5
        wigner = torch.zeros(self.num_edges, 9, 9, dtype=torch.float64)
        for start, size in ((0, 1), (1, 3), (4, 5)):
            wigner[:, start : start + size, start : start + size] = torch.randn(
                self.num_edges, size, size, dtype=torch.float64
            )
        wigner_inv = wigner.transpose(1, 2).contiguous()

        outputs = []
        for module in (general, compact):
            _, prepared_inv = module.backend.prepare_wigner(
                wigner, wigner_inv, module.mappingReduced, None
            )
            outputs.append(module(x, x_edge, scatter_target, prepared_inv))

        # Confirm the compact path was actually exercised
        assert prepared_inv.shape == (self.num_edges, 35)
        torch.testing.assert_close(outputs[1], outputs[0])
