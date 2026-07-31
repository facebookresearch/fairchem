"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.nn.embedding import ChgSpinEmbedding, DatasetEmbedding


@pytest.mark.gpu()
@pytest.mark.compile_gpu()
@pytest.mark.parametrize("embedding_type", ["pos_emb", "lin_emb"])
def test_spin_embedding_fullgraph_preserves_input(embedding_type, compile_reset_state):
    layer = ChgSpinEmbedding(
        embedding_type=embedding_type,
        embedding_target="spin",
        embedding_size=8,
        grad=False,
    ).cuda()
    spin = torch.tensor([0, 1, 2, 0], device="cuda")
    original = spin.clone()

    expected = layer(spin.clone())
    actual = torch.compile(layer, fullgraph=True)(spin)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(spin, original)
    if embedding_type == "pos_emb":
        zero_spin = actual[spin == 0]
        torch.testing.assert_close(zero_spin, torch.zeros_like(zero_spin))


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
