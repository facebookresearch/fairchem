"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.models.uma.nn.embedding import DatasetEmbedding


class TestDatasetEmbedding:
    """Test the DatasetEmbedding class."""

    def test_embeddings_trainable_when_grad_true(self):
        """Test that embeddings have requires_grad=True when grad=True."""
        dataset_list = ["oc20", "omat", "omol"]
        embedding_size = 64

        layer = DatasetEmbedding(
            embedding_size=embedding_size,
            grad=True,
            dataset_list=dataset_list,
        )

        # Check all embedding parameters have requires_grad=True
        for dataset in dataset_list:
            for param in layer.dataset_emb_dict[dataset].parameters():
                assert (
                    param.requires_grad is True
                ), f"Expected requires_grad=True for dataset '{dataset}'"

    def test_embeddings_not_trainable_when_grad_false(self):
        """Test that embeddings have requires_grad=False when grad=False."""
        dataset_list = ["oc20", "omat", "omol"]
        embedding_size = 64

        layer = DatasetEmbedding(
            embedding_size=embedding_size,
            grad=False,
            dataset_list=dataset_list,
        )

        # Check all embedding parameters have requires_grad=False
        for dataset in dataset_list:
            for param in layer.dataset_emb_dict[dataset].parameters():
                assert (
                    param.requires_grad is False
                ), f"Expected requires_grad=False for dataset '{dataset}'"

    def test_dataset_mapping(self):
        """Test that dataset_mapping correctly maps one dataset to another's embedding."""
        dataset_list = ["omol", "omat", "oc20", "oc20_subset"]
        embedding_size = 64
        dataset_mapping = {"oc20_subset": "oc20"}

        # Instance 1: no mapping
        torch.manual_seed(42)
        layer_no_mapping = DatasetEmbedding(
            embedding_size=embedding_size,
            grad=False,
            dataset_list=dataset_list,
        )
        layer_no_mapping.eval()

        # Instance 2: with mapping
        torch.manual_seed(42)
        layer_with_mapping = DatasetEmbedding(
            embedding_size=embedding_size,
            grad=False,
            dataset_list=dataset_list,
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
