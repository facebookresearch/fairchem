"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Fused feature extraction for the umas_flash execution mode.

``FlashFeatures`` replaces the body of ``eSCNMDBackbone.forward`` between graph
generation and the final norm. It keeps the backbone's own pointwise modules
(``norm_1``, ``norm_2``, ``atom_wise``, ``norm``) and swaps everything else for
Triton kernels that never stage Wigner matrices, radial embeddings or rotated
edge messages to HBM.

Weights are repacked once, in ``prepare_for_inference``, after any MOLE merge:

* the two complex SO(2) convolutions become single real block matrices, so each
  m-order is one GEMM;
* the first radial linear is split into the Gaussian half (kept as a GEMM) and
  the two element-embedding halves, which depend only on atomic number and are
  therefore precomputed into (max_num_elements, C) lookup tables.

Graph parallelism: node tensors are this rank's slice, edge indices are global.
Each layer all-gathers the node features before the edge gather (matching
``Edgewise.forward``) and hands ``data_dict["scatter_target"]`` -- the local row
each edge writes to -- to the scatter kernels. Only the allgather/index_split
GP configuration is supported; see the check in ``FlashFeatures.forward``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from fairchem.core.common import gp_utils
from fairchem.core.models.uma.flash.custom_ops import (
    fused_gating_split,
    gather_wigner_split,
    init_node_scatter,
    layer_norm_silu,
    precompute_geometry,
    radial_stage1,
    rotate_scatter_split,
)

# Radial MLP layout produced by RadialMLP: Linear, LayerNorm, SiLU, Linear,
# LayerNorm, SiLU, Linear.
_RAD_LIN_1, _RAD_NORM_1, _RAD_LIN_2, _RAD_NORM_2, _RAD_LIN_3 = 0, 1, 3, 4, 6


def add_csd_to_l0(x: torch.Tensor, csd_emb: torch.Tensor, batch: torch.Tensor):
    """
    Add the per-system charge/spin/dataset embedding to the l=0 channel.

    Out of place with respect to ``x``, which is the output of a norm module
    that autograd still needs -- but in place on our own clone, which nothing
    else holds a reference to.

    The obvious spelling is ``cat((x[:, :1] + csd, x[:, 1:]), dim=1)``. It
    reads worse than it runs: the concatenation copies eight ninths of the
    tensor for no reason, and its backward is two slice-gradient kernels rather
    than the identity. Measured at N=20k, C=128 the clone is ~5% quicker
    forward and ~45% quicker backward, with no change in peak memory.
    """
    out = x.clone()
    out[:, 0, :] += csd_emb.index_select(0, batch)
    return out


def _pack_complex_block(fc_weight: torch.Tensor, out_half: int, dtype) -> torch.Tensor:
    """
    Fold a complex SO(2) m-order convolution into one real matrix.

    ``fc_weight`` stacks the real and imaginary output rows. A complex product
    (R + iI)(a + ib) is the real matrix [[R, -I], [I, R]] applied to [a; b], so
    the m-block becomes a single GEMM over the concatenated real/imaginary
    inputs.
    """
    W_r = fc_weight[:out_half, :]
    W_i = fc_weight[out_half:, :]
    in_half = W_r.shape[1]
    packed = torch.zeros(
        (2 * out_half, 2 * in_half), device=fc_weight.device, dtype=torch.float32
    )
    packed[:out_half, :in_half] = W_r
    packed[:out_half, in_half:] = -W_i
    packed[out_half:, :in_half] = W_i
    packed[out_half:, in_half:] = W_r
    return packed.to(dtype)


class _FlashRadialMLP(nn.Module):
    """
    Radial MLP with the element-embedding half of its first linear baked into
    per-element tables.

    The input the eager model builds is ``[gaussian | W_src[z_i] | W_tgt[z_j]]``
    and only ever enters through this first linear. ``W_src[z] @ W1_src.T``
    depends on the atomic number alone, so it is precomputed once per element,
    shrinking the per-edge GEMM from (B + 2 * D) x C to B x C. Exact up to
    float re-association.

    The tables are then consumed inside the stage-1 kernel rather than gathered
    into [E, H] tensors first, which is where most of this stage's time was
    going.
    """

    def __init__(self, rad_net, num_gaussians: int, src_emb, tgt_emb, dtype):
        super().__init__()
        lin1 = rad_net[_RAD_LIN_1]
        B, D = num_gaussians, src_emb.shape[1]
        # ValueError rather than assert: this is a model-shape contract, and
        # asserts vanish under python -O.
        if lin1.in_features != B + 2 * D:
            raise ValueError(
                f"radial MLP expects [gaussian | source | target] input of "
                f"width {B + 2 * D}, got {lin1.in_features}"
            )
        W1 = lin1.weight.data.float()
        self.register_buffer("W_gauss", W1[:, :B].contiguous().to(dtype))
        self.register_buffer(
            "table_src", (src_emb.float() @ W1[:, B : B + D].T).contiguous().to(dtype)
        )
        self.register_buffer(
            "table_tgt",
            (tgt_emb.float() @ W1[:, B + D : B + 2 * D].T).contiguous().to(dtype),
        )
        self.register_buffer("bias_1", lin1.bias.data.clone().to(dtype))
        # Take eps from the modules rather than assuming a default: LayerNorm
        # ships 1e-5, and a smaller value visibly shifts the output whenever the
        # pre-norm variance is small.
        self.norm_1_eps = float(rad_net[_RAD_NORM_1].eps)
        self.norm_2_eps = float(rad_net[_RAD_NORM_2].eps)
        self.register_buffer(
            "norm_1_weight", rad_net[_RAD_NORM_1].weight.data.clone().to(dtype)
        )
        self.register_buffer(
            "norm_1_bias", rad_net[_RAD_NORM_1].bias.data.clone().to(dtype)
        )
        self.register_buffer(
            "weight_2", rad_net[_RAD_LIN_2].weight.data.clone().to(dtype)
        )
        self.register_buffer("bias_2", rad_net[_RAD_LIN_2].bias.data.clone().to(dtype))
        self.register_buffer(
            "norm_2_weight", rad_net[_RAD_NORM_2].weight.data.clone().to(dtype)
        )
        self.register_buffer(
            "norm_2_bias", rad_net[_RAD_NORM_2].bias.data.clone().to(dtype)
        )
        self.register_buffer(
            "weight_3", rad_net[_RAD_LIN_3].weight.data.clone().to(dtype)
        )
        self.register_buffer("bias_3", rad_net[_RAD_LIN_3].bias.data.clone().to(dtype))

    def hidden(self, gauss, z_i, z_j):
        """
        Everything up to the last linear, i.e. the [E, H] hidden state.

        The layers stop here and hand ``weight_3`` to the gather kernel, which
        applies it a C-wide slice at a time. Only the edge-degree embedding,
        whose consumer is a different kernel, needs the full output.
        """
        # One kernel for the GEMM, both element lookups, the norm and the
        # activation. Spelled out in torch it is seven [E, H] tensors through
        # HBM for a contraction only 32 deep; see kernels/radial.py.
        h = radial_stage1(
            gauss,
            self.W_gauss,
            self.bias_1,
            self.table_src,
            self.table_tgt,
            z_i,
            z_j,
            self.norm_1_weight,
            self.norm_1_bias,
            self.norm_1_eps,
        )
        return layer_norm_silu(
            F.linear(h, self.weight_2, self.bias_2),
            self.norm_2_weight,
            self.norm_2_bias,
            self.norm_2_eps,
        )

    def forward(self, gauss, z_i, z_j):
        return F.linear(self.hidden(gauss, z_i, z_j), self.weight_3, self.bias_3)


class _FlashLayer(nn.Module):
    """
    Packed SO(2) weights for one eSCN-MD block.

    The block's own ``norm_1`` / ``norm_2`` / ``atom_wise`` are reused in place
    rather than copied, so this holds only what the kernels consume.
    """

    def __init__(
        self, block, sphere_channels: int, num_gaussians: int, src, tgt, dtype
    ):
        super().__init__()
        self.C = sphere_channels
        so2_1 = block.edge_wise.so2_conv_1
        so2_2 = block.edge_wise.so2_conv_2
        C = sphere_channels

        self.radial = _FlashRadialMLP(
            so2_1.rad_func.net, num_gaussians, src, tgt, dtype
        )

        self.register_buffer("W_conv1_m0", so2_1.fc_m0.weight.data.clone().to(dtype))
        self.register_buffer("b_conv1_m0", so2_1.fc_m0.bias.data.clone().to(dtype))
        self.register_buffer(
            "W_conv1_m1",
            _pack_complex_block(so2_1.so2_m_conv[0].fc.weight.data, 2 * C, dtype),
        )
        self.register_buffer(
            "W_conv1_m2",
            _pack_complex_block(so2_1.so2_m_conv[1].fc.weight.data, 1 * C, dtype),
        )

        self.register_buffer("W_conv2_m0", so2_2.fc_m0.weight.data.clone().to(dtype))
        self.register_buffer("b_conv2_m0", so2_2.fc_m0.bias.data.clone().to(dtype))
        self.register_buffer(
            "W_conv2_m1",
            _pack_complex_block(so2_2.so2_m_conv[0].fc.weight.data, 2 * C, dtype),
        )
        self.register_buffer(
            "W_conv2_m2",
            _pack_complex_block(so2_2.so2_m_conv[1].fc.weight.data, 1 * C, dtype),
        )

    def forward(
        self,
        x_in,
        gauss,
        wigner,
        env,
        idx_i,
        idx_j,
        z_i,
        z_j,
        csd_emb,
        batch,
        block,
        scatter_target,
        num_nodes_full,
    ):
        x_res = x_in
        x_norm = add_csd_to_l0(block.norm_1(x_in), csd_emb, batch)

        # Under graph parallelism the edge gather reads nodes owned by other
        # ranks, so features are gathered up front exactly as Edgewise does;
        # the backward of this gather is the matching reduce-scatter.
        if gp_utils.initialized():
            x_full = gp_utils.gather_from_model_parallel_region_sum_grad(
                x_norm, num_nodes_full
            )
        else:
            x_full = x_norm

        # The radial's last linear is applied inside the gather, so what
        # crosses this boundary is [E, H] rather than [E, 12C].
        rad_hidden = self.radial.hidden(gauss, z_i, z_j)

        X_m0, X_m1, X_m2 = gather_wigner_split(
            x_full,
            idx_i,
            idx_j,
            wigner,
            rad_hidden,
            self.radial.weight_3,
            self.radial.bias_3,
        )
        Y_m0 = F.linear(X_m0, self.W_conv1_m0, self.b_conv1_m0)
        Y_m1 = F.linear(X_m1, self.W_conv1_m1)
        Y_m2 = F.linear(X_m2, self.W_conv1_m2)

        Z_m0, Z_m1, Z_m2 = fused_gating_split(Y_m0, Y_m1, Y_m2, self.C)
        Z_out_m0 = F.linear(Z_m0, self.W_conv2_m0, self.b_conv2_m0)
        Z_out_m1 = F.linear(Z_m1, self.W_conv2_m1)
        Z_out_m2 = F.linear(Z_m2, self.W_conv2_m2)

        # Y (11C) and Z_out (9C) both stay alive for the backward. Rebuilding
        # Z_out from Y instead was measured at -26% peak memory for +13% wall
        # time; activation_checkpointing gets -72% for +55%, so it is the
        # better lever for anyone who is memory bound and this is not worth
        # charging every caller for.
        x_mid = rotate_scatter_split(
            Z_out_m0, Z_out_m1, Z_out_m2, wigner, env, x_res, scatter_target
        )
        return block.atom_wise(block.norm_2(x_mid)) + x_mid


class FlashFeatures(nn.Module):
    """
    Packed weights plus the fused forward for the umas_flash backend.

    Built by ``UMASFlashBackend.prepare_model_for_inference`` and attached to
    the backbone, so it is constructed once per process. That matters for
    multi-GPU: every worker of a ``ParallelMLIPPredictUnit`` instantiates its
    own predict unit and therefore its own copy, with no driver-side patching.
    """

    def __init__(
        self,
        model,
        dtype: torch.dtype | None = None,
        checkpointing: bool = False,
    ):
        super().__init__()
        if dtype is None:
            dtype = next(model.parameters()).dtype
        self.checkpointing = checkpointing

        smearing = model.distance_expansion
        num_gaussians = smearing.offset.shape[0]
        src = model.source_embedding.weight.data
        tgt = model.target_embedding.weight.data

        self.sphere_channels = model.sphere_channels
        self.cutoff = float(model.cutoff)
        self.coeff_g = float(smearing.coeff)
        # The edge-degree embedding rescales its sum aggregation; folding the
        # reciprocal into the kernel saves a pass over the node tensor.
        self.inv_rescale = 1.0 / float(model.edge_degree_embedding.rescale_factor)
        self.register_buffer("offset_g", smearing.offset.data.clone().to(dtype))
        self.register_buffer(
            "W_sphere", model.sphere_embedding.weight.data.clone().to(dtype)
        )

        self.edge_degree_radial = _FlashRadialMLP(
            model.edge_degree_embedding.rad_func.net, num_gaussians, src, tgt, dtype
        )
        self.layers = nn.ModuleList(
            [
                _FlashLayer(
                    block, model.sphere_channels, num_gaussians, src, tgt, dtype
                )
                for block in model.blocks
            ]
        )

    def forward(self, model, data_dict, graph_dict, csd_mixed_emb) -> torch.Tensor:
        edge_index = graph_dict["edge_index"]
        idx_i, idx_j = edge_index[0], edge_index[1]

        # Checked here rather than in prepare_model_for_inference, which runs
        # before the predict unit moves the model to its device.
        if not data_dict["pos"].is_cuda:
            raise ValueError(
                "umas_flash launches Triton kernels and needs CUDA tensors, "
                f"got positions on {data_dict['pos'].device}"
            )

        if gp_utils.initialized():
            gp_config = gp_utils.get_gp_config()
            # The gather below is an all-gather whose output is ranks
            # concatenated in order, so it only equals global node order when
            # the partition is a contiguous index split. Spatial partitions and
            # the all-to-all collective would need gp_ctx routing instead.
            if (
                gp_config.mode != gp_utils.GPMode.ALLGATHER
                or gp_config.partition != gp_utils.GPPartition.INDEX_SPLIT
            ):
                raise ValueError(
                    "umas_flash supports graph parallelism only with "
                    f"mode='allgather' and partition='index_split', got "
                    f"mode={gp_config.mode!r}, partition={gp_config.partition!r}."
                )

        # Local row each edge scatters into. Upstream builds it per edge rather
        # than as a scalar offset so non-contiguous partitions work; the
        # kernels take it in place of the global target index.
        scatter_target = data_dict["scatter_target"]

        # Node quantities are this rank's slice; edge endpoints are global ids.
        z_local = data_dict["atomic_numbers"]
        z_full = data_dict["atomic_numbers_full"]
        batch = data_dict["batch"]
        num_nodes_local = z_local.shape[0]
        num_nodes_full = z_full.shape[0]

        # edge_distance_vec carries the position and cell dependency, so forces
        # and stress come out of autograd rather than being reconstructed here.
        edge_vec = graph_dict["edge_distance_vec"].contiguous()
        gauss, wigner, env = precompute_geometry(
            edge_vec, self.offset_g, self.coeff_g, self.cutoff
        )

        z_i, z_j = z_full[idx_i], z_full[idx_j]

        rad_out = self.edge_degree_radial(gauss, z_i, z_j)
        x_node = init_node_scatter(
            rad_out,
            wigner,
            env,
            z_local,
            batch,
            self.W_sphere,
            csd_mixed_emb,
            scatter_target,
            num_nodes_local,
            self.inv_rescale,
        )

        for layer, block in zip(self.layers, model.blocks):
            args = (
                x_node,
                gauss,
                wigner,
                env,
                idx_i,
                idx_j,
                z_i,
                z_j,
                csd_mixed_emb,
                batch,
                block,
                scatter_target,
                num_nodes_full,
            )
            if self.checkpointing:
                # Nothing inside the layer survives the call, so the backward
                # replays the whole layer from x_node. Costs one extra forward
                # and holds one layer's activations instead of every layer's.
                x_node = checkpoint(layer, *args, use_reentrant=False)
            else:
                x_node = layer(*args)
            x_node = model.balance_channels(
                x_node,
                charge=data_dict["charge"],
                spin=data_dict["spin"],
                natoms=data_dict["natoms"],
                batch=batch,
            )

        return model.norm(x_node)
