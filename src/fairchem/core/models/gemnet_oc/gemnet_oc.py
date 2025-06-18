"""
Copyright (c) Meta, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from e3nn import o3

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import (
    conditional_grad,
    segment_coo,
)
from fairchem.core.graph.compute import generate_graph
from fairchem.core.graph.radius_graph_pbc import get_max_neighbors_mask
from fairchem.core.models.base import BackboneInterface, HeadInterface
from fairchem.core.modules.scaling.compat import load_scales_compat

from .initializers import get_initializer
from .interaction_indices import get_mixed_triplets, get_quadruplets, get_triplets
from .layers.atom_update_block import OutputBlock
from .layers.base_layers import Dense, ResidualLayer
from .layers.efficient import BasisEmbedding
from .layers.embedding_block import AtomEmbedding, EdgeEmbedding
from .layers.force_scaler import ForceStressScaler
from .layers.interaction_block import InteractionBlock
from .layers.radial_basis import RadialBasis
from .layers.spherical_basis import CircularBasisLayer, SphericalBasisLayer
from .utils import (
    get_angle,
    get_edge_id,
    get_inner_idx,
    inner_product_clamped,
    mask_neighbors,
    repeat_blocks,
)

DTYPE_DICT = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


@registry.register_model("gemnet_oc_backbone")
class GemNetOCBackbone(nn.Module):
    """
    Arguments
    ---------
    num_spherical: int
        Controls maximum frequency.
    num_radial: int
        Controls maximum frequency.
    num_blocks: int
        Number of building blocks to be stacked.

    emb_size_atom: int
        Embedding size of the atoms.
    emb_size_edge: int
        Embedding size of the edges.
    emb_size_trip_in: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        before the bilinear layer.
    emb_size_trip_out: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        after the bilinear layer.
    emb_size_quad_in: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        before the bilinear layer.
    emb_size_quad_out: int
        (Down-projected) embedding size of the quadruplet edge embeddings
        after the bilinear layer.
    emb_size_aint_in: int
        Embedding size in the atom interaction before the bilinear layer.
    emb_size_aint_out: int
        Embedding size in the atom interaction after the bilinear layer.
    emb_size_rbf: int
        Embedding size of the radial basis transformation.
    emb_size_cbf: int
        Embedding size of the circular basis transformation (one angle).
    emb_size_sbf: int
        Embedding size of the spherical basis transformation (two angles).

    num_before_skip: int
        Number of residual blocks before the first skip connection.
    num_after_skip: int
        Number of residual blocks after the first skip connection.
    num_concat: int
        Number of residual blocks after the concatenation.
    num_atom: int
        Number of residual blocks in the atom embedding blocks.
    num_output_afteratom: int
        Number of residual blocks in the output blocks
        after adding the atom embedding.
    num_atom_emb_layers: int
        Number of residual blocks for transforming atom embeddings.
    num_global_out_layers: int
        Number of final residual blocks before the output.

    regress_forces: bool
        Whether to predict forces. Default: True
    direct_forces: bool
        If True predict forces based on aggregation of interatomic directions.
        If False predict forces based on negative gradient of energy potential.
    use_pbc: bool
        Whether to use periodic boundary conditions.
    use_pbc_single:
        Process batch PBC graphs one at a time
    scale_backprop_forces: bool
        Whether to scale up the energy and then scales down the forces
        to prevent NaNs and infs in backpropagated forces.

    cutoff: float
        Embedding cutoff for interatomic connections and embeddings in Angstrom.
    cutoff_qint: float
        Quadruplet interaction cutoff in Angstrom.
        Optional. Uses cutoff per default.
    cutoff_aeaint: float
        Edge-to-atom and atom-to-edge interaction cutoff in Angstrom.
        Optional. Uses cutoff per default.
    cutoff_aint: float
        Atom-to-atom interaction cutoff in Angstrom.
        Optional. Uses maximum of all other cutoffs per default.
    max_neighbors: int
        Maximum number of neighbors for interatomic connections and embeddings.
    max_neighbors_qint: int
        Maximum number of quadruplet interactions per embedding.
        Optional. Uses max_neighbors per default.
    max_neighbors_aeaint: int
        Maximum number of edge-to-atom and atom-to-edge interactions per embedding.
        Optional. Uses max_neighbors per default.
    max_neighbors_aint: int
        Maximum number of atom-to-atom interactions per atom.
        Optional. Uses maximum of all other neighbors per default.
    enforce_max_neighbors_strictly: bool
        When subselected edges based on max_neighbors args, arbitrarily
        select amongst degenerate edges to have exactly the correct number.
    rbf: dict
        Name and hyperparameters of the radial basis function.
    rbf_spherical: dict
        Name and hyperparameters of the radial basis function used as part of the
        circular and spherical bases.
        Optional. Uses rbf per default.
    envelope: dict
        Name and hyperparameters of the envelope function.
    cbf: dict
        Name and hyperparameters of the circular basis function.
    sbf: dict
        Name and hyperparameters of the spherical basis function.
    extensive: bool
        Whether the output should be extensive (proportional to the number of atoms)
    forces_coupled: bool
        If True, enforce that |F_st| = |F_ts|. No effect if direct_forces is False.
    output_init: str
        Initialization method for the final dense layer.
    activation: str
        Name of the activation function.
    scale_file: str
        Path to the pytorch file containing the scaling factors.

    quad_interaction: bool
        Whether to use quadruplet interactions (with dihedral angles)
    atom_edge_interaction: bool
        Whether to use atom-to-edge interactions
    edge_atom_interaction: bool
        Whether to use edge-to-atom interactions
    atom_interaction: bool
        Whether to use atom-to-atom interactions

    scale_basis: bool
        Whether to use a scaling layer in the raw basis function for better
        numerical stability.
    qint_tags: list
        Which atom tags to use quadruplet interactions for.
        0=sub-surface bulk, 1=surface, 2=adsorbate atoms.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        num_blocks: int,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip_in: int,
        emb_size_trip_out: int,
        emb_size_quad_in: int,
        emb_size_quad_out: int,
        emb_size_aint_in: int,
        emb_size_aint_out: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_sbf: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        num_output_afteratom: int,
        num_atom_emb_layers: int = 0,
        num_global_out_layers: int = 2,
        regress_forces: bool = True,
        direct_forces: bool = False,
        regress_stress: bool = False,
        direct_stress: bool = False,
        use_pbc: bool = True,
        use_pbc_single: bool = False,
        scale_backprop_forces: bool = False,
        cutoff: float = 6.0,
        cutoff_qint: float | None = None,
        cutoff_aeaint: float | None = None,
        cutoff_aint: float | None = None,
        max_neighbors: int = 50,
        max_neighbors_qint: int | None = None,
        max_neighbors_aeaint: int | None = None,
        max_neighbors_aint: int | None = None,
        enforce_max_neighbors_strictly: bool = True,
        rbf: dict[str, str] | None = None,
        rbf_spherical: dict | None = None,
        envelope: dict[str, str | int] | None = None,
        cbf: dict[str, str] | None = None,
        sbf: dict[str, str] | None = None,
        extensive: bool = True,
        forces_coupled: bool = False,
        output_init: str = "HeOrthogonal",
        activation: str = "silu",
        quad_interaction: bool = False,
        atom_edge_interaction: bool = False,
        edge_atom_interaction: bool = False,
        atom_interaction: bool = False,
        scale_basis: bool = False,
        qint_tags: list | None = None,
        num_elements: int = 119,
        otf_graph: bool = False,
        scale_file: str | None = None,
        use_torch_sparse: bool = False,
        **kwargs,  # backwards compatibility with deprecated arguments
    ) -> None:
        if qint_tags is None:
            qint_tags = [0, 1, 2]
        if sbf is None:
            sbf = {"name": "spherical_harmonics"}
        if cbf is None:
            cbf = {"name": "spherical_harmonics"}
        if envelope is None:
            envelope = {"name": "polynomial", "exponent": 5}
        if rbf is None:
            rbf = {"name": "gaussian"}
        super().__init__()
        if len(kwargs) > 0:
            logging.warning(f"Unrecognized arguments: {list(kwargs.keys())}")
        assert num_blocks > 0
        self.num_blocks = num_blocks
        self.extensive = extensive

        self.activation = activation
        self.atom_edge_interaction = atom_edge_interaction
        self.edge_atom_interaction = edge_atom_interaction
        self.atom_interaction = atom_interaction
        self.quad_interaction = quad_interaction
        self.qint_tags = torch.tensor(qint_tags)
        self.otf_graph = otf_graph
        self.use_torch_sparse = use_torch_sparse
        if not rbf_spherical:
            rbf_spherical = rbf

        self.set_cutoffs(cutoff, cutoff_qint, cutoff_aeaint, cutoff_aint)
        self.set_max_neighbors(
            max_neighbors,
            max_neighbors_qint,
            max_neighbors_aeaint,
            max_neighbors_aint,
        )
        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly
        self.use_pbc = use_pbc
        self.use_pbc_single = use_pbc_single

        self.direct_forces = direct_forces
        self.direct_stress = direct_stress
        self.regress_stress = regress_stress
        self.forces_coupled = forces_coupled
        self.regress_forces = regress_forces
        self.scaler = ForceStressScaler(enabled=scale_backprop_forces)

        self.init_basis_functions(
            num_radial,
            num_spherical,
            rbf,
            rbf_spherical,
            envelope,
            cbf,
            sbf,
            scale_basis,
        )
        self.init_shared_basis_layers(
            num_radial, num_spherical, emb_size_rbf, emb_size_cbf, emb_size_sbf
        )

        # Embedding blocks
        self.atom_emb = AtomEmbedding(emb_size_atom, num_elements)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        # Interaction Blocks
        int_blocks = []
        for _ in range(num_blocks):
            int_blocks.append(
                InteractionBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip_in=emb_size_trip_in,
                    emb_size_trip_out=emb_size_trip_out,
                    emb_size_quad_in=emb_size_quad_in,
                    emb_size_quad_out=emb_size_quad_out,
                    emb_size_a2a_in=emb_size_aint_in,
                    emb_size_a2a_out=emb_size_aint_out,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_sbf=emb_size_sbf,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    num_atom_emb_layers=num_atom_emb_layers,
                    quad_interaction=quad_interaction,
                    atom_edge_interaction=atom_edge_interaction,
                    edge_atom_interaction=edge_atom_interaction,
                    atom_interaction=atom_interaction,
                    activation=activation,
                )
            )
        self.int_blocks = torch.nn.ModuleList(int_blocks)

        out_blocks = []
        for _ in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    nHidden_afteratom=num_output_afteratom,
                    activation=activation,
                    direct_forces=direct_forces,
                )
            )
        self.out_blocks = torch.nn.ModuleList(out_blocks)

        out_mlp_E = [
            Dense(
                emb_size_atom * (num_blocks + 1),
                emb_size_atom,
                activation=activation,
            )
        ] + [
            ResidualLayer(
                emb_size_atom,
                activation=activation,
            )
            for _ in range(num_global_out_layers)
        ]
        self.out_mlp_E = torch.nn.Sequential(*out_mlp_E)
        self.out_energy = Dense(emb_size_atom, 1, bias=False, activation=None)
        if direct_forces:
            out_mlp_F = [
                Dense(
                    emb_size_edge * (num_blocks + 1),
                    emb_size_edge,
                    activation=activation,
                )
            ] + [
                ResidualLayer(
                    emb_size_edge,
                    activation=activation,
                )
                for _ in range(num_global_out_layers)
            ]
            self.out_mlp_F = torch.nn.Sequential(*out_mlp_F)
            self.out_forces = Dense(emb_size_edge, 1, bias=False, activation=None)

        out_initializer = get_initializer(output_init)
        self.out_energy.reset_parameters(out_initializer)
        if direct_forces:
            self.out_forces.reset_parameters(out_initializer)

        load_scales_compat(self, scale_file)

    def set_cutoffs(self, cutoff, cutoff_qint, cutoff_aeaint, cutoff_aint):
        self.cutoff = cutoff

        if (
            not (self.atom_edge_interaction or self.edge_atom_interaction)
            or cutoff_aeaint is None
        ):
            self.cutoff_aeaint = self.cutoff
        else:
            self.cutoff_aeaint = cutoff_aeaint
        if not self.quad_interaction or cutoff_qint is None:
            self.cutoff_qint = self.cutoff
        else:
            self.cutoff_qint = cutoff_qint
        if not self.atom_interaction or cutoff_aint is None:
            self.cutoff_aint = max(
                self.cutoff,
                self.cutoff_aeaint,
                self.cutoff_qint,
            )
        else:
            self.cutoff_aint = cutoff_aint

        assert self.cutoff <= self.cutoff_aint
        assert self.cutoff_aeaint <= self.cutoff_aint
        assert self.cutoff_qint <= self.cutoff_aint

    def set_max_neighbors(
        self,
        max_neighbors,
        max_neighbors_qint,
        max_neighbors_aeaint,
        max_neighbors_aint,
    ):
        self.max_neighbors = max_neighbors

        if (
            not (self.atom_edge_interaction or self.edge_atom_interaction)
            or max_neighbors_aeaint is None
        ):
            self.max_neighbors_aeaint = self.max_neighbors
        else:
            self.max_neighbors_aeaint = max_neighbors_aeaint
        if not self.quad_interaction or max_neighbors_qint is None:
            self.max_neighbors_qint = self.max_neighbors
        else:
            self.max_neighbors_qint = max_neighbors_qint
        if not self.atom_interaction or max_neighbors_aint is None:
            self.max_neighbors_aint = max(
                self.max_neighbors,
                self.max_neighbors_aeaint,
                self.max_neighbors_qint,
            )
        else:
            self.max_neighbors_aint = max_neighbors_aint

        assert self.max_neighbors <= self.max_neighbors_aint
        assert self.max_neighbors_aeaint <= self.max_neighbors_aint
        assert self.max_neighbors_qint <= self.max_neighbors_aint

    def init_basis_functions(
        self,
        num_radial,
        num_spherical,
        rbf,
        rbf_spherical,
        envelope,
        cbf,
        sbf,
        scale_basis,
    ):
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=self.cutoff,
            rbf=rbf,
            envelope=envelope,
            scale_basis=scale_basis,
        )
        radial_basis_spherical = RadialBasis(
            num_radial=num_radial,
            cutoff=self.cutoff,
            rbf=rbf_spherical,
            envelope=envelope,
            scale_basis=scale_basis,
        )
        if self.quad_interaction:
            radial_basis_spherical_qint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_qint,
                rbf=rbf_spherical,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_qint = CircularBasisLayer(
                num_spherical,
                radial_basis=radial_basis_spherical_qint,
                cbf=cbf,
                scale_basis=scale_basis,
            )

            self.sbf_basis_qint = SphericalBasisLayer(
                num_spherical,
                radial_basis=radial_basis_spherical,
                sbf=sbf,
                scale_basis=scale_basis,
            )
        if self.atom_edge_interaction:
            self.radial_basis_aeaint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_aeaint,
                rbf=rbf,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_aeint = CircularBasisLayer(
                num_spherical,
                radial_basis=radial_basis_spherical,
                cbf=cbf,
                scale_basis=scale_basis,
            )
        if self.edge_atom_interaction:
            self.radial_basis_aeaint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_aeaint,
                rbf=rbf,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            radial_basis_spherical_aeaint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_aeaint,
                rbf=rbf_spherical,
                envelope=envelope,
                scale_basis=scale_basis,
            )
            self.cbf_basis_eaint = CircularBasisLayer(
                num_spherical,
                radial_basis=radial_basis_spherical_aeaint,
                cbf=cbf,
                scale_basis=scale_basis,
            )
        if self.atom_interaction:
            self.radial_basis_aint = RadialBasis(
                num_radial=num_radial,
                cutoff=self.cutoff_aint,
                rbf=rbf,
                envelope=envelope,
                scale_basis=scale_basis,
            )

        self.cbf_basis_tint = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_spherical,
            cbf=cbf,
            scale_basis=scale_basis,
        )

    def init_shared_basis_layers(
        self,
        num_radial,
        num_spherical,
        emb_size_rbf,
        emb_size_cbf,
        emb_size_sbf,
    ):
        # Share basis down projections across all interaction blocks
        if self.quad_interaction:
            self.mlp_rbf_qint = Dense(
                num_radial,
                emb_size_rbf,
                activation=None,
                bias=False,
            )
            self.mlp_cbf_qint = BasisEmbedding(num_radial, emb_size_cbf, num_spherical)
            self.mlp_sbf_qint = BasisEmbedding(
                num_radial, emb_size_sbf, num_spherical**2
            )

        if self.atom_edge_interaction:
            self.mlp_rbf_aeint = Dense(
                num_radial,
                emb_size_rbf,
                activation=None,
                bias=False,
            )
            self.mlp_cbf_aeint = BasisEmbedding(num_radial, emb_size_cbf, num_spherical)
        if self.edge_atom_interaction:
            self.mlp_rbf_eaint = Dense(
                num_radial,
                emb_size_rbf,
                activation=None,
                bias=False,
            )
            self.mlp_cbf_eaint = BasisEmbedding(num_radial, emb_size_cbf, num_spherical)
        if self.atom_interaction:
            self.mlp_rbf_aint = BasisEmbedding(num_radial, emb_size_rbf)

        self.mlp_rbf_tint = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf_tint = BasisEmbedding(num_radial, emb_size_cbf, num_spherical)

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )

        # Set shared parameters for better gradients
        self.shared_parameters = [
            (self.mlp_rbf_tint.linear.weight, self.num_blocks),
            (self.mlp_cbf_tint.weight, self.num_blocks),
            (self.mlp_rbf_h.linear.weight, self.num_blocks),
            (self.mlp_rbf_out.linear.weight, self.num_blocks + 1),
        ]
        if self.quad_interaction:
            self.shared_parameters += [
                (self.mlp_rbf_qint.linear.weight, self.num_blocks),
                (self.mlp_cbf_qint.weight, self.num_blocks),
                (self.mlp_sbf_qint.weight, self.num_blocks),
            ]
        if self.atom_edge_interaction:
            self.shared_parameters += [
                (self.mlp_rbf_aeint.linear.weight, self.num_blocks),
                (self.mlp_cbf_aeint.weight, self.num_blocks),
            ]
        if self.edge_atom_interaction:
            self.shared_parameters += [
                (self.mlp_rbf_eaint.linear.weight, self.num_blocks),
                (self.mlp_cbf_eaint.weight, self.num_blocks),
            ]
        if self.atom_interaction:
            self.shared_parameters += [
                (self.mlp_rbf_aint.weight, self.num_blocks),
            ]

    def calculate_quad_angles(
        self,
        V_st,
        V_qint_st,
        quad_idx,
    ):
        """Calculate angles for quadruplet-based message passing.

        Arguments
        ---------
        V_st: Tensor, shape = (nAtoms, 3)
            Normalized directions from s to t
        V_qint_st: Tensor, shape = (nAtoms, 3)
            Normalized directions from s to t for the quadruplet
            interaction graph
        quad_idx: dict of torch.Tensor
            Indices relevant for quadruplet interactions.

        Returns
        -------
        cosφ_cab: Tensor, shape = (num_triplets_inint,)
            Cosine of angle between atoms c -> a <- b.
        cosφ_abd: Tensor, shape = (num_triplets_qint,)
            Cosine of angle between atoms a -> b -> d.
        angle_cabd: Tensor, shape = (num_quadruplets,)
            Dihedral angle between atoms c <- a-b -> d.
        """
        # ---------------------------------- d -> b -> a ---------------------------------- #
        V_ba = V_qint_st[quad_idx["triplet_in"]["out"]]
        # (num_triplets_qint, 3)
        V_db = V_st[quad_idx["triplet_in"]["in"]]
        # (num_triplets_qint, 3)
        cosφ_abd = inner_product_clamped(V_ba, V_db)
        # (num_triplets_qint,)

        # Project for calculating dihedral angle
        # Cross product is the same as projection, just 90° rotated
        V_db_cross = torch.cross(V_db, V_ba, dim=-1)  # a - b -| d
        V_db_cross = V_db_cross[quad_idx["trip_in_to_quad"]]
        # (num_quadruplets,)

        # --------------------------------- c -> a <- b ---------------------------------- #
        V_ca = V_st[quad_idx["triplet_out"]["out"]]  # (num_triplets_in, 3)
        V_ba = V_qint_st[quad_idx["triplet_out"]["in"]]  # (num_triplets_in, 3)
        cosφ_cab = inner_product_clamped(V_ca, V_ba)  # (n4Triplets,)

        # Project for calculating dihedral angle
        # Cross product is the same as projection, just 90° rotated
        V_ca_cross = torch.cross(V_ca, V_ba, dim=-1)  # c |- a - b
        V_ca_cross = V_ca_cross[quad_idx["trip_out_to_quad"]]
        # (num_quadruplets,)

        # -------------------------------- c -> a - b <- d -------------------------------- #
        half_angle_cabd = get_angle(V_ca_cross, V_db_cross)
        # (num_quadruplets,)
        angle_cabd = half_angle_cabd
        # Ignore parity and just use the half angle.

        return cosφ_cab, cosφ_abd, angle_cabd

    def select_symmetric_edges(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
        reorder_idx: torch.Tensor,
        opposite_neg,
    ) -> torch.Tensor:
        """Use a mask to remove values of removed edges and then
        duplicate the values for the correct edge direction.

        Arguments
        ---------
        tensor: torch.Tensor
            Values to symmetrize for the new tensor.
        mask: torch.Tensor
            Mask defining which edges go in the correct direction.
        reorder_idx: torch.Tensor
            Indices defining how to reorder the tensor values after
            concatenating the edge values of both directions.
        opposite_neg: bool
            Whether the edge in the opposite direction should use the
            negative tensor value.

        Returns
        -------
        tensor_ordered: torch.Tensor
            A tensor with symmetrized values.
        """
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * opposite_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        return tensor_cat[reorder_idx]

    def symmetrize_edges(
        self,
        graph,
        batch_idx,
    ):
        """
        Symmetrize edges to ensure existence of counter-directional edges.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors.
        We only use i->j edges here. So we lose some j->i edges
        and add others by making it symmetric.
        """
        num_atoms = batch_idx.shape[0]
        new_graph = {}

        # Generate mask
        mask_sep_atoms = graph["edge_index"][0] < graph["edge_index"][1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (graph["cell_offset"][:, 0] < 0)
            | ((graph["cell_offset"][:, 0] == 0) & (graph["cell_offset"][:, 1] < 0))
            | (
                (graph["cell_offset"][:, 0] == 0)
                & (graph["cell_offset"][:, 1] == 0)
                & (graph["cell_offset"][:, 2] < 0)
            )
        )
        mask_same_atoms = graph["edge_index"][0] == graph["edge_index"][1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_directed = graph["edge_index"][mask[None, :].expand(2, -1)].view(
            2, -1
        )

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [edge_index_directed, edge_index_directed.flip(0)],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(
                graph["num_neighbors"].size(0),
                device=graph["edge_index"].device,
            ),
            graph["num_neighbors"],
        )
        batch_edge = batch_edge[mask]
        # segment_coo assumes sorted batch_edge
        # Factor 2 since this is only one half of the edges
        ones = batch_edge.new_ones(1).expand_as(batch_edge)
        new_graph["num_neighbors"] = 2 * segment_coo(
            ones, batch_edge, dim_size=graph["num_neighbors"].size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            torch.div(new_graph["num_neighbors"], 2, rounding_mode="floor"),
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_directed.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        new_graph["edge_index"] = edge_index_cat[:, edge_reorder_idx]
        new_graph["cell_offset"] = self.select_symmetric_edges(
            graph["cell_offset"], mask, edge_reorder_idx, True
        )
        new_graph["distance"] = self.select_symmetric_edges(
            graph["distance"], mask, edge_reorder_idx, False
        )
        new_graph["vector"] = self.select_symmetric_edges(
            graph["vector"], mask, edge_reorder_idx, True
        )
        # Indices for swapping c->a and a->c (for symmetric MP)
        # To obtain these efficiently and without any index assumptions,
        # we get order the counter-edge IDs and then
        # map this order back to the edge IDs.
        # Double argsort gives the desired mapping
        # from the ordered tensor to the original tensor.
        edge_ids = get_edge_id(
            new_graph["edge_index"], new_graph["cell_offset"], num_atoms
        )
        order_edge_ids = torch.argsort(edge_ids)
        inv_order_edge_ids = torch.argsort(order_edge_ids)
        edge_ids_counter = get_edge_id(
            new_graph["edge_index"].flip(0),
            -new_graph["cell_offset"],
            num_atoms,
        )
        order_edge_ids_counter = torch.argsort(edge_ids_counter)
        id_swap = order_edge_ids_counter[inv_order_edge_ids]

        return new_graph, id_swap

    def subselect_edges(
        self,
        data_dict,
        graph,
        cutoff=None,
        max_neighbors=None,
    ):
        """Subselect edges using a stricter cutoff and max_neighbors."""
        subgraph = graph.copy()

        if cutoff is not None:
            edge_mask = subgraph["distance"] <= cutoff

            subgraph["edge_index"] = subgraph["edge_index"][:, edge_mask]
            subgraph["cell_offset"] = subgraph["cell_offset"][edge_mask]
            subgraph["num_neighbors"] = mask_neighbors(
                subgraph["num_neighbors"], edge_mask
            )
            subgraph["distance"] = subgraph["distance"][edge_mask]
            subgraph["vector"] = subgraph["vector"][edge_mask]

        if max_neighbors is not None:
            edge_mask, subgraph["num_neighbors"] = get_max_neighbors_mask(
                natoms=data_dict["atomic_numbers"].size(0),
                index=subgraph["edge_index"][1],
                atom_distance=subgraph["distance"],
                max_num_neighbors_threshold=max_neighbors,
                enforce_max_strictly=self.enforce_max_neighbors_strictly,
            )
            if not torch.all(edge_mask):
                subgraph["edge_index"] = subgraph["edge_index"][:, edge_mask]
                subgraph["cell_offset"] = subgraph["cell_offset"][edge_mask]
                subgraph["distance"] = subgraph["distance"][edge_mask]
                subgraph["vector"] = subgraph["vector"][edge_mask]

        empty_image = subgraph["num_neighbors"] == 0
        if torch.any(empty_image):
            logging.warning(f"An image has no neighbors: {data_dict}")
        return subgraph

    def generate_graph_dict(self, data_dict, cutoff, max_neighbors):
        """Generate a radius/nearest neighbor graph."""
        otf_graph = cutoff > 6 or max_neighbors > 50 or self.otf_graph
        graph = self._generate_graph(
            data_dict,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            otf_graph=otf_graph,
        )
        # These vectors actually point in the opposite direction.
        # But we want to use col as idx_t for efficient aggregation.
        edge_vector = -graph["edge_distance_vec"] / graph["edge_distance"][:, None]
        cell_offsets = -graph["cell_offsets"]  # a - c + offset

        graph = {
            "edge_index": graph["edge_index"],
            "distance": graph["edge_distance"],
            "vector": edge_vector,
            "cell_offset": cell_offsets,
            "num_neighbors": graph["neighbors"],
        }

        # Mask interaction edges if required
        select_cutoff = None if otf_graph or np.isclose(cutoff, 6) else cutoff
        select_neighbors = None if otf_graph or max_neighbors == 50 else max_neighbors
        return self.subselect_edges(
            data_dict=data_dict,
            graph=graph,
            cutoff=select_cutoff,
            max_neighbors=select_neighbors,
        )

    def subselect_graph(
        self,
        data_dict,
        graph,
        cutoff,
        max_neighbors,
        cutoff_orig,
        max_neighbors_orig,
    ):
        """If the new cutoff and max_neighbors is different from the original,
        subselect the edges of a given graph.
        """
        # Check if embedding edges are different from interaction edges
        select_cutoff = None if np.isclose(cutoff, cutoff_orig) else cutoff
        if max_neighbors == max_neighbors_orig:
            select_neighbors = None
        else:
            select_neighbors = max_neighbors

        return self.subselect_edges(
            data_dict=data_dict,
            graph=graph,
            cutoff=select_cutoff,
            max_neighbors=select_neighbors,
        )

    def get_graphs_and_indices(self, data_dict):
        """Generate embedding and interaction graphs and indices."""
        num_atoms = data_dict["atomic_numbers"].size(0)

        # Atom interaction graph is always the largest
        if (
            self.atom_edge_interaction
            or self.edge_atom_interaction
            or self.atom_interaction
        ):
            a2a_graph = self.generate_graph_dict(
                data_dict, self.cutoff_aint, self.max_neighbors_aint
            )
            main_graph = self.subselect_graph(
                data_dict,
                a2a_graph,
                self.cutoff,
                self.max_neighbors,
                self.cutoff_aint,
                self.max_neighbors_aint,
            )
            a2ee2a_graph = self.subselect_graph(
                data_dict,
                a2a_graph,
                self.cutoff_aeaint,
                self.max_neighbors_aeaint,
                self.cutoff_aint,
                self.max_neighbors_aint,
            )
        else:
            main_graph = self.generate_graph_dict(
                data_dict, self.cutoff, self.max_neighbors
            )
            a2a_graph = {}
            a2ee2a_graph = {}
        if self.quad_interaction:
            if (
                self.atom_edge_interaction
                or self.edge_atom_interaction
                or self.atom_interaction
            ):
                qint_graph = self.subselect_graph(
                    data_dict,
                    a2a_graph,
                    self.cutoff_qint,
                    self.max_neighbors_qint,
                    self.cutoff_aint,
                    self.max_neighbors_aint,
                )
            else:
                assert self.cutoff_qint <= self.cutoff
                assert self.max_neighbors_qint <= self.max_neighbors
                qint_graph = self.subselect_graph(
                    data_dict,
                    main_graph,
                    self.cutoff_qint,
                    self.max_neighbors_qint,
                    self.cutoff,
                    self.max_neighbors,
                )

            # Only use quadruplets for certain tags
            self.qint_tags = self.qint_tags.to(qint_graph["edge_index"].device)
            tags_s = data_dict["tags"][qint_graph["edge_index"][0]]
            tags_t = data_dict["tags"][qint_graph["edge_index"][1]]
            qint_tag_mask_s = (tags_s[..., None] == self.qint_tags).any(dim=-1)
            qint_tag_mask_t = (tags_t[..., None] == self.qint_tags).any(dim=-1)
            qint_tag_mask = qint_tag_mask_s | qint_tag_mask_t
            qint_graph["edge_index"] = qint_graph["edge_index"][:, qint_tag_mask]
            qint_graph["cell_offset"] = qint_graph["cell_offset"][qint_tag_mask, :]
            qint_graph["distance"] = qint_graph["distance"][qint_tag_mask]
            qint_graph["vector"] = qint_graph["vector"][qint_tag_mask, :]
            del qint_graph["num_neighbors"]
        else:
            qint_graph = {}

        # Symmetrize edges for swapping in symmetric message passing
        main_graph, id_swap = self.symmetrize_edges(main_graph, data_dict["batch"])

        trip_idx_e2e = get_triplets(
            main_graph, num_atoms=num_atoms, use_torch_sparse=self.use_torch_sparse
        )

        # Additional indices for quadruplets
        if self.quad_interaction:
            quad_idx = get_quadruplets(
                main_graph,
                qint_graph,
                num_atoms,
                use_torch_sparse=self.use_torch_sparse,
            )
        else:
            quad_idx = {}

        if self.atom_edge_interaction:
            trip_idx_a2e = get_mixed_triplets(
                a2ee2a_graph,
                main_graph,
                num_atoms=num_atoms,
                return_agg_idx=True,
                use_torch_sparse=self.use_torch_sparse,
            )
        else:
            trip_idx_a2e = {}
        if self.edge_atom_interaction:
            trip_idx_e2a = get_mixed_triplets(
                main_graph,
                a2ee2a_graph,
                num_atoms=num_atoms,
                return_agg_idx=True,
                use_torch_sparse=self.use_torch_sparse,
            )
            # a2ee2a_graph['edge_index'][1] has to be sorted for this
            a2ee2a_graph["target_neighbor_idx"] = get_inner_idx(
                a2ee2a_graph["edge_index"][1], dim_size=num_atoms
            )
        else:
            trip_idx_e2a = {}
        if self.atom_interaction:
            # a2a_graph['edge_index'][1] has to be sorted for this
            a2a_graph["target_neighbor_idx"] = get_inner_idx(
                a2a_graph["edge_index"][1], dim_size=num_atoms
            )

        return (
            main_graph,
            a2a_graph,
            a2ee2a_graph,
            qint_graph,
            id_swap,
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
            quad_idx,
        )

    def get_bases(
        self,
        main_graph,
        a2a_graph,
        a2ee2a_graph,
        qint_graph,
        trip_idx_e2e,
        trip_idx_a2e,
        trip_idx_e2a,
        quad_idx,
        num_atoms,
    ):
        """Calculate and transform basis functions."""
        basis_rad_main_raw = self.radial_basis(main_graph["distance"])

        # Calculate triplet angles
        cosφ_cab = inner_product_clamped(
            main_graph["vector"][trip_idx_e2e["out"]],
            main_graph["vector"][trip_idx_e2e["in"]],
        )
        basis_rad_cir_e2e_raw, basis_cir_e2e_raw = self.cbf_basis_tint(
            main_graph["distance"], cosφ_cab
        )

        if self.quad_interaction:
            # Calculate quadruplet angles
            cosφ_cab_q, cosφ_abd, angle_cabd = self.calculate_quad_angles(
                main_graph["vector"],
                qint_graph["vector"],
                quad_idx,
            )

            basis_rad_cir_qint_raw, basis_cir_qint_raw = self.cbf_basis_qint(
                qint_graph["distance"], cosφ_abd
            )
            basis_rad_sph_qint_raw, basis_sph_qint_raw = self.sbf_basis_qint(
                main_graph["distance"],
                cosφ_cab_q[quad_idx["trip_out_to_quad"]],
                angle_cabd,
            )
        if self.atom_edge_interaction:
            basis_rad_a2ee2a_raw = self.radial_basis_aeaint(a2ee2a_graph["distance"])
            cosφ_cab_a2e = inner_product_clamped(
                main_graph["vector"][trip_idx_a2e["out"]],
                a2ee2a_graph["vector"][trip_idx_a2e["in"]],
            )
            basis_rad_cir_a2e_raw, basis_cir_a2e_raw = self.cbf_basis_aeint(
                main_graph["distance"], cosφ_cab_a2e
            )
        if self.edge_atom_interaction:
            cosφ_cab_e2a = inner_product_clamped(
                a2ee2a_graph["vector"][trip_idx_e2a["out"]],
                main_graph["vector"][trip_idx_e2a["in"]],
            )
            basis_rad_cir_e2a_raw, basis_cir_e2a_raw = self.cbf_basis_eaint(
                a2ee2a_graph["distance"], cosφ_cab_e2a
            )
        if self.atom_interaction:
            basis_rad_a2a_raw = self.radial_basis_aint(a2a_graph["distance"])

        # Shared Down Projections
        bases_qint = {}
        if self.quad_interaction:
            bases_qint["rad"] = self.mlp_rbf_qint(basis_rad_main_raw)
            bases_qint["cir"] = self.mlp_cbf_qint(
                rad_basis=basis_rad_cir_qint_raw,
                sph_basis=basis_cir_qint_raw,
                idx_sph_outer=quad_idx["triplet_in"]["out"],
            )
            bases_qint["sph"] = self.mlp_sbf_qint(
                rad_basis=basis_rad_sph_qint_raw,
                sph_basis=basis_sph_qint_raw,
                idx_sph_outer=quad_idx["out"],
                idx_sph_inner=quad_idx["out_agg"],
            )

        bases_a2e = {}
        if self.atom_edge_interaction:
            bases_a2e["rad"] = self.mlp_rbf_aeint(basis_rad_a2ee2a_raw)
            bases_a2e["cir"] = self.mlp_cbf_aeint(
                rad_basis=basis_rad_cir_a2e_raw,
                sph_basis=basis_cir_a2e_raw,
                idx_sph_outer=trip_idx_a2e["out"],
                idx_sph_inner=trip_idx_a2e["out_agg"],
            )
        bases_e2a = {}
        if self.edge_atom_interaction:
            bases_e2a["rad"] = self.mlp_rbf_eaint(basis_rad_main_raw)
            bases_e2a["cir"] = self.mlp_cbf_eaint(
                rad_basis=basis_rad_cir_e2a_raw,
                sph_basis=basis_cir_e2a_raw,
                idx_rad_outer=a2ee2a_graph["edge_index"][1],
                idx_rad_inner=a2ee2a_graph["target_neighbor_idx"],
                idx_sph_outer=trip_idx_e2a["out"],
                idx_sph_inner=trip_idx_e2a["out_agg"],
                num_atoms=num_atoms,
            )
        if self.atom_interaction:
            basis_a2a_rad = self.mlp_rbf_aint(
                rad_basis=basis_rad_a2a_raw,
                idx_rad_outer=a2a_graph["edge_index"][1],
                idx_rad_inner=a2a_graph["target_neighbor_idx"],
                num_atoms=num_atoms,
            )
        else:
            basis_a2a_rad = None

        bases_e2e = {}
        bases_e2e["rad"] = self.mlp_rbf_tint(basis_rad_main_raw)
        bases_e2e["cir"] = self.mlp_cbf_tint(
            rad_basis=basis_rad_cir_e2e_raw,
            sph_basis=basis_cir_e2e_raw,
            idx_sph_outer=trip_idx_e2e["out"],
            idx_sph_inner=trip_idx_e2e["out_agg"],
        )

        basis_atom_update = self.mlp_rbf_h(basis_rad_main_raw)
        basis_output = self.mlp_rbf_out(basis_rad_main_raw)

        return (
            basis_rad_main_raw,
            basis_atom_update,
            basis_output,
            bases_qint,
            bases_e2e,
            bases_a2e,
            bases_e2a,
            basis_a2a_rad,
        )

    def _generate_graph(self, data_dict, cutoff, max_neighbors, otf_graph):
        """Generate a radius/nearest neighbor graph, following escn_md.py style."""
        # OTF graph construction
        if otf_graph:
            pbc = None
            if getattr(self, "always_use_pbc", False):
                # Always use PBC for internal graph gen
                pbc = torch.ones(
                    len(data_dict["pos"]),
                    3,
                    dtype=torch.bool,
                    device=data_dict["pos"].device,
                )
            else:
                # Use PBC from input data
                assert (
                    "pbc" in data_dict
                ), "Since always_use_pbc is False, pbc conditions must be supplied by the input data"
                pbc = data_dict["pbc"]
            assert (
                pbc.all() or (~pbc).all()
            ), "We can only accept pbc that is all true or all false"
            logging.debug(
                f"Using radius graph gen version {getattr(self, 'radius_pbc_version', 1)}"
            )
            graph_dict = generate_graph(
                data_dict,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                enforce_max_neighbors_strictly=getattr(
                    self, "enforce_max_neighbors_strictly", True
                ),
                radius_pbc_version=getattr(self, "radius_pbc_version", 1),
                pbc=pbc,
            )
        else:
            # This assumes edge_index is provided
            assert (
                "edge_index" in data_dict
            ), "otf_graph is false, need to provide edge_index as input!"
            cell_per_edge = data_dict["cell"].repeat_interleave(
                data_dict["nedges"], dim=0
            )
            shifts = torch.einsum(
                "ij,ijk->ik",
                data_dict["cell_offsets"].to(cell_per_edge.dtype),
                cell_per_edge,
            )
            edge_distance_vec = (
                data_dict["pos"][data_dict["edge_index"][0]]
                - data_dict["pos"][data_dict["edge_index"][1]]
                + shifts
            )  # [n_edges, 3]
            edge_distance = torch.linalg.norm(
                edge_distance_vec, dim=-1, keepdim=False
            )  # [n_edges, 1]
            graph_dict = {
                "edge_index": data_dict["edge_index"],
                "edge_distance": edge_distance,
                "edge_distance_vec": edge_distance_vec,
                "node_offset": 0,
            }
        # Optionally add more fields as in escn_md.py
        return graph_dict

    @conditional_grad(torch.enable_grad())
    def forward(self, data_dict: dict) -> dict[str, torch.Tensor]:
        self.dtype = data_dict["pos"].dtype

        if (self.dtype != torch.float32) and (data_dict["pos"].dtype != self.dtype):
            data_dict["pos"] = data_dict["pos"].to(self.dtype)
            data_dict["cell"] = data_dict["cell"].to(self.dtype)

        pos = data_dict["pos"]
        batch = data_dict["batch"]
        cell = data_dict["cell"]

        atomic_numbers = data_dict["atomic_numbers"].long()
        num_atoms = atomic_numbers.shape[0]

        data_dict["tags"] = torch.ones(num_atoms, dtype=torch.long, device=pos.device)

        if self.regress_stress and not self.direct_stress:
            displacement = torch.zeros(
                batch.max() + 1, 3, 3, device=pos.device, dtype=self.dtype
            )
            displacement.requires_grad_(True)
            displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
            pos.requires_grad_(True)
            pos = pos + torch.einsum("be,bec->bc", pos, displacement[batch])
            cell = cell.view(-1, 3, 3)
            cell = cell + torch.matmul(cell, displacement)
            data_dict["pos"] = pos
            data_dict["cell"] = cell
            data_dict["displacement"] = displacement

        (
            main_graph,
            a2a_graph,
            a2ee2a_graph,
            qint_graph,
            id_swap,
            trip_idx_e2e,
            trip_idx_a2e,
            trip_idx_e2a,
            quad_idx,
        ) = self.get_graphs_and_indices(data_dict)
        _, idx_t = main_graph["edge_index"]

        (
            basis_rad_raw,
            basis_atom_update,
            basis_output,
            bases_qint,
            bases_e2e,
            bases_a2e,
            bases_e2a,
            basis_a2a_rad,
        ) = self.get_bases(
            main_graph=main_graph,
            a2a_graph=a2a_graph,
            a2ee2a_graph=a2ee2a_graph,
            qint_graph=qint_graph,
            trip_idx_e2e=trip_idx_e2e,
            trip_idx_a2e=trip_idx_a2e,
            trip_idx_e2a=trip_idx_e2a,
            quad_idx=quad_idx,
            num_atoms=num_atoms,
        )

        # Embedding block
        h = self.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, basis_rad_raw, main_graph["edge_index"])
        # (nEdges, emb_size_edge)

        x_E, x_F = self.out_blocks[0](h, m, basis_output, idx_t)
        # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
        xs_E, xs_F = [x_E], [x_F]

        for i in range(self.num_blocks):
            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                bases_qint=bases_qint,
                bases_e2e=bases_e2e,
                bases_a2e=bases_a2e,
                bases_e2a=bases_e2a,
                basis_a2a_rad=basis_a2a_rad,
                basis_atom_update=basis_atom_update,
                edge_index_main=main_graph["edge_index"],
                a2ee2a_graph=a2ee2a_graph,
                a2a_graph=a2a_graph,
                id_swap=id_swap,
                trip_idx_e2e=trip_idx_e2e,
                trip_idx_a2e=trip_idx_a2e,
                trip_idx_e2a=trip_idx_e2a,
                quad_idx=quad_idx,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            x_E, x_F = self.out_blocks[i + 1](h, m, basis_output, idx_t)
            # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)
            xs_E.append(x_E)
            xs_F.append(x_F)

        return {
            "xs_E": xs_E,
            "xs_F": xs_F,
            "edge_vec": main_graph["vector"],
            "edge_idx": idx_t,
            "num_neighbors": main_graph["num_neighbors"],
        }

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@registry.register_model("gemnet_oc_force_head_dtype")
class GemNetOCForceHead(nn.Module, HeadInterface):
    def __init__(
        self,
        backbone,
        num_global_out_layers: int,
        output_init: str = "HeOrthogonal",
        dtype: str | None = None,
    ):
        super().__init__()

        self.dtype = DTYPE_DICT[dtype] if dtype is not None else torch.float32
        if dtype is not None:
            torch.set_default_dtype(self.dtype)

        self.direct_forces = backbone.direct_forces
        self.forces_coupled = backbone.forces_coupled

        emb_size_edge = backbone.edge_emb.dense.linear.out_features
        if self.direct_forces:
            backbone.out_mlp_F = None
            backbone.out_forces = None
            out_mlp_F = [
                Dense(
                    emb_size_edge * (len(backbone.int_blocks) + 1),
                    emb_size_edge,
                    activation=backbone.activation,
                )
            ] + [
                ResidualLayer(
                    emb_size_edge,
                    activation=backbone.activation,
                )
                for _ in range(num_global_out_layers)
            ]
            self.out_mlp_F = torch.nn.Sequential(*out_mlp_F)
            self.out_forces = Dense(
                emb_size_edge,
                1,
                bias=False,
                activation=None,
            )
            out_initializer = get_initializer(output_init)
            self.out_forces.reset_parameters(out_initializer)

    def forward(
        self, data_dict, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.direct_forces:
            with torch.cuda.amp.autocast(False):
                x_F = self.out_mlp_F(torch.cat(emb["xs_F"], dim=-1).to(self.dtype))
                F_st = self.out_forces(x_F)

            if self.forces_coupled:  # enforce F_st = F_ts
                nEdges = emb["edge_idx"].shape[0]
                id_undir = repeat_blocks(
                    emb["num_neighbors"] // 2,
                    repeats=2,
                    continuous_indexing=True,
                )
                # Aggregate forces for coupled/symmetric edges (mean aggregation)
                F_st_agg = torch.zeros(
                    int(nEdges / 2), 1, device=F_st.device, dtype=F_st.dtype
                )
                count = torch.zeros(
                    int(nEdges / 2), 1, device=F_st.device, dtype=F_st.dtype
                )
                ones = torch.ones_like(F_st)
                F_st_agg.scatter_add_(0, id_undir.unsqueeze(-1), F_st)
                count.scatter_add_(0, id_undir.unsqueeze(-1), ones)
                F_st_agg = F_st_agg / count.clamp(min=1)  # mean of the two directions
                F_st = F_st_agg[id_undir]  # (nEdges, 1)

            # map forces in edge directions
            F_st_vec = F_st[:, :, None] * emb["edge_vec"][:, None, :]
            # (nEdges, 1, 3)
            F_t = torch.zeros(
                data_dict["atomic_numbers"].long().shape[0],
                1,
                3,
                device=F_st_vec.device,
                dtype=F_st_vec.dtype,
            )
            F_t.scatter_add_(
                0,
                emb["edge_idx"].unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 3),
                F_st_vec,
            )
            return {"forces": F_t.squeeze(1)}  # (num_atoms, 3)
        return {}


@registry.register_model("gemnet_oc_energy_and_grad_force_head_dtype")
class GemNetOCEnergyAndGradForceHead(nn.Module, HeadInterface):
    def __init__(
        self,
        backbone: BackboneInterface,
        num_global_out_layers: int,
        output_init: str = "HeOrthogonal",
        dtype: str | None = None,
    ):
        super().__init__()
        self.extensive = backbone.extensive

        self.regress_forces = backbone.regress_forces
        self.direct_forces = backbone.direct_forces
        self.scaler = backbone.scaler

        self.dtype = DTYPE_DICT[dtype] if dtype is not None else torch.float32
        if dtype is not None:
            torch.set_default_dtype(self.dtype)

        backbone.out_mlp_E = None
        backbone.out_energy = None

        out_mlp_E = [
            Dense(
                backbone.atom_emb.emb_size * (len(backbone.int_blocks) + 1),
                backbone.atom_emb.emb_size,
                activation=backbone.activation,
            )
        ] + [
            ResidualLayer(
                backbone.atom_emb.emb_size,
                activation=backbone.activation,
            )
            for _ in range(num_global_out_layers)
        ]
        self.out_mlp_E = torch.nn.Sequential(*out_mlp_E)

        self.out_energy = Dense(
            backbone.atom_emb.emb_size,
            1,
            bias=False,
            activation=None,
        )

        out_initializer = get_initializer(output_init)
        self.out_energy.reset_parameters(out_initializer)

    @conditional_grad(torch.enable_grad())
    def forward(
        self, data_dict, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Global output block for final predictions
        x_E = self.out_mlp_E(torch.cat(emb["xs_E"], dim=-1))
        with torch.cuda.amp.autocast(False):
            E_t = self.out_energy(x_E.to(self.dtype))

        nMolecules = torch.max(data_dict["batch"]) + 1
        if self.extensive:
            E_t_agg = torch.zeros(nMolecules, 1, device=E_t.device, dtype=E_t.dtype)
            E_t_agg.scatter_add_(0, data_dict["batch"].unsqueeze(-1), E_t)
        else:
            # For mean, we need to use scatter_add and then divide by count
            E_t_agg = torch.zeros(nMolecules, 1, device=E_t.device, dtype=E_t.dtype)
            count = torch.zeros(nMolecules, 1, device=E_t.device, dtype=E_t.dtype)
            ones = torch.ones_like(E_t)
            E_t_agg.scatter_add_(0, data_dict["batch"].unsqueeze(-1), E_t)
            count.scatter_add_(0, data_dict["batch"].unsqueeze(-1), ones)
            E_t_agg = E_t_agg / count.clamp(min=1)  # avoid division by zero

        outputs = {"energy": E_t_agg.squeeze(1)}  # (num_molecules)

        if self.regress_forces and not self.direct_forces:
            F_t, virials = self.scaler.calc_and_update(
                outputs["energy"],
                data_dict["pos"],
                data_dict["displacement"],
                self.training,
            )
            outputs["forces"] = F_t.squeeze(1)
        return outputs


@registry.register_model("gemnet_oc_energy_and_grad_force_stress_head_dtype")
class GemNetOCEnergyAndGradForceStressHead(nn.Module, HeadInterface):
    def __init__(
        self,
        backbone: BackboneInterface,
        num_global_out_layers: int,
        output_init: str = "HeOrthogonal",
        dtype: str | None = None,
    ):
        super().__init__()
        self.extensive = backbone.extensive

        self.regress_forces = backbone.regress_forces
        self.direct_forces = backbone.direct_forces
        self.scaler = backbone.scaler

        self.dtype = DTYPE_DICT[dtype] if dtype is not None else torch.float32
        if dtype is not None:
            torch.set_default_dtype(self.dtype)

        backbone.out_mlp_E = None
        backbone.out_energy = None

        out_mlp_E = [
            Dense(
                backbone.atom_emb.emb_size * (len(backbone.int_blocks) + 1),
                backbone.atom_emb.emb_size,
                activation=backbone.activation,
            )
        ] + [
            ResidualLayer(
                backbone.atom_emb.emb_size,
                activation=backbone.activation,
            )
            for _ in range(num_global_out_layers)
        ]
        self.out_mlp_E = torch.nn.Sequential(*out_mlp_E)

        self.out_energy = Dense(
            backbone.atom_emb.emb_size,
            1,
            bias=False,
            activation=None,
        )

        out_initializer = get_initializer(output_init)
        self.out_energy.reset_parameters(out_initializer)

    @conditional_grad(torch.enable_grad())
    def forward(
        self, data_dict, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Global output block for final predictions
        x_E = self.out_mlp_E(torch.cat(emb["xs_E"], dim=-1))
        with torch.cuda.amp.autocast(False):
            E_t = self.out_energy(x_E.to(self.dtype))

        nMolecules = torch.max(data_dict["batch"]) + 1
        if self.extensive:
            E_t_agg = torch.zeros(nMolecules, 1, device=E_t.device, dtype=E_t.dtype)
            E_t_agg.scatter_add_(0, data_dict["batch"].unsqueeze(-1), E_t)
        else:
            # For mean, we need to use scatter_add and then divide by count
            E_t_agg = torch.zeros(nMolecules, 1, device=E_t.device, dtype=E_t.dtype)
            count = torch.zeros(nMolecules, 1, device=E_t.device, dtype=E_t.dtype)
            ones = torch.ones_like(E_t)
            E_t_agg.scatter_add_(0, data_dict["batch"].unsqueeze(-1), E_t)
            count.scatter_add_(0, data_dict["batch"].unsqueeze(-1), ones)
            E_t_agg = E_t_agg / count.clamp(min=1)  # avoid division by zero

        outputs = {"energy": E_t_agg.squeeze(1)}  # (num_molecules)

        if self.regress_forces and not self.direct_forces:
            F_t, virials = self.scaler.calc_and_update(
                outputs["energy"],
                data_dict["pos"],
                data_dict["displacement"],
                self.training,
            )
            outputs["forces"] = F_t.squeeze(1)

            volume = torch.linalg.det(data_dict["cell"]).abs().unsqueeze(-1)
            stress = virials / volume.view(-1, 1, 1)
            stress = torch.where(
                torch.abs(stress) < 1e10, stress, torch.zeros_like(stress)
            )
            outputs["stress"] = stress
        return outputs


@registry.register_model("rank2_decomp_head_dtype")
class Rank2DecompositionEdgeBlock(nn.Module, HeadInterface):
    """
    Output block for predicting rank-2 tensors (stress, dielectric tensor, etc).
    Decomposes a rank-2 symmetric tensor into irrep degree 0 and 2.
    """

    def __init__(
        self,
        backbone: BackboneInterface,
        output_name: str,
        num_global_out_layers: int,
        output_init: str = "HeOrthogonal",
        edge_level: bool = False,
        extensive: bool = False,
        dtype: str | None = None,
    ):
        super().__init__()
        emb_size_edge = backbone.edge_emb.dense.linear.out_features
        self.edge_level = edge_level
        self.extensive = extensive
        self.output_name = output_name
        self.dtype = DTYPE_DICT[dtype] if dtype is not None else torch.float32
        if dtype is not None:
            torch.set_default_dtype(self.dtype)

        out_mlp_scalar = [
            Dense(
                emb_size_edge * (len(backbone.int_blocks) + 1),
                emb_size_edge,
                activation=backbone.activation,
            )
        ] + [
            ResidualLayer(
                emb_size_edge,
                activation=backbone.activation,
            )
            for _ in range(num_global_out_layers)
        ]
        self.out_mlp_scalar = torch.nn.Sequential(*out_mlp_scalar)
        self.out_mlp_scalar_final = Dense(
            emb_size_edge,
            1,
            bias=False,
            activation=None,
        )
        out_mlp_irrep2 = [
            Dense(
                emb_size_edge * (len(backbone.int_blocks) + 1),
                emb_size_edge,
                activation=backbone.activation,
            )
        ] + [
            ResidualLayer(
                emb_size_edge,
                activation=backbone.activation,
            )
            for _ in range(num_global_out_layers)
        ]
        self.out_mlp_irrep2 = torch.nn.Sequential(*out_mlp_irrep2)
        self.out_mlp_irrep2_final = Dense(
            emb_size_edge,
            1,
            bias=False,
            activation=None,
        )
        out_initializer = get_initializer(output_init)
        self.out_mlp_irrep2_final.reset_parameters(out_initializer)
        self.out_mlp_scalar_final.reset_parameters(out_initializer)

        # Change of basis obtained by stacking the C-G coefficients
        self.change_mat = torch.transpose(
            torch.tensor(
                [
                    [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                    [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                    [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                    [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                    [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                    [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                    [
                        -(6 ** (-0.5)),
                        0,
                        0,
                        0,
                        2 * 6 ** (-0.5),
                        0,
                        0,
                        0,
                        -(6 ** (-0.5)),
                    ],
                    [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                    [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
                ]
            ).detach(),
            0,
            1,
        )

    def forward(
        self, data_dict, emb: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        edge_distance_vec = emb["edge_vec"]
        x_edge = torch.cat(emb["xs_F"], dim=-1).to(self.dtype)
        edge_index = emb["edge_idx"]

        # Calculate spherical harmonics of degree 2 of the points sampled
        sphere_irrep2 = o3.spherical_harmonics(
            2, edge_distance_vec, True
        ).detach()  # (nEdges, 5)

        if self.edge_level:
            # MLP at edge level before pooling.
            with torch.cuda.amp.autocast(False):
                # Irrep 0 prediction
                edge_scalar = x_edge
                edge_scalar = self.out_mlp_scalar(edge_scalar)
                edge_scalar = self.out_mlp_scalar_final(edge_scalar)
                # Irrep 2 prediction
                edge_irrep2 = (
                    sphere_irrep2[:, :, None] * x_edge[:, None, :]
                )  # (nEdges, 5, emb_size)
                edge_irrep2 = self.out_mlp_irrep2(edge_irrep2)
                edge_irrep2 = self.out_mlp_irrep2_final(edge_irrep2)

            # Edge to node aggregation using scatter operations
            node_scalar = torch.zeros(
                emb["xs_E"][0].shape[0],
                edge_scalar.shape[1],
                device=edge_scalar.device,
                dtype=edge_scalar.dtype,
            )
            count_scalar = torch.zeros(
                emb["xs_E"][0].shape[0],
                1,
                device=edge_scalar.device,
                dtype=edge_scalar.dtype,
            )
            ones_scalar = torch.ones(
                edge_scalar.shape[0],
                1,
                device=edge_scalar.device,
                dtype=edge_scalar.dtype,
            )
            node_scalar.scatter_add_(
                0,
                edge_index.unsqueeze(-1).expand(-1, edge_scalar.shape[1]),
                edge_scalar,
            )
            count_scalar.scatter_add_(0, edge_index.unsqueeze(-1), ones_scalar)
            node_scalar = node_scalar / count_scalar.clamp(min=1)  # mean aggregation

            node_irrep2 = torch.zeros(
                emb["xs_E"][0].shape[0],
                edge_irrep2.shape[1],
                edge_irrep2.shape[2],
                device=edge_irrep2.device,
                dtype=edge_irrep2.dtype,
            )
            count_irrep2 = torch.zeros(
                emb["xs_E"][0].shape[0],
                1,
                1,
                device=edge_irrep2.device,
                dtype=edge_irrep2.dtype,
            )
            ones_irrep2 = torch.ones(
                edge_irrep2.shape[0],
                1,
                1,
                device=edge_irrep2.device,
                dtype=edge_irrep2.dtype,
            )
            node_irrep2.scatter_add_(
                0,
                edge_index.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, edge_irrep2.shape[1], edge_irrep2.shape[2]),
                edge_irrep2,
            )
            count_irrep2.scatter_add_(
                0, edge_index.unsqueeze(-1).unsqueeze(-1), ones_irrep2
            )
            node_irrep2 = node_irrep2 / count_irrep2.clamp(min=1)  # mean aggregation
        else:
            # Edge to node aggregation first, then MLP
            edge_irrep2 = (
                sphere_irrep2[:, :, None] * x_edge[:, None, :]
            )  # (nEdges, 5, emb_size)

            # Edge to node aggregation using scatter operations
            node_scalar = torch.zeros(
                emb["xs_E"][0].shape[0],
                x_edge.shape[1],
                device=x_edge.device,
                dtype=x_edge.dtype,
            )
            count_scalar = torch.zeros(
                emb["xs_E"][0].shape[0], 1, device=x_edge.device, dtype=x_edge.dtype
            )
            ones_scalar = torch.ones(
                x_edge.shape[0], 1, device=x_edge.device, dtype=x_edge.dtype
            )
            node_scalar.scatter_add_(
                0, edge_index.unsqueeze(-1).expand(-1, x_edge.shape[1]), x_edge
            )
            count_scalar.scatter_add_(0, edge_index.unsqueeze(-1), ones_scalar)
            node_scalar = node_scalar / count_scalar.clamp(min=1)  # mean aggregation

            node_irrep2 = torch.zeros(
                emb["xs_E"][0].shape[0],
                edge_irrep2.shape[1],
                edge_irrep2.shape[2],
                device=edge_irrep2.device,
                dtype=edge_irrep2.dtype,
            )
            count_irrep2 = torch.zeros(
                emb["xs_E"][0].shape[0],
                1,
                1,
                device=edge_irrep2.device,
                dtype=edge_irrep2.dtype,
            )
            ones_irrep2 = torch.ones(
                edge_irrep2.shape[0],
                1,
                1,
                device=edge_irrep2.device,
                dtype=edge_irrep2.dtype,
            )
            node_irrep2.scatter_add_(
                0,
                edge_index.unsqueeze(-1)
                .unsqueeze(-1)
                .expand(-1, edge_irrep2.shape[1], edge_irrep2.shape[2]),
                edge_irrep2,
            )
            count_irrep2.scatter_add_(
                0, edge_index.unsqueeze(-1).unsqueeze(-1), ones_irrep2
            )
            node_irrep2 = node_irrep2 / count_irrep2.clamp(min=1)  # mean aggregation

            with torch.cuda.amp.autocast(False):
                # Irrep 0 prediction
                node_scalar = self.out_mlp_scalar(node_scalar)
                node_scalar = self.out_mlp_scalar_final(node_scalar)

                # Irrep 2 prediction
                node_irrep2 = self.out_mlp_irrep2(node_irrep2)
                node_irrep2 = self.out_mlp_irrep2_final(node_irrep2)

        # Final aggregation from nodes to molecules
        batch_size = data_dict["batch"].max() + 1
        if self.extensive:
            scalar = torch.zeros(
                batch_size, device=node_scalar.device, dtype=node_scalar.dtype
            )
            scalar.scatter_add_(0, data_dict["batch"], node_scalar.view(-1))

            irrep2 = torch.zeros(
                batch_size, 5, device=node_irrep2.device, dtype=node_irrep2.dtype
            )
            irrep2.scatter_add_(
                0,
                data_dict["batch"].unsqueeze(-1).expand(-1, 5),
                node_irrep2.view(-1, 5),
            )
        else:
            # For mean aggregation
            scalar = torch.zeros(
                batch_size, device=node_scalar.device, dtype=node_scalar.dtype
            )
            count_scalar_final = torch.zeros(
                batch_size, device=node_scalar.device, dtype=node_scalar.dtype
            )
            ones_scalar_final = torch.ones_like(node_scalar.view(-1))
            scalar.scatter_add_(0, data_dict["batch"], node_scalar.view(-1))
            count_scalar_final.scatter_add_(0, data_dict["batch"], ones_scalar_final)
            scalar = scalar / count_scalar_final.clamp(min=1)

            irrep2 = torch.zeros(
                batch_size, 5, device=node_irrep2.device, dtype=node_irrep2.dtype
            )
            count_irrep2_final = torch.zeros(
                batch_size, 1, device=node_irrep2.device, dtype=node_irrep2.dtype
            )
            ones_irrep2_final = torch.ones(
                node_irrep2.shape[0],
                1,
                device=node_irrep2.device,
                dtype=node_irrep2.dtype,
            )
            irrep2.scatter_add_(
                0,
                data_dict["batch"].unsqueeze(-1).expand(-1, 5),
                node_irrep2.view(-1, 5),
            )
            count_irrep2_final.scatter_add_(
                0, data_dict["batch"].unsqueeze(-1), ones_irrep2_final
            )
            irrep2 = irrep2 / count_irrep2_final.clamp(min=1)

        return {
            f"{self.output_name}_isotropic": scalar.reshape(-1),
            f"{self.output_name}_anisotropic": irrep2,
        }
