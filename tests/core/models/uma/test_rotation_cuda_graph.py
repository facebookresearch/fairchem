"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import functools
import os
import random
import tempfile

import numpy as np
import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from fairchem.core.common.profiler_utils import get_profile_schedule
from fairchem.core.models.uma.common.rotation import (
    eulers_to_wigner,
    init_edge_rot_euler_angles,
)
from fairchem.core.models.uma.common.rotation_cuda_graph import RotMatWignerCudaGraph

JD_path = "src/fairchem/core/models/uma/Jd.pt"


def seed_everywhere(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_jds(lmax):
    Jd_list = torch.load(JD_path)
    Jd_buffers = [Jd_list[l].to(dtype=torch.float).cuda() for l in range(lmax + 1)]
    return Jd_buffers


def get_rotmat_and_wigner(edge_distance_vecs, jds):
    euler_angles = init_edge_rot_euler_angles(edge_distance_vecs)
    wigner = eulers_to_wigner(
        euler_angles,
        0,
        len(jds) - 1,
        jds,
    )
    wigner_inv = torch.transpose(wigner, 1, 2).contiguous()

    return wigner, wigner_inv


@pytest.mark.gpu()
@pytest.mark.parametrize("lmax", [2, 3, 4, 6])
def test_rotation_no_graph_matches_graph_basic(lmax):
    seed_everywhere()
    jds = get_jds(lmax=lmax)
    edge_dist_vec = torch.rand(torch.Size([1200, 3]), requires_grad=True).cuda()
    edge_dist_vec[0:100, :] = edge_dist_vec.new_tensor([1, 0, 0])
    edge_dist_vec[101:200, :] = edge_dist_vec.new_tensor([0, 1, 0])
    edge_dist_vec[201:300, :] = edge_dist_vec.new_tensor([0, 0, 1])
    graph_obj = RotMatWignerCudaGraph()
    graph_obj._capture_graph(edge_dist_vec, jds)
    seed_everywhere()
    wigner_graph, wigner_inv_graph = graph_obj.get_rotmat_and_wigner(edge_dist_vec, jds)
    seed_everywhere()
    wigner, wigner_inv = get_rotmat_and_wigner(edge_dist_vec, jds)

    assert torch.allclose(wigner_graph, wigner, atol=1e-7)
    assert torch.allclose(wigner_inv_graph, wigner_inv, atol=1e-7)


@pytest.mark.gpu()
def test_rotation_no_graph_matches_graph_shape_change():
    seed_everywhere()
    lmax = 4
    jds = get_jds(lmax=lmax)
    edge_dist_vec = torch.rand(torch.Size([120000, 3]), requires_grad=True).cuda()
    graph_obj = RotMatWignerCudaGraph()
    wigner_graph, wigner_inv_graph = graph_obj.get_rotmat_and_wigner(edge_dist_vec, jds)
    assert graph_obj.graph_capture_count == 1
    # now pass in a smaller input, should not trigger a graph capture
    seed_everywhere()
    edge_dist_vec2 = torch.rand(torch.Size([110000, 3]), requires_grad=True).cuda()
    wigner_graph, wigner_inv_graph = graph_obj.get_rotmat_and_wigner(
        edge_dist_vec2, jds
    )
    seed_everywhere()
    wigner, wigner_inv = get_rotmat_and_wigner(edge_dist_vec2, jds)
    assert graph_obj.graph_capture_count == 1
    assert torch.allclose(wigner_graph, wigner, atol=1e-7)
    assert torch.allclose(wigner_inv_graph, wigner_inv, atol=1e-7)
    # now pass in a large input, should trigger a graph capture
    edge_dist_vec3 = torch.rand(torch.Size([130000, 3]), requires_grad=True).cuda()
    wigner_graph, wigner_inv_graph = graph_obj.get_rotmat_and_wigner(
        edge_dist_vec3, jds
    )
    assert graph_obj.graph_capture_count == 2
    assert wigner_graph.shape[0] == edge_dist_vec3.shape[0]
    assert wigner_inv_graph.shape[0] == edge_dist_vec3.shape[0]


def wigner_grad(wigner, wigner_inv, edge_dist_vec):
    sum = wigner.sum() + wigner_inv.sum()
    grad = torch.autograd.grad(sum, edge_dist_vec, create_graph=False)
    return grad


@pytest.mark.gpu()
def test_rotation_graph_grads():
    lmax = 4
    jds = get_jds(lmax=lmax)
    seed_everywhere()
    edge_dist_vec = torch.rand(torch.Size([120000, 3]), requires_grad=True).cuda()
    graph_obj = RotMatWignerCudaGraph()
    graph_obj._capture_graph(edge_dist_vec, jds)
    seed_everywhere()
    wigner_graph, wigner_inv_graph = graph_obj.get_rotmat_and_wigner(edge_dist_vec, jds)
    grad_graph = wigner_grad(wigner_graph, wigner_inv_graph, edge_dist_vec)

    seed_everywhere()
    # edge_dist_vec = torch.rand(torch.Size([120000, 3]), requires_grad=True).cuda()
    wigner, wigner_inv = get_rotmat_and_wigner(edge_dist_vec, jds)
    grad_no_graph = wigner_grad(wigner, wigner_inv, edge_dist_vec)
    assert torch.allclose(grad_graph[0], grad_no_graph[0], atol=1e-3)


def make_profile(fn, output_path):
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    profile_schedule, total_profile_steps = get_profile_schedule(active=5)

    def trace_handler(p, output_path):
        print(f"Saving trace in {output_path}")
        p.export_chrome_trace(output_path)

    tc = functools.partial(trace_handler, output_path=output_path)

    with profile(
        activities=activities,
        schedule=profile_schedule,
        on_trace_ready=tc,
    ) as p:
        for _ in range(total_profile_steps):
            fn()
            torch.cuda.synchronize()
            p.step()


@pytest.mark.gpu()
def test_generate_traces():
    seed_everywhere()
    temp_dir = tempfile.TemporaryDirectory(delete=False)
    print("traces stored at temp dir", temp_dir)
    lmax = 2
    jds = get_jds(lmax=lmax)
    graph_obj = RotMatWignerCudaGraph()
    edge_dist_vec = torch.rand(torch.Size([120000, 3]), requires_grad=True).cuda()

    def call_fn():
        wigner_graph, wigner_inv_graph = graph_obj.get_rotmat_and_wigner(
            edge_dist_vec, jds
        )
        wigner_grad(wigner_graph, wigner_inv_graph, edge_dist_vec)

    # these traces should be ~12ms long for each step on H100
    make_profile(
        call_fn,
        output_path=os.path.join(temp_dir.name, "trace_cuda_graph.json"),
    )
