
import torch

__all__ = [ "segment_mm"]

# This code is derived from Deep Graph Library DGL, licensed under the Apache License 2.0.
# See https://www.apache.org/licenses/LICENSE-2.0 for more information.
# https://github.com/dmlc/dgl
class SEGMENTMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, seglen_A):
        A=A.contiguous()
        B=B.contiguous()
        seglen_A=seglen_A.contiguous()
        if B.dim() != 3:
            raise ValueError("segment_mm expects B to be a 3D tensor.")
        C = torch.empty((A.shape[0], B.shape[2]), device=A.device, dtype=A.dtype)
        torch.ops.fairchem_cpp.segment_mm(A, B, C, seglen_A, False)
        ctx.backward_cache = A, B, seglen_A
        return C
    
    @staticmethod
    def backward(ctx, dZ):
        dZ=dZ.contiguous()
        A, B, seglen_A = ctx.backward_cache
        A_grad = B_grad = None
        if ctx.needs_input_grad[0]:
            #  Compute A_grad = Out_grad * B^T
            A_grad = torch.empty(A.shape, device=A.device, dtype=A.dtype)
            torch.ops.fairchem_cpp.segment_mm(dZ, B, A_grad, seglen_A, True)
        if ctx.needs_input_grad[1]:
            #  Compute B_grad = A^T * Out_grad
            B_grad = torch.empty(B.shape, device=B.device, dtype=B.dtype)
            torch.ops.fairchem_cpp.segment_mm_backward(A, dZ, B_grad, seglen_A)
        return A_grad, B_grad, None
    
def segment_mm(A, B, seglen_A):
    if A.device.type == "cpu":
        C = []
        off = 0
        for i in range(B.shape[0]):
            C.append(A[off : off + seglen_A[i]] @ B[i])
            off += seglen_A[i]
        return torch.cat(C)
    else:
        #if autocasting make sure weights are same type
        B=B.to(A.dtype)
        return SEGMENTMM.apply(A,B,seglen_A)