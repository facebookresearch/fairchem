#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <stdio.h>
#include <torch/extension.h>

// This code is derived from Deep Graph Library DGL, licensed under the Apache License 2.0.
// See https://www.apache.org/licenses/LICENSE-2.0 for more information.
// https://github.com/dmlc/dgl

namespace fairchem_cpp {

#define CUBLAS_CHECK(expr) \
  do { \
    cublasStatus_t status = (expr); \
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS error: ", status); \
  } while (0)

// Forward kernel
template <typename TorchDType, typename CUDADtype>
void SegmentMM(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& seglen,
    bool b_trans,cublasDataType_t type, cublasComputeType_t compute) {

    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    const CUDADtype* A_data = reinterpret_cast<CUDADtype*>(A.data_ptr<TorchDType>());
    const CUDADtype* B_data = reinterpret_cast<CUDADtype*>(B.data_ptr<TorchDType>());
    CUDADtype* C_data = reinterpret_cast<CUDADtype*>(C.data_ptr<TorchDType>());

    const int32_t* seglen_data = seglen.data_ptr<int32_t>();

    int64_t A_offset = 0, B_offset = 0, C_offset = 0;
    int64_t num_rel = seglen.numel();

    int64_t m_offset = 0; //just for sanity check
    for (int64_t etype = 0; etype < num_rel; ++etype) {
        int64_t m = seglen_data[etype];
        TORCH_CHECK((m_offset + m <= A.size(0)),"Segment index out of bound of A->shape[0]." )
        int64_t n = B.size(2);
        int64_t k = B.size(1);

        int ldb = n, lda = k, ldc = n;
        cublasOperation_t transB = CUBLAS_OP_N; 

        if (b_trans) {
            transB = CUBLAS_OP_T;
            ldb = n, lda = n, ldc = k;
            std::swap(n, k);
        }

        CUBLAS_CHECK(cublasGemmEx(
            handle, transB, CUBLAS_OP_N, n, m, k,
            &alpha,
            B_data + B_offset, type, ldb,
            A_data + A_offset, type, lda,
            &beta,
            C_data + C_offset, type, ldc,
            compute,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        A_offset += m * k;
        B_offset += k * n;
        C_offset += m * n;
        m_offset += m;
    }
}

// Backward kernel (grad wrt B)
template <typename TorchDType, typename CUDADtype>
void SegmentMMBackwardB(
    const at::Tensor& A,
    const at::Tensor& dC,
    at::Tensor& dB,
    const at::Tensor& seglen,cublasDataType_t type, cublasComputeType_t compute) {

    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    const CUDADtype* A_data = reinterpret_cast<CUDADtype*>(A.data_ptr<TorchDType>());
    const CUDADtype* dC_data = reinterpret_cast<CUDADtype*>(dC.data_ptr<TorchDType>());
    CUDADtype* dB_data = reinterpret_cast<CUDADtype*>(dB.data_ptr<TorchDType>());

    const int32_t* seglen_data = seglen.data_ptr<int32_t>();

    int64_t A_offset = 0, dC_offset = 0, dB_offset = 0;
    int64_t num_rel = seglen.numel();

    int64_t k_offset = 0;
    for (int64_t etype = 0; etype < num_rel; ++etype) {
        int64_t m = dC.size(1);         // rows of dC
        int64_t n = A.size(1);          // cols of A
        int64_t k = seglen_data[etype]; // batch size
        TORCH_CHECK((k_offset + k <= A.size(0)),"Segment index out of bound of A->shape[0]." )

        int lddC = m, ldA = n, lddB = m;

        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
            &alpha,
            dC_data + dC_offset, type, lddC,
            A_data + A_offset, type, ldA,
            &beta,
            dB_data + dB_offset, type, lddB,
            compute,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        dC_offset += m * k;
        A_offset += n * k;
        dB_offset += m * n;
        k_offset += k;
    }
}

void segment_mm_dispatch(const at::Tensor& A, const at::Tensor& B, at::Tensor& C, const at::Tensor& seglen, bool b_trans) {
    TORCH_CHECK(seglen.dtype() == at::kInt);
    TORCH_CHECK(A.dtype() == B.dtype())
    TORCH_CHECK(A.dtype() == C.dtype())

    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "dC must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "dB must be a CUDA tensor");
    TORCH_CHECK(seglen.is_cpu(), "seglen must be a CPU tensor");
    
    if (A.scalar_type() == at::ScalarType::Float) {
        SegmentMM<float,float>(A, B, C, seglen,  b_trans, CUDA_R_32F,CUBLAS_COMPUTE_32F );
    } else if (A.scalar_type() == at::ScalarType::Half) {
        SegmentMM<at::Half,__half>(A, B, C, seglen,  b_trans, CUDA_R_16F ,CUBLAS_COMPUTE_16F);
    } else if (A.scalar_type() == at::ScalarType::BFloat16) {
        SegmentMM<at::BFloat16, __nv_bfloat16>(A, B, C, seglen,  b_trans, CUDA_R_16BF, CUBLAS_COMPUTE_32F);
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}

void segment_mm_backward_dispatch(const at::Tensor& A, const at::Tensor& dC, at::Tensor& dB, const at::Tensor& seglen) {
    TORCH_CHECK(seglen.dtype() == at::kInt);
    TORCH_CHECK(A.dtype() == dC.dtype())
    TORCH_CHECK(A.dtype() == dB.dtype())

    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(dC.is_cuda(), "dC must be a CUDA tensor");
    TORCH_CHECK(dB.is_cuda(), "dB must be a CUDA tensor");
    TORCH_CHECK(seglen.is_cpu(), "seglen must be a CPU tensor");

    if (A.scalar_type() == at::ScalarType::Float) {
        SegmentMMBackwardB<float,float>(A, dC, dB, seglen,CUDA_R_32F,CUBLAS_COMPUTE_32F);
    } else if (A.scalar_type() == at::ScalarType::Half) {
        SegmentMMBackwardB<at::Half, __half>(A, dC, dB, seglen,CUDA_R_16F ,CUBLAS_COMPUTE_16F);
    } else if (A.scalar_type() == at::ScalarType::BFloat16) {
        SegmentMMBackwardB<at::BFloat16,__nv_bfloat16>(A, dC, dB, seglen,CUDA_R_16BF, CUBLAS_COMPUTE_32F);
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}
  
TORCH_LIBRARY_IMPL(fairchem_cpp, CUDA, m) {
    m.impl("segment_mm", segment_mm_dispatch);
    m.impl("segment_mm_backward", segment_mm_backward_dispatch);
  }
  
}