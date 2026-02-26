#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>

// This code is derived from Deep Graph Library DGL, licensed under the Apache License 2.0.
// See https://www.apache.org/licenses/LICENSE-2.0 for more information.
// https://github.com/dmlc/dgl

// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

namespace fairchem_cpp {

#define CUBLAS_CHECK(expr) \
  do { \
    cublasStatus_t status = (expr); \
    TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS error: ", status); \
  } while (0)

constexpr bool use_grouped_gemm = true;

// Ensures that the pointer mode is set to HOST, which is required for cublasGemmGroupedBatchedEx
class CublasPointerModeGuard {
 public:
  explicit CublasPointerModeGuard(cublasHandle_t handle) : handle_(handle) {
    CUBLAS_CHECK(cublasGetPointerMode(handle_, &old_mode_));
    if (old_mode_ != CUBLAS_POINTER_MODE_HOST) {
      CUBLAS_CHECK(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST));
      restore_ = true;
    }
  }

  ~CublasPointerModeGuard() {
    if (restore_) {
      // Best-effort restore in destructor.
      cublasSetPointerMode(handle_, old_mode_);
    }
  }

 private:
  cublasHandle_t handle_;
  cublasPointerMode_t old_mode_;
  bool restore_{false};
};

template <typename T>
inline T one_scalar();

template <typename T>
inline T zero_scalar();

template <>
inline float one_scalar<float>() {
  return 1.0f;
}

template <>
inline float zero_scalar<float>() {
  return 0.0f;
}

template <>
inline __half one_scalar<__half>() {
  return __float2half(1.0f);
}

template <>
inline __half zero_scalar<__half>() {
  return __float2half(0.0f);
}

template <typename PtrType>
at::Tensor copy_pointer_array_to_device(
    const std::vector<PtrType>& host_ptrs,
    const at::Tensor& like_tensor,
    cudaStream_t stream) {
  TORCH_CHECK(sizeof(void*) == sizeof(int64_t), "Expected 64-bit pointers");
  auto dev_ptrs = at::empty(
      {static_cast<int64_t>(host_ptrs.size())},
      at::TensorOptions().device(like_tensor.device()).dtype(at::kLong));
  if (!host_ptrs.empty()) {
    cudaError_t err = cudaMemcpyAsync(
        dev_ptrs.data_ptr<int64_t>(),
        host_ptrs.data(),
        host_ptrs.size() * sizeof(void*),
        cudaMemcpyHostToDevice,
        stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
  }
  return dev_ptrs;
}


template <typename TorchDType, typename CUDADtype>
void SegmentMMGrouped(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& seglen,
    bool b_trans,
    cublasDataType_t type,
    cublasComputeType_t compute) {

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    CublasPointerModeGuard pointer_mode_guard(handle);

    const CUDADtype* A_data = reinterpret_cast<CUDADtype*>(A.data_ptr<TorchDType>());
    const CUDADtype* B_data = reinterpret_cast<CUDADtype*>(B.data_ptr<TorchDType>());
    CUDADtype* C_data = reinterpret_cast<CUDADtype*>(C.data_ptr<TorchDType>());

    const int32_t* seglen_data = seglen.data_ptr<int32_t>();

    int64_t A_offset = 0, B_offset = 0, C_offset = 0;
    int64_t num_rel = seglen.numel();
    int64_t m_offset = 0; //just for sanity check
    if (num_rel == 0) {return; } // empty tensor, early return

    std::vector<cublasOperation_t> transa(num_rel), transb(num_rel);
    std::vector<int> m(num_rel), n(num_rel), k(num_rel);
    std::vector<int> lda(num_rel), ldb(num_rel), ldc(num_rel);
    std::vector<int> group_size(num_rel, 1);
    std::vector<const void*> A_ptrs(num_rel), B_ptrs(num_rel);
    std::vector<void*> C_ptrs(num_rel);

    const int64_t B_n = B.size(2);
    const int64_t B_k = B.size(1);

    for (int64_t etype = 0; etype < num_rel; ++etype) {
        int64_t seg_m = seglen_data[etype];
        TORCH_CHECK(seg_m >= 0, "Segment length must be non-negative.");
        TORCH_CHECK(m_offset + seg_m <= A.size(0), "Segment index out of bound of A->shape[0].");

        // build up A/B/C pointers on the host
        A_ptrs[etype] = B_data + B_offset;
        B_ptrs[etype] = A_data + A_offset;
        C_ptrs[etype] = C_data + C_offset;

        if (!b_trans) {
            // Equivalent to current cublasGemmEx path:
            // C[m x n] = A_input[m x k] * B[k x n]
            transa[etype] = CUBLAS_OP_N;
            transb[etype] = CUBLAS_OP_N;
            m[etype] = static_cast<int>(B_n);
            n[etype] = static_cast<int>(seg_m);
            k[etype] = static_cast<int>(B_k);
            lda[etype] = static_cast<int>(B_n);
            ldb[etype] = static_cast<int>(B_k);
            ldc[etype] = static_cast<int>(B_n);

            A_offset += seg_m * B_k;
            B_offset += B_k * B_n;
            C_offset += seg_m * B_n;
        } else {
            // Equivalent to current cublasGemmEx path for A_grad:
            // A_grad[m x k_out] = dZ[m x n_out] * B^T[n_out x k_out]
            transa[etype] = CUBLAS_OP_T;
            transb[etype] = CUBLAS_OP_N;
            m[etype] = static_cast<int>(B_k);
            n[etype] = static_cast<int>(seg_m);
            k[etype] = static_cast<int>(B_n);
            lda[etype] = static_cast<int>(B_n);
            ldb[etype] = static_cast<int>(B_n);
            ldc[etype] = static_cast<int>(B_k);

            A_offset += seg_m * B_n;
            B_offset += B_k * B_n;
            C_offset += seg_m * B_k;
        }
        m_offset += seg_m;
    }

    // cublasGemmGroupedBatchedEx requires pointers to be on the device. Sending asyncly to the device
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    at::Tensor dA_ptrs = copy_pointer_array_to_device(A_ptrs, C, stream);
    at::Tensor dB_ptrs = copy_pointer_array_to_device(B_ptrs, C, stream);
    at::Tensor dC_ptrs = copy_pointer_array_to_device(C_ptrs, C, stream);

    if (type == CUDA_R_16F && compute == CUBLAS_COMPUTE_16F) {
        std::vector<__half> alpha(num_rel, one_scalar<__half>());
        std::vector<__half> beta(num_rel, zero_scalar<__half>());
        CUBLAS_CHECK(cublasGemmGroupedBatchedEx(
            handle,
            transa.data(),
            transb.data(),
            m.data(),
            n.data(),
            k.data(),
            alpha.data(),
            reinterpret_cast<const void* const*>(dA_ptrs.data_ptr<int64_t>()),
            type,
            lda.data(),
            reinterpret_cast<const void* const*>(dB_ptrs.data_ptr<int64_t>()),
            type,
            ldb.data(),
            beta.data(),
            reinterpret_cast<void* const*>(dC_ptrs.data_ptr<int64_t>()),
            type,
            ldc.data(),
            static_cast<int>(group_size.size()),
            group_size.data(),
            compute)
        );
    } else {
        // Float/BFloat16 grouped GEMM use float scaling factors.
        std::vector<float> alpha(num_rel, one_scalar<float>());
        std::vector<float> beta(num_rel, zero_scalar<float>());
        CUBLAS_CHECK(cublasGemmGroupedBatchedEx(
            handle,
            transa.data(),
            transb.data(),
            m.data(),
            n.data(),
            k.data(),
            alpha.data(),
            reinterpret_cast<const void* const*>(dA_ptrs.data_ptr<int64_t>()),
            type,
            lda.data(),
            reinterpret_cast<const void* const*>(dB_ptrs.data_ptr<int64_t>()),
            type,
            ldb.data(),
            beta.data(),
            reinterpret_cast<void* const*>(dC_ptrs.data_ptr<int64_t>()),
            type,
            ldc.data(),
            static_cast<int>(group_size.size()),
            group_size.data(),
            compute)
        );
    }
}

template <typename TorchDType, typename CUDADtype>
void SegmentMMBackwardBGrouped(
    const at::Tensor& A,
    const at::Tensor& dC,
    at::Tensor& dB,
    const at::Tensor& seglen,
    cublasDataType_t type,
    cublasComputeType_t compute) {

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CublasPointerModeGuard pointer_mode_guard(handle);

  const CUDADtype* A_data = reinterpret_cast<CUDADtype*>(A.data_ptr<TorchDType>());
  const CUDADtype* dC_data = reinterpret_cast<CUDADtype*>(dC.data_ptr<TorchDType>());
  CUDADtype* dB_data = reinterpret_cast<CUDADtype*>(dB.data_ptr<TorchDType>());

  const int32_t* seglen_data = seglen.data_ptr<int32_t>();

  int64_t A_offset = 0, dC_offset = 0, dB_offset = 0;
  int64_t num_rel = seglen.numel();
  if (num_rel == 0) {return; } // empty tensor, early return

  std::vector<cublasOperation_t> transa(num_rel, CUBLAS_OP_N);
  std::vector<cublasOperation_t> transb(num_rel, CUBLAS_OP_T);
  std::vector<int> m(num_rel), n(num_rel), k(num_rel);
  std::vector<int> lda(num_rel), ldb(num_rel), ldc(num_rel);
  std::vector<int> group_size(num_rel, 1);
  std::vector<const void*> A_ptrs(num_rel), B_ptrs(num_rel);
  std::vector<void*> C_ptrs(num_rel);

  int64_t k_offset = 0;
  const int64_t m_const = dC.size(1); // rows of dC
  const int64_t n_const = A.size(1); // cols of A

  // build up A/B/C pointers on the host
  for (int64_t etype = 0; etype < num_rel; ++etype) {
    int64_t seg_k = seglen_data[etype]; // batch size
    TORCH_CHECK(k_offset + seg_k <= A.size(0), "Segment index out of bound of A->shape[0].");

    m[etype] = static_cast<int>(m_const);
    n[etype] = static_cast<int>(n_const);
    k[etype] = static_cast<int>(seg_k);
    lda[etype] = static_cast<int>(m_const);
    ldb[etype] = static_cast<int>(n_const);
    ldc[etype] = static_cast<int>(m_const);

    A_ptrs[etype] = dC_data + dC_offset;
    B_ptrs[etype] = A_data + A_offset;
    C_ptrs[etype] = dB_data + dB_offset;

    dC_offset += m_const * seg_k;
    A_offset += n_const * seg_k;
    dB_offset += m_const * n_const;
    k_offset += seg_k;
  }

  // cublasGemmGroupedBatchedEx requires pointers to be on the device. Sending asyncly to the device
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  at::Tensor dA_ptrs = copy_pointer_array_to_device(A_ptrs, dB, stream);
  at::Tensor dB_ptrs = copy_pointer_array_to_device(B_ptrs, dB, stream);
  at::Tensor dC_ptrs = copy_pointer_array_to_device(C_ptrs, dB, stream);

  if (type == CUDA_R_16F && compute == CUBLAS_COMPUTE_16F) {
    std::vector<__half> alpha(num_rel, one_scalar<__half>());
    std::vector<__half> beta(num_rel, zero_scalar<__half>());
    CUBLAS_CHECK(cublasGemmGroupedBatchedEx(
        handle,
        transa.data(),
        transb.data(),
        m.data(),
        n.data(),
        k.data(),
        alpha.data(),
        reinterpret_cast<const void* const*>(dA_ptrs.data_ptr<int64_t>()),
        type,
        lda.data(),
        reinterpret_cast<const void* const*>(dB_ptrs.data_ptr<int64_t>()),
        type,
        ldb.data(),
        beta.data(),
        reinterpret_cast<void* const*>(dC_ptrs.data_ptr<int64_t>()),
        type,
        ldc.data(),
        static_cast<int>(group_size.size()),
        group_size.data(),
        compute)
    );
  } else {
    std::vector<float> alpha(num_rel, one_scalar<float>());
    std::vector<float> beta(num_rel, zero_scalar<float>());
    CUBLAS_CHECK(cublasGemmGroupedBatchedEx(
        handle,
        transa.data(),
        transb.data(),
        m.data(),
        n.data(),
        k.data(),
        alpha.data(),
        reinterpret_cast<const void* const*>(dA_ptrs.data_ptr<int64_t>()),
        type,
        lda.data(),
        reinterpret_cast<const void* const*>(dB_ptrs.data_ptr<int64_t>()),
        type,
        ldb.data(),
        beta.data(),
        reinterpret_cast<void* const*>(dC_ptrs.data_ptr<int64_t>()),
        type,
        ldc.data(),
        static_cast<int>(group_size.size()),
        group_size.data(),
        compute)
    );
  }
}

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
        if (use_grouped_gemm) {
            SegmentMMGrouped<float,float>(A, B, C, seglen,  b_trans, CUDA_R_32F,CUBLAS_COMPUTE_32F );
        } else {
            SegmentMM<float,float>(A, B, C, seglen,  b_trans, CUDA_R_32F,CUBLAS_COMPUTE_32F );
        }
    } else if (A.scalar_type() == at::ScalarType::Half) {
        if (use_grouped_gemm) {
            SegmentMMGrouped<at::Half,__half>(A, B, C, seglen,  b_trans, CUDA_R_16F ,CUBLAS_COMPUTE_32F);
        } else {
            SegmentMM<at::Half,__half>(A, B, C, seglen,  b_trans, CUDA_R_16F ,CUBLAS_COMPUTE_16F);
        }
    } else if (A.scalar_type() == at::ScalarType::BFloat16) {
        if (use_grouped_gemm) {
            SegmentMMGrouped<at::BFloat16, __nv_bfloat16>(A, B, C, seglen,  b_trans, CUDA_R_16BF, CUBLAS_COMPUTE_32F);
        } else {
            SegmentMM<at::BFloat16, __nv_bfloat16>(A, B, C, seglen,  b_trans, CUDA_R_16BF, CUBLAS_COMPUTE_32F);
        }
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
        if (use_grouped_gemm) {
            SegmentMMBackwardBGrouped<float,float>(A, dC, dB, seglen,CUDA_R_32F,CUBLAS_COMPUTE_32F);
        } else {
            SegmentMMBackwardB<float,float>(A, dC, dB, seglen,CUDA_R_32F,CUBLAS_COMPUTE_32F);
        }
    } else if (A.scalar_type() == at::ScalarType::Half) {
        if (use_grouped_gemm) {
            SegmentMMBackwardBGrouped<at::Half, __half>(A, dC, dB, seglen,CUDA_R_16F ,CUBLAS_COMPUTE_32F);
        } else {
            SegmentMMBackwardB<at::Half, __half>(A, dC, dB, seglen,CUDA_R_16F ,CUBLAS_COMPUTE_16F);
        }
    } else if (A.scalar_type() == at::ScalarType::BFloat16) {
        if (use_grouped_gemm) {
            SegmentMMBackwardBGrouped<at::BFloat16,__nv_bfloat16>(A, dC, dB, seglen,CUDA_R_16BF, CUBLAS_COMPUTE_32F);
        } else {
            SegmentMMBackwardB<at::BFloat16,__nv_bfloat16>(A, dC, dB, seglen,CUDA_R_16BF, CUBLAS_COMPUTE_32F);
        }
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
}

TORCH_LIBRARY_IMPL(fairchem_cpp, CUDA, m) {
    m.impl("segment_mm", segment_mm_dispatch);
    m.impl("segment_mm_backward", segment_mm_backward_dispatch);
  }

}