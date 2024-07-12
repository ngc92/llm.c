/*
AdamW kernel
*/

// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"
#include <type_traits>

// ----------------------------------------------------------------------------
// CUDA kernels

// Implements linear interpolation using only two floating-point operations (as opposed to three in a naive implementation).
// Reference: https://developer.nvidia.com/blog/lerp-faster-cuda
__device__ float lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

template <typename Tp, typename Tg, typename Tm>
__device__ void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, Tm* m_memory, float* m_scales,
                             float* v_memory, size_t num_parameters,
                             float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                             float grad_scale, unsigned int seed) {
    // do we need to do any scaling of the momentum? Only if Tm != float;
    constexpr bool M_SCALING = !std::is_same_v<Tm, float>;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_parameters) { return; }  // guard

    // get the gradient, m, and v for this parameter
    float grad = grad_scale * (float)grads_memory[idx];

    Tm m_scaled = m_memory[idx];
    float m_scale = M_SCALING ? m_scales[idx/32] : 1.f;
    float m = (float)m_scaled * m_scale;

    float v = v_memory[idx];
    // update the first moment (momentum)
    m = lerp(grad, m, beta1);
    // get the largest (absolute value) in the current warp as the new scale
    if(M_SCALING) {
        float largest_m = warpReduceMax(fabs(m));
        // we want: largest / scale_factor == HALF_MAX
        // In references, I've seen m/MAX + 1e-8, but that
        // changes the algorithm a bit, I believe, effectively
        // adding a tiny decay.
        m_scale = fmax(largest_m / 65504.f, 1e-8f);
        m_memory[idx] = m / m_scale;

        if (threadIdx.x % WARP_SIZE == 0) {
            m_scales[idx / 32] = m_scale;
        }
    } else {
        m_memory[idx] = m;
    }
    // update the second moment (RMSprop)
    v = lerp(grad * grad, v, beta2);
    v_memory[idx] = v;
    m /= beta1_correction;  // m_hat
    v /= beta2_correction;  // v_hat
    // fetch the old value of this parameter as a float, from either source
    float old_param = (master_params_memory != NULL) ? master_params_memory[idx] : (float)params_memory[idx];
    // update this parameter
    float param = old_param - (learning_rate * (m / (sqrtf(v) + eps) + weight_decay * old_param));
    // update our low precision version of the parameters using stochastic rounding
    // this will be used in the next forward pass
    stochastic_rounding(param, &params_memory[idx], seed);
    // write the full, float version of the param into our master copy, if we maintain one
    // this will be used in the next update
    if (master_params_memory != NULL) { master_params_memory[idx] = param; }
}

template <typename Tp, typename Tg, typename Tm>
__global__ void adamw_kernel3(Tp* params_memory, float* master_params_memory, Tg* grads_memory, Tm* m_memory, float* m_scales,
                              float* v_memory, size_t num_parameters,
                              ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,
                              float learning_rate, float beta1, float beta2, float beta1_correction, float beta2_correction, float eps, float weight_decay,
                              float grad_scale, unsigned int seed) {
    adamw_update(params_memory + blockIdx.y * w_stride,
                 master_params_memory ? master_params_memory + blockIdx.y * s_stride : NULL,
                 grads_memory + blockIdx.y * g_stride,
                 m_memory + blockIdx.y * s_stride,
                 m_scales + blockIdx.y * s_stride / 32,
                 v_memory + blockIdx.y * s_stride,
                 num_parameters, learning_rate, beta1, beta2, beta1_correction, beta2_correction, eps, weight_decay, grad_scale,
                 seed
                 );
}

template <typename Tp, typename Tg>
void adamw_update(Tp* params_memory, float* master_params_memory, Tg* grads_memory, void* m_memory, float* m_scales, DType m_type, float* v_memory, size_t num_parameters,
                  ptrdiff_t w_stride, ptrdiff_t g_stride, ptrdiff_t s_stride,  int num_slices, float learning_rate, float beta1, float beta2, int t, float eps, float weight_decay,
                  float grad_scale, unsigned int seed, cudaStream_t stream) {
    // AdamW update
    int block_size = 512;
    int num_blocks = CEIL_DIV(num_parameters, block_size);
    float beta1_correction = 1.0f - powf(beta1, t);
    float beta2_correction = 1.0f - powf(beta2, t);
    if(m_type == DType::FP32) {
        adamw_kernel3<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>(params_memory, master_params_memory,
                                                                               grads_memory,
                                                                               (float*)m_memory, m_scales, v_memory,
                                                                               num_parameters, w_stride, g_stride,
                                                                               s_stride,
                                                                               learning_rate, beta1, beta2,
                                                                               beta1_correction, beta2_correction, eps,
                                                                               weight_decay,
                                                                               grad_scale, seed);
    } else if (m_type == DType::FP16) {
        adamw_kernel3<<<dim3(num_blocks, num_slices), block_size, 0, stream>>>(params_memory, master_params_memory,
                                                                               grads_memory,
                                                                               (__half*)m_memory, m_scales, v_memory,
                                                                               num_parameters, w_stride, g_stride,
                                                                               s_stride,
                                                                               learning_rate, beta1, beta2,
                                                                               beta1_correction, beta2_correction, eps,
                                                                               weight_decay,
                                                                               grad_scale, seed);
    }
    cudaCheck(cudaGetLastError());
}