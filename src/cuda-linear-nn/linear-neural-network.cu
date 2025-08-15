// linear_neural_network.cu
#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>
#include "../sm-centric_macros.hh"
#include "linear_relu_linear_sigmoid.hh"

// Device function for linear forward
__device__ float linear_forward_elem(
    const float* W, const float* A, const float* b,
    int in_features, int out_features, int batch_size,
    int row, int col)
{
    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i) {
        sum += W[row * in_features + i] * A[i * batch_size + col];
    }
    return sum + b[row];
}

// Device function for ReLU
__device__ float relu_elem(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Device function for Sigmoid
__device__ float sigmoid_elem(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Device function for Tanh (optional extra activation)
__device__ float tanh_elem(float x) {
    return tanhf(x);
}

// The kernel for a deeper linear neural network
__global__ void linear_neural_network(
    // Layer 1
    const float* W1, const float* b1, int in_features1, int out_features1,
    // Layer 2
    const float* W2, const float* b2, int in_features2, int out_features2,
    // Layer 3
    const float* W3, const float* b3, int in_features3, int out_features3,
    // Layer 4
    const float* W4, const float* b4, int in_features4, int out_features4,
    // Input and output
    const float* input,              // [in_features1 x batch_size]
    float* output,                   // [out_features4 x batch_size]
    int batch_size,
    // SMC-centric parameters
    unsigned int* SMC_workerCount,
    unsigned int* SMC_newChunkSeq,
    unsigned int SMC_chunksPerSM,
    unsigned int SMC_workersNeeded,
    int* sm_usage_log,
    int N
) {
    int offsetInCTA = threadIdx.x;
    int SMC_chunkID = 0;

    SMC_Begin

    // Only allow certain SMs to execute, if using SMC logic
    if (SMC_smid >= 8) return;

    // Log which SM is processing
    if (offsetInCTA == 0) atomicAdd(&sm_usage_log[SMC_smid], 1);

    // SMC-centric scheduling (pseudo—replace with your macro or logic)
    if (SMC_chunkID * blockDim.x + threadIdx.x < N) {
        int out4_row = blockIdx.y * blockDim.y + threadIdx.y;    // Output neuron of layer 4
        int batch_col = blockIdx.x * blockDim.x + threadIdx.x;   // Batch/sample index

        if (out4_row < out_features4 && batch_col < batch_size) {
            // Layer 1: Linear + ReLU
            float lin1_out[128]; // For out_features1 <= 128
            for (int i = 0; i < out_features1; ++i) {
                lin1_out[i] = linear_forward_elem(
                    W1, input, b1,
                    in_features1, out_features1, batch_size,
                    i, batch_col
                );
            }
            for (int i = 0; i < out_features1; ++i) {
                lin1_out[i] = relu_elem(lin1_out[i]);
            }

            // Layer 2: Linear + Sigmoid
            float lin2_out[128]; // For out_features2 <= 128
            for (int i = 0; i < out_features2; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < in_features2; ++j) {
                    sum += W2[i * in_features2 + j] * lin1_out[j];
                }
                lin2_out[i] = sum + b2[i];
            }
            for (int i = 0; i < out_features2; ++i) {
                lin2_out[i] = sigmoid_elem(lin2_out[i]);
            }

            // Layer 3: Linear + Tanh (extra, slows things more)
            float lin3_out[128]; // For out_features3 <= 128
            for (int i = 0; i < out_features3; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < in_features3; ++j) {
                    sum += W3[i * in_features3 + j] * lin2_out[j];
                }
                lin3_out[i] = sum + b3[i];
            }
            for (int i = 0; i < out_features3; ++i) {
                lin3_out[i] = tanh_elem(lin3_out[i]);
            }

            // Layer 4: Linear + Sigmoid (output)
            float lin4_sum = 0.0f;
            for (int i = 0; i < in_features4; ++i) {
                lin4_sum += W4[out4_row * in_features4 + i] * lin3_out[i];
            }
            lin4_sum += b4[out4_row];
            float final_act = sigmoid_elem(lin4_sum);

            // Output
            output[out4_row * batch_size + batch_col] = final_act;
        }

        SMC_End
    }
}
