// linear_relu_linear_sigmoid.cu
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

// The fused kernel
__global__ void linear_relu_linear_sigmoid(
    // First linear layer weights, bias, dims
    const float* W1, const float* b1, int in_features1, int out_features1,
    // Second linear layer weights, bias, dims
    const float* W2, const float* b2, int in_features2, int out_features2,
    // Input and output
    const float* input,              // [in_features1 x batch_size]
    float* output,                   // [out_features2 x batch_size]
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
	    
    //if (threadIdx.x == 0) {
        //printf("LOGGING: block %d on SM %d\n", blockIdx.x, SMC_smid);
        //atomicAdd(&sm_usage_log[SMC_smid], 1);
    //}

    // Example: log which SM is processing (if using macro for SMC_smid)
    if (offsetInCTA == 0) atomicAdd(&sm_usage_log[SMC_smid], 1);

    // SMC-centric scheduling (pseudo—replace with your macro or logic)
    if (SMC_chunkID * blockDim.x + threadIdx.x < N) {
        int out2_row = blockIdx.y * blockDim.y + threadIdx.y;    // Output neuron of second linear
        int batch_col = blockIdx.x * blockDim.x + threadIdx.x;   // Batch/sample index

        if (out2_row < out_features2 && batch_col < batch_size) {
            // Step 1: First Linear Layer
            float lin1_out[128]; // For out_features1 <= 128
            for (int i = 0; i < out_features1; ++i) {
                lin1_out[i] = linear_forward_elem(
                    W1, input, b1,
                    in_features1, out_features1, batch_size,
                    i, batch_col
                );
            }

            // Step 2: ReLU
            for (int i = 0; i < out_features1; ++i) {
                lin1_out[i] = relu_elem(lin1_out[i]);
            }

            // Step 3: Second Linear Layer
            float lin2_sum = 0.0f;
            for (int i = 0; i < in_features2; ++i) {
                lin2_sum += W2[out2_row * in_features2 + i] * lin1_out[i];
            }
            lin2_sum += b2[out2_row];

            // Step 4: Sigmoid
            float final_act = sigmoid_elem(lin2_sum);

            // Output
            output[out2_row * batch_size + batch_col] = final_act;

	    // Print for debugging
            //printf("final_act (row %d, col %d): %f\n", out2_row, batch_col, final_act);
        }

    SMC_End
   }
}
