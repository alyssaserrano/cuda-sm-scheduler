//linear_relu_linear_sigmoid.hh
#pragma once
#include <cuda_runtime.h>

// Host launcher declaration (add SMC args if needed)
void launchLinearReluLinearSigmoid(
    const float* d_W1, const float* d_b1, int in_features1, int out_features1,
    const float* d_W2, const float* d_b2, int in_features2, int out_features2,
    const float* d_input, float* d_output, int batch_size,
    unsigned int* d_SMC_workerCount,
    unsigned int* d_SMC_newChunkSeq,
    unsigned int SMC_chunksPerSM,
    unsigned int SMC_workersNeeded,
    int* d_sm_usage_log,
    int N,
    cudaStream_t stream
);

// Kernel declaration (must match your .cu implementation)
__global__ void linear_relu_linear_sigmoid(
    const float* W1, const float* b1, int in_features1, int out_features1,
    const float* W2, const float* b2, int in_features2, int out_features2,
    const float* input, float* output, int batch_size,
    unsigned int* SMC_workerCount,
    unsigned int* SMC_newChunkSeq,
    unsigned int SMC_chunksPerSM,
    unsigned int SMC_workersNeeded,
    int* sm_usage_log,
    int N
);
