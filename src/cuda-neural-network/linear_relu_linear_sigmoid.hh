// linear_relu_linear_sigmoid.hh
#pragma once

__global__ void linear_relu_linear_sigmoid(
    const float* W1, const float* b1, int in_features1, int out_features1,
    const float* W2, const float* b2, int in_features2, int out_features2,
    const float* input, float* output, int batch_size,
    unsigned int* SMC_workerCount, unsigned int* SMC_newChunkSeq,
    unsigned int SMC_chunksPerSM, unsigned int SMC_workersNeeded,
    int* sm_usage_log,
    int N
);

