// nn_launcher.hh
#pragma once

void launchLinearNeuralNetwork(
    const float* d_W1, const float* d_b1, int in_features1, int out_features1,
    const float* d_W2, const float* d_b2, int in_features2, int out_features2,
    const float* d_W3, const float* d_b3, int in_features3, int out_features3,
    const float* d_W4, const float* d_b4, int in_features4, int out_features4,
    const float* d_input, float* d_output, int batch_size,
    unsigned int* d_SMC_workerCount, unsigned int* d_SMC_newChunkSeq,
    unsigned int SMC_chunksPerSM, unsigned int SMC_workersNeeded,
    int* d_sm_usage_log,
    int blocksToLaunch,
    int threadsPerBlock,
    int N,
    cudaStream_t stream
);
