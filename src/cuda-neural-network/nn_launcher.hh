// nn_launcher.hh
#pragma once

void launchLinearReluLinearSigmoid(
    const float* d_W1, const float* d_b1, int in_features1, int out_features1,
    const float* d_W2, const float* d_b2, int in_features2, int out_features2,
    const float* d_input, float* d_output, int batch_size,
    unsigned int* d_SMC_workerCount, unsigned int* d_SMC_newChunkSeq,
    unsigned int SMC_chunksPerSM, unsigned int SMC_workersNeeded,
    int* d_sm_usage_log,
    int blocksToLaunch,
    int threadsPerBlock,
    int N
);
