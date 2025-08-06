// fused_cnn.hh
#pragma once

#include <cuda_runtime.h>

void launchFusedNeuralNetwork(
    // Convolution params
    const float* d_conv_weights, const float* d_conv_bias,
    int C_in, int C_out, int H, int W, int K_h, int K_w,

    // BatchNorm params
    const float* d_bn_gamma, const float* d_bn_beta,
    const float* d_bn_mean, const float* d_bn_var,
    float bn_epsilon,

    // Input/output
    const float* d_input, float* d_output,

    // Fully Connected params
    const float* d_fc_weights, const float* d_fc_bias,
    int fc_in, int num_classes,

    // SM-centric scheduler params
    unsigned int* d_SMC_workerCount, unsigned int* d_SMC_newChunkSeq,
    unsigned int SMC_chunksPerSM, unsigned int SMC_workersNeeded,
    int* d_sm_usage_log,

    // Kernel launch params
    int blocksToLaunch,
    int threadsPerBlock,
    int batch_size,

    cudaStream_t stream
);

__global__ void fused_nn_kernel(
    const float* input,
    const float* conv_weights, const float* conv_bias,
    const float* bn_gamma, const float* bn_beta,
    const float* bn_mean, const float* bn_var, float bn_epsilon,
    int C_in, int C_out,
    int H, int W, int K_h, int K_w,
    const float* fc_weights, const float* fc_bias,
    int fc_in, int num_classes,
    float* output,
    unsigned int* SMC_workerCount, unsigned int* SMC_newChunkSeq,
    unsigned int SMC_chunksPerSM, unsigned int SMC_workersNeeded,
    int* sm_usage_log,
    float* conv_bn_relu,
    float* pool_out,
    float* fc_out,
    int batch_size
);
