// fused_cnn_launcher.cu
#include "fused_cnn.hh"
#include <cuda_runtime.h>
#include <cstdio>

void launchFusedNeuralNetwork(
    const float* d_conv_weights, const float* d_conv_bias,
    int C_in, int C_out, int H, int W, int K_h, int K_w,
    const float* d_bn_gamma, const float* d_bn_beta,
    const float* d_bn_mean, const float* d_bn_var,
    float bn_epsilon,
    const float* d_input, float* d_output,
    const float* d_fc_weights, const float* d_fc_bias,
    int fc_in, int num_classes,
    unsigned int* d_SMC_workerCount, unsigned int* d_SMC_newChunkSeq,
    unsigned int SMC_chunksPerSM, unsigned int SMC_workersNeeded,
    int* d_sm_usage_log,
    int blocksToLaunch,
    int threadsPerBlock,
    int batch_size,
    cudaStream_t stream
) {
    int H_out = H - K_h + 1;
    int W_out = W - K_w + 1;
    int H_pool = H_out / 2;
    int W_pool = W_out / 2;

    float* d_conv_bn_relu;
    float* d_pool_out;
    float* d_fc_out;

    cudaMalloc(&d_conv_bn_relu, C_out * H_out * W_out * sizeof(float));
    cudaMalloc(&d_pool_out, C_out * H_pool * W_pool * sizeof(float));
    cudaMalloc(&d_fc_out, num_classes * sizeof(float));

    fused_nn_kernel<<<blocksToLaunch, threadsPerBlock, 0, stream>>>(
        d_input,
        d_conv_weights, d_conv_bias,
        d_bn_gamma, d_bn_beta, d_bn_mean, d_bn_var, bn_epsilon,
        C_in, C_out, H, W, K_h, K_w,
        d_fc_weights, d_fc_bias,
        fc_in, num_classes,
        d_output,
        d_SMC_workerCount, d_SMC_newChunkSeq,
        SMC_chunksPerSM, SMC_workersNeeded,
        d_sm_usage_log,
        d_conv_bn_relu,
        d_pool_out,
        d_fc_out,
	batch_size
    );

    cudaFree(d_conv_bn_relu);
    cudaFree(d_pool_out);
    cudaFree(d_fc_out);
}
