// nn_launcher.cu 
#include "linear_relu_linear_sigmoid.hh"
#include <cuda_runtime.h>
#include <cstdio>
#include "../sm-centric_macros.hh"

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
)
{
    // Launch the kernel
    linear_neural_network<<<blocksToLaunch, threadsPerBlock, 0, stream>>>(
        d_W1, d_b1, in_features1, out_features1,
        d_W2, d_b2, in_features2, out_features2,
        d_W3, d_b3, in_features3, out_features3,
        d_W4, d_b4, in_features4, out_features4,
        d_input, d_output, batch_size,
        d_SMC_workerCount, d_SMC_newChunkSeq,
        SMC_chunksPerSM, SMC_workersNeeded,
        d_sm_usage_log,
        N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[HOST] CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    } else {
        printf("[HOST] Kernel launch was successful.\n");
    }
}
