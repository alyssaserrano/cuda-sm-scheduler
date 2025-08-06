// nn_launcher.cu 
#include "linear_relu_linear_sigmoid.hh"
#include <cuda_runtime.h>
#include <cstdio>
#include "../sm-centric_macros.hh"

void launchLinearReluLinearSigmoid(
    const float* d_W1, const float* d_b1, int in_features1, int out_features1,
    const float* d_W2, const float* d_b2, int in_features2, int out_features2,
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
    printf("[HOST] Launching fused kernel now...\n");
    // Launch the kernel
    linear_relu_linear_sigmoid<<<blocksToLaunch, threadsPerBlock, 0, stream>>>(
        d_W1, d_b1, in_features1, out_features1,
        d_W2, d_b2, in_features2, out_features2,
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

