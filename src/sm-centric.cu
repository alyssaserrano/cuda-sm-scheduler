// sm-centric.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <climits>
#include "cuda-neural-network/linear_relu_linear_sigmoid.hh"
#include "cuda-neural-network/nn_launcher.hh"

/**** MACROS ****/
#define SMC_init(K) \
	unsigned int SMC_workersNeeded = SMC_numNeeded(); \
	unsigned int* SMC_newChunkSeq = SMC_buildChunkSeq((K), SMC_workersNeeded); \
	unsigned int* SMC_workerCount = SMC_initiateArray(SMC_workersNeeded);

#define SMC_getSMid \
    uint SMC_smid; \
    asm("mov.u32 %0, %smid;" : "=r"(SMC_smid) );

#define SMC_Begin \
    __shared__ int SMC_workingCTAs; \
    SMC_getSMid; \
    if (offsetInCTA == 0) \
        SMC_workingCTAs = atomicInc(&SMC_workerCount[SMC_smid], INT_MAX); \
    __syncthreads(); \
    if (SMC_workingCTAs >= SMC_workersNeeded) return; \
    int SMC_chunksPerCTA = SMC_chunksPerSM / SMC_workersNeeded; \
    int SMC_startChunkIDidx = SMC_smid * SMC_chunksPerSM + SMC_workingCTAs * SMC_chunksPerCTA; \
    for (int SMC_chunkIDidx = SMC_startChunkIDidx; \
         SMC_chunkIDidx < SMC_startChunkIDidx + SMC_chunksPerCTA; \
         SMC_chunkIDidx++) { \
        SMC_chunkID = SMC_newChunkSeq[SMC_chunkIDidx];

#define SMC_End }

/**** HELPER FUNCTIONS ****/
unsigned int SMC_numNeeded() {
    int nSM = 0;
    cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, 0);
    return nSM;
}

unsigned int* SMC_buildChunkSeq(unsigned int totalChunks, unsigned int numSMs) {
    unsigned int* chunkSeq = new unsigned int[totalChunks];
    for (unsigned int i = 0; i < totalChunks; i++) {
        chunkSeq[i] = i;
    }
    return chunkSeq;
}

unsigned int* SMC_initiateArray(unsigned int numSMs) {
    unsigned int* workerCount = new unsigned int[numSMs];
    for (unsigned int i = 0; i < numSMs; i++) {
        workerCount[i] = 0;
    }
    return workerCount;
}


/**** GPU-side Code with SM Logging ****/
__global__
void kernel1(float* input, float* output, int N,
             unsigned int* SMC_workerCount, unsigned int* SMC_newChunkSeq, 
             unsigned int SMC_chunksPerSM, unsigned int SMC_workersNeeded,
             int* sm_usage_log) {
    
    int offsetInCTA = threadIdx.x;
    int SMC_chunkID = 0;
    
    SMC_Begin

    // Ensure only SMs 0-7 execute this kernel
    if (SMC_smid >= 8) return;

    // Log which SM is processing this kernel
    if (offsetInCTA == 0) {
        atomicAdd(&sm_usage_log[SMC_smid], 1);
        // Optional: print from device (can be noisy)
        // printf("Kernel1: SM %d, Block %d, Working CTAs %d\n", SMC_smid, blockIdx.x, SMC_workingCTAs);
    }

    // Process work
    if (SMC_chunkID * blockDim.x + threadIdx.x < N) {
        int globalIndex = SMC_chunkID * blockDim.x + threadIdx.x;
        output[globalIndex] = input[globalIndex] * 2.0f;

	// Check SM IDs that completed the work
        //printf("SM ID: %d processed work in kernel 1, Block %d, Working CTAs %d\n", SMC_smid, blockIdx.x, SMC_workingCTAs);
        if (offsetInCTA == 0) {
                printf("SM ID: %d processed work in kernel 1\n", SMC_smid);
        }
    }

    SMC_End
}

__global__
void kernel2(float* input, float* output, int N,
             unsigned int* SMC_workerCount, unsigned int* SMC_newChunkSeq, 
             unsigned int SMC_chunksPerSM, unsigned int SMC_workersNeeded,
             int* sm_usage_log) {
    
    int offsetInCTA = threadIdx.x;
    int SMC_chunkID = 0;
    
    SMC_Begin

    // Ensure only SMs 8-15 execute this kernel
    if (SMC_smid < 8) return;

    // Log which SM is processing this kernel
    if (offsetInCTA == 0) {
        atomicAdd(&sm_usage_log[SMC_smid + 16], 1); // Offset by 16 to separate from kernel1
        // Optional: print from device
        // printf("Kernel2: SM %d, Block %d, Working CTAs %d\n", SMC_smid, blockIdx.x, SMC_workingCTAs);
    }

    // Process work
    if (SMC_chunkID * blockDim.x + threadIdx.x < N) {
        int globalIndex = SMC_chunkID * blockDim.x + threadIdx.x;
        output[globalIndex] = input[globalIndex] * 3.0f;

	// Check SM IDs that completed the work
	//printf("SM ID: %d processed work in kernel 2, Block %d, Working CTAs %d\n", SMC_smid, blockIdx.x, SMC_workingCTAs);
	if (offsetInCTA == 0) {
		printf("SM ID: %d processed work in kernel 2\n", SMC_smid);
        }
    }

    SMC_End
}

int main(void) {
    int N = 1024;
    int threadsPerBlock = 256;
    unsigned int K = (N + threadsPerBlock - 1) / threadsPerBlock;

    // SM-centric initialization
    SMC_init(K);
    
    unsigned int chunksPerSM = K / SMC_workersNeeded;
    if (K % SMC_workersNeeded != 0) chunksPerSM++;

    // Sanity check
    printf("=== SM-Centric Kernel Execution Analysis ===\n");
    printf("Total Jobs: %u, Total chunks: %u, workers: %u, Workers per SM: %u\n",N , K, SMC_workersNeeded, chunksPerSM);

    // Allocate SM usage logging arrays
    int *d_sm_usage_log;
    cudaMalloc(&d_sm_usage_log, 32 * sizeof(int)); // 16 SMs * 2 kernels
    cudaMemset(d_sm_usage_log, 0, 32 * sizeof(int));

    // Allocate device memory for SM-centric parameters
    unsigned int *d_SMC_workerCount1, *d_SMC_newChunkSeq1, *d_SMC_workerCount2, *d_SMC_newChunkSeq2;
    
    cudaMalloc(&d_SMC_workerCount1, SMC_workersNeeded * sizeof(unsigned int));
    cudaMalloc(&d_SMC_newChunkSeq1, K * sizeof(unsigned int));
    cudaMalloc(&d_SMC_workerCount2, SMC_workersNeeded * sizeof(unsigned int));
    cudaMalloc(&d_SMC_newChunkSeq2, K * sizeof(unsigned int));

    // Copy data to device
    cudaMemcpy(d_SMC_workerCount1, SMC_workerCount, SMC_workersNeeded * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SMC_newChunkSeq1, SMC_newChunkSeq, K * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SMC_workerCount2, SMC_workerCount, SMC_workersNeeded * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SMC_newChunkSeq2, SMC_newChunkSeq, K * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Linear-Relu-Linear-Sigmoid Neural network dimensions
    int in_features1 = 8, out_features1 = 16;
    int in_features2 = out_features1, out_features2 = 4;
    int batch_size = N / in_features1; // Make sure N is divisible by in_features1!
    int NN = batch_size * in_features1;

    // Allocate and initialize host memory for weights, biases, input, output
    std::vector<float> W1(in_features1 * out_features1, 0.05f);
    std::vector<float> b1(out_features1, 0.0f);
    std::vector<float> W2(in_features2 * out_features2, 0.1f);
    std::vector<float> b2(out_features2, 0.0f);
    std::vector<float> nn_input(NN, 1.0f);
    std::vector<float> nn_output(out_features2 * batch_size, 0.0f);

    // Allocate device memory for neural network
    float *d_W1, *d_b1, *d_W2, *d_b2, *d_nn_input, *d_nn_output;
    cudaMalloc(&d_W1, in_features1 * out_features1 * sizeof(float));
    cudaMalloc(&d_b1, out_features1 * sizeof(float));
    cudaMalloc(&d_W2, in_features2 * out_features2 * sizeof(float));
    cudaMalloc(&d_b2, out_features2 * sizeof(float));
    cudaMalloc(&d_nn_input, NN * sizeof(float));
    cudaMalloc(&d_nn_output, out_features2 * batch_size * sizeof(float));

    // Copy host data to device
    cudaMemcpy(d_W1, W1.data(), in_features1 * out_features1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1.data(), out_features1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2.data(), in_features2 * out_features2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2.data(), out_features2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nn_input, nn_input.data(), NN * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    int blocksToLaunch = std::max(K, (unsigned int)16);
    
    printf("\n=== Launching Kernels ===\n");
    printf("Blocks per kernel: %d\n", blocksToLaunch);
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Launching kernel with %d blocks. Expecting %u workers needed.\n", blocksToLaunch, SMC_numNeeded());
    
    // Launch the fused neural network kernel using the launcher
    launchLinearReluLinearSigmoid(
        d_W1, d_b1, in_features1, out_features1,
        d_W2, d_b2, in_features2, out_features2,
        d_nn_input, d_nn_output, batch_size,
        d_SMC_workerCount1, d_SMC_newChunkSeq1, // <--- reused here
        chunksPerSM, SMC_workersNeeded,
        d_sm_usage_log,
        blocksToLaunch,
        threadsPerBlock,
        N
    );

    cudaDeviceSynchronize();

    // Copy back and analyze SM usage
    std::vector<int> sm_usage_log(32);
    cudaMemcpy(sm_usage_log.data(), d_sm_usage_log, 32 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n=== SM Usage Analysis ===\n");
    printf("Kernel1 (should use SMs 0-7):\n");
    for (int i = 0; i < 16; i++) {
        if (sm_usage_log[i] > 0) {
            printf("  SM %d: %d CTAs processed\n", i, sm_usage_log[i]);
        }
    }
    
    printf("\nKernel2 (should use SMs 8-15):\n");
    for (int i = 16; i < 32; i++) {
        if (sm_usage_log[i] > 0) {
            printf("  SM %d: %d CTAs processed\n", i-16, sm_usage_log[i]);
        }
    }

    // Cleanup
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_nn_input); cudaFree(d_nn_output);
    cudaFree(d_SMC_workerCount1); cudaFree(d_SMC_newChunkSeq1);
    cudaFree(d_SMC_workerCount2); cudaFree(d_SMC_newChunkSeq2);
    cudaFree(d_sm_usage_log);
    
    delete[] SMC_newChunkSeq;
    delete[] SMC_workerCount;

    printf("\nDONE!\n");
    return 0;
}
