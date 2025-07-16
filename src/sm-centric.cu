#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <climits>

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
    
    printf("=== SM-Centric Kernel Execution Analysis ===\n");
    printf("Total chunks: %u, SMs: %u, Chunks per SM: %u\n", K, SMC_workersNeeded, chunksPerSM);

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

    // Allocate kernel data
    float *d_input1, *d_output1, *d_input2, *d_output2;
    cudaMalloc(&d_input1, N * sizeof(float));
    cudaMalloc(&d_output1, N * sizeof(float));
    cudaMalloc(&d_input2, N * sizeof(float));
    cudaMalloc(&d_output2, N * sizeof(float));

    std::vector<float> input(N, 1.0f);
    cudaMemcpy(d_input1, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernels
    int blocksToLaunch = std::max(K, (unsigned int)16);
    
    printf("\n=== Launching Kernels ===\n");
    printf("Blocks per kernel: %d\n", blocksToLaunch);
    printf("Threads per block: %d\n", threadsPerBlock);
    
    kernel1<<<blocksToLaunch, threadsPerBlock>>>(d_input1, d_output1, N, d_SMC_workerCount1, d_SMC_newChunkSeq1, chunksPerSM, 1, d_sm_usage_log);
    kernel2<<<blocksToLaunch, threadsPerBlock>>>(d_input2, d_output2, N, d_SMC_workerCount2, d_SMC_newChunkSeq2, chunksPerSM, 1, d_sm_usage_log);

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

    // Verify results
    std::vector<float> output1(N), output2(N);
    cudaMemcpy(output1.data(), d_output1, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output2.data(), d_output2, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("\n=== Results Verification ===\n");
    printf("Sample outputs - Kernel1: %.2f, Kernel2: %.2f\n", output1[0], output2[0]);

    // Cleanup
    cudaFree(d_input1); cudaFree(d_output1); cudaFree(d_input2); cudaFree(d_output2);
    cudaFree(d_SMC_workerCount1); cudaFree(d_SMC_newChunkSeq1);
    cudaFree(d_SMC_workerCount2); cudaFree(d_SMC_newChunkSeq2);
    cudaFree(d_sm_usage_log);
    
    delete[] SMC_newChunkSeq;
    delete[] SMC_workerCount;

    printf("\nDONE!\n");
    return 0;
}
