#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <climits>

/**** MACROS (Based on Paper's Listing 1) ****/
// Get the ID of the current SM
#define SMC_getSMid \
    uint SMC_smid; \
    asm("mov.u32 %0, %smid;" : "=r"(SMC_smid) );

// Begin macro for SM partition logic
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

// End macro for loop closure
#define SMC_End }

/**** HELPER FUNCTIONS ****/
// Get number of SMs on the current device
unsigned int SMC_numNeeded() {
    int nSM = 0;
    cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, 0);
    return nSM;
}

// Build chunk sequence array (host-side)
unsigned int* SMC_buildChunkSeq(unsigned int totalChunks, unsigned int numSMs) {
    unsigned int* chunkSeq = new unsigned int[totalChunks];
    // Simple round-robin assignment of chunks to SMs
    for (unsigned int i = 0; i < totalChunks; i++) {
        chunkSeq[i] = i; // Original chunk ID
    }
    return chunkSeq;
}

// Initialize worker count array (host-side)
unsigned int* SMC_initiateArray(unsigned int numSMs) {
    unsigned int* workerCount = new unsigned int[numSMs];
    for (unsigned int i = 0; i < numSMs; i++) {
        workerCount[i] = 0;
    }
    return workerCount;
}

/**** GPU-side Code ****/
__global__
void kernel1(float* input, float* output, int N,
             unsigned int* SMC_workerCount, unsigned int* SMC_newChunkSeq, 
             unsigned int SMC_chunksPerSM, unsigned int SMC_workersNeeded) {
    
    // Required variables for SM-centric transformation
    int offsetInCTA = threadIdx.x;
    int SMC_chunkID = 0; // Will be set in macro loop
    
    SMC_Begin // Start SM-centric logic

    SMC_getSMid; // Retrieve SM ID

    // Ensure only SMs 0-7 execute this kernel
    if (SMC_smid >= 8) return;

    // Process work based on SM-centric chunkID
    if (SMC_chunkID * blockDim.x + threadIdx.x < N) {
        int globalIndex = SMC_chunkID * blockDim.x + threadIdx.x;
        output[globalIndex] = input[globalIndex] * 2.0f; // Example computation
    }

    SMC_End // End SM-centric logic
}

__global__
void kernel2(float* input, float* output, int N,
             unsigned int* SMC_workerCount, unsigned int* SMC_newChunkSeq, 
             unsigned int SMC_chunksPerSM, unsigned int SMC_workersNeeded) {
    
    // Required variables for SM-centric transformation
    int offsetInCTA = threadIdx.x;
    int SMC_chunkID = 0; // Will be set in macro loop
    
    SMC_Begin // Start SM-centric logic

    SMC_getSMid; // Retrieve SM ID

    // Ensure only SMs 8-15 execute this kernel
    if (SMC_smid < 8) return;

    // Process work based on SM-centric chunkID
    if (SMC_chunkID * blockDim.x + threadIdx.x < N) {
        int globalIndex = SMC_chunkID * blockDim.x + threadIdx.x;
        output[globalIndex] = input[globalIndex] * 3.0f; // Example computation
    }

    SMC_End // End SM-centric logic
}

/**** HOST-side Code ****/
int main(void) {
    int N = 1024; // Total workload size
    int threadsPerBlock = 256;
    unsigned int K = (N + threadsPerBlock - 1) / threadsPerBlock; // Total number of chunks (blocks)
    
    // SMC_init equivalent (from paper's Listing 1)
    unsigned int SMC_workersNeeded = SMC_numNeeded();
    unsigned int* SMC_newChunkSeq = SMC_buildChunkSeq(K, SMC_workersNeeded);
    unsigned int* SMC_workerCount = SMC_initiateArray(SMC_workersNeeded);
    
    unsigned int chunksPerSM = K / SMC_workersNeeded;
    if (K % SMC_workersNeeded != 0) chunksPerSM++; // Round up
    
    printf("Total chunks: %u, SMs: %u, Chunks per SM: %u\n", K, SMC_workersNeeded, chunksPerSM);

    // Allocate device memory for SM-centric parameters
    unsigned int *d_SMC_workerCount1, *d_SMC_newChunkSeq1, *d_SMC_workerCount2, *d_SMC_newChunkSeq2;
    
    // For kernel1 (SMs 0-7) - assuming at least 8 SMs
    cudaMalloc(&d_SMC_workerCount1, SMC_workersNeeded * sizeof(unsigned int));
    cudaMalloc(&d_SMC_newChunkSeq1, K * sizeof(unsigned int));
    
    // For kernel2 (SMs 8-15) - assuming at least 16 SMs
    cudaMalloc(&d_SMC_workerCount2, SMC_workersNeeded * sizeof(unsigned int));
    cudaMalloc(&d_SMC_newChunkSeq2, K * sizeof(unsigned int));

    // Copy initialized data to device memory
    cudaMemcpy(d_SMC_workerCount1, SMC_workerCount, SMC_workersNeeded * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SMC_newChunkSeq1, SMC_newChunkSeq, K * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_SMC_workerCount2, SMC_workerCount, SMC_workersNeeded * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_SMC_newChunkSeq2, SMC_newChunkSeq, K * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Allocate device memory for the kernel's data
    float *d_input1, *d_output1, *d_input2, *d_output2;
    cudaMalloc(&d_input1, N * sizeof(float));
    cudaMalloc(&d_output1, N * sizeof(float));
    cudaMalloc(&d_input2, N * sizeof(float));
    cudaMalloc(&d_output2, N * sizeof(float));

    // Initialize input data on host
    std::vector<float> input(N, 1.0f);
    cudaMemcpy(d_input1, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel1 (SMs 0-7)
    // Note: We launch with enough blocks to ensure SM coverage
    int blocksToLaunch = std::max(K, (unsigned int)16); // Ensure enough blocks for all SMs
    kernel1<<<blocksToLaunch, threadsPerBlock>>>(d_input1, d_output1, N, d_SMC_workerCount1, d_SMC_newChunkSeq1, chunksPerSM, 1);

    // Launch kernel2 (SMs 8-15)
    kernel2<<<blocksToLaunch, threadsPerBlock>>>(d_input2, d_output2, N, d_SMC_workerCount2, d_SMC_newChunkSeq2, chunksPerSM, 1);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Synchronize and check for execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Verify results (optional)
    std::vector<float> output1(N), output2(N);
    cudaMemcpy(output1.data(), d_output1, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output2.data(), d_output2, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Sample outputs - Kernel1: %.2f, Kernel2: %.2f\n", output1[0], output2[0]);

    // Cleanup
    cudaFree(d_input1);
    cudaFree(d_output1);
    cudaFree(d_input2);
    cudaFree(d_output2);
    cudaFree(d_SMC_workerCount1);
    cudaFree(d_SMC_newChunkSeq1);
    cudaFree(d_SMC_workerCount2);
    cudaFree(d_SMC_newChunkSeq2);
    
    // Free host memory
    delete[] SMC_newChunkSeq;
    delete[] SMC_workerCount;

    printf("DONE!\n");
    return 0;
}
