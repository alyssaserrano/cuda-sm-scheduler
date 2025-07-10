#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>

/**** MACROS ****/
// Get the ID of the current SM
#define __SMC_getSMid                     \
    uint __SMC_smid;                     \
    asm("mov.u32 %0, %smid;" : "=r"(__SMC_smid) )

// Initialize the SM-control framework
#define __SMC_init                                      \
    unsigned int __SMC_workersNeeded = __SMC_numNeeded(); \
    unsigned int* __SMC_newChunkSeq = __SMC_buildChunkSeq(); \
    unsigned int* __SMC_workerCount = __SMC_initiateArray();

// Begin macro for SM partition logic
#define __SMC_Begin                                         \
    __shared__ int __SMC_workingCTAs;                       \
    __SMC_getSMid;                                          \
    if (offsetInCTA == 0)                                   \
        __SMC_workingCTAs =                                 \
            atomicInc(&__SMC_workerCount[__SMC_smid], INT_MAX); \
    __syncthreads();                                        \
    if (__SMC_workingCTAs >= __SMC_workersNeeded) return;   \
    int __SMC_chunksPerCTA =                                \
        __SMC_chunksPerSM / __SMC_workersNeeded;            \
    int __SMC_startChunkIDidx = __SMC_smid * __SMC_chunksPerSM + \
        __SMC_workingCTAs * __SMC_chunksPerCTA;             \
    for (int __SMC_chunkIDidx = __SMC_startChunkIDidx;      \
         __SMC_chunkIDidx < __SMC_startChunkIDidx + __SMC_chunksPerCTA; \
         __SMC_chunkIDidx++) {                              \
        __SMC_chunkID = __SMC_newChunkSeq[__SMC_chunkIDidx];

// End macro for loop closure
#define __SMC_End }

// Test kernels for now
/**** GPU-side Code ****/
__global__
void kernel1(float* input, float* output, int N,
             unsigned int* SMC_chunkCount, unsigned int* SMC_newChunkSeq, unsigned int SMC_chunksPerSM) {
    __SMC_Begin // Start SM-centric logic

    __SMC_getSMid; // Retrieve SM ID

    // Ensure only SMs 0-7 execute this kernel
    if (__SMC_smid >= 8) return;

    int idx = threadIdx.x + blockDim.x * blockIdx.x; // Compute thread index (original kernel logic)

    // Replace original chunk ID with SM-centric chunkID
    int chunkID = __SMC_chunkID;

    // Process work based on SM-centric chunkID
    if (chunkID * blockDim.x + threadIdx.x < N) {
        int globalIndex = chunkID * blockDim.x + threadIdx.x; // Map chunkID to global index
        output[globalIndex] = input[globalIndex] * 2.0f; // Example computation
    }

    __SMC_End // End SM-centric logic
}

__global__
void kernel2(float* input, float* output, int N,
             unsigned int* SMC_chunkCount, unsigned int* SMC_newChunkSeq, unsigned int SMC_chunksPerSM) {
    __SMC_Begin // Start SM-centric logic

    __SMC_getSMid; // Retrieve SM ID

    // Ensure only SMs 8-15 execute this kernel
    if (__SMC_smid < 8) return;

    int idx = threadIdx.x + blockDim.x * blockIdx.x; // Compute thread index (original kernel logic)

    // Replace original chunk ID with SM-centric chunkID
    int chunkID = __SMC_chunkID;

    // Process work based on SM-centric chunkID
    if (chunkID * blockDim.x + threadIdx.x < N) {
        int globalIndex = chunkID * blockDim.x + threadIdx.x; // Map chunkID to global index
        output[globalIndex] = input[globalIndex] * 3.0f; // Example computation
    }

    __SMC_End // End SM-centric logic
}

int main(void) {
    int N = 1024; // Total workload size
    int threadsPerBlock = 256;
    unsigned int K = (N + threadsPerBlock - 1) / threadsPerBlock; // Total number of chunks (blocks)
    unsigned int M = __SMC_numNeeded(); // Number of SMs on the GPU
    unsigned int chunksPerSM = K / M;

    // Allocate and initialize SM-centric parameters
    unsigned int *SMC_chunkCount1, *SMC_newChunkSeq1, *SMC_chunkCount2, *SMC_newChunkSeq2;
    cudaMalloc(&SMC_chunkCount1, 8 * sizeof(unsigned int)); // SMs 0-7
    cudaMalloc(&SMC_newChunkSeq1, K * sizeof(unsigned int));
    cudaMalloc(&SMC_chunkCount2, 8 * sizeof(unsigned int)); // SMs 8-15
    cudaMalloc(&SMC_newChunkSeq2, K * sizeof(unsigned int));

    std::vector<unsigned int> hostChunkCount1(8, 0);
    std::vector<unsigned int> hostChunkCount2(8, 0);
    std::vector<unsigned int> hostNewChunkSeq1(K);
    std::vector<unsigned int> hostNewChunkSeq2(K);

    // Initialize chunk mapping for kernel1 (SMs 0-7)
    for (unsigned int i = 0; i < K; ++i) {
        hostNewChunkSeq1[i] = (i % 8); // Assign chunks to SMs 0-7
    }

    // Initialize chunk mapping for kernel2 (SMs 8-15)
    for (unsigned int i = 0; i < K; ++i) {
        hostNewChunkSeq2[i] = (i % 8) + 8; // Assign chunks to SMs 8-15
    }

    // Copy initialized data to device memory
    cudaMemcpy(SMC_chunkCount1, hostChunkCount1.data(), 8 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(SMC_newChunkSeq1, hostNewChunkSeq1.data(), K * sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaMemcpy(SMC_chunkCount2, hostChunkCount2.data(), 8 * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(SMC_newChunkSeq2, hostNewChunkSeq2.data(), K * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Allocate device memory for the original kernel's data
    float *d_input1, *d_output1, *d_input2, *d_output2;
    cudaMalloc(&d_input1, N * sizeof(float));
    cudaMalloc(&d_output1, N * sizeof(float));
    cudaMalloc(&d_input2, N * sizeof(float));
    cudaMalloc(&d_output2, N * sizeof(float));

    // Initialize input data on host (example)
    std::vector<float> input(N, 1.0f);
    cudaMemcpy(d_input1, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, input.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel1 (SMs 0-7)
    kernel1<<<K, threadsPerBlock>>>(d_input1, d_output1, N, SMC_chunkCount1, SMC_newChunkSeq1, chunksPerSM);

    // Launch kernel2 (SMs 8-15)
    kernel2<<<K, threadsPerBlock>>>(d_input2, d_output2, N, SMC_chunkCount2, SMC_newChunkSeq2, chunksPerSM);

    // Synchronize and cleanup
    cudaDeviceSynchronize();
    cudaFree(d_input1);
    cudaFree(d_output1);
    cudaFree(d_input2);
    cudaFree(d_output2);
    cudaFree(SMC_chunkCount1);
    cudaFree(SMC_newChunkSeq1);
    cudaFree(SMC_chunkCount2);
    cudaFree(SMC_newChunkSeq2);

    printf("DONE!\n");
    return 0;
}
