#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include "libsmctrl.h" // libsmctrl library

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
__global__
void compute_heavy_kernel(float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0)
        printf("[stream B] running!\n");

    if (idx < N) {
        float sum = 0.0f;
        for (int i = 0; i < 1000; ++i) {
            sum += sinf(idx * 0.001f + i);
        }
        output[idx] = sum;
    }
}

__global__
void saxpy(int n, float a, float *x, float *y)
{
        int i = blockIdx.x*blockDim.x + threadIdx.x;    // Adding the block dimension of x and thread of x together.
        if (i == 0){                             // one print per kernel launch
                printf("[stream A] running!\n");
        }

        if (i < n) y[i] = a*x[i] + y[i];                // If the sum of i is greater than n then add arrays x and y.
}

/**** CPU-side code ****/ 
int main(void){
	// Function set ups
	int N = 1<<20;                        // Bitwise operation
  	const size_t size = N * sizeof(float);
  	float *x, *y, *d_x, *d_y;
  	int numRuns = 1000;                   // For average measurement.
  	std::vector<float> latencies;         // Dynamic vector for times.

  	x = (float*)malloc(N*sizeof(float));  // Allocates memory for x and y (arrays) on host
  	y = (float*)malloc(N*sizeof(float));

  	cudaMalloc(&d_x, N*sizeof(float));    // Allocates memory for x and y on device (gpu) with pointers from host to device.
  	cudaMalloc(&d_y, N*sizeof(float));

  	for (int i = 0; i < N; i++) { // Initializes arrays
    	x[i] = 1.0f;
    	y[i] = 2.0f;
  	}

  	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);  // Copies arrays from host to device
  	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

	// 1. Number of chunks (K) and SMs (M)
	unsigned int K = 128;			// job chunks
	unsigned int M = __SMC_numNeeded();	// number of SMs

	// 2. Compute work per SM
	unsigned int chunksPerSM = K / M;

	// 3. Initialize SMC controller
	__SMC_init;

	// 4. Launch Kernels with libsmctrl
	libsmctrl_set_global_mask(~0x1ull);	// Allow work on only TPC 0

	cudaStream_t stream_A, stream_B;	// Create streams for each kernel.
	cudaStreamCreate(&stream_A);
	cudaStreamCreate(&stream_B);

	libsmctrl_set_stream_mask(stream_A, ~0xf0ull);	// disable 0-3
	libsmctrl_set_stream_mask(stream_B, ~0x0full);	// disable 4-7

	saxpy<<<(N+255)/256, 256, 0, stream_A>>>(N, 2.0f, d_x, d_y);
	compute_heavy_kernel<<<(N+255)/256, 256, 0, stream_B>>>(d_y, N);

	cudaStreamSynchronize(stream_A);
	cudaStreamSynchronize(stream_B);

	cudaDeviceSynchronize();

	printf("DONE! \n");
	cudaFree(d_x);		// Release memory of both device and host.
	cudaFree(d_y);
	free(x);
	free(y);

	return 0;
}

/**** GPU-side code ****/
__global__
void smc_kernel (unsigned int *__SMC_chunkCount, unsigned int *__SMC_chunkSeq, unsigned int __SMC_chunksPerSM){
	__SMC_Begin

	// original kernel body using __SMC_chunkID
	printf("Handling chunk %d on SM %d\n", __SMC_chunkID, __SMC_smid);

	__SMC_End
}
