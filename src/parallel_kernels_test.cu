#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include "libsmctrl.h"	// libsmctrl library

// __global__ is keyword to define kernel function
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
	int i = blockIdx.x*blockDim.x + threadIdx.x;	// Adding the block dimension of x and thread of x together.
	if (i == 0){                             // one print per kernel launch
        	printf("[stream A] running!\n");
	}

	if (i < n) y[i] = a*x[i] + y[i];		// If the sum of i is greater than n then add arrays x and y.
}

int main(void){

  // SET UP FOR HOST DEVICE
  int N = 1<<20;			// Bitwise operation
  const size_t size = N * sizeof(float);
  float *x, *y, *d_x, *d_y;
  int numRuns = 1000;			// For average measurement.
  std::vector<float> latencies;		// Dynamic vector for times.

  x = (float*)malloc(N*sizeof(float));	// Allocates memory for x and y (arrays) on host
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float));	// Allocates memory for x and y on device (gpu) with pointers from host to device.
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {	// Initializes arrays
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);	// Copies arrays from host to device
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  // LIBSMCTRL INSTANTIATION
  libsmctrl_set_global_mask(~0x1ull);	// Allow work on only TPC 0 
  
  // Create streams for each kernel.
  cudaStream_t stream_A, stream_B;
  cudaStreamCreate(&stream_A);
  cudaStreamCreate(&stream_B);

  // Partition the TPCs
  libsmctrl_set_stream_mask(stream_A, ~0xf0ull);	// disable 0-3
  libsmctrl_set_stream_mask(stream_B, ~0x0full);	// disable 4-7

  // Launch kernels
  saxpy<<<(N+255)/256, 256, 0, stream_A>>>(N, 2.0f, d_x, d_y);
  compute_heavy_kernel<<<(N+255)/256, 256, 0, stream_B>>>(d_y, N);

  // Stream sync of stream A
  cudaStreamSynchronize(stream_A);
  cudaStreamSynchronize(stream_B);

  cudaDeviceSynchronize();

  // Benchmark of latency.
  /*
  for(int run = 0; run < numRuns; run++){
	  // Reset output buffer before each run
          cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

	  cudaEvent_t start, stop;	// Instantiate cuda Events
	  cudaEventCreate(&start);
	  cudaEventCreate(&stop);

	  cudaEventRecord(start, stream_A);

	  saxpy<<<(N+255)/256, 256, 0, stream_A>>>(N, 2.0f, d_x, d_y);
	  compute_heavy_kernel<<<(N + 255) / 256, 256, 0, stream_B>>>(d_y, N);

	  cudaEventRecord(stop, stream_B);

	  //cudaEventSynchronize(start);	// Optional to sync
	  cudaEventSynchronize(stop);	// Necessary to sync to get accurate measurment.

	  float dt_ms = 0;	// Reset the ms
	  cudaEventElapsedTime(&dt_ms, start, stop);
	  latencies.push_back(dt_ms);

	  cudaEventDestroy(start);
	  cudaEventDestroy(stop);
	  
  }

  // Calculate average latency.
  float sum = std::accumulate(latencies.begin(), latencies.end(), 0.0f);
  float mean = sum / latencies.size();

	float variance = 0.0f;
  for (float t : latencies)
	variance += (t - mean) * (t - mean);
  variance /= latencies.size();

  float min_latency = *std::min_element(latencies.begin(), latencies.end());
  float max_latency = *std::max_element(latencies.begin(), latencies.end());

  // Write latencies to a csv file for plotting.
  std::ofstream outfile("latencies.csv");
  for (float t: latencies) {
	  outfile << t << std::endl;
  }
  outfile.close();
	
  // Output results
  printf("Average latency: %.6f ms\n", mean);
  printf("Variance       : %.6f ms^2\n", variance);
  printf("Min latency    : %.6f ms\n", min_latency);
  printf("Max latency    : %.6f ms\n", max_latency);

  // Remaining code for dummy program.
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);	// Copies results from device to host

  // The max error should be 0.0000
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);*/

  printf("DONE! \n");
  cudaFree(d_x);	// Typical protocol of releasing memory of both device and host.
  cudaFree(d_y);
  free(x);
  free(y);

  return 0;
}
