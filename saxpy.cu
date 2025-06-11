#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>

// __global__ is keyword to define kernel function
__global__
void saxpy(int n, float a, float *x, float *y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;	// Adding the block dimension of x and thread of x together.
	if (i < n) y[i] = a*x[i] + y[i];		// If the sum of i is greater than n then add arrays x and y.
}

int main(void){

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

  // Benchmark of latency.
  for(int run = 0; run < numRuns; run++){
	  // Reset output buffer before each run
          cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);

	  cudaEvent_t start, stop;	// Instantiate cuda Events
	  cudaEventCreate(&start);
	  cudaEventCreate(&stop);

	  cudaEventRecord(start);

	  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

	  cudaEventRecord(stop);

	  cudaEventSynchronize(start);	// Optional to sync
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
  printf("SAXPY Latency Benchmark (%d runs)\n", numRuns);
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
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);	// Typical protocol of releasing memory of both device and host.
  cudaFree(d_y);
  free(x);
  free(y);

  return 0;
}
