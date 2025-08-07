// sm-centric.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <climits>
#include <iostream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <random>
#include "libsmctrl.h"
#include "sm-centric_macros.hh"
#include "cuda-neural-network/linear_relu_linear_sigmoid.hh"
#include "cuda-neural-network/nn_launcher.hh"
#include "cuda_cnn/fused_cnn.hh"

/**** MACROS ****/
/*#define SMC_init(K) \
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

#define SMC_End }*/

/**** HELPER FUNCTIONS ****/
void print_timestamp(const char* label) {
    using std::chrono::system_clock;
    auto now = system_clock::now();
    std::time_t now_time = system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::cout << "[" << label << "] "
              << std::put_time(std::localtime(&now_time), "%F %T")
              << "." << std::setfill('0') << std::setw(3) << ms.count()
              << std::endl;
}

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
/*__global__
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
}*/

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

    // Fused CNN Neural Network
    int C_in = 3, C_out = 8, H = 32, W = 32, K_h = 3, K_w = 3;
    float bn_epsilon = 1e-5;
    int fc_in = (H/2)*(W/2)*C_out; // after 2x2 maxpool, adjust as needed
    int num_classes = 10;
    int cnn_batch_size = batch_size; // or set as needed

    std::vector<float> conv_weights(C_out * C_in * K_h * K_w);
    std::vector<float> conv_bias(C_out);
    std::vector<float> bn_gamma(C_out, 1.0f);
    std::vector<float> bn_beta(C_out, 0.0f);
    std::vector<float> bn_mean(C_out, 0.0f);
    std::vector<float> bn_var(C_out, 1.0f);
    std::vector<float> cnn_input(C_in * H * W * cnn_batch_size);
    std::vector<float> fc_weights(num_classes * fc_in);
    std::vector<float> fc_bias(num_classes);
    std::vector<float> cnn_output(num_classes * cnn_batch_size, 0.0f);

    // Random initialization for weights, biases, and input
    std::mt19937 rng(42); // fixed seed for reproducibility
    std::uniform_real_distribution<float> wdist(-0.1f, 0.1f);
    std::uniform_real_distribution<float> idist(0.0f, 1.0f);

    for (auto& w : conv_weights) w = wdist(rng);
    for (auto& b : conv_bias) b = wdist(rng); // or 0.0f if you prefer
    for (auto& x : cnn_input) x = idist(rng);
    for (auto& w : fc_weights) w = wdist(rng);
    for (auto& b : fc_bias) b = wdist(rng); // or 0.0f if you prefer

    float *d_conv_weights, *d_conv_bias, *d_bn_gamma, *d_bn_beta, *d_bn_mean, *d_bn_var;
    float *d_cnn_input, *d_fc_weights, *d_fc_bias, *d_cnn_output;

    cudaMalloc(&d_conv_weights, conv_weights.size() * sizeof(float));
    cudaMalloc(&d_conv_bias, conv_bias.size() * sizeof(float));
    cudaMalloc(&d_bn_gamma, bn_gamma.size() * sizeof(float));
    cudaMalloc(&d_bn_beta, bn_beta.size() * sizeof(float));
    cudaMalloc(&d_bn_mean, bn_mean.size() * sizeof(float));
    cudaMalloc(&d_bn_var, bn_var.size() * sizeof(float));
    cudaMalloc(&d_cnn_input, cnn_input.size() * sizeof(float));
    cudaMalloc(&d_fc_weights, fc_weights.size() * sizeof(float));
    cudaMalloc(&d_fc_bias, fc_bias.size() * sizeof(float));
    cudaMalloc(&d_cnn_output, cnn_output.size() * sizeof(float));

    cudaMemcpy(d_conv_weights, conv_weights.data(), conv_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv_bias, conv_bias.data(), conv_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_gamma, bn_gamma.data(), bn_gamma.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_beta, bn_beta.data(), bn_beta.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_mean, bn_mean.data(), bn_mean.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bn_var, bn_var.data(), bn_var.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cnn_input, cnn_input.data(), cnn_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc_weights, fc_weights.data(), fc_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc_bias, fc_bias.data(), fc_bias.size() * sizeof(float), cudaMemcpyHostToDevice);

    /*** LAUNCH KERNELS USING LIBSMCTRL LIBRARY***/
    int blocksToLaunch = std::max(K, (unsigned int)16) / 2;	// Had to divide by 2 since libsmctrl is a TPC/SM relationship.
    								// Therefore diving by 2 ensures that we split the amount of work blocks
    								// we need to do for each set of SMs.

    // LIBSMCTRL INSTANTIATION
    libsmctrl_set_global_mask(~0x1ull);	// Allow work on only TPC 0

    cudaStream_t stream_A, stream_B;
    cudaStreamCreate(&stream_A);
    cudaStreamCreate(&stream_B);

    // Partition TPCs for each kernel
    libsmctrl_set_stream_mask(stream_B, ~0xf0ull);	// disable 0-3
    libsmctrl_set_stream_mask(stream_A, ~0x0full);	// disable 4-7

    printf("\n=== Launching Kernels ===\n");
    printf("Blocks per kernel: %d\n", blocksToLaunch);
    printf("Threads per block: %d\n", threadsPerBlock);
   
    // Number of Latency runs
    const int numRuns = 100;
    std::vector<float> latencies_kernel1, latencies_kernel2;

    for(int run = 0; run < numRuns; run++){
	  // Measure latency for linear_relu_linear_sigmoid neural network.
	  cudaEvent_t start1, stop1;
	  cudaEventCreate(&start1);
	  cudaEventCreate(&stop1);

	  cudaEventRecord(start1, stream_A);

    //printf("Launching kernel with %d blocks. Expecting %u workers needed.\n", blocksToLaunch, SMC_numNeeded());
    // Launch the fused neural network kernel using the launcher
    print_timestamp("Launching LinearReluLinearSigmoid kernel");
    launchLinearReluLinearSigmoid(
        d_W1, d_b1, in_features1, out_features1,
        d_W2, d_b2, in_features2, out_features2,
        d_nn_input, d_nn_output, batch_size,
        d_SMC_workerCount1, d_SMC_newChunkSeq1,
        chunksPerSM, SMC_workersNeeded,
        d_sm_usage_log,
        blocksToLaunch,
        threadsPerBlock,
        N,
	stream_A
    );

    cudaEventRecord(stop1, stream_A);
    cudaEventSynchronize(stop1);

    float ms1 = 0;
    cudaEventElapsedTime(&ms1, start1, stop1);
    latencies_kernel1.push_back(ms1);

    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);

    // Measure latency for FusedNeuralNetwork kernel
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2, stream_B);

    // Launch fused CNN kernel
    //print_timestamp("Launching FusedCNN kernel");
    launchFusedNeuralNetwork(
        d_conv_weights, d_conv_bias,
        C_in, C_out, H, W, K_h, K_w,
        d_bn_gamma, d_bn_beta, d_bn_mean, d_bn_var, bn_epsilon,
        d_cnn_input, d_cnn_output,
        d_fc_weights, d_fc_bias,
        fc_in, num_classes,
        d_SMC_workerCount2, d_SMC_newChunkSeq2,
        chunksPerSM, SMC_workersNeeded,
        d_sm_usage_log,
        blocksToLaunch,
        threadsPerBlock,
        cnn_batch_size,
	stream_B
    );

    cudaEventRecord(stop2, stream_B);
    cudaEventSynchronize(stop2);

    float ms2 = 0;
    cudaEventElapsedTime(&ms2, start2, stop2);
    latencies_kernel2.push_back(ms2);

    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    } 


    // Write results to CSV
    std::ofstream outfile("sm_centric_latencies.csv");
    outfile << "LinearReluLinearSigmoid,FusedCNN\n";
    for (int i = 0; i < numRuns; ++i) {
        outfile << latencies_kernel1[i] << "," << latencies_kernel2[i] << "\n";
    }
    outfile.close();

    // Post-processing
    cudaStreamSynchronize(stream_A);
    cudaStreamSynchronize(stream_B);

    // Copy and print neural network output
    std::vector<float> h_nn_output(out_features2 * batch_size);
    cudaMemcpy(h_nn_output.data(), d_nn_output, out_features2 * batch_size * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n=== Neural Network Output ===\n");
    for (int i = 0; i < out_features2; ++i) {
        printf("Class %d: ", i);
        for (int j = 0; j < batch_size; ++j) {
            printf("%8.5f ", h_nn_output[i * batch_size + j]);
        }
        printf("\n");
    }

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
    cudaStreamDestroy(stream_A);
    cudaStreamDestroy(stream_B);
    cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_nn_input); cudaFree(d_nn_output);
    cudaFree(d_SMC_workerCount1); cudaFree(d_SMC_newChunkSeq1);
    cudaFree(d_SMC_workerCount2); cudaFree(d_SMC_newChunkSeq2);
    cudaFree(d_sm_usage_log);

    // Cleanup for CNN network
    cudaFree(d_conv_weights); cudaFree(d_conv_bias);
    cudaFree(d_bn_gamma); cudaFree(d_bn_beta);
    cudaFree(d_bn_mean); cudaFree(d_bn_var);
    cudaFree(d_cnn_input); cudaFree(d_fc_weights);
    cudaFree(d_fc_bias); cudaFree(d_cnn_output);
    
    delete[] SMC_newChunkSeq;
    delete[] SMC_workerCount;

    printf("\nDONE!\n");
    return 0;
}
