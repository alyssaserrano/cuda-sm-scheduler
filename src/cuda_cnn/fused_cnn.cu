// fused_cnn.cu
#include <stdio.h>
#include "sm-centric_macros.hh"
#include <cuda_runtime.h>
#include <math.h>

__device__ float convolution_forward_elem(
    const float* input,
    const float* weights,
    const float* bias,
    int C_in, int C_out,
    int H, int W,
    int K_h, int K_w,
    int out_channel, int h, int w)
{
    float sum = 0.0f;
    for (int c = 0; c < C_in; ++c)
        for (int kh = 0; kh < K_h; ++kh)
            for (int kw = 0; kw < K_w; ++kw) {
                int in_h = h + kh;
                int in_w = w + kw;
                sum += weights[out_channel * (C_in * K_h * K_w) + c * (K_h * K_w) + kh * K_w + kw] *
                       input[c * (H * W) + in_h * W + in_w];
            }
    sum += bias[out_channel];
    return sum;
}

__device__ float batchnorm_forward_elem(
    float x,
    float gamma, float beta,
    float mean, float var, float epsilon)
{
    return gamma * ((x - mean) / sqrtf(var + epsilon)) + beta;
}

static __device__ float relu_elem(float x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ float maxpool2x2_forward_elem(
    const float* input, int H, int W, int h, int w)
{
    float maxval = -INFINITY;
    for (int kh = 0; kh < 2; ++kh)
        for (int kw = 0; kw < 2; ++kw) {
            int in_h = h * 2 + kh;
            int in_w = w * 2 + kw;
            if (in_h < H && in_w < W)
                maxval = fmaxf(maxval, input[in_h * W + in_w]);
        }
    return maxval;
}

__device__ float fc_forward_elem(
    const float* input,
    const float* weights,
    const float* bias,
    int out_feature, int in_features)
{
    float sum = 0.0f;
    for (int i = 0; i < in_features; ++i)
        sum += weights[out_feature * in_features + i] * input[i];
    sum += bias[out_feature];
    return sum;
}

__device__ void softmax_forward(
    const float* input, float* output, int num_classes)
{
    float maxval = -INFINITY;
    for (int i = 0; i < num_classes; ++i)
        maxval = fmaxf(maxval, input[i]);
    float sum = 0.0f;
    for (int i = 0; i < num_classes; ++i)
        sum += expf(input[i] - maxval);
    for (int i = 0; i < num_classes; ++i)
        output[i] = expf(input[i] - maxval) / sum;
}

__global__ void fused_nn_kernel(
    const float* input,
    const float* conv_weights, const float* conv_bias,
    const float* bn_gamma, const float* bn_beta,
    const float* bn_mean, const float* bn_var, float bn_epsilon,
    int C_in, int C_out,
    int H, int W, int K_h, int K_w,
    const float* fc_weights, const float* fc_bias,
    int fc_in, int num_classes,
    float* output,
    unsigned int* SMC_workerCount, unsigned int* SMC_newChunkSeq,
    unsigned int SMC_chunksPerSM, unsigned int SMC_workersNeeded,
    int* sm_usage_log,
    float* conv_bn_relu,
    float* pool_out,
    float* fc_out,
    int batch_size 
) {
    int batch_idx = blockIdx.x; // One block per batch element
    if (batch_idx >= batch_size) return;

    int offsetInCTA = threadIdx.x;
    int SMC_chunkID = 0;

    SMC_Begin

    // Only run on SMS >= 8
    if (SMC_smid < 8) return;

    if (offsetInCTA == 0) atomicAdd(&sm_usage_log[SMC_smid + 16], 1);

    // 1. Convolution + BatchNorm + ReLU
    int H_out = H - K_h + 1;
    int W_out = W - K_w + 1;

    // Pointers to per-batch input and intermediates
    const float* input_batch = input + batch_idx * C_in * H * W;
    float* conv_bn_relu_batch = conv_bn_relu + batch_idx * C_out * H_out * W_out;
    float* pool_out_batch = pool_out + batch_idx * C_out * (H_out / 2) * (W_out / 2);
    float* fc_out_batch = fc_out + batch_idx * num_classes;

    for (int c = 0; c < C_out; ++c)
        for (int h = 0; h < H_out; ++h)
            for (int w = 0; w < W_out; ++w) {
                float conv_val = convolution_forward_elem(
                    input_batch, conv_weights, conv_bias,
                    C_in, C_out, H, W, K_h, K_w,
                    c, h, w);
                float bn_val = batchnorm_forward_elem(conv_val, bn_gamma[c], bn_beta[c], bn_mean[c], bn_var[c], bn_epsilon);
                conv_bn_relu_batch[c * (H_out * W_out) + h * W_out + w] = relu_elem(bn_val);
            }

    // 2. MaxPool (for each channel)
    int H_pool = H_out / 2;
    int W_pool = W_out / 2;
    for (int c = 0; c < C_out; ++c)
        for (int h = 0; h < H_pool; ++h)
            for (int w = 0; w < W_pool; ++w)
                pool_out_batch[c * (H_pool * W_pool) + h * W_pool + w] =
                    maxpool2x2_forward_elem(
                        &conv_bn_relu_batch[c * (H_out * W_out)],
                        H_out, W_out, h, w);

    // 3. Flatten for FC
    for (int cls = 0; cls < num_classes; ++cls)
        fc_out_batch[cls] = fc_forward_elem(
            pool_out_batch, fc_weights, fc_bias, cls, fc_in);

    // 4. Softmax
    softmax_forward(fc_out_batch, fc_out_batch, num_classes);

    // Output: Write to global output for this batch
    for (int c = 0; c < num_classes; ++c) {
        output[batch_idx * num_classes + c] = fc_out_batch[c];
    }

    SMC_End
}
