# gpu-benchmark

This project implements 1) baseline sm partition policy and 2) novel sm partition policy. Furthermore, benchmarks GPU kernel latency variance with ml models. It is intended as a controlled dummy workload to evaluate SM/TPC partitioning strategies on NVIDIA GPUs.

The project supports latency/latency variance measurement, error checking, and can serve as a foundation for more complex GPU benchmarking.

---

## Project Purpose

The purpose of this project is to:

- Implementation of a simulated sm partition policy.
- Novel sm partition policy.
- Benchmark CUDA kernel execution latency/latency variance.
- Analyze the impact of GPU sm partitioning on performance.
- Provide a reproducible framework for evaluating SM/TPC isolation strategies.

This benchmark is suitable for Jetson-class embedded devices.

---

## Building the Benchmark

### Requirements

- A system with an NVIDIA GPU
- CUDA Toolkit (version 12.6 or compatible)
- `nvcc` compiler available in your system path

### Build Instructions

To compile the benchmark:

```bash
nvcc -o saxpy saxpy.cu

