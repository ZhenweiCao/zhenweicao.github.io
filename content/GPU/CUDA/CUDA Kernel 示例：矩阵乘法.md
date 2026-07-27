---
aliases:
  - matmul.cu
updated: 2026-05-16
tags:
  - gpu-computing
  - cuda-programming
  - gemm-optimization
  - gpu-programming
---
# CUDA Kernel 示例：矩阵乘法

相关主笔记：

- [[GPU 知识库索引]]
- [[CUDA 编程基础]]
- [[CUDA 线程配置与占用率]]
- [[CUDA Shared Memory 与 Bank Conflict]]
- [[CUDA GEMM 矩阵乘法优化指南]]

文件：`matmul.cu`

## 概述

实现两个矩阵乘法的 CUDA kernel，包含：
- **Naive 版本**：基础实现，每个线程计算 C 的一个元素
- **共享内存优化版本**：使用分块 (tiling) 技术，利用共享内存减少全局内存访问

矩阵约定：

```text
A: M x K
B: K x N
C: M x N
C[row, col] = sum_k A[row, k] * B[k, col]
```

这份代码适合作为教学骨架。它没有使用 register tiling、vectorized load、double buffering、Tensor Core 或 CUTLASS/CuTe，因此不应拿它和 cuBLAS 直接比较生产性能。完整优化路线见 [[CUDA GEMM 矩阵乘法优化指南]]。

## 性能边界

| 版本 | 优点 | 局限 |
|------|------|------|
| Naive | 代码最直观，便于验证索引 | 全局内存复用差，访存瓶颈明显 |
| Shared Memory Tiling | A/B tile 在 block 内复用，减少 HBM 访问 | 每线程只算一个输出，算术强度仍低 |
| 高性能 GEMM | 需要 register tiling、Tensor Core、流水线 | 代码复杂，建议参考 CUTLASS / cuBLAS |

## 完整代码

```cpp
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// 错误检查宏
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
            }                                                                 \
    } while (0)

// 基础矩阵乘法 kernel (naive 实现)
template <typename T>
__global__ void matmul_naive(const T *A, const T *B, T *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        T sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 共享内存优化的矩阵乘法 kernel
template <typename T, int BLOCK_SIZE>
__global__ void matmul_shared(const T *A, const T *B, T *C, int M, int N, int K) {
    __shared__ T sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ T sB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    T sum = 0;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // 加载 A 的块到共享内存
        if (row < M && t * BLOCK_SIZE + tx < K) {
            sA[ty][tx] = A[row * K + t * BLOCK_SIZE + tx];
        } else {
            sA[ty][tx] = 0;
        }

        // 加载 B 的块到共享内存
        if (t * BLOCK_SIZE + ty < K && col < N) {
            sB[ty][tx] = B[(t * BLOCK_SIZE + ty) * N + col];
        } else {
            sB[ty][tx] = 0;
        }

        __syncthreads();

        // 计算部分和
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 主机端矩阵乘法（用于验证结果）
template <typename T>
void matmul_cpu(const T *A, const T *B, T *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            T sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// 初始化矩阵
template <typename T>
void init_matrix(T *mat, int size, unsigned int seed = 42) {
    srand(seed);
    for (int i = 0; i < size; ++i) {
        mat[i] = (T)(rand() % 100) / 10.0f;
    }
}

// 验证结果
template <typename T>
bool verify_result(const T *C_gpu, const T *C_cpu, int M, int N, T tolerance = 1e-3) {
    bool correct = true;
    int error_count = 0;
    const int max_errors_to_print = 5;

    for (int i = 0; i < M * N; ++i) {
        T diff = fabs(C_gpu[i] - C_cpu[i]);
        if (diff > tolerance) {
            correct = false;
            if (error_count < max_errors_to_print) {
                int row = i / N;
                int col = i % N;
                printf("  Error at [%d,%d]: GPU=%.6f, CPU=%.6f, diff=%.6f\\n",
                       row, col, C_gpu[i], C_cpu[i], diff);
            }
            error_count++;
        }
    }

    if (!correct) {
        printf("  Total errors: %d out of %d elements\\n", error_count, M * N);
    }

    return correct;
}

// 测量 GPU kernel 执行时间
float measure_gpu_time(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
}

// 主函数
int main(int argc, char **argv) {
    // 解析命令行参数
    int M = 512;
    int N = 512;
    int K = 512;
    int kernel_choice = 0;  // 0=naive, 1=shared

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) kernel_choice = atoi(argv[4]);

    printf("Matrix Multiplication: %dx%d * %dx%d = %dx%d\\n", M, K, K, N, M, N);
    printf("Kernel: %s\\n", kernel_choice == 0 ? "naive" : "shared memory");
    printf("Total elements: A=%d, B=%d, C=%d\\n\\n", M*K, K*N, M*N);

    // 分配主机内存
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C_gpu = (float *)malloc(size_C);
    float *h_C_cpu = (float *)malloc(size_C);

    // 初始化矩阵
    printf("Initializing matrices...\\n");
    init_matrix(h_A, M * K, 42);
    init_matrix(h_B, K * N, 123);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size_B));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size_C));

    // 拷贝数据到设备
    printf("Copying data to GPU...\\n");
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // 设置 kernel 启动参数
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("\\nLaunching kernel with grid (%d, %d), block (%d, %d)...\\n",
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x, threadsPerBlock.y);

    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 启动 kernel
    CHECK_CUDA(cudaEventRecord(start));

    if (kernel_choice == 0) {
        matmul_naive<float><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    } else {
        const int BLOCK_SIZE = 16;
        matmul_shared<float, BLOCK_SIZE><<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float gpu_time = measure_gpu_time(start, stop);
    printf("GPU kernel time: %.3f ms\\n", gpu_time);

    // 计算 FLOPS
    double flops = 2.0 * M * N * K;
    double gflops = (flops / 1e9) / (gpu_time / 1000.0);
    printf("Performance: %.2f GFLOPS\\n", gflops);

    // 拷贝结果回主机
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost));

    // 在 CPU 上计算参考结果
    printf("\\nComputing CPU reference result...\\n");
    matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);

    // 验证结果
    printf("Verifying results...\\n");
    bool correct = verify_result(h_C_gpu, h_C_cpu, M, N, 1e-3);

    if (correct) {
        printf("\\n✓ Verification PASSED - GPU results match CPU reference!\\n");
    } else {
        printf("\\n✗ Verification FAILED - GPU results do not match!\\n");
    }

    // 清理资源
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return correct ? 0 : 1;
}
```

## 使用方法

### 编译

```bash
nvcc -o matmul matmul.cu
```

### 运行

```bash
# 默认配置 (512x512x512, naive kernel)
./matmul

# 自定义矩阵大小
./matmul 1024 1024 1024

# 使用共享内存优化版本
./matmul 1024 1024 1024 1
```

参数说明：
- `argv[1]` = M (默认 512)
- `argv[2]` = N (默认 512)
- `argv[3]` = K (默认 512)
- `argv[4]` = kernel 选择 (0=naive, 1=shared memory)

## 输出示例

```
Matrix Multiplication: 512x512 * 512x512 = 512x512
Kernel: naive
Total elements: A=262144, B=262144, C=262144

Initializing matrices...
Copying data to GPU...

Launching kernel with grid (32, 32), block (16, 16)...
GPU kernel time: 2.345 ms
Performance: 114.24 GFLOPS

Computing CPU reference result...
Verifying results...

✓ Verification PASSED - GPU results match CPU reference!
```

## 核心流程图解

```
矩阵分块计算 (共享内存版本):

A (M x K)              B (K x N)              C (M x N)
┌──────┬──────┐       ┌──────┬──────┐       ┌──────┬──────┐
│ A11  │ A12  │   ×   │ B11  │ B12  │   =   │ C11  │ C12  │
├──────┼──────┤       ├──────┼──────┤       ├──────┼──────┤
│ A21  │ A22  │       │ B21  │ B22  │       │ C21  │ C22  │
└──────┴──────┘       └──────┴──────┘       └──────┴──────┘

C11 = A11×B11 + A12×B21  (通过循环加载小块到共享内存计算)

对于 C 的每个 BLOCK_SIZE x BLOCK_SIZE 块:
  遍历 K 维度,每次加载 BLOCK_SIZE 大小的子块到 sA 和 sB
    1. sA[ty][tx] = A[row][t*BLOCK_SIZE + tx]  // 加载 A 块
    2. sB[ty][tx] = B[t*BLOCK_SIZE + ty][col]  // 加载 B 块
    3. __syncthreads()                          // 同步
    4. for k in 0..BLOCK_SIZE-1:
         sum += sA[ty][k] * sB[k][tx]            // 计算部分点积
    5. __syncthreads()                            // 同步，准备下一轮
  循环结束后 sum 包含完整点积
  C[row][col] = sum                                // 写回结果
```
