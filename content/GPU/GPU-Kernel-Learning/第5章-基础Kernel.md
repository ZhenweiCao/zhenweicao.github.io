---
title: "第 5 章 - 基础 Kernel 实现"
content_type: guide
maturity: reviewed
updated: 2026-07-27
publish: true
tags:
  - gpu-computing
  - gpu-programming
  - concept-note
---
# 第 5 章 - 基础 Kernel 实现

> 本章开始实战！我们将从最简单的 Kernel 开始，逐步实现经典的 GPU 算法

## 本章定位

本章负责把前面学到的 thread/block、shared memory、coalescing 和同步真正用到 kernel 里。矩阵乘法的完整优化路线以 [[CUDA GEMM 矩阵乘法优化指南]] 为准，本章只保留初学者能手写和验证的版本。

配套主文档：

- [[CUDA Kernel 示例：向量加法]]
- [[CUDA Kernel 示例：归约求和]]
- [[CUDA Kernel 示例：矩阵乘法]]
- [[CUDA GEMM 矩阵乘法优化指南]]

## 学习目标

- 理解矩阵乘法的基本原理
- 掌握从 naive 版本到 shared memory 优化的演进过程
- 学会分析性能瓶颈
- 能够独立实现优化的矩阵乘法 Kernel

## 1. 矩阵乘法基础

### 1.1 什么是矩阵乘法？

给定两个矩阵 A (M×K) 和 B (K×N)，它们的乘积 C = A × B 是一个 (M×N) 的矩阵：

```text
C[i][j] = Σ A[i][k] × B[k][j]  (k 从 0 到 K-1)
```

**可视化表示**：

```text
        A (M×K)           B (K×N)
    ┌──────────┐      ┌──────────┐
    │ a00 a01  │      │ b00 b01  │
    │ a10 a11  │  ×   │ b10 b11  │  =  C (M×N)
    │  ...     │      │  ...     │
    └──────────┘      └──────────┘

C[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + ... + A[0][K-1]*B[K-1][0]
```

### 1.2 计算复杂度

- 每个元素需要 K 次乘法和 K-1 次加法
- 总计算量：O(M × N × K)
- 对于方阵 (N×N)：O(N³)

## 2. Naive 矩阵乘法实现

### 2.1 一维索引映射

GPU 使用一维线程索引，我们需要将二维矩阵映射到一维：

```cpp
// 二维索引转一维
int row = idx / N;      // 行号
int col = idx % N;      // 列号

// 一维索引转二维
idx = row * N + col;
```

### 2.2 Naive 版本代码

```cpp
// 代码位置：[[matmul.cu|GPU/CUDA/matmul.cu]] 中的 matmul_naive

__global__ void matmul_naive(float* A, float* B, float* C,
                              int M, int N, int K) {
    // 计算全局线程索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查
    if (row < M && col < N) {
        float sum = 0.0f;

        // 计算 C[row][col] = Σ A[row][k] * B[k][col]
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}
```

### 2.3 Naive 版本的问题

**内存访问模式分析**：

```text
矩阵 A 的访问：A[row * K + k]
- 同一线程内：顺序访问 (k 递增) ✓
- 同一 warp 内：不同线程访问不同 row，导致非合并访问 ✗

矩阵 B 的访问：B[k * N + col]
- 同一线程内：跨步访问 (stride = N) ✗
- 同一 warp 内：相邻线程访问相邻元素 ✓
```

**性能瓶颈**：

1. **全局内存带宽限制**：每次从 global memory 读取数据
2. **数据复用率低**：每个元素从全局内存读取多次
3. **非合并访问**：矩阵 A 的访问模式不够理想

## 3. Shared Memory 优化版本

### 3.1 核心思想

**Shared Memory 的特点**：
- **延迟比全局内存低约 10~20 倍，带宽高约 10 倍**（"快 100 倍"是过于粗略的口语化说法，请按 NCU 的实测指标判断）
- 每个线程块（block）私有
- 需要手动管理数据加载

**优化策略**：

```text
将矩阵分块（tiling），每块加载到 shared memory：

1. 将 C 的一个 tile 计算所需的数据加载到 shared memory
2. 在 shared memory 中进行计算
3. 重复直到完成整个 tile 的计算
```

### 3.2 分块策略可视化

```text
假设 TILE_SIZE = 2, 矩阵 A (4×4), B (4×4)

A 被分成 4 个 tile:          B 被分成 4 个 tile:
┌─────┬─────┐               ┌─────┬─────┐
│ A00 │ A01 │               │ B00 │ B01 │
│     │     │               │     │     │
├─────┼─────┤               ├─────┼─────┤
│ A10 │ A11 │               │ B10 │ B11 │
│     │     │               │     │     │
└─────┴─────┘               └─────┴─────┘

计算 C00 需要：
C00 = A00 × B00 + A01 × B10  (tile 级别的乘法)
```

### 3.3 Shared Memory 版本代码

```cpp
// 代码位置：[[matmul.cu|GPU/CUDA/matmul.cu]] 中的 matmul_shared

#define TILE_SIZE 16  // 根据硬件调整

__global__ void matmul_shared(float* A, float* B, float* C,
                               int M, int N, int K) {
    // 声明 shared memory
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // 计算线程在 block 中的位置
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 计算线程在全局内存中的起始位置
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // 遍历所有 tile
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载 A 的 tile 到 shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        // 加载 B 的 tile 到 shared memory
        if (t * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // 等待 block 内所有线程完成加载
        __syncthreads();

        // 在 shared memory 中进行矩阵乘法
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 等待 block 内所有线程完成计算
        __syncthreads();
    }

    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 3.4 数据复用分析

```text
对于 TILE_SIZE = 16：

- 每个线程计算 1 个输出元素
- 每个 tile 需要加载 16×16 的 A 和 B
- 每个元素被复用 16 次（因为要做 16 次乘法）
- 全局内存访问减少到原来的 1/16
```

**访问次数对比**：

| 版本 | A 的访问次数 | B 的访问次数 | 总计 |
|------|-------------|-------------|------|
| Naive | M×N×K | M×N×K | 2×M×N×K |
| Shared | (M×N×K)/16 | (M×N×K)/16 | (M×N×K)/8 |

## 4. 进阶优化技巧

### 4.1 避免 Bank Conflict

**问题**：

```cpp
// ❌ 可能产生 bank conflict
__shared__ float As[16][16];
// 当多个线程访问同一列时，会发生 bank conflict
float val = As[threadIdx.y][threadIdx.x];  // 列访问冲突
```

**解决方案 - Padding**：

```cpp
// ✅ 添加 padding 避免 bank conflict
__shared__ float As[16][17];  // 多一列用于 padding
// 现在相邻列在不同 bank 中
```

### 4.2 使用向量化加载

```cpp
// 使用 float4 一次性加载 4 个元素
__global__ void matmul_vectorized(float* A, float* B, float* C,
                                   int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 向量化加载（假设内存对齐）
        float4 a_vec = reinterpret_cast<float4*>(
            A + row * K + t * TILE_SIZE)[tx / 4];
        // ... 处理 4 个元素

        // 在 shared memory 中计算
        // ...
    }
}
```

### 4.3 循环展开

```cpp
// 手动展开内部循环
#pragma unroll
for (int k = 0; k < TILE_SIZE; k += 4) {
    sum += As[ty][k] * Bs[k][tx];
    sum += As[ty][k+1] * Bs[k+1][tx];
    sum += As[ty][k+2] * Bs[k+2][tx];
    sum += As[ty][k+3] * Bs[k+3][tx];
}
```

## 5. 完整代码示例

参见代码文件：
- [[matmul.cu|GPU/CUDA/matmul.cu]]：包含 naive 与 shared memory 两个版本。

## 6. 性能对比

### 6.1 理论性能分析

对于 N×N 矩阵，分块大小 T：

| 指标 | Naive | Shared Memory |
|------|-------|---------------|
| 全局内存读取 | 2N³ | 2N³/T |
| 计算强度 | 低 | T 倍提升 |
| 理论加速比 | 1x | ~T 倍 |

### 6.2 实际性能（RTX 4090, 1024×1024 矩阵）

| 版本 | 耗时 (ms) | GFLOPS | 相对加速 |
|------|----------|--------|---------|
| Naive | 2.5 | 860 | 1.0x |
| Shared (T=16) | 0.18 | 12000 | 13.9x |
| Shared + Padding | 0.16 | 13500 | 15.6x |
| Shared + Unroll | 0.14 | 15400 | 17.9x |

> **口径说明**：上表数字为**估算/示意**而非严格实测——RTX 4090 FP32 dense 峰值约 82 TFLOPS，1024³ 问题尺寸太小，且未注明编译参数、warmup、是否使用 Tensor Core。若要复现实测，请配合 `nsys`/`ncu` 重新测量并标注完整 runtime 配置。课程层只用它说明优化方向上的差距数量级。

## 7. 调试和验证

### 7.1 CPU 验证函数

```cpp
void matmul_cpu(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
```

### 7.2 结果验证

```cpp
bool verify(float* gpu_result, float* cpu_result, int size, float tolerance = 1e-4) {
    for (int i = 0; i < size; i++) {
        if (fabs(gpu_result[i] - cpu_result[i]) > tolerance) {
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n",
                   i, gpu_result[i], cpu_result[i]);
            return false;
        }
    }
    return true;
}
```

## 8. 练习

1. **基础题**：实现 naive 版本，并用小矩阵（32×32）验证正确性
2. **进阶题**：实现 shared memory 版本，对比性能
3. **挑战题**：尝试不同的 TILE_SIZE（8, 16, 32），找出最优值
4. **思考题**：为什么 padding 可以避免 bank conflict？

## 9. 关键知识点总结

| 概念 | 说明 |
|------|------|
| 分块（Tiling）| 将大矩阵分成小块，提高数据复用 |
| Shared Memory | 快速的片上内存，需要手动管理 |
| __syncthreads() | 块内同步，确保数据加载完成 |
| Bank Conflict | 共享内存访问冲突，会降低带宽 |
| Padding | 添加额外元素避免 bank conflict |

## 相关文档

- [[第3章-GPU硬件|第 3 章：GPU 硬件]]
- [[第4章-优化技巧|第 4 章：优化技巧]]
- [[CUDA Kernel 示例：矩阵乘法]]
- [[CUDA GEMM 矩阵乘法优化指南]]
- [[CUDA Shared Memory 与 Bank Conflict]]
