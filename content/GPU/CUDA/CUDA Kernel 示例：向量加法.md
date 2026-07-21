---
aliases:
  - vect_add.cu
updated: 2026-05-16
tags:
  - gpu-computing
  - cuda-programming
  - gpu-programming
---
# CUDA Kernel 示例：向量加法

相关主笔记：

- [[GPU 知识库索引]]
- [[CUDA 编程基础]]
- [[CUDA 线程配置与占用率]]

## 概览

这个示例比较向量加法 kernel 的几种写法：

| 版本 | 核心点 | 适合记忆的结论 |
|------|--------|----------------|
| `add_basic` | 每线程处理一个元素 | 最小 kernel 骨架 |
| `add_grid_stride` | grid-stride loop | 更稳健，适配任意规模 |
| `add_unroll4` | 每线程展开处理 4 个元素 | 减少循环与调度开销 |
| `add_float4` | 使用 `float4` 向量化访存 | 要求对齐，提升带宽利用率 |
| `add_ldg` | 使用只读缓存加载 `x` | 适合只读数据路径 |

优先把 `add_grid_stride` 当作通用模板；当确认瓶颈在全局内存带宽时，再考虑展开和向量化。

## 工程注意事项

- **grid-stride loop 是默认安全模板**：它允许 grid 大小不必精确覆盖全部数据，也让同一个 kernel 在不同 GPU 上更容易复用。
- **`float4` 需要对齐**：`cudaMalloc` 返回的指针通常满足足够对齐，但如果对指针做过偏移，必须确认地址仍然 16-byte aligned。
- **余数处理要单独检查**：向量化版本一次处理 4 个 `float`，当 `n % 4 != 0` 时需要标量处理尾部元素。
- **`__ldg` 的收益依架构而定**：现代 GPU 的缓存路径和编译器优化已经更智能。只读数据可以先用 `const __restrict__` 表达意图，再通过 profiling 判断是否需要显式 `__ldg`。
- **不要只看 kernel 时间**：向量加法通常是 memory-bound，端到端性能还会被 H2D/D2H 拷贝、stream 同步、数据规模影响。

## 代码

```cpp

#include "kernels.cuh"

// 基础版本 - 每个线程处理一个元素
__global__ void add_basic(int n, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = x[i] + y[i];
}

// 网格步长循环版本 - 每个线程处理多个元素
__global__ void add_grid_stride(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

// 展开版本 - 每次处理4个元素
__global__ void add_unroll4(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // 每次处理4个元素
    for (int i = index; i < n; i += stride * 4) {
        if (i < n) y[i] = x[i] + y[i];
        if (i + stride < n) y[i + stride] = x[i + stride] + y[i + stride];
        if (i + 2 * stride < n) y[i + 2 * stride] = x[i + 2 * stride] + y[i + 2 * stride];
        if (i + 3 * stride < n) y[i + 3 * stride] = x[i + 3 * stride] + y[i + 3 * stride];
    }
}

// 向量化版本 - 使用float4处理4个float
__global__ void add_float4(int n, float *x, float *y)
{
    // 将float指针转换为float4指针
    float4 *x4 = (float4*)x;
    float4 *y4 = (float4*)y;

    // 计算float4的个数（假设n是4的倍数）
    int n4 = n / 4;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n4) {
        float4 xv = x4[i];
        float4 yv = y4[i];

        yv.x += xv.x;
        yv.y += xv.y;
        yv.z += xv.z;
        yv.w += xv.w;

        y4[i] = yv;
    }

    // 处理剩余的元素（如果n不是4的倍数）
    int remainder_start = n4 * 4;
    int remainder_idx = remainder_start + threadIdx.x;
    if (remainder_idx < n && blockIdx.x == 0) {
        y[remainder_idx] = x[remainder_idx] + y[remainder_idx];
    }
}

// LDG优化版本 - 使用只读缓存加载x
__global__ void add_ldg(int n, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float xv = __ldg(x + i);  // 通过只读数据缓存加载
        y[i] += xv;
    }
}

// Kernel配置表
static const KernelConfig kernel_configs[] = {
    {"basic",       add_basic,       256, 1},
    {"grid_stride", add_grid_stride, 256, 1},
    {"unroll4",     add_unroll4,     256, 1},
    {"float4",      add_float4,      256, 1},
    {"ldg",         add_ldg,         256, 1},
};

const KernelConfig* get_kernels(int *count)
{
    if (count)
        *count = sizeof(kernel_configs) / sizeof(KernelConfig);
    return kernel_configs;
}

```
