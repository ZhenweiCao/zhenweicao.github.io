---
aliases:
  - CUDA编程
updated: 2026-06-14
tags:
  - gpu-computing
  - cuda-programming
  - concept-note
---
# CUDA 编程基础

## 定位

这篇作为 CUDA 学习的入口笔记，负责串起执行模型、线程组织、基础 kernel 写法、Thrust 以及后续性能优化主题。

相关主笔记：

- [[GPU 知识库索引]]
- [[CUDA 线程配置与占用率]]
- [[CUDA Stream 与异步执行]]
- [[CUDA Shared Memory 与 Bank Conflict]]
- [[CUDA GEMM 矩阵乘法优化指南]]
- [[CUDA Kernel 示例：向量加法]]
- [[CUDA Kernel 示例：归约求和]]
- [[CUDA Kernel 示例：矩阵乘法]]

## CUDA 软件栈

- Application
- CUDA Libraries
- CUDA Runtime
- CUDA Driver

![[file-20260414214157941.png|411x354]]

从上到下可以理解为：应用调用 CUDA Runtime 或 CUDA Libraries，Runtime 通过 Driver API 让 GPU 执行 kernel。日常开发中，多数代码直接使用 CUDA Runtime API；性能敏感场景再考虑 Driver API、cuBLAS、CUTLASS 或自定义 kernel。

## 执行模型

一个 kernel launch 会创建一个 grid。grid 由多个 block 组成，block 内部再由 thread 组成。硬件调度时，线程以 32 个为一组形成 warp，warp 是 GPU 实际发射指令的基本单位。

```
Grid
  └── Block
        └── Warp
              └── Thread
```

每个 block 是一批线程的执行单元。block 内线程可以通过 Shared Memory 共享数据，并通过 `__syncthreads()` 设置同步点；block 之间默认不能直接同步。

对于一个大小为 `(Dx, Dy)` 的二维 block，线程索引是 `(x, y)`，线性线程 ID 是：

```cpp
int tid = x + y * Dx;
```

对于大小为 `(Dx, Dy, Dz)` 的三维 block：

```cpp
int tid = x + y * Dx + z * Dx * Dy;
```

![[file-20260414214157940.png|369x481]]

### 几个容易混淆的点

- **Block 是调度和同步边界**：同一个 block 内的线程可以通过 shared memory 通信，也可以用 `__syncthreads()` 做 block 级同步。
- **Block 之间默认独立**：不同 block 之间不应依赖执行顺序。若需要跨 block 汇总，通常拆成多个 kernel，或使用 cooperative groups / cluster 等更高级机制。
- **Warp 是硬件执行单位**：warp 大小是 32。一个 block 的线程会按线性 thread id 切成多个 warp。
- **不要依赖隐式 warp 同步**：Volta 之后支持 Independent Thread Scheduling，老式“同一个 warp 天然锁步所以不用同步”的写法更容易出错；warp 内交换数据时优先用带 mask 的 `__shfl_sync` / `__syncwarp`。

## 内存层次速记

| 层级 | 可见范围 | 典型用途 | 注意点 |
|------|----------|----------|--------|
| Register | 单线程 | 局部变量、累加器 | 每线程寄存器太多会降低 occupancy |
| Shared Memory | 同一 block | tile 缓存、块内归约、数据重排 | 需要关注 [[CUDA Shared Memory 与 Bank Conflict]] |
| L1 / L2 Cache | SM / 全 GPU | 缓存全局内存访问 | 命中率取决于访问模式 |
| Global Memory / HBM | 全 GPU | 输入输出张量、模型权重 | 带宽高但延迟大，优先合并访问 |
| Constant / Texture | 特定只读路径 | 小型只读参数、特殊访问模式 | 现代编译器常能自动选择合适路径 |

CUDA 优化的核心不是“让线程越多越好”，而是让每次 HBM 读写产生更多有效计算，并尽量让 warp 内访问合并。

## Kernel 编写骨架

```cpp
__global__ void kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = x[i] * 2.0f;
    }
}

int block = 256;
int grid = (n + block - 1) / block;
kernel<<<grid, block>>>(x, n);
```

选择 `block` 和 `grid` 时，优先读 [[CUDA 线程配置与占用率]]。当数据规模很大或希望 kernel 对不同 GPU 更稳健时，常用 grid-stride loop，示例见 [[CUDA Kernel 示例：向量加法]]。

### 最小主机端流程

```cpp
float *d_x = nullptr;
cudaMalloc(&d_x, n * sizeof(float));
cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

kernel<<<grid, block>>>(d_x, n);
cudaError_t launch_err = cudaGetLastError();
cudaError_t sync_err = cudaDeviceSynchronize();

cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
cudaFree(d_x);
```

实战中至少检查两类错误：

- `cudaGetLastError()`：检查 kernel launch 参数、资源限制等启动错误。
- `cudaDeviceSynchronize()`：让异步 kernel 结束，并暴露运行时错误。

`cudaMemcpy` 默认是同步拷贝；如果使用 stream 和 `cudaMemcpyAsync`，需要把同步关系写清楚，否则计时和数据可见性很容易判断错。

更多 stream、event、默认流和 copy/compute overlap 的细节见 [[CUDA Stream 与异步执行]]。

## Thrust 速记

Thrust 是 CUDA 的 C++ 并行算法库，适合快速写出可靠的 GPU 数据处理流程。

常用容器：

```cpp
thrust::host_vector<float> h;
thrust::device_vector<float> d;
thrust::universal_vector<float> u;
```

常用 iterator：

```cpp
thrust::constant_iterator<int> c(1);
thrust::counting_iterator<int> idx(0);
thrust::transform_iterator it(idx, op);
```

常用算法：

```cpp
thrust::tabulate(first, last, op);
thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
thrust::reduce_by_key(keys.begin(), keys.end(), values.begin(), out_keys.begin(), out_values.begin());
```

需要极致性能时，Thrust 可作为 baseline；确认瓶颈后再改写为自定义 kernel、cuBLAS/CUTLASS 或框架算子。

## 可靠参考

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA Technical Blog: CUDA Refresher](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/)
