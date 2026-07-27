---
aliases:
  - CTA 入门
  - CUDA CTA
  - Thread Block Cluster 入门
updated: 2026-06-14
tags:
  - gpu-computing
  - cuda-programming
  - concept-note
---
# CUDA CTA 与 Thread Block Cluster 入门

## 定位

这篇是给 GPU 初学者看的 CTA 背景笔记。读 [[nvjet_kernel_naming]] 里 `2cta`、读 [[CUDA GEMM 矩阵乘法优化指南]] 里的 cluster / DSMEM / `cta_group::2` 前，先把这篇过一遍。

相关主笔记：

- [[CUDA 编程基础]]
- [[CUDA 线程配置与占用率]]
- [[CUDA Shared Memory 与 Bank Conflict]]
- [[CUDA GEMM 矩阵乘法优化指南]]
- [[GPU 硬件架构背景与编程范式]]

## CTA 是什么

CTA 是 **Cooperative Thread Array**，在 PTX 文档里的概念。CUDA C++ 日常更常说 **thread block**。

可以先粗略记成：

```text
CTA ≈ CUDA block
```

一个 CUDA kernel launch 的层次是：

```text
Grid
  └── Block / CTA
        └── Warp，通常 32 个 thread
              └── Thread
```

每个 CTA 里有一组线程。这组线程可以：

- 共享同一个 block 的 shared memory。
- 用 `__syncthreads()` 做 block 级同步。
- 用 `threadIdx.x/y/z` 区分自己在 CTA 内的位置。

普通情况下，不同 CTA 之间不能随便同步，也不能假设执行顺序。

## CTA、Warp、SM 的关系

三个词很容易混：

| 名词 | 所属层次 | 初学者解释 |
|------|----------|------------|
| Thread | 软件执行单元 | 一条 CUDA 线程，有自己的寄存器和 thread index。 |
| Warp | 硬件调度单元 | 通常 32 个 thread 一起发射同一条指令。 |
| CTA / Block | CUDA 线程组织单元 | 一组 thread，被调度到同一个 SM 上，共享 shared memory。 |
| SM | GPU 硬件执行单元 | Streaming Multiprocessor，真正执行 CTA、warp 和 Tensor Core 指令。 |

普通 CUDA kernel 里，一个 CTA 会被调度到一个 SM 上执行。同一个 SM 可以同时驻留多个 CTA，前提是寄存器、shared memory、最大线程数等资源放得下。

```text
GPU
 ├── SM0: CTA0, CTA3, ...
 ├── SM1: CTA1, CTA4, ...
 └── SM2: CTA2, CTA5, ...
```

这也是 [[CUDA 线程配置与占用率]] 要关注 block size、寄存器和 shared memory 的原因。

## 为什么 GEMM 总在讲 CTA tile

GEMM 是：

```text
C[M, N] = A[M, K] × B[K, N]
```

高性能 GEMM 会把 C 矩阵切成 tile，让一个 CTA 负责一个输出子块：

```text
C 矩阵
┌────────┬────────┬────────┐
│ CTA 0  │ CTA 1  │ CTA 2  │
├────────┼────────┼────────┤
│ CTA 3  │ CTA 4  │ CTA 5  │
└────────┴────────┴────────┘
```

每个 CTA 会把自己需要的 A/B tile 搬进 shared memory，然后 CTA 内的 warp 协作调用 Tensor Core。

所以 kernel 名称里的 `128x256`、`256x240` 这类字段，通常不是矩阵总大小，而是 CTA 或 CTA group 负责的局部 tile 大小。

## Thread Block Cluster 是什么

从 Hopper，也就是 compute capability 9.0 开始，CUDA 编程模型引入了 **Thread Block Cluster**。Blackwell 也沿用同一套机制，没有放宽其上限。

Cluster 可以理解为：

```text
多个 CTA 被绑定成一组，同时调度，并允许组内同步和通信
```

普通 CTA 模型：

```text
CTA A 和 CTA B 默认独立
不能假设它们同时运行
不能直接同步
不能直接读写对方 shared memory
```

Cluster 模型：

```text
同一个 cluster 里的 CTA 会被协同调度
可以用 Cooperative Groups 做 cluster 级同步
可以访问 Distributed Shared Memory
```

NVIDIA 文档里把 cluster 内跨 CTA 可见的 shared memory 称为 **Distributed Shared Memory**，简称 DSMEM。

![[GPU/Drawings/Thread Block Cluster 与 DSMEM.svg]]

### Cluster 关键限制（必读）

| 维度 | 限制 |
|------|------|
| **Cluster size 上限** | **portable 上限 = 8 CTA**；通过 `cudaFuncSetAttribute(cudaFuncAttributeNonPortableClusterSizeAllowed, 1)` 可 opt-in 到 **16 CTA**。Hopper 与 Blackwell 都遵守该上限，**Blackwell 没有把它扩大**。 |
| **物理位置** | cluster 内所有 CTA 必须落在**同一个 GPC**（GPU Processing Cluster）内的不同 SM 上；**不能跨 GPC**，更不可能跨 GPU。 |
| **DSMEM 物理路径** | cluster 内 CTA 间的 SMEM 访问走**片内 SM-to-SM fabric**（cluster local network），**不走 NVLink**——NVLink 是 GPU 之间的互联，与 cluster 内通信无关。读到"cluster 间通信走 NVLink 5.0"是误解。 |
| **Per-CTA SMEM 容量** | 每个 CTA 自己的 SMEM 仍是 228 KB（Hopper / Blackwell opt-in 上限），**不会因为 cluster 变大而扩展**。 |
| **Cluster-aggregate SMEM** | cluster 可见 SMEM = N × per-CTA SMEM（N = cluster size）。例如 4-CTA cluster 上，CTA 0 可通过 DSMEM 访问其他 3 个 CTA 的 SMEM，相当于片上有 4×228 = 912 KB 可达数据；但单 CTA 占用、寄存器分配、occupancy 仍按 228 KB 计。 |
| **同步** | 需用 `cluster.sync()`（Cooperative Groups）或 `mbarrier` 做 cluster 级同步；普通 `__syncthreads()` 只在单 CTA 内有效。 |
| **DSMEM 对齐与一致性** | DSMEM 访问需走 cluster local address space（通过 `__cluster_relative_addr` / `cluster_local_ptr` 转换）；跨 CTA 写后读需配 `mbarrier` 或 `cluster.sync()` 才保证可见。 |

这里要特别注意：**CTA 的优势不是"跨 SM 使用 shared memory"**。CTA/block 的基本优势是同一个 block 内的线程可以共享本 CTA 的 local shared memory；跨 CTA、跨 SM 访问彼此 shared memory，是 Thread Block Cluster + DSMEM 提供的能力。

换句话说：

```text
CTA / block:
  一个 CTA 驻留在一个 SM 上
  CTA 内线程共享本 CTA 的 shared memory（228 KB 上限）
  同步用 __syncthreads()

Thread Block Cluster:
  多个 CTA（portable ≤ 8, opt-in ≤ 16）协同调度到同一 GPC 内的一组 SM
  cluster 内 CTA 可以通过 DSMEM 访问彼此的 shared memory（走片内 SM-to-SM fabric）
  同步用 cluster.sync() 或 mbarrier
  跨 GPC / 跨 GPU 不属于 cluster 范畴
```

## 和 cooperative launch 的区别

`Thread Block Cluster` 和 cooperative launch 都是在突破“普通 block 之间不能随便同步”的边界，但它们解决的是不同范围的问题。

| 机制 | 同步范围 | 调度保证 | 主要代价 |
|------|----------|----------|----------|
| 普通 kernel | block 内 `__syncthreads()` | 不同 block 可任意顺序执行 | 没有跨 block 同步 |
| cooperative launch + `grid.sync()` | 整个 grid | 需要满足全 grid 可协作执行的启动约束 | grid size 受限，启动方式特殊 |
| Thread Block Cluster + `cluster.sync()` | 同一 cluster 内多个 CTA | cluster 内 CTA 同时调度到同一 GPC 内 | cluster size 受限，调度粒度变大 |
| 多 kernel 拆分 | kernel 边界全局同步 | 前一个 kernel 完成后下一个才开始 | 多一次 launch，数据要落到 global memory |

因此，cluster 不是“全 grid 同步”的替代品。它更像是在 block 和 grid 之间插入一个中间层：

```text
block 内协作:
  shared memory + __syncthreads()

cluster 内协作:
  DSMEM + cluster.sync()

grid 内协作:
  cooperative launch + grid.sync()

kernel 间协作:
  拆成多个 kernel，kernel launch 边界同步
```

普通 grid 里仍然可以有很多 cluster；这些 cluster 之间还是像普通 block 一样，不保证执行顺序，也不能直接 `cluster.sync()` 到一起。

## 最小代码骨架

Cluster 侧的 CUDA C++ API 来自 Cooperative Groups。kernel 内先拿到当前 cluster，再读自己的 cluster 内 block rank：

```cpp
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void cluster_kernel(float* out) {
    extern __shared__ float smem[];

    cg::cluster_group cluster = cg::this_cluster();
    int rank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x;

    // 每个 CTA 初始化自己的 local shared memory。
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        smem[i] = static_cast<float>(rank);
    }

    // 确保 cluster 内所有 CTA 都已经启动，并且 local SMEM 初始化完成。
    cluster.sync();

    // 访问 cluster 内 rank 0 的 shared memory。
    float* rank0_smem = cluster.map_shared_rank(smem, 0);
    float v = rank0_smem[threadIdx.x % 256];

    // 跨 CTA 访问结束前再同步，避免某个 CTA 提前退出导致远端 SMEM 生命周期结束。
    cluster.sync();

    if (threadIdx.x == 0) {
        out[blockIdx.x] = v + cluster_size;
    }
}
```

host 侧可以用 extensible launch 设置 cluster 维度。注意 `dynamicSmemBytes` 仍然是 **per block** 的 shared memory 大小，不是整个 cluster 的总大小。

```cpp
cudaLaunchConfig_t config = {};
config.gridDim = grid_dim;
config.blockDim = block_dim;
config.dynamicSmemBytes = per_block_smem_bytes;

cudaLaunchAttribute attrs[1];
attrs[0].id = cudaLaunchAttributeClusterDimension;
attrs[0].val.clusterDim.x = cluster_size;
attrs[0].val.clusterDim.y = 1;
attrs[0].val.clusterDim.z = 1;

config.attrs = attrs;
config.numAttrs = 1;

cudaLaunchKernelEx(&config, cluster_kernel, out);
```

如果 kernel 需要较大的动态 shared memory，还要配合 `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` 显式放开 per-block 动态 SMEM 上限。

## DSMEM 访问生命周期

DSMEM 的本质不是新开一块独立 SRAM，而是把同一 cluster 内多个 CTA 的 local shared memory 暴露成一个可互访的逻辑地址空间。

一个安全的 DSMEM 访问流程通常是：

```text
1. 每个 CTA 初始化自己的 local SMEM
2. cluster.sync()
   -> 保证 cluster 内所有 CTA 已经启动，并且初始化完成
3. 通过 cluster.map_shared_rank(ptr, rank) 获得远端 CTA 的 SMEM 指针
4. 读 / 写 / atomic 访问 DSMEM
5. cluster.sync() 或 mbarrier
   -> 保证跨 CTA DSMEM 访问完成
6. CTA 才能退出
```

这里最容易踩的坑是第 5 步：如果 CTA 0 还在读 CTA 1 的 shared memory，而 CTA 1 已经执行结束退出，那么 CTA 1 的 shared memory 生命周期就结束了，远端访问语义就不成立。所以 DSMEM 代码里通常需要前后两个 cluster 级同步点：一个保证大家都“活着且初始化完”，另一个保证远端访问结束后再退出。

## 什么时候值得用

Thread Block Cluster 适合解决“一个 CTA 太小，整个 grid 又太大”的中间层协作问题。

适合用 cluster 的情况：

- 单 CTA shared memory 放不下完整工作集，但 `N × per-CTA SMEM` 能放下。
- 多个 CTA 处理同一个大 tile，需要复用同一份 A/B tile 或中间结果。
- 需要 cluster 内同步，而不想拆成多个 kernel。
- Hopper/Blackwell GEMM 中配合 TMA multicast、WGMMA、`cta_group::2` 或更大的 tile 复用。
- histogram、stencil、block-level producer-consumer、跨 CTA partial reduction 等局部协作。

不适合用 cluster 的情况：

- 数据没有明显跨 CTA 复用，DSMEM 访问只是在制造额外同步。
- 工作粒度很小，cluster 调度和同步开销会吃掉收益。
- 需要全 grid 同步；这不是 cluster 的职责。
- 依赖跨 GPC / 跨 GPU 通信；cluster 只在单 GPU 的 GPC 内。
- 访问模式随机且细碎，远端 DSMEM 延迟可能不如直接重算或重排 tile。

调优时可以先问三个问题：

```text
1. 这个工作单元是否真的需要多个 CTA 协作？
2. DSMEM 带来的片上复用，能否抵消 cluster.sync 和调度约束？
3. cluster size 变大后，会不会增加 tail effect 或降低可并行的 cluster 数？
```

## `2cta` 可以怎么理解

当你在 kernel 名称里看到：

```text
..._2cta_...
```

可以先读成：

```text
这个 kernel 变种可能让 2 个 CTA 协同处理一个工作单元
```

它不是：

- 2 个 thread。
- 2 个 warp。
- 固定只占用 2 个 CUDA core。

它更像是：

```text
2 个 block / CTA 组成一个更大的执行协作单元
```

在 Blackwell GEMM 里，官方 CUTLASS 文档也能看到 `tcgen05.mma` 指令族带有 `cta_group::1` / `cta_group::2` 这种范围标记。读 `2cta` 时，可以把它和这类“一个 Tensor Core 工作跨 1 个或 2 个 CTA 组织”的路线联系起来。

## 为什么要让多个 CTA 协同

单 CTA 的限制主要来自片上资源：

- 一个 CTA 能用的 shared memory 有上限。
- 一个 CTA 内的线程数有上限。
- 一个 CTA 能覆盖的 tile 太小，A/B 数据复用率可能不够。

多个 CTA 协同时，可以：

- 做更大的 tile，提高数据复用。
- 通过 cluster / DSMEM 共享部分片上数据。
- 配合 TMA multicast，减少多 CTA 重复从 HBM 读同一份数据。
- 在大 GEMM 中更容易形成高吞吐的 Tensor Core 流水线。

但它也有代价：

- 占用更多 SM 和 shared memory。
- 需要 cluster 级同步，调度更重。
- 小矩阵时可能不如单 CTA 或小 tile 灵活。

所以 `2cta` 常见于大 tile、高吞吐 GEMM；decode 或小 batch 场景可能更偏小 tile。

## CTA 与 Occupancy

Occupancy 不是“线程越多越好”，而是每个 SM 能同时驻留多少活跃 warp。

CTA 会消耗三类关键资源：

```text
每 CTA 线程数
每 CTA shared memory
每线程寄存器数 × 每 CTA 线程数
```

资源消耗越大，每个 SM 能同时驻留的 CTA 可能越少。GEMM kernel 经常故意用较大的 tile 和较多寄存器来换取数据复用与 Tensor Core 吞吐，所以 occupancy 不一定最高，但整体性能可能更好。

新手可以记住：

```text
CTA size / tile size / shared memory / register pressure 是一组联动参数
```

不要单独看一个数字判断性能。

## 常见问题

| 问题 | 答案 |
|------|------|
| CTA 和 block 是同一个东西吗 | 日常 CUDA C++ 里基本可以等价理解；CTA 是 PTX/底层文档常用词。 |
| CTA 和 warp 是同一个东西吗 | 不是。一个 CTA 里通常有多个 warp，一个 warp 通常 32 个 thread。 |
| CTA 一定等于一个 SM 吗 | 一个 CTA 会在一个 SM 上执行，但一个 SM 可同时驻留多个 CTA。 |
| CTA 的优势是跨 SM 使用 shared memory 吗 | 不是。CTA 的优势是 block 内共享 local shared memory；跨 CTA/SM 访问 shared memory 是 cluster/DSMEM 的能力。 |
| 不同 CTA 能同步吗 | 普通 block 不能；cluster/cooperative groups 或 kernel 边界可以。 |
| `2cta` 一定比单 CTA 快吗 | 不一定。大 GEMM 可能受益，小矩阵可能亏。 |

## 参考

- [PTX ISA: Cooperative Thread Arrays](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cooperative-thread-arrays)
- [CUDA Programming Model: Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)
- [CUDA Programming Guide: Distributed Shared Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#distributed-shared-memory)
- [CUDA Programming Guide: Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [NVIDIA CUTLASS: Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
