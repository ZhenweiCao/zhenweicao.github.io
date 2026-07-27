---
aliases:
  - GPU 术语表
  - CUDA 初学者术语表
updated: 2026-06-14
tags:
  - gpu-computing
  - cuda-programming
  - concept-note
---
# GPU 初学者术语表

这篇是读 GPU/CUDA 文档前的缓冲层。遇到陌生词先来这里确认它属于哪一层，再回到具体文档。

相关入口：

- [[GPU 知识库索引]]
- [[GPU Kernel 学习路线]]
- [[CUDA 编程基础]]
- [[GPU 硬件架构背景与编程范式]]

## 一张层次图

![[GPU/Drawings/GPU 硬件层次总览.svg]]

## 执行层次

| 术语 | 初学者解释 | 常见误区 |
|------|------------|----------|
| Kernel | 在 GPU 上执行的函数。CPU 发起一次 kernel launch，GPU 并行跑很多线程。 | kernel 不是操作系统内核。 |
| Grid | 一次 kernel launch 的全部 thread block 集合。 | grid 不是硬件单元。 |
| CUDA Stream | CUDA runtime 的按序工作队列，用来提交 kernel、异步拷贝、event、graph 等操作。 | stream 不是 SM/warp；多个 stream 只是表达可并发，不保证一定并发。 |
| Block / Thread Block | CUDA 编程层的一组线程。block 内线程可以共享 shared memory、用 `__syncthreads()` 同步。 | block 之间默认不能直接同步。 |
| CTA | Cooperative Thread Array，PTX/底层文档里对 block 的叫法。 | CTA 和 block 大多数时候可等价理解。 |
| Thread Block Cluster | Hopper 之后引入的更高一级分组，由多个 CTA 组成；落在同一 GPC 内的多个 SM 上。 | Cluster size 上限：portable ≤ 8 CTA、opt-in ≤ 16 CTA；Blackwell 未扩大。**不跨 GPC、不走 NVLink**。 |
| Warp | GPU 硬件调度单位，NVIDIA GPU 通常 32 个 thread。 | 线程不是一个个完全独立发射，warp 才是调度核心。 |
| Thread | CUDA 的最小软件执行单元，有自己的 thread index 和寄存器状态。 | thread 很轻量，不要按 CPU thread 理解。 |

执行层次可以这样背：

```text
Grid
  -> CTA / Block
      -> Warp
          -> Thread
```

## 硬件层次

| 术语 | 初学者解释 | 读什么 |
|------|------------|--------|
| GPU | 整颗加速器，包含很多 SM、L2、HBM 控制器等。 | [[NVIDIA GPU 架构与规格]] |
| SM | Streaming Multiprocessor，真正执行 CTA/warp 的硬件单元。 | [[GPU 硬件架构背景与编程范式]] |
| CUDA Core | 执行 FP32/INT32 等标量/SIMT 指令的单元。 | [[CUDA 编程基础]] |
| Tensor Core | 执行矩阵乘加 MMA 的专用单元，服务 GEMM、attention、convolution。 | [[CUDA GEMM 矩阵乘法优化指南]] |
| Warp Scheduler | SM 内选择 ready warp 发射指令的调度器。 | [[Nsight Compute NCU 分析方法与优化思路]] |
| LD/ST Unit | 处理 load/store 指令的单元，负责 global/shared/local memory 访问。 | [[CUDA Shared Memory 与 Bank Conflict]] |

## 存储层次

| 术语 | 作用 | 重点 |
|------|------|------|
| Register | 每个 thread 私有，最快但数量有限。 | 寄存器太多会降低 occupancy，甚至 spill 到 local memory。 |
| Shared Memory / SMEM | 同一个 CTA 内共享的片上 SRAM；在现代 NVIDIA GPU 上常与 L1 共享 SM 内 unified data cache 资源。 | 适合 tile 缓存和 block 内协作，但要注意 bank conflict 和 carveout。 |
| L1 Cache | SM 附近的自动缓存路径，和 shared memory 属于同一物理层级口径。 | 访问模式会影响命中率；容量可能和 SMEM carveout 互相影响。 |
| L2 Cache | 全 GPU 共享缓存，所有 SM 都能通过 L2 访问 global memory。 | KV cache、权重、跨 SM 复用常会受 L2 影响。 |
| Global Memory / HBM | GPU 显存，容量大、带宽高，但延迟远高于片上存储。 | 高性能 kernel 的核心是减少重复 HBM 访问。 |
| DSMEM | Distributed Shared Memory，cluster 内多个 CTA 的 shared memory 组成的逻辑空间。 | 走片内 SM-to-SM fabric，**与 NVLink 无关**；per-CTA SMEM 仍是 228 KB，cluster aggregate = N × 228 KB。 |

一句话：**Shared Memory 和 L1 在硬件位置上是同层的 SM 内片上资源；区别在于 Shared Memory 由 kernel 显式管理，L1 由硬件缓存机制自动管理。**

存储层次可以这样背：

```text
Register 最快最小
Shared Memory / L1 是 SM 内同层片上资源：前者显式，后者自动缓存
L2 全 GPU 共享缓存
HBM 最大但最远
```

## 性能术语

| 术语 | 解释 | 常见优化动作 |
|------|------|--------------|
| Coalescing | 同一个 warp 的相邻线程访问连续内存，让 global memory transaction 更高效。 | 调整数据布局、线程映射、向量化 load/store。 |
| Bank Conflict | 同一 warp 访问 shared memory 时，多个线程打到同一 bank 的不同地址导致串行化。 | padding、swizzle、转置存储。 |
| Occupancy | 活跃 warp 数 / SM 最大 warp 数。 | 调整 block size、寄存器、shared memory。注意 occupancy 高不等于性能一定高。 |
| Arithmetic Intensity | 每读写一个 byte 能做多少 FLOPs。 | 增大 tile 复用、减少 HBM 重读。 |
| Memory-bound | 性能主要受 HBM/L2/SMEM 带宽或延迟限制。 | 合并访问、缓存复用、shared memory tiling。 |
| Compute-bound | 性能主要受 CUDA Core/Tensor Core 吞吐限制。 | Tensor Core、tile shape、减少非计算指令。 |
| Tail Effect | 最后一波 CTA 数量不足，部分 SM 空闲。 | persistent kernel、调整 grid、合并小任务。 |

优化时按这个闭环走：

![[GPU/Drawings/CUDA Kernel 优化闭环.svg]]

## Tensor Core 与现代 GEMM 术语

| 术语 | 解释 | 读什么 |
|------|------|--------|
| GEMM | General Matrix Multiply，通用矩阵乘。大模型线性层/MLP/QKV projection 的主力。 | [[CUDA GEMM 矩阵乘法优化指南]] |
| Tile | 把大矩阵切成小块，放到 shared memory 或寄存器中复用。 | [[CUDA Kernel 示例：矩阵乘法]] |
| MMA | Matrix Multiply-Accumulate，Tensor Core 执行的矩阵乘加指令族。 | [[GPU 硬件架构背景与编程范式]] |
| WMMA | CUDA C++ 暴露的 warp 级 matrix API，适合理解 Tensor Core 入门。 | [[CUDA GEMM 矩阵乘法优化指南]] |
| `mma.sync` | Ampere 等架构常见的 PTX warp-level MMA 指令。 | [[CUDA GEMM 矩阵乘法优化指南]] |
| `cp.async` | Ampere 的 global-to-shared 异步拷贝，用于构建软件流水线。 | [[GPU 硬件架构背景与编程范式]] |
| TMA | Hopper 的 Tensor Memory Accelerator，用于 bulk tensor async copy。 | [[GPU 硬件架构背景与编程范式]] |
| WGMMA | Hopper 的 warpgroup MMA。一个 warpgroup 通常是 4 个 warp / 128 个线程。 | [[GPU 硬件架构背景与编程范式]] |
| `tcgen05.mma` | Blackwell SM100 GEMM 的第五代 Tensor Core 指令路线。 | [[GPU 硬件架构背景与编程范式]] |
| Block Scale | 低精度 GEMM（MXFP / NVFP4）中多个元素共享一个缩放因子的量化方式。缩放因子格式为 **E8M0（8-bit exponent-only）或 FP8(E4M3)，不是 FP16**。MXFP4 块=32 元素，NVFP4 块=16 元素。 | [[CUDA GEMM 矩阵乘法优化指南]] §13.2 |

## Profiling 术语

| 术语 | 解释 |
|------|------|
| Nsight Systems / nsys | 看系统级 timeline：kernel 顺序、CPU/GPU 交互、通信、空洞。 |
| Nsight Compute / ncu | 看单个 kernel 的详细指标：SM、memory、warp stall、Tensor Core。 |
| Warmup | 正式计时前先跑几轮，排除首次 JIT、cache cold start、allocator 初始化等影响。 |
| JIT | Just-in-Time compilation，运行时编译。不要把所有“首次慢”都叫 JIT。 |
| Heuristics | 库根据 shape/dtype/layout 选择算法或 kernel，不一定发生编译。 |

相关：

- [[Nsight Compute NCU 分析方法与优化思路]]
- [[CUDA JIT、AOT 与 Kernel 选择机制]]
- [[nvjet_kernel_naming]]

## 最小阅读顺序

如果你只想先打通概念，按这个顺序：

1. 本篇：先认词。
2. [[CUDA 编程基础]]：把词放进代码。
3. [[CUDA 线程配置与占用率]]：理解 block/grid 和 SM 资源。
4. [[CUDA Shared Memory 与 Bank Conflict]]：理解片上复用。
5. [[CUDA Kernel 示例：矩阵乘法]]：把 tile 写出来。
6. [[CUDA GEMM 矩阵乘法优化指南]]：再看 Tensor Core 和现代 GEMM。
