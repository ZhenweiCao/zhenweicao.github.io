---
aliases:
  - GPU 硬件地图
  - CUDA 硬件心智模型
updated: 2026-05-30
tags:
  - gpu-computing
  - gpu-architecture
  - cuda-programming
  - architecture-note
---
# GPU 硬件背景地图

## 定位

这篇是 GPU 初学者的硬件地图。目标不是背规格，而是建立一个稳定心智模型：当你看到 CUDA kernel、Nsight 指标、GEMM tile、Tensor Core、HBM、NVLink 时，能知道它们分别落在哪一层。

相关主文档：

- [[GPU 知识库索引]]
- [[GPU 初学者术语表]]
- [[CUDA 编程基础]]
- [[GPU 硬件架构背景与编程范式]]
- [[NVIDIA GPU 架构与规格]]
- [[Nsight Compute NCU 分析方法与优化思路]]

## 总图

![[GPU/Drawings/GPU 硬件背景地图.svg]]

## 1. 软件入口：谁发起 GPU 工作

GPU 不会自己开始执行。通常是 CPU 侧程序通过 CUDA Runtime、CUDA Driver、PyTorch、Triton、cuBLAS/cuBLASLt、TensorRT-LLM 等入口发起工作。

常见路径：

```text
Python / C++
  -> CUDA Runtime / Driver
  -> kernel launch 或 library call
  -> GPU 执行一个或多个 kernel
```

初学者先记住：你在 Python 里写一行 `torch.matmul`，底层可能触发 cuBLASLt 选择一个 GEMM kernel；你在 CUDA C++ 里写 `kernel<<<grid, block>>>`，则直接发起一个自定义 kernel。

## 2. 执行层次：谁在并行执行

CUDA 的执行层次是：

```text
Grid
  -> CTA / Block
      -> Warp
          -> Thread
```

对应直觉：

- **Grid**：一次 kernel launch 的全部工作。
- **CTA / Block**：线程协作单位，同一 block 内可以共享 shared memory 和同步。
- **Warp**：NVIDIA GPU 的硬件调度单位，通常 32 个 thread。
- **Thread**：最小软件执行单元，通过 `threadIdx` / `blockIdx` 找到自己负责的数据。

这解释了为什么 CUDA kernel 里经常先写索引计算：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

这句话不是普通数组下标技巧，而是把“很多 thread”映射到“很多数据元素”。

### CTA、block、cluster 先放在同一张图里

初学 CUDA 时可以把 **CTA 和 thread block 当成同一件事的两种叫法**：CUDA C++ 常说 block，PTX、硬件调度和架构文档常说 CTA。一个 CTA/block 通常被调度到一个 SM 上执行，block 内线程共享同一块 shared memory，并能用 `__syncthreads()` 同步。

Thread Block Cluster 是 Hopper 之后多出来的一层：它不是"一个 CTA 跨多个 SM"，而是"多个 CTA 被协同调度到**同一个 GPC**（GPU Processing Cluster）内的一组 SM"。cluster 内 CTA 可以通过 Distributed Shared Memory 访问彼此的 shared memory，**且这一访问走片内 SM-to-SM fabric，与 NVLink 无关**。Cluster size 上限：portable ≤ 8 CTA、opt-in ≤ 16 CTA。详细约束见 [[CUDA CTA 与 Thread Block Cluster 入门]]。

## 3. SM 内部：代码在哪里跑

SM（Streaming Multiprocessor）是 GPU 的基本执行单元。一个 CTA/block 会被调度到一个 SM 上执行，一个 SM 可以同时驻留多个 CTA，前提是寄存器、shared memory、线程数等资源放得下。

你可以先把一个现代 NVIDIA SM 理解为几类部件：

| 部件 | 作用 | 初学者关注点 |
|------|------|--------------|
| Warp scheduler | 选择 ready warp 发射指令 | warp stall 为什么发生。 |
| Register file | 存 thread 私有变量和 accumulator | 寄存器太多会降低 occupancy。 |
| Shared Memory / L1 | SM 内 unified data cache / 片上 SRAM 口径；SMEM 是显式 scratchpad，L1 是自动缓存路径。 | tiling、bank conflict、carveout、缓存命中。 |
| CUDA Core | 执行标量/SIMT 指令 | elementwise、普通 FP32/INT32 操作。 |
| Tensor Core | 执行矩阵乘加 MMA | GEMM、attention、convolution、MoE。 |
| LD/ST Unit | 处理 load/store | coalescing、global/shared memory 访问。 |

更详细的图见 [[GPU 硬件架构背景与编程范式]] 中的 SM 内部执行路径。

## 4. 内存层次：数据在哪里

GPU kernel 的很多优化本质是数据移动优化。先记这个速度/容量直觉：

```text
Register：最快，thread 私有，容量最小
Shared Memory / L1：同在 SM 附近的片上资源；SMEM 显式管理，L1 自动缓存
L2 Cache：全 GPU 共享缓存
HBM / Global Memory：容量最大，带宽高，但延迟远高于片上资源
```

这里要小心“层级”这个词：从**物理位置**看，Shared Memory 和 L1 都在 SM 附近，很多现代架构上还共享同一类 unified data cache 资源；从**编程模型**看，Shared Memory 是 block/CTA 显式读写的地址空间，L1 是硬件自动管理的缓存。

优化时常见问题：

- 相邻 thread 是否访问连续 global memory，也就是 coalescing。
- 同一个 warp 访问 shared memory 时是否发生 bank conflict。判定要点：访问"同一 Bank 不同 word"才冲突；命中"同一 Bank 同一 word"会触发 broadcast/multicast，不冲突。详见 [[CUDA Shared Memory 与 Bank Conflict]]。
- 是否把反复使用的数据从 HBM 搬到 shared memory/register 里复用。
- 是否因为寄存器或 shared memory 使用太多，导致 SM 上可驻留 CTA/warp 太少。

## 5. 系统互联：多 GPU 怎么连

单 GPU 内部主要看 SM、L2、HBM；多 GPU 推理/训练还要看互联。

| 层次 | 常见硬件 | 影响 |
|------|----------|------|
| CPU-GPU | PCIe、NVLink-C2C | host/device 数据传输、Grace Blackwell 共享系统设计。 |
| GPU-GPU 单机 | NVLink、NVSwitch | tensor parallel、pipeline parallel、KV/权重分布。 |
| 多机 | InfiniBand、Ethernet、NCCL | expert parallel、data parallel、跨节点通信。 |

> 具体 NVLink / PCIe 各代带宽数字（单向 / 双向口径）以 [[NVIDIA GPU 架构与规格]] 为唯一来源；本表只做心智模型，不维护数字。常见误读："cluster 间通信走 NVLink"——cluster 是片内概念，DSMEM 走 SM-to-SM fabric，与 NVLink 无关。

所以做 LLM 推理时，不要只看单个 kernel 的 TFLOPS。decode 阶段可能被 KV cache/HBM 限制，多卡 MoE 可能被 all-to-all 通信限制。

## 6. Profiling：怎么验证理解是否正确

硬件地图最终要落到指标上。

| 问题 | 工具 | 看什么 |
|------|------|--------|
| 端到端慢在哪里 | Nsight Systems / `nsys` | CPU gap、kernel timeline、通信、同步。 |
| 单个 kernel 慢在哪里 | Nsight Compute / `ncu` | SM active、memory throughput、warp stall、Tensor Core 利用率。 |
| 首轮为什么慢 | 日志 + cache + profiler | JIT、cuBLASLt heuristics、allocator warmup、cache cold start。 |

优化闭环见：

![[GPU/Drawings/CUDA Kernel 优化闭环.svg]]

## 初学者常见误区

| 误区 | 更准确的理解 |
|------|--------------|
| 线程越多越快 | 线程要足够多，但还受访存、寄存器、SMEM、occupancy 和调度影响。 |
| block 就是 SM | 一个 block/CTA 会在一个 SM 上执行，但一个 SM 可以驻留多个 block。 |
| shared memory 是全 GPU 共享 | 普通 shared memory 是 block 内共享；cluster/DSMEM 才能跨 cluster 内 CTA。 |
| Tensor Core 自动加速所有代码 | 只有满足 dtype、tile、layout、对齐和库/指令路径时才会用上。 |
| 峰值 TFLOPS 等于实际性能 | 实际性能还取决于 shape、batch、访存、调度、通信和 kernel 选择。 |

## 推荐阅读顺序

1. 本篇：先建立硬件地图。
2. [[GPU 初学者术语表]]：把术语对齐。
3. [[CUDA 编程基础]]：把执行层次写成代码。
4. [[CUDA 线程配置与占用率]]：理解 block/grid 与 SM 资源。
5. [[CUDA Shared Memory 与 Bank Conflict]]：理解片上数据复用。
6. [[GPU 硬件架构背景与编程范式]]：进入 Tensor Core、TMA、WGMMA、Blackwell。
7. [[NVIDIA GPU 架构与规格]]：查具体产品规格和官方资料。
