---
aliases:
  - GPU Kernel学习路线
updated: 2026-05-19
tags:
  - gpu-computing
  - cuda-programming
  - triton-kernel
  - learning-roadmap
---
# GPU Kernel 学习路线

这篇只做学习路线入口。主题分类和 canonical 文档以 [[GPU 知识库索引]] 为准；可按周推进的课程层见 [[GPU/GPU-Kernel-Learning/README|GPU Kernel 系统学习课程]]。

## 总路线

![[GPU/Drawings/GPU 初学者学习路径.svg]]

学习顺序不要从 Tensor Core 或 FlashAttention 开始。对初学者更稳的路径是：

```text
硬件地图 -> CUDA 最小程序 -> 线程配置 -> Shared Memory -> 基础 kernel -> GEMM -> profiling -> LLM kernel
```

## 阶段 0：先建立硬件地图

目标：知道 GPU 为什么适合高吞吐并行，以及常见术语在哪一层。

读：

- [[GPU 初学者术语表]]
- [[GPU 硬件背景地图]]
- [[GPU 硬件架构背景与编程范式]]
- [[NVIDIA GPU 架构与规格]]
- [[CUDA CTA 与 Thread Block Cluster 入门]]

必须搞清楚：

- GPU / SM / warp / thread 的层次。
- CTA 和 block 基本等价，cluster 是多个 CTA 的更高一级协作单位。
- HBM、L2、shared memory、register 的速度和作用不同。

## 阶段 1：写出第一个 CUDA Kernel

目标：能独立写、编译、运行最小 CUDA 程序。

读：

- [[CUDA 编程基础]]
- [[CUDA Kernel 示例：向量加法]]

练：

- 一维 vector add。
- grid-stride loop。
- `cudaGetLastError()` 和 `cudaDeviceSynchronize()` 的错误检查。

## 阶段 2：理解线程配置和 occupancy

目标：不再只会复制 `block=256`，知道 block/grid 和 SM 资源之间的关系。

读：

- [[CUDA 线程配置与占用率]]
- [[Nsight Compute NCU 分析方法与优化思路]] 的入门部分。

练：

- 改 block size，观察 kernel 时间。
- 理解一个 SM 可同时驻留多少 block/warp。
- 区分 occupancy、SM active 和实际性能。

## 阶段 3：掌握 shared memory

目标：理解为什么高性能 kernel 不是每次都直接读 HBM。

读：

- [[CUDA Shared Memory 与 Bank Conflict]]
- [[CUDA Kernel 示例：归约求和]]
- [[CUDA Kernel 示例：矩阵乘法]]

练：

- block reduce。
- naive matmul 和 tiled matmul 对比。
- 观察 coalescing、bank conflict、同步开销。

## 阶段 4：进入 GEMM 和 Tensor Core

目标：知道 GEMM 优化从 naive 到 Tensor Core 的主线，不急着手写最底层 PTX。

读：

- [[CUDA GEMM 矩阵乘法优化指南]]
- [[GPU 硬件架构背景与编程范式]] 中 Ampere/Hopper/Blackwell 演进部分。

先理解：

- CTA tile、warp tile、MMA tile 的分层。
- `cp.async`、TMA、WGMMA、`tcgen05` 分别解决什么问题。
- 为什么 tile、layout、pipeline、register、SMEM 是一组联动参数。

## 阶段 5：建立 profiling 闭环

目标：从“感觉优化了”变成“用指标证明优化了”。

![[GPU/Drawings/CUDA Kernel 优化闭环.svg]]

读：

- [[Nsight Compute NCU 分析方法与优化思路]]
- [[CUDA JIT、AOT 与 Kernel 选择机制]]
- [[nvjet_kernel_naming]]

练：

- 用 nsys 看 timeline。
- 用 ncu 看 memory throughput、SM active、warp stall、tensor pipe。
- 区分首次 JIT/heuristics/warmup 和真实 kernel 耗时。

## 阶段 6：进入 LLM Kernel

目标：把 CUDA 基础映射到 LLM 推理里的实际算子。

方向：

- RMSNorm / LayerNorm：reduction + elementwise。
- Softmax / Online Softmax：数值稳定 + reduction。
- Attention / FlashAttention：tile、shared memory、online softmax、KV cache。
- MoE / grouped GEMM：小矩阵、调度、通信和低精度。

相关入口：

- [[GPU 学习任务]]
- [[CUDA GEMM 矩阵乘法优化指南]]
- [[GPU 硬件架构背景与编程范式]]

## 推荐节奏

| 周期 | 目标 | 输出 |
|------|------|------|
| 第 1 周 | CUDA 基础和 vector add | 能写最小 kernel，并解释 grid/block/thread。 |
| 第 2 周 | 线程配置和 shared memory | 能写 reduction，并解释 occupancy 与 bank conflict。 |
| 第 3-4 周 | matmul 和 GEMM 优化主线 | 能写 naive/tiled matmul，知道生产 GEMM 为什么复杂。 |
| 第 5 周 | profiling | 能用 nsys/ncu 判断 memory-bound、compute-bound 或资源 stall。 |
| 第 6 周以后 | LLM kernel | 挑 RMSNorm、softmax、attention、MoE 中一个方向深入。 |

## 组织提醒

- 主题知识优先沉淀到 [[GPU 知识库索引]] 中列出的 canonical 文档。
- 章节式课程放在 `GPU/GPU-Kernel-Learning/`，用于循序渐进学习；不要让它和主笔记维护两份互相冲突的结论。
- 新图优先放 `GPU/Drawings/`，使用普通 SVG，避免 Mermaid 作为主图。
