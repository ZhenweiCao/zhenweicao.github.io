---
title: "GPU Kernel 系统学习课程"
content_type: index
maturity: reviewed
publish: true
aliases:
  - GPU Kernel Learning
  - GPU Kernel 系统课程
updated: 2026-05-19
tags:
  - gpu-computing
  - cuda-programming
  - learning-roadmap
---
# GPU Kernel 系统学习课程

这组文档是面向初学者的课程层。它的作用不是替代 [[GPU 知识库索引]] 里的主文档，而是把主文档串成一条可执行的学习路线：

```text
先按章节建立直觉和动手经验
再回到 GPU/ 目录主文档补严谨概念、性能方法和架构细节
```

主入口：

- [[GPU 知识库索引]]：主题分类和 canonical 文档入口。
- [[GPU Kernel 学习路线]]：按阶段学习 GPU kernel 的路线。
- [[GPU 初学者术语表]]：遇到术语先查这里。

## 学习路线图

![[GPU/Drawings/GPU 初学者学习路径.svg]]

推荐顺序：

```text
环境配置
  -> 第1章 基础概念
  -> 第2章 CUDA 入门
  -> 第3章 GPU 硬件
  -> 第4章 优化技巧
  -> 第5章 基础 Kernel
  -> 第6章 大模型 Kernel
  -> 第7章 高级主题
```

## 章节目录

| 章节 | 本章目标 | 配套主文档 | 学完后应该能做什么 |
|------|----------|------------|--------------------|
| [[环境配置指南]] | 准备 CUDA 编译和运行环境 | [[CUDA 编程基础]] | 能确认 GPU、driver、CUDA Toolkit、`nvcc` 可用。 |
| [[第1章-基础概念]] | 建立 GPU/CPU、并行、kernel 的直觉 | [[GPU 初学者术语表]]、[[GPU 硬件架构背景与编程范式]] | 能解释 kernel、grid、block、thread、memory hierarchy。 |
| [[第2章-CUDA入门]] | 写第一个 CUDA 程序 | [[CUDA 编程基础]]、[[CUDA Kernel 示例：向量加法]] | 能写 vector add，知道 host/device memory copy 和错误检查。 |
| [[第3章-GPU硬件]] | 理解 SM、warp、memory、occupancy | [[CUDA 线程配置与占用率]]、[[CUDA Shared Memory 与 Bank Conflict]] | 能解释为什么 block size、寄存器、SMEM 会影响性能。 |
| [[第4章-优化技巧]] | 学会从内存、计算、资源三个方向优化 | [[CUDA Kernel 性能瓶颈定位流程]]、[[Nsight Compute NCU 分析方法与优化思路]]、[[GPU 初学者术语表]] | 能用 profiling 闭环提出一个优化假设。 |
| [[第5章-基础Kernel]] | 实现和优化基础 kernel，重点是 matmul | [[CUDA Kernel 示例：矩阵乘法]]、[[CUDA GEMM 矩阵乘法优化指南]] | 能写 naive/tiled matmul，并说明 shared memory tiling 的价值。 |
| [[第6章-大模型Kernel]] | 把 CUDA 基础映射到 softmax、norm、attention | [[GPU 学习任务]]、[[CUDA GEMM 矩阵乘法优化指南]] | 能看懂 LLM kernel 的数据流，不再把所有问题都当 GEMM。 |
| [[第7章-高级主题]] | Tensor Core、多 GPU、低精度、融合、CUTLASS | [[GPU 硬件架构背景与编程范式]]、[[CUDA JIT、AOT 与 Kernel 选择机制]] | 知道高级主题的入口和边界，不急着手写底层 PTX。 |

## 每章怎么学

每一章按同一个节奏走：

1. 先读本章，抓住“为什么需要这个概念”。
2. 跳到配套主文档，确认严谨定义和常见误区。
3. 做本章练习或对应 kernel 示例。
4. 用 [[CUDA Kernel 性能瓶颈定位流程]] 的模板记录一次指标，再回到 [[Nsight Compute NCU 分析方法与优化思路]] 查指标口径。
5. 回来补一段自己的笔记：shape、GPU 型号、耗时、瓶颈判断、下一步优化。

优化时按这个闭环：

![[GPU/Drawings/CUDA Kernel 优化闭环.svg]]

## 推荐节奏

| 周期 | 学习内容 | 输出 |
|------|----------|------|
| 第 1 周 | 环境配置、第1章、第2章 | 跑通 vector add，能解释 grid/block/thread。 |
| 第 2 周 | 第3章、第4章 | 能解释 warp、SMEM、occupancy、coalescing、bank conflict。 |
| 第 3 周 | 第5章 | 跑通 naive matmul 和 shared-memory matmul，写一份性能对比。 |
| 第 4 周 | GEMM 主文档 + NCU | 能按 [[CUDA Kernel 性能瓶颈定位流程]] 判断一次 kernel 是 memory-bound、compute-bound 还是资源受限。 |
| 第 5 周以后 | 第6章、第7章 | 选择 softmax、RMSNorm、attention 或 MoE 中一个方向深入。 |

## 目录内代码

目前可直接参考的本地代码：

- [[matmul.cu|GPU/CUDA/matmul.cu]]：包含 naive matmul 和 shared-memory matmul。
- [[reduction_sum.cu|GPU/CUDA/reduction_sum.cu]]：归约求和示例。

如果后续新增代码，建议放在 `GPU/CUDA/` 或专门的 `GPU/CUDA/code/` 下，并从对应章节和主文档双向链接。

## 维护规则

- 课程章节负责“从零开始讲清楚”，语言可以更直观。
- `GPU/` 根目录主文档负责“严谨、可查、可复用”，不要在章节里维护另一份互相冲突的结论。
- 复杂流程图放 `GPU/Drawings/`，使用普通 SVG 或 Excalidraw 源文件；不要在 GPU 目录新增 Mermaid。
- 章节里出现新术语时，优先补到 [[GPU 初学者术语表]]，再在具体章节解释。
