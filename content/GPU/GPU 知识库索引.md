---
aliases:
  - GPU
  - CUDA
updated: 2026-06-14
tags:
  - gpu-computing
  - index-note
  - reference-note
---
# GPU 知识库索引

这个目录现在统一承载 GPU 硬件架构、CUDA 编程、kernel 优化、profiling 和 GPU 相关官方资料。根目录只保留总入口、术语表、学习路线和任务清单；具体主题笔记进入对应子目录。本索引是 GPU/CUDA 学习的总入口；[[编程语言索引]] 只保留语言/API 侧反链。

## 目录分层

| 目录 | 定位 | 说明 |
|------|------|------|
| `GPU/` | 索引与高层入口 | 索引、术语表、学习路线、任务清单；不放长篇专题和 PDF 原件。 |
| `GPU/Hardware/` | 硬件视角 | 入门地图、架构演进、规格表——规格类数据的唯一来源。 |
| `GPU/CUDA/` | CUDA 编程与 kernel 笔记 | 执行模型、线程配置、shared memory、thread block cluster、GEMM、示例和 `.cu` 代码。 |
| `GPU/Profiling/` | NCU 与瓶颈分析 | NCU 方法、性能定位流程、Python NCU 操作手册、实测案例。 |
| `GPU/Runtime/` | 运行时 / 库 / trace 读法 | JIT/AOT、PDL、Nsight trace 中的 nvjet kernel 命名解读。 |
| `GPU/References/` | 官方 PDF / 书籍原件 | CUDA Programming Guide、Blackwell 技术概览、H200 datasheet 等，只做引用源。 |
| `GPU/Drawings/` | 稳定 SVG 图 | 面向 Obsidian/浏览器/PDF 导出的普通 SVG，不用 Mermaid 作为主图。 |
| `GPU/GPU-Kernel-Learning/` | 章节式系统课程 | 面向初学者的课程层，负责把 canonical 文档串成可执行学习路径。 |
| `GPU/_archive/` | 非正文归档 | `.edtz` 等编辑快照，不作为 canonical 文档入口。 |

## 整理边界

- 新增主题笔记优先放到 `Hardware/`、`CUDA/`、`Runtime/`、`Profiling/` 之一；根目录只放入口类文档。
- `GPU/GPU-Kernel-Learning/` 是课程层和阅读顺序层，长期结论回链到 canonical 文档，不在课程章节里重复维护完整定义。
- 官方 PDF 原件统一放在 `GPU/References/`；摘录后的规格数字仍以 [[NVIDIA GPU 架构与规格]] 为唯一维护点。
- `.edtz` 等编辑器/同步快照放入 `GPU/_archive/`，除恢复历史内容外不要引用。

## 新手路线

![](</GPU/Drawings/GPU 初学者学习路径.svg>)

建议先走这条路径：

1. [[GPU 硬件背景地图]]：用一张地图把软件入口、执行层次、SM、内存、互联和 profiling 连起来。
2. [[GPU 初学者术语表]]：把 SM、warp、CTA、SMEM、Tensor Core 这些词先对齐。
3. [[GPU 硬件架构背景与编程范式]]：再理解 GPU / SM / warp / memory hierarchy 与 Ampere/Hopper/Blackwell 演进。
4. [[CUDA 编程基础]]：写第一个 kernel，理解 grid/block/thread。
5. [[CUDA 线程配置与占用率]]：学会 block/grid size、occupancy 和 wave。
6. [[CUDA Shared Memory 与 Bank Conflict]]：理解 shared memory、bank conflict 和 coalescing。
7. [[CUDA Kernel 示例：向量加法]]、[[CUDA Kernel 示例：归约求和]]、[[CUDA Kernel 示例：矩阵乘法]]：从小例子练手。
8. [[CUDA GEMM 矩阵乘法优化指南]]：进入 GEMM、Tensor Core、Hopper/Blackwell。
9. [[CUDA Kernel 性能瓶颈定位流程]]：按 NCU 指标定位单个 kernel 的瓶颈和优化方向。
10. [[Nsight Compute NCU 分析方法与优化思路]]：理解 NCU section、指标口径和采集命令。

## 主题分类

### 1. 入门与执行模型

- [[CUDA 编程基础]]：CUDA 软件栈、kernel launch、grid/block/thread、内存拷贝。
- [[GPU 硬件背景地图]]：面向初学者的硬件全局地图（位于 `Hardware/`）。
- [[GPU 初学者术语表]]：把常见硬件、执行、存储和 profiling 术语放到一页。
- [[CUDA 线程配置与占用率]]：block/grid 选择、occupancy、wave、资源约束。
- [[CUDA Stream 与异步执行]]：stream、event、default stream、copy/compute overlap 和 CUDA Graph capture。
- [[CUDA CTA 与 Thread Block Cluster 入门]]：CTA/block、cluster、DSMEM 和 `2cta`。
- [[GPU Kernel 学习路线]]：外部资源和 CUDA/Triton 学习建议。
- [[GPU 学习任务]]：待做 kernel 练习和专项任务。

### 2. 硬件架构与代际演进（位于 `Hardware/`）

- [[NVIDIA GPU 架构与规格]]：H200、GB200、GB300、DGX B300、系统规格口径和官方 PDF——规格类数字的唯一权威来源。
- [[GPU 通信机制与优化技术]]：NVLink、GPUDirect RDMA、PCIe P2P、NCCL/NVSHMEM 软件栈与多机优化 checklist。
- [[GPU 硬件架构背景与编程范式]]：SM、Tensor Core、MMA/PTX、Ampere/Hopper/Blackwell 编程范式演进。
- [[Blackwell 架构新特性与 Kernel 编程]]：`tcgen05.mma`、TMEM、CLC、`cta_group::2`、NVFP4 block scaling、PDL/GDC 的 kernel 编程级整理。
- [[CUDA Shared Memory 与 Bank Conflict]]：shared memory bank、冲突模式和规避（位于 `CUDA/`，作为 shared memory canonical）。

### 3. Kernel 优化与 GEMM

- [[CUDA GEMM 矩阵乘法优化指南]]：GEMM 从 naive 到 Tensor Core、Hopper、Blackwell 的完整参考文档。
- [[Blackwell 架构新特性与 Kernel 编程]]：面向 SM100/Blackwell kernel 的底层机制与实现 checklist。
- [[CUDA Kernel 示例：矩阵乘法]]：教学版 naive / shared tile matmul（位于 `CUDA/`）。

优化闭环：

![](</GPU/Drawings/CUDA Kernel 优化闭环.svg>)

### 4. Profiling 与瓶颈分析（位于 `Profiling/`）

- [[CUDA Kernel 性能瓶颈定位流程]]：面向单个 kernel 的诊断顺序、证据链速查表。
- [[Nsight Compute NCU 分析方法与优化思路]]：**NCU 方法的唯一权威**——section 含义、指标口径、命令、瓶颈→优化映射。
- [[复杂 Python 进程选择性 NCU Profiling 操作手册]]：复杂 Python/PyTorch 程序中只抓目标 kernel 的 NCU 操作手册。
- [[NCU_ANALYSIS]]：Naive vs Tiled MatMul 的 NCU 报告拆解案例。

### 5. 运行时、库与 trace 读法（位于 `Runtime/`）

- [[CUDA JIT、AOT 与 Kernel 选择机制]]：区分 PTX JIT、NVRTC、框架 JIT 和 cuBLASLt heuristics——JIT/AOT canonical。
- [[CUDA Stream 与异步执行]]：CUDA runtime 的异步提交、stream 顺序、event 依赖、默认流语义和 overlap 排障。
- [[CUDA PDL Programmatic Dependent Launch]]：Hopper/Blackwell 上 same-stream dependent kernel 的提前启动机制。
- [[nvjet_kernel_naming]]：解释 Nsight trace 里的 `nvjet_*` cuBLASLt 内部 GEMM kernel 名称。

### 6. CUDA 示例代码（位于 `CUDA/`）

- [[CUDA Kernel 示例：向量加法]]
- [[CUDA Kernel 示例：归约求和]]
- [[CUDA Kernel 示例：矩阵乘法]]

## 阅读路线

| 目标 | 建议路径 |
|------|----------|
| 快速入门 CUDA | [[GPU 初学者术语表]] -> [[CUDA 编程基础]] -> [[CUDA Kernel 示例：向量加法]] -> [[CUDA 线程配置与占用率]] |
| 补硬件架构背景 | [[GPU 硬件背景地图]] -> [[NVIDIA GPU 架构与规格]] -> [[GPU 硬件架构背景与编程范式]] |
| 理解多卡/多机通信 | [[GPU 通信机制与优化技术]] -> [[NVIDIA GPU 架构与规格]]（NVLink 带宽） |
| 理解性能瓶颈 | [[CUDA Kernel 性能瓶颈定位流程]] -> [[Nsight Compute NCU 分析方法与优化思路]] -> [[CUDA 线程配置与占用率]] -> [[CUDA Shared Memory 与 Bank Conflict]] |
| 学 GEMM 优化 | [[CUDA Kernel 示例：矩阵乘法]] -> [[CUDA GEMM 矩阵乘法优化指南]] -> [[NVIDIA GPU 架构与规格]] |
| 做 LLM kernel | [[GPU 硬件架构背景与编程范式]] -> [[CUDA GEMM 矩阵乘法优化指南]] -> [[GPU 学习任务]] 中的 online softmax / RMSNorm / MHA |
| 做 NCU profiling | [[CUDA Kernel 性能瓶颈定位流程]] -> [[Nsight Compute NCU 分析方法与优化思路]] -> [[复杂 Python 进程选择性 NCU Profiling 操作手册]] -> [[NCU_ANALYSIS]] |
| 理解 stream / 异步执行 | [[CUDA 编程基础]] -> [[CUDA Stream 与异步执行]] -> [[CUDA PDL Programmatic Dependent Launch]] -> [[CUDA Kernel 性能瓶颈定位流程]] |
| 看 Nsight 里的库 kernel 名字 | [[CUDA JIT、AOT 与 Kernel 选择机制]] -> [[CUDA PDL Programmatic Dependent Launch]] -> [[CUDA CTA 与 Thread Block Cluster 入门]] -> [[nvjet_kernel_naming]] -> [[Nsight Compute NCU 分析方法与优化思路]] |
| 查 PyTorch / CUDA API 写法 | [[编程语言索引]] -> [[PyTorch 张量与模块速查]] -> [[CUDA 编程基础]] |

## 可靠性约定

- 硬件规格优先引用 NVIDIA 官方产品页或本地官方 PDF，所有规格数字以 [[NVIDIA GPU 架构与规格]] 为唯一来源。
- 教学 kernel 明确标注性能边界，不把示例代码当生产最优实现。
- 涉及 PyTorch 行为时优先参考官方 docs，并注明版本敏感点。
- 主入口以本索引列出的规范文档为准。

## Canonical 文档约定

- `GPU/` 根目录只保留总入口、术语表、学习路线和任务清单；长篇专题文档进入对应子目录。
- `[[CUDA GEMM 矩阵乘法优化指南]]` 位于 `GPU/CUDA/`，是 GEMM 完整主文档，避免重复维护两份长文。
- `[[CUDA 编程基础]]`、`[[CUDA 线程配置与占用率]]`、`[[CUDA Shared Memory 与 Bank Conflict]]`、`[[CUDA CTA 与 Thread Block Cluster 入门]]` 和三个 `CUDA Kernel 示例` 位于 `GPU/CUDA/`，用作 CUDA 编程主线和代码练习。
- `[[NVIDIA GPU 架构与规格]]`（位于 `Hardware/`）是所有硬件规格数字的唯一来源；其他文档涉及具体 TFLOPS / SMEM 容量 / NVLink 带宽时，应 wikilink 回到这里，不在本地重复维护。
- `[[Nsight Compute NCU 分析方法与优化思路]]`（位于 `Profiling/`）是 NCU 方法权威；瓶颈→优化映射表只在此处维护，其他 profiling 文档以 wikilink 引用。
- `[[CUDA JIT、AOT 与 Kernel 选择机制]]`（位于 `Runtime/`）是 JIT/AOT canonical。
- `[[CUDA Stream 与异步执行]]`（位于 `Runtime/`）是 stream、event、default stream 和 overlap 排障 canonical。
- `GPU/GPU-Kernel-Learning/` 是课程式学习层，负责循序渐进；严谨定义和长期维护的结论仍回到上面的 canonical 文档。

## 辅助入口

- [[编程语言索引]]
- [[PyTorch 张量与模块速查]]
- [[GPU 学习任务]]
- [[GPU/GPU-Kernel-Learning/README|GPU Kernel Learning 章节式教程]]
- [[GPU/modern-gpu-programming-for-mlsys/README|Modern GPU Programming for MLSys（English Obsidian edition）]]

## 官方资料

- [[GPU/References/CUDA Programming Guide.pdf|CUDA Programming Guide.pdf]]
- [[GPU/References/NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf|NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf]]
- [[GPU/References/cudabook.pdf|cudabook.pdf]]
- [[GPU/References/NVIDIA Blackwell Architecture Technical Overview.pdf|NVIDIA Blackwell Architecture Technical Overview.pdf]]
- [[GPU/References/hpc-datasheet-sc24-h200-datasheet-3002446.pdf|hpc-datasheet-sc24-h200-datasheet-3002446.pdf]]
