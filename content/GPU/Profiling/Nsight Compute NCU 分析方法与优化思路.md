---
aliases:
  - NCU 分析方法
  - Nsight Compute 指标速查
  - CUDA Kernel Profiling
updated: 2026-05-30
tags:
  - gpu-computing
  - cuda-programming
  - performance-profiling
---
# Nsight Compute NCU 分析方法与优化思路

这篇记录如何用 `ncu` 分析 CUDA kernel，以及拿到瓶颈证据后应该优先尝试哪些优化。它不是指标字典，而是一条工作流：先定位 kernel，再判断瓶颈类型，最后把指标映射到代码修改。

相关入口：

- [[GPU 知识库索引]]
- [[CUDA 线程配置与占用率]]
- [[CUDA Shared Memory 与 Bank Conflict]]
- [[CUDA GEMM 矩阵乘法优化指南]]
- [[CUDA Kernel 性能瓶颈定位流程]]
- [[复杂 Python 进程选择性 NCU Profiling 操作手册]]
- [[GPU 硬件架构背景与编程范式]]
- [[CUDA 编程基础]]
- [[NCU_ANALYSIS]]

## NCU 适合分析什么

`ncu` / NVIDIA Nsight Compute 是 kernel 级 profiler，适合回答：

- 某个 CUDA kernel 慢在哪里。
- SM、Tensor Core、memory pipeline、L2、DRAM、shared memory 是否被打满。
- occupancy、register、shared memory、block/grid 配置是否限制并行度。
- warp 为什么发不出指令：访存依赖、同步、指令管线、分支发散等。
- 源码或 SASS 中哪几行贡献了主要 stall、访存或指令开销。

它不适合单独回答：

- 端到端请求延迟为什么高。
- CPU 调度、Python overhead、kernel launch gap、NCCL 通信排队、跨进程同步问题。

这些问题先用 Nsight Systems / `nsys` 或业务 trace 找到时间线，再用 `ncu` 钻进关键 kernel。

## 基本流程

![[GPU/Drawings/NCU Kernel 分析闭环.svg]]

一个健康的分析习惯是：每轮只验证一个假设。不要一次收集 `--set full`、改 tile、改 block size、改数据布局、再换 dtype，否则很难解释收益来自哪里。

## NCU 报告的背景地图

NCU 报告不是一张“分数表”，而是一组从不同角度观察同一个 kernel 的 section。初学者可以先记这张表：

| Section | 先回答什么问题 | 常见下一步 |
|---------|----------------|------------|
| SpeedOfLight | compute / memory 资源谁接近峰值，瓶颈大方向在哪里。 | 展开 breakdown，确认是 SM、L1TEX、L2 还是 DRAM。 |
| LaunchStats | kernel launch 形态是否合理：grid、block、寄存器、shared memory。 | 看 grid 是否太小、block 是否太大、资源是否限制驻留 block。 |
| Occupancy | SM 上有多少 active warp 用来隐藏延迟。 | 区分 theoretical occupancy 和 achieved occupancy。 |
| MemoryWorkloadAnalysis | 访存压力落在哪一级：L1TEX、shared memory、L2、DRAM。 | 看 sector/request、hit rate、bank conflict、load/store pattern。 |
| ComputeWorkloadAnalysis | 计算指令和管线利用情况。 | 看 tensor pipe、FP/INT/SFU、load/store pipe 是否失衡。 |
| SchedulerStats | warp scheduler 是否持续发射指令。 | 如果 eligible warp 少，再看 WarpStateStats。 |
| WarpStateStats | warp 为什么不能 issue 下一条指令。 | long scoreboard、barrier、dependency、pipe throttle 等。 |
| Source / SASS | 热点指标落在哪一行源码或哪类 SASS 指令。 | 修改具体 load/store、barrier、layout、tile 或指令路径。 |
| Roofline | 算术强度和 achieved FLOPS 是否贴近算力/带宽 roof。 | 验证 compute-bound、memory-bound 或离 roof 很远。 |

一句话顺序：

```text
先看 SOL 定大方向
-> 再看 Launch / Occupancy 判断硬件是否被喂饱
-> 再看 Memory / Compute 定具体资源
-> 再看 Scheduler / Warp State 解释为什么发不出指令
-> 最后回到 Source / SASS 改代码
```

`[[NCU_ANALYSIS]]` 是一个 naive matmul 和 tiled matmul 的完整 NCU 报告拆解案例，可以作为本文方法的配套练习。

如果你只想快速定位一个 kernel 的瓶颈，不想先读完本文所有指标解释，先看 [[CUDA Kernel 性能瓶颈定位流程]]；本文负责解释每个 NCU section 和指标口径。

如果目标程序是复杂 Python/PyTorch 进程，只想抓其中少数 kernel，先看 [[复杂 Python 进程选择性 NCU Profiling 操作手册]]，里面整理了 NVTX、kernel name、`launch-skip/count`、`cudaProfilerStart/Stop` 和子进程过滤。

## 准备工作

### 编译信息

为了让 NCU 把 SASS / PTX / CUDA-C 关联起来，编译时建议带上 source line 信息：

```bash
nvcc -O3 -lineinfo ...
```

如果是 PyTorch extension、Triton 或框架 JIT kernel，也要尽量保留源码路径、kernel name 和 NVTX range。NCU 的 Source 页面只有在可解析源码信息时，才能更好地把指标落到源码行。

### 基准稳定性

- 固定输入规模、batch、sequence length、dtype、GPU 型号和驱动/CUDA 版本。
- 做 warmup，避免首次 JIT、cache cold start、lazy init 混入结果。
- 单独记录 baseline：端到端 latency、目标 kernel 时间、吞吐、正确性误差。
- `ncu` 可能使用 replay 多次采集指标，profiling 本身会改变运行时间；不要把 NCU 下的 wall time 当线上延迟。
- 多进程、多 GPU、NCCL/NVSHMEM 等强依赖并发的 kernel 要谨慎，因为 profiling 可能序列化或需要 lockstep 设置。

## 常用命令

先看版本和可用 set / section：

```bash
ncu --version
ncu --list-sets
ncu --list-sections
```

粗看所有 kernel，不建议用于大型模型全量长跑：

```bash
ncu --set basic -o ncu_basic ./app args...
```

只采一个 kernel，减少 profiling 开销：

```bash
ncu --set full \
  --kernel-name regex:".*my_kernel.*" \
  --launch-skip 10 \
  --launch-count 1 \
  -f -o ncu_my_kernel \
  ./app args...
```

只收关键 section，适合迭代优化：

```bash
ncu \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --kernel-name regex:".*my_kernel.*" \
  --launch-count 1 \
  -f -o ncu_target \
  ./app args...
```

按 NVTX range 过滤，适合 PyTorch / LLM serving：

```bash
ncu --nvtx \
  --nvtx-include "decode/" \
  --kernel-name regex:".*attention.*" \
  --launch-count 1 \
  -f -o ncu_decode_attention \
  python bench.py
```

导出或命令行查看已有报告：

```bash
ncu --import ncu_target.ncu-rep --page details
ncu --import ncu_target.ncu-rep --page raw --csv
```

针对某个假设补采指标：

```bash
# 展开 Memory Throughput 到底被谁拉高
ncu --metrics breakdown:gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
  --kernel-name regex:".*my_kernel.*" ./app

# 看 L1 / L2 / DRAM 三层 throughput
ncu --metrics \
  l1tex__throughput.avg.pct_of_peak_sustained_active,\
  lts__throughput.avg.pct_of_peak_sustained_elapsed,\
  gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
  --kernel-name regex:".*my_kernel.*" ./app

# 看 shared memory bank conflict
ncu --metrics \
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
  --kernel-name regex:".*my_kernel.*" ./app

# 看 scheduler issue 和 active warps
ncu --metrics \
  sm__issue_active.avg.pct_of_peak_sustained_elapsed,\
  sm__warps_active.avg.pct_of_peak_sustained_active \
  --kernel-name regex:".*my_kernel.*" ./app
```

## 阅读顺序

### 1. 先看 kernel 时间和 launch 形态

关注：

- kernel duration。
- grid/block 维度。
- 每线程寄存器数。
- 每 block shared memory。
- theoretical occupancy 和 achieved occupancy。
- waves / tail effect：block 数是否太少，最后一波是否浪费很多 SM。

常见结论：

- kernel 很短但 launch 很多：可能是 launch overhead 或框架调度问题，先回到 `nsys` 看 timeline，而不是钻单个 kernel。
- grid 太小：SM 没喂饱，增加并行任务数、拆分更细粒度 tile，或者把多个独立任务合并成更大的 grid。
- register / shared memory 限制 occupancy：要判断这是必要的 tile 代价，还是可减少的资源浪费。

#### LaunchStats 背后的几个公式

NCU 的 LaunchStats 和 Occupancy 不是孤立的。它们共同回答：一个 SM 同时能放多少个 block/CTA、多少个 warp。

常用公式：

```text
warps_per_block = ceil(threads_per_block / 32)

resident_blocks_per_sm
  = min(
      block_limit_registers,
      block_limit_shared_memory,
      block_limit_warps,
      block_limit_blocks,
      block_limit_barriers,
      launch_bounds_or_cluster_limit
    )

theoretical_active_warps_per_sm
  = resident_blocks_per_sm * warps_per_block

theoretical_occupancy
  = theoretical_active_warps_per_sm / max_warps_per_sm

waves_per_sm
  = grid_blocks / (num_sms * resident_blocks_per_sm)
```

`resident_blocks_per_sm` 不是硬件规格里的绝对最大 block/SM，而是这个 kernel 在当前寄存器、shared memory、warp 数、barrier、cluster 等约束下，一个 SM 最多能同时驻留的 block 数。`num_sms * resident_blocks_per_sm` 就是一波最多能同时跑的 block 数。

读法：

- `Block Limit Registers` 低：每线程寄存器太多，SM 放不下更多 block。
- `Block Limit Shared Mem` 低：每 block shared memory 太多，限制驻留 block。
- `Block Limit Warps` 低：block 太大，warp 数本身把 SM 塞满。
- `Waves Per SM` 很小：总 block 数太少，SM 没有足够工作。
- `Waves Per SM` 不是整数：最后一波可能只有部分 SM 有活干，形成 tail effect。

例子：

```text
grid_blocks = 4096
num_sms = 148
resident_blocks_per_sm = 2

一波最多同时跑的 block 数 = 148 * 2 = 296
waves_per_sm = 4096 / 296 = 13.84
```

直觉上就是：前面 13 波都能把 148 个 SM 基本铺满，每个 SM 同时跑 2 个 block；最后还有 0.84 波的剩余 block。`0.84` 足够大，说明尾波只有轻微浪费。如果是 `13.05`，最后一波就很瘦，tail effect 会更明显。

一个例子：1024 threads/block 等于 32 warps/block。如果某个 GPU 最大 64 warps/SM，那么只要能驻留 2 个 block/SM，理论 occupancy 就已经是 100%。即使 `Block Limit Registers = 2` 看起来像“寄存器限制很严”，它限制的是 block 数；但两个大 block 已经把 64 个 warp 槽位填满了。这时继续追 occupancy 没意义，应该看 issue active、stall reason、访存和计算管线。

### 2. Speed Of Light 判断大方向

Speed Of Light 常缩写成 SOL。这里的 “Speed Of Light” 不是物理光速，而是 Nsight Compute 对硬件理论/持续峰值吞吐的比喻：**某类硬件资源在这个 kernel 运行期间，达到了自身峰值能力的多少百分比**。

可以把它当成第一眼的性能罗盘：

```text
SOL = achieved throughput / peak throughput
```

常见字段包括 SM / Compute throughput、Memory throughput、L2 throughput、DRAM throughput 等。100% 不代表“程序完美”，而是说明某个资源已经接近它在该口径下的峰值；低百分比也不一定坏，可能只是这个 kernel 本来不使用那类资源。

用 SpeedOfLight 先判断：

- SM / Compute throughput 高，memory throughput 低：更像 compute-bound。
- DRAM / L2 / memory throughput 高，SM throughput 上不去：更像 memory-bound。
- 两者都不高：常见是 latency hiding 不足、同步/分支/依赖、grid 太小、tail wave、指令混合不理想。

读法示例：

| 看到的现象 | 初步判断 | 下一步看什么 |
|------------|----------|--------------|
| SM throughput 很高，DRAM/L2 不高 | 更像算力或指令管线受限 | Tensor Core 利用率、指令 mix、tile shape、SchedulerStats。 |
| DRAM throughput 很高，SM throughput 不高 | 更像 HBM 带宽受限 | coalescing、重复读取、数据复用、Roofline。 |
| L1/TEX/memory pipeline 高，但 DRAM 不高 | 可能是缓存/访存管线被小而散的请求堵住 | MemoryWorkloadAnalysis、transaction、sector、load/store pattern。 |
| SM 和 Memory 都低 | 可能不是峰值吞吐问题，而是喂不饱硬件 | Occupancy、eligible warps、stall reason、grid size、tail effect。 |

不要只看一个百分比。比如 memory throughput 高不一定等于 DRAM 带宽打满，可能是 memory pipeline 被大量小而散的访存指令堵住；SM throughput 高也不等于 Tensor Core 一定吃满，可能是其他 SM 子管线接近峰值。要打开 SpeedOfLight 的 breakdown，再结合后面的 MemoryWorkloadAnalysis、SchedulerStats 和 Roofline。

#### SM throughput 怎么理解

NCU 的 `SM Throughput` 通常对应：

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
```

它是一个 throughput metric，表示 SM 子系统在 kernel elapsed cycles 内接近持续峰值能力的程度。它不是 FLOPS 利用率，也不是 Tensor Core 利用率。

概念上可以这样理解：

```text
sm__throughput
  ≈ max(SM breakdown 中各个子管线/子活动各自的 % of peak)
```

实际哪些子指标参与计算会随 GPU 架构和 NCU 版本变化，不建议背死。需要确认时，用：

```bash
ncu --metrics breakdown:sm__throughput.avg.pct_of_peak_sustained_elapsed ...
```

读法：

- `SM Throughput` 高：说明至少有某些 SM 内子管线/活动接近峰值，但不代表 Tensor Core、FMA 或所有计算单元都吃满。
- `SM Throughput` 低：说明 SM 高层活动不饱，可能是 grid 太小、tail effect、eligible warp 少、依赖链、同步、访存等待、分支等。
- 两个 kernel 的 `SM Throughput` 接近：只能说明平均 SM 高层资源压力相近。谁更快，还要看 duration / elapsed cycles / 完成的工作量。

访存较慢会影响 `SM Throughput`，但多半是间接影响：

- 如果 warp 在等 global/local memory，scheduler 没有足够 eligible warp 可发，SM 子管线空转，`SM Throughput` 会下降。
- 如果 occupancy 和独立工作足够多，访存延迟被隐藏，`SM Throughput` 可能仍然不低。
- 如果 load/store、shared memory 或 L1TEX 路径本身很忙，瓶颈可能同时表现为 memory throughput / L1TEX throughput 高，而不是简单表现为 SM throughput 低。

因此，判断“是不是访存拖慢了 SM”，要把 `SM Throughput` 和 `Issue Active`、`eligible warps`、`Long Scoreboard`、`Memory Throughput`、`L1/TEX`、`L2`、`DRAM` 放在一起看。

#### Memory throughput 和 DRAM throughput 的区别

这两个指标最容易混：

| 指标 | 看的是哪里 | 回答的问题 |
|------|------------|------------|
| Memory Throughput | 整个 memory subsystem 的高层 SOL 概览，可能由 L1/TEX、shared memory、L2、DRAM、内部端口/链路等子指标中的高贡献项决定。 | 这个 kernel 有没有把某段内存系统资源逼近峰值？ |
| DRAM Throughput | 设备显存/HBM 侧吞吐，主要是 L2 和 device memory 之间真实发生的 off-chip 数据传输。 | 这个 kernel 有没有真的大量访问 HBM？ |

所以：

- **Memory Throughput 高，DRAM Throughput 也高**：更像 HBM 带宽受限。优先看 global memory 访问量、coalescing、L2 hit rate、重复读取、数据类型和 tile 复用。
- **Memory Throughput 高，DRAM Throughput 不高**：不一定是 HBM 瓶颈，可能是 L1/TEX、shared memory、L2、memory pipeline 或端口压力高。优先看 MemoryWorkloadAnalysis 里的 L1/TEX、L2、Shared Memory、bank conflict、sectors/request、transaction pattern。
- **Memory Throughput 低，DRAM Throughput 低**：通常不是吞吐型 memory-bound，可能是 occupancy、eligible warp、同步、依赖链、分支、grid 太小或 tail effect。

一句话：**Memory Throughput 是“内存系统哪里忙不忙”的总览；DRAM Throughput 是“有没有真的打到 HBM/显存带宽”的专项指标。**

NCU 里的 `Memory Throughput` 通常对应：

```text
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
```

它的计算口径不是“所有内存层级 GB/s 相加”，而是 Nsight Compute 的 throughput metric 口径：

```text
先把多个底层 counter 各自换算成 % of peak
再从这些 constituent / breakdown 指标里取最高贡献项作为高层 throughput
```

概念上可以理解成：

```text
Memory Throughput
  = max(
      L1/TEX 相关 throughput,
      shared memory / L1 data pipe 相关 throughput,
      L2 相关 throughput,
      DRAM / framebuffer 相关 throughput,
      memory request/access internal activity ...
    )
```

实际包含哪些 breakdown 指标会随 GPU 架构和 NCU 版本变化，不建议背死。需要看当前报告时，用：

```bash
ncu --metrics breakdown:gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed ...
```

或者在 Nsight Compute UI 的 Speed Of Light / Details 里展开 `Memory Throughput Breakdown`。

例如：

```text
L1/TEX throughput: 82%
L2 throughput:     35%
DRAM throughput:   18%
Memory Throughput: 82%
```

这个结果说明内存系统的高层 SOL 被 L1/TEX 或 SM 内存管线拉高，但不能说 HBM 打满。要判断 HBM 带宽，优先看 `gpu__dram_throughput...`、DRAM read/write bytes、L2 hit rate 和 sectors/request。

#### L1/TEX 里的 TEX 是什么

TEX 原本指 texture pipeline / texture unit。现代 NCU 里常见的 `L1/TEX` 或 `l1tex__*`，更准确地说是 **SM 附近的 L1TEX 子系统**，它把 L1 data cache、texture cache、shared memory 相关数据通路放在同一个硬件/指标口径下观察。

所以 `L1/TEX throughput` 不等于“texture 访问很多”。它可能来自：

- global / local memory 访问经过 L1TEX 管线。
- texture / surface 访问经过 TEX 管线。
- shared memory 访问使用同一套 shared data / bank 资源。
- L1TEX 内部 request、sector、wavefront、bank read/write、pipe active 等活动。

因此你的理解大方向是对的：**它覆盖的是 L1 层级附近的综合活动，shared memory 和 L1 cache 相关活动都可能贡献到 L1/TEX throughput。** 但要注意两点：

1. 它不是“所有 L1 访问量的字节求和”，而是多个底层活动各自换算成 `% of peak` 后形成的 throughput 指标。
2. 它不只统计 cache hit，也可能反映 request、sector、pipeline、bank、writeback 等内部活动。

想知道到底是哪一类活动把它拉高，看 breakdown 里的具体名字：

| breakdown 名字线索 | 常见含义 |
|--------------------|----------|
| `data_bank_reads` / `data_bank_writes` | shared memory / L1TEX data bank 读写压力。 |
| `data_pipe_lsu_wavefronts` | load/store 单元发到 L1TEX 的数据管线活动。 |
| `tex` / `surface` | texture 或 surface 路径活动。 |
| `sectors` / `requests` | cache line sector 或请求数量，常用来看访问是否碎。 |
| `writeback` | 写回相关活动。 |

#### L1 hit rate 为 0 不一定是坏事

NCU 里某些 `l1tex__t_*_hit_rate` 指标主要描述 texture/global/local 这类走 L1TEX cache 路径的访问。它不是“所有 SM 内片上访问”的统一命中率。

典型例子：

```text
naive matmul:
  A/B 反复从 global memory 读取
  -> 可能看到较高 L1 hit rate

tiled matmul:
  A/B tile 先从 global memory 搬进 shared memory
  后续计算主要读 shared memory
  -> 某些 L1 cache hit rate 可能很低甚至为 0
```

这不代表 tiled kernel 没有利用片上存储。它只是把复用从“硬件 L1 cache 自动命中”改成了“shared memory 显式复用”。这时更应该看：

- shared memory load/store 指标。
- `data_bank_reads` / `data_bank_writes`。
- bank conflict。
- barrier stall。
- global load/store 总量是否减少。

所以读 NCU 内存指标时要先分清：

| 代码访问 | 常见路径 | 更该看什么 |
|----------|----------|------------|
| global / local load-store | L1TEX -> L2 -> DRAM | L1/L2 hit rate、sectors/request、DRAM throughput、long scoreboard。 |
| shared memory load-store | SM 内 shared data bank | bank read/write、bank conflict、barrier stall、shared load/store hotspot。 |
| texture / surface | TEX -> L1TEX -> L2 -> DRAM | TEX 相关 request、sector、hit rate。 |

#### Workload Distribution 活跃周期怎么看

有些报告会列出 SM、L1、L2、DRAM 的 active cycles 或 active rate。它回答的不是“传了多少字节”，而是“这一级硬件在 kernel 期间有多长时间处于忙碌状态”。

常见读法：

- DRAM active 很低，L1TEX 很高：大概率不是 HBM 带宽瓶颈，压力更可能在 L1TEX、shared memory、访问模式或 pipe。
- L2 active 明显高于 L1TEX：可能是很多 SM 的请求汇聚到 L2，或者 L1 复用不足。
- L1TEX 和 SM active 都高：不一定坏，可能是 kernel 工作很饱满；要结合 duration 和 issue active 看是否真的卡住。
- active cycles 高不等于 throughput 高，throughput 还要结合峰值口径和请求大小。

### 3. 用 Roofline 确认算力/带宽边界

Roofline 把 arithmetic intensity 和 achieved FLOPS 放在同一张图里：

- 点在斜线附近：更接近带宽受限，优化数据移动。
- 点在平顶附近：更接近计算受限，优化 Tensor Core / 指令吞吐。
- 点离两条 roof 都远：通常不是单纯带宽或算力峰值问题，要看调度、occupancy、stall 和代码结构。

> **算力 roof 口径**：NCU 默认计算并报告的 FLOPS 是 **dense** 口径（不计入 2:4 结构化稀疏的等效翻倍）。如果 kernel 走了 `__nv_sparse_meta` / cuSPARSELt 稀疏 GEMM 路径，需要单独读 sparse 指标或手工乘以 2 才能与 NVIDIA 官方页 sparse TFLOPS 对齐。涉及具体硬件峰值数字时，统一参考 [[NVIDIA GPU 架构与规格]] §"Tensor Core 代际"。

对 LLM 推理尤其要分阶段：

- Prefill：大 GEMM / attention tile 更可能接近 compute-bound 或 Tensor Core pipeline。
- Decode：单 token、小 batch、长 KV 读取，常转向 memory-bound / latency-bound。
- MoE：除了 expert GEMM，还要看 dispatch、combine、load balance 和跨卡通信；单个 kernel 的 NCU 只解释局部。

### 4. 再看 Scheduler / Warp State

SchedulerStats 和 WarpStateStats 用来回答“为什么 issue 不出去”：

- eligible warps 少：没有足够可发射 warp，可能 occupancy 不足或依赖链太长。
- issue slot 空：latency hiding 不足，继续看 stall reason。
- stall long scoreboard 高：常见是等待 global/local memory 依赖。
- stall barrier 高：同步太频繁或 block 内负载不均。
- stall math pipe / tensor pipe / MIO throttle 高：对应指令管线拥塞，要看指令类型和数据流。

注意：stall reason 只是症状。官方文档也强调，只有当 scheduler 不能持续 issue 时，stall reason 才值得重点追。否则“有 stall”不等于它限制了性能。

### 5. 最后落到 Source / SASS

Source 页面要回答：

- 哪几行产生最多 global load/store、shared load/store、branch、barrier。
- 热点行对应的 SASS 指令是什么，是否走了预期的 `LDG`、`LDS`、`STG`、`MMA`、`WGMMA`、`LDMatrix`、`CP.ASYNC` / TMA。
- 是否出现 local memory load/store，提示 register spill。
- 编译器是否把循环展开、向量化、常量传播做出来了。

源码行级信息特别适合优化小 kernel；对 CUTLASS/Triton/框架生成 kernel，则更适合结合生成后的 SASS 和模板参数看。

## 瓶颈到优化动作

| 观察 | 常见原因 | 优先优化 |
|---|---|---|
| SM 和 memory 利用率都低 | grid 太小、tail wave、launch 过碎、依赖链长 | 增大 grid、改善任务划分、合并小 kernel、增加独立工作、减少串行依赖 |
| DRAM throughput 高 | HBM 带宽受限 | 减少读写字节、融合算子、向量化访存、使用更低精度、提高数据复用、压缩 KV/activation |
| Memory busy 高但 DRAM 带宽不高 | 访存指令太多、小访问、非合并、地址分散 | coalescing、`float4`/`half2` 向量化、调整 layout、减少 pointer chasing、批量化访问 |
| L2 hit rate 低 | 工作集大、复用距离长、访问随机 | blocking/tiling、重排数据、提高局部性、复用 shared/register、减少重复读 |
| Shared bank conflict 高 | shared memory layout 与 warp 访问模式冲突 | padding、swizzle、改变 tile layout、使用 `ldmatrix` 友好布局 |
| Achieved occupancy 低 | register/SMEM/block 上限限制 | 调整 block size、减少寄存器 live range、减小 tile、拆 kernel、使用 `launch_bounds`；但不要盲目追 100% |
| Long scoreboard 高 | 等待 global/local memory 数据 | 提前加载、软件流水、增加 independent work、提高 occupancy、减少 cache miss、避免 spill |
| Barrier stall 高 | `__syncthreads()` 频繁或 warp 间负载不均 | 减少同步、warp-level primitive、双缓冲、重新划分 block 内工作 |
| Branch divergence 高 | 同一 warp 内控制流分裂 | 数据重排、按分支拆 kernel、使用 predication、让 warp 内处理同类元素 |
| Tensor pipe 利用率低 | 没走 Tensor Core、tile 太小、维度不对齐、数据供应不足 | 使用 cuBLAS/CUTLASS/Triton Tensor Core 路径、对齐 M/N/K、扩大 tile、pipeline global/shared/register |
| Tensor pipe 高但总性能低 | Tensor Core 在算，但被 epilogue/访存/同步拖住 | 融合 epilogue、减少写回、改善 shared/register pipeline、检查 accumulator 和 store 布局 |
| Local memory 访问高 | register spill | 减少 live variable、拆函数/拆 kernel、减小 unroll/tile、检查编译器寄存器限制 |
| Instruction throughput 受限 | div/mod、复杂地址计算、特殊函数太多 | 用位运算/乘加替代、预计算、近似函数、减少索引表达式复杂度 |

## 常见分析模板

### Memory-bound kernel

判断链：

```text
SpeedOfLight: memory 高、compute 低
-> Roofline: 靠近 memory roof
-> MemoryWorkloadAnalysis: DRAM/L2 哪层高
-> Source: 哪些 load/store 行最热
```

优化顺序：

1. 少搬：融合算子、减少中间结果、低精度、只读必要字段。
2. 搬得顺：coalesced、对齐、向量化、连续布局。
3. 多复用：shared memory tile、register blocking、L2 locality。
4. 藏延迟：提高 occupancy、prefetch、double buffering、增加独立计算。

### Compute-bound GEMM / Tensor Core kernel

判断链：

```text
SpeedOfLight: compute / tensor pipe 高
-> Roofline: 接近 compute roof
-> ComputeWorkloadAnalysis: tensor / fp pipe 利用
-> Source/SASS: 是否是预期 MMA/WGMMA 指令
```

优化顺序：

1. 确认真的走 Tensor Core：dtype、layout、对齐、矩阵维度是否满足要求。
2. 提高 tile 复用：CTA tile、warp tile、MMA tile 分层合理。
3. 做流水：global->shared、shared->register、MMA 重叠。
4. 降低非 matmul 开销：epilogue fusion、减少 mask/branch/rescale。
5. 接受较低 occupancy：GEMM 常用更大 tile 和更多寄存器换取数据复用，25%-50% occupancy 也可能很好。

### Latency-bound / issue 不足

判断链：

```text
SpeedOfLight: compute 和 memory 都不高
-> SchedulerStats: eligible warps 少 / issue slot 空
-> WarpStateStats: long scoreboard、barrier、wait、dependency
-> LaunchStats/Occupancy: 并行度是否被资源限制
```

优化顺序：

1. 增加可运行 warp：调 block size、寄存器、shared memory。
2. 增加每 warp 独立工作：unroll、thread coarsening、预取下一批数据。
3. 缩短依赖链：重排计算，减少 load 后立即使用。
4. 减少同步和分支：warp primitive、分支分组、减少 block-wide barrier。

### Shared memory / bank conflict

判断链：

```text
MemoryWorkloadAnalysis / SourceCounters
-> shared load/store 相关指标异常
-> bank conflict 指标高
-> 源码定位 shared memory 索引表达式
```

优化顺序：

1. 画出同一个 warp 的 lane -> shared address -> bank 映射。
2. 对二维 tile 加 padding，例如 `TILE_DIM + 1`。
3. 改 shared layout 或 swizzle，让连续 lane 落到不同 bank。
4. 对 Tensor Core 路径检查 `ldmatrix` 期望的数据布局。

## 实战案例：Naive vs Tiled MatMul 怎么读

`[[NCU_ANALYSIS]]` 里对比了 2048x2048 FP32 naive matmul 和 tiled matmul。这个案例适合用来理解：**高 Memory Throughput 不等于 HBM 带宽瓶颈**。

报告里的关键现象：

| 指标现象 | 怎么读 |
|----------|--------|
| Tiled 比 Naive 更快，但两者 DRAM Throughput 都很低 | 性能差异不是由 HBM 带宽打满造成的。 |
| Memory Throughput 主要跟 L1/TEX Throughput 接近 | 高层 memory SOL 被 L1TEX / shared data path 拉高。 |
| Tiled 使用 shared memory 后，某些 L1 cache hit rate 可能很低 | 这不是坏事，因为复用转移到了 shared memory 显式路径。 |
| Achieved Occupancy 接近满载 | 后续优化不要继续追 occupancy，要看 issue active、barrier、shared memory、FMA/Tensor pipe。 |
| Tiled 的 Issue Active 可能低于 Naive | tiled kernel 增加了 load tile、`__syncthreads()`、shared memory 访问，调度器发射节奏可能更不连续。 |

这个案例的正确学习方式：

```text
先看结果：Tiled 更快
-> 看 SOL：Memory 高但 DRAM 很低
-> 展开 breakdown：主要是 L1TEX/shared path 高
-> 看 Launch/Occupancy：并行度足够
-> 看 Scheduler/Warp State：同步、访存、issue 是否影响节奏
-> 回到源码：shared tile、bank conflict、barrier、thread tiling 怎么改
```

不要从单一指标推出结论。例如：

- 看到 `DRAM Throughput` 低，不能说“这个 kernel 没有访存压力”，只能说“HBM 侧压力低”。
- 看到 `Memory Throughput` 高，不能说“HBM 打满”，要看 breakdown。
- 看到 `L1 hit rate` 低，不能说“cache 用得差”，要确认是否改走 shared memory。
- 看到 `Occupancy` 高，不能说“性能已经好”，还要看 issue active 和 stall。

## LLM 推理场景的读法

| 场景 | 典型瓶颈 | NCU 关注点 | 可能动作 |
|---|---|---|---|
| Prefill GEMM / MLP | Tensor Core、HBM 供数、epilogue | tensor pipe、roofline、global/shared pipeline | FP8/FP4、CUTLASS/Triton 参数、tile、fusion |
| Prefill Attention | QK/PV matmul、softmax 非 matmul 开销、shared/register | tensor pipe、barrier、shared conflict、source | FlashAttention 类分块、online softmax、减少 rescale/store |
| Decode Attention | KV cache 读带宽、L2 命中、访存延迟 | DRAM/L2、long scoreboard、memory instruction | GQA/MQA、KV cache layout、paged cache locality、KV 量化 |
| RMSNorm / activation | memory-bound、小算子 launch | DRAM、vector load/store、kernel duration | fusion、向量化、persistent/thread coarsening |
| MoE expert GEMM | 小 GEMM、grouped GEMM、load imbalance | kernel 分布、tensor pipe、tail wave | grouped GEMM、expert batching、EPLB、量化 |
| 通信相关 kernel | NCCL/NVSHMEM 进度、网络拓扑 | 单看 NCU 不够 | 先用 `nsys` + NCCL 日志，再局部看 kernel |

## 记录模板

```markdown
## Profiling Case: <kernel>

- 环境：GPU / CUDA / driver / ncu / clocks
- 输入：shape、dtype、batch、seq、num heads、hidden
- Baseline：端到端、kernel time、吞吐、正确性
- 命令：`ncu ...`
- 关键指标：
  - SpeedOfLight:
  - Roofline:
  - Occupancy:
  - Memory:
  - Scheduler / Warp State:
  - Source hotspot:
- 判断：compute-bound / memory-bound / latency-bound / launch-bound
- 修改：
- 结果：
- 结论：保留 / 回滚 / 下一步
```

## 注意事项

- 不要把 NCU 的单 kernel 结论直接推广到端到端系统。LLM serving 的排队、调度、KV cache 命中率、通信和 CPU overhead 往往在 NCU 外面。
- 不要盲目追 occupancy。occupancy 是隐藏延迟的手段，不是最终目标；Tensor Core kernel 为了 tile 复用牺牲 occupancy 很常见。
- 不要只看 stall reason 排名。先确认 scheduler 是否真的 issue 不足，再解释 stall。
- 不同 GPU 架构的峰值、cache 层级、Tensor Core 指令和指标含义会变化，跨架构比较要保留 GPU 型号和 NCU 版本。
- `--set full` 很慢，且可能 replay 很多次。日常迭代优先收小 section。

## 参考

- [NVIDIA Nsight Compute CLI Documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [NVIDIA Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html)
- [NVIDIA Nsight Compute Roofline Analysis Blog](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/)
