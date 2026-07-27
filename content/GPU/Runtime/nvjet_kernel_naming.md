---
aliases:
  - nvjet kernel 命名
  - cuBLASLt nvjet
updated: 2026-05-30
tags:
  - gpu-computing
  - performance-profiling
  - gemm-optimization
  - cublaslt
---
# cuBLASLt nvjet Kernel 名称速查

## 定位

这篇笔记用于解释 Nsight Systems / Nsight Compute trace 里看到的 `nvjet_*` kernel 名称。先给结论：

- `nvjet_*` 更适合理解为 **cuBLASLt 内部 GEMM kernel 名称前缀**，不是一个单独的 attention 库。
- `nvjet` 不是 NVIDIA 公开文档中稳定定义的 API 名称，字段含义只能作为 **profile 读图和性能分析线索**，不要当成公开 ABI。
- 不要把 `nvjet` 直接展开成 `JIT`。CUDA 里真正的 JIT 指 PTX/runtime codegen 等编译路径，见 [[CUDA JIT、AOT 与 Kernel 选择机制]]。
- 名字里出现 `2cta` 时，优先联想到 [[CUDA CTA 与 Thread Block Cluster 入门]] 里的 CTA / cluster / `cta_group::2` 这条线，而不是"2 个线程"。

**版本下界**：`nvjet_*` 前缀大约从 **CUDA Toolkit 12.x** 开始在 cuBLASLt 中出现（与 Hopper / Blackwell 上新的 GEMM 实现路径绑定）；CUDA 11.x 及之前的 cuBLAS/cuBLASLt 用不同的内部命名（如 `ampere_*` / `volta_*`）。如果你的 trace 里完全没看到 `nvjet_*`，先检查 CUDA Toolkit / driver 版本。

相关主笔记：

- [[CUDA JIT、AOT 与 Kernel 选择机制]]
- [[CUDA CTA 与 Thread Block Cluster 入门]]
- [[CUDA GEMM 矩阵乘法优化指南]]
- [[GPU 硬件架构背景与编程范式]]
- [[Nsight Compute NCU 分析方法与优化思路]]

## 如何确认它属于 cuBLASLt

在装有 CUDA 的 Linux 环境里，可以用二进制字符串搜索确认 `nvjet_*` 是否出现在 cuBLASLt 中：

```bash
strings /usr/local/cuda/lib64/libcublasLt.so | grep '^nvjet_'

# 或者直接粗略搜索
grep -a nvjet /usr/local/cuda/lib64/libcublasLt.so
```

如果 trace 里调用栈能看到 PyTorch / SGLang / vLLM 最终走到 `cublasLtMatmul`，那 `nvjet_*` 大概率就是 cuBLASLt 为某个 GEMM 形状选择的内部 kernel。

## 例子

常见名称形如：

```text
nvjet_sm100_tst_256x240_64x4_2x1_2cta_v_bz_TNT
nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT
nvjet_sm100_tst_24x64_64x16_4x1_v_bz_TNN
```

以第一个为例，可以先按下列方式切字段：

```text
nvjet _ sm100 _ tst _ 256x240 _ 64x4 _ 2x1 _ 2cta _ v _ bz _ TNT
  1      2      3       4        5      6      7     8   9    10
```

## 字段解读

> 可靠性说明：`sm100`、tile 形状、`2cta` 这类字段比较容易和公开 CUDA/CUTLASS 概念对应；`tst`、`v/h`、`bz` 等更像 NVIDIA 内部命名，下面只作为经验解读。

| 字段 | 示例 | 怎么读 | 可靠性 |
|------|------|--------|--------|
| 1 | `nvjet` | cuBLASLt 内部 GEMM kernel family 前缀。 | 中 |
| 2 | `sm100` | 目标 GPU 架构，SM100 对应 Blackwell。 | 高 |
| 3 | `tst` / `hsh` | 内部 kernel 子类型或调度族。不要强行展开成固定术语。 | 低 |
| 4 | `256x240` | 输出 tile 的 M×N 尺寸，表示一个 CTA 或一组 CTA 负责的 C 子块形状。 | 中高 |
| 5 | `64x4` | 常可读作 K tile 为 64，pipeline stages 为 4。`64x6`、`64x16` 同理。 | 中 |
| 6 | `2x1` | CTA 内或 warp-group 级的 tile 划分提示，例如 M/N 方向的分工比例。 | 中低 |
| 7 | `2cta` | 两个 CTA 协同的 kernel 变种，可能对应 cluster / `cta_group::2` 一类路径。 | 中 |
| 8 | `v` / `h` | 内部 layout 方向提示，可粗略记为 vertical / horizontal 变种。 | 低 |
| 9 | `bz` | 内部数据排列或 swizzle/layout 标记。 | 低 |
| 10 | `TNT` / `TNN` | A/B/输出相关的转置或布局标记。不要简单理解成“用户代码里把 C 转置了”。 | 中 |

### `TNT` / `TNN` 怎么理解

GEMM 的数学形式是：

```text
D = alpha * op(A) * op(B) + beta * C
```

cuBLASLt 同时支持不同的 `op(A)`、`op(B)`、matrix layout、leading dimension、epilogue 和输出布局。kernel 名称末尾的 `T/N` 串一般反映内部布局选择，但不一定等于 Python 或 CUDA 调用层面显式写了 `transpose()`。

新手读 trace 时可以先用这个简化模型：

```text
TNT / TNN 约等于：这颗 kernel 针对某组 A/B/输出布局优化过
```

如果要精确判断，需要结合 `cublasLtMatmulDesc`、输入 tensor stride、layout order、profile 调用栈和 cuBLASLt log。

## 从 GEMM 视角理解这些名字

GEMM 是：

```text
C[M, N] = A[M, K] × B[K, N]
```

高性能 kernel 不会让一个线程直接算一个完整输出矩阵，而是把输出 C 切成很多 tile：

```text
C 的一个 tile，例如 128×256 或 256×240
  ↑
由一个 CTA 或一组 CTA 负责
  ↑
CTA 内再分给 warp / warpgroup
  ↑
Tensor Core 反复计算更小的 MMA tile
```

所以 `128x256_64x6` 可以先读成：

```text
输出 tile: 128×256
K 方向每次处理: 64
流水线 stage: 6
```

这不是矩阵本身的总大小，而是 kernel 内部一次处理的局部块大小。

## 为什么有这么多变种

cuBLASLt 的目标是为不同 GEMM 形状、数据类型、布局和硬件选择合适 kernel。NVIDIA 官方文档把这条路径称为 heuristics：cuBLASLt 会根据 problem size、GPU 配置和其他参数选择 matmul kernel，并维护 heuristics cache。

这和“运行时真的编译出一颗新 kernel”不是一回事：

- **CUDA PTX JIT**：driver 把 PTX 编译成具体 GPU 的 binary，首次加载可能变慢。
- **NVRTC / Triton / torch.compile**：运行时生成或编译 CUDA/PTX/IR。
- **cuBLASLt heuristics**：在库已有的算法和 kernel 实现中选一个合适的，结果可以缓存。

所以看到 `nvjet_*` 时，更稳妥的说法是：

```text
cuBLASLt 为当前 matmul problem 选择了这个内部 GEMM kernel
```

不要直接说：

```text
cuBLASLt 现场 JIT 编译了 nvjet kernel
```

## 典型变种对比

| Kernel 片段 | 粗略 tile | K / stages | 2cta | 可能场景 |
|-------------|-----------|------------|------|----------|
| `tst_256x240_64x4_2x1_2cta_v_bz_TNT` | 256×240 | 64 / 4 | 是 | 大矩阵 GEMM，prefill 或 FFN 一类高吞吐路径 |
| `tst_128x256_64x6_2x1_2cta_v_bz_TNT` | 128×256 | 64 / 6 | 是 | 大矩阵 GEMM，attention projection / MLP 常见形状 |
| `tst_256x128_64x5_2x2_2cta_h_bz_TNT` | 256×128 | 64 / 5 | 是 | 另一种 M/N 比例和内部 layout |
| `tst_24x64_64x16_4x1_v_bz_TNN` | 24×64 | 64 / 16 | 否 | 小 M/N 形状，decode 或小 batch GEMM |
| `tst_32x64_64x16_4x1_v_bz_TNN` | 32×64 | 64 / 16 | 否 | 小 tile GEMM，减少小矩阵下的空转 |

这些“可能场景”只是根据 tile 大小和 LLM workload 形态做的读图推断。真正归因要看调用栈和 matmul 输入 shape。

## `2cta` 对性能意味着什么

CTA 是 thread block 的 PTX 名称。普通 CUDA 心智模型里，一个 block/CTA 被调度到一个 SM 上，block 内线程共享 shared memory。

Hopper 之后引入 Thread Block Cluster：多个 CTA 可以组成 cluster，被协同调度，并能通过 distributed shared memory 通信。Blackwell 的 `tcgen05.mma` 文档里也能看到 `cta_group::1` / `cta_group::2` 这样的指令范围。

因此名字里的 `2cta` 可以先理解为：

```text
这个 kernel 变种可能让 2 个 CTA 协同处理一个更大的工作单元
```

它的好处通常是：

- 能处理更大的 tile，提高 A/B 数据复用。
- 可以配合 TMA / cluster shared memory 减少重复搬运。
- 在大矩阵上更容易喂饱 Tensor Core。

代价是：

- 单个工作单元占用更多 SM/片上资源。
- 小矩阵或 decode 场景可能不划算。
- 调度和同步更复杂，不能只看 `2cta` 判断一定更快。

## Prefill 和 Decode 为什么会选不同 tile

LLM 推理里常见两类 GEMM 形态：

| 阶段 | 典型 shape 特点 | kernel 倾向 |
|------|-----------------|-------------|
| Prefill | M 通常较大，例如 seq_len × hidden | 大 tile、高 Tensor Core 吞吐、可能用 `2cta` |
| Decode | 每步新增 token 少，M 可能很小 | 小 tile、减少 SM 空转、避免过大的协同开销 |

所以同一个 Transformer forward 的 trace 里看到多种 `nvjet_*`，不代表同一个矩阵乘被拆成很多 nvjet kernel。更常见的情况是：

```text
Q projection      → 一个 GEMM，选一种 kernel
KV projection     → 一个 GEMM，选另一种 kernel
FFN gate/up       → 一个 GEMM，选另一种 kernel
FFN down          → 一个 GEMM，选另一种 kernel
Decode 小 batch   → 小 tile kernel
```

## 看 trace 时的检查顺序

1. 先看调用栈：是否来自 `cublasLtMatmul`。
2. 再看输入 shape：M/N/K 分别是多少，是否是 prefill、decode、MoE expert GEMM。
3. 看 kernel 名字：tile、K、stage、`2cta` 只作为提示。
4. 看耗时和占比：它是不是端到端瓶颈。
5. 用 NCU 看指标：Tensor Core 利用率、HBM/L2 吞吐、shared memory stall、occupancy。
6. 如果首轮慢，区分 CUDA JIT、框架编译、cuBLASLt heuristics cache、allocator warmup 和 cache cold start。

## 常见误区

| 误区 | 更准确的说法 |
|------|--------------|
| `nvjet` 就是 JIT | `nvjet` 是观察到的内部 kernel 前缀；JIT 是另一套编译机制。 |
| `nvjet` 是 attention kernel | 它通常是 cuBLASLt GEMM kernel；attention projection、FFN 都可能触发。 |
| `2cta` 是 2 个线程 | CTA 基本等价于 CUDA block，`2cta` 是两个 block/CTA 协同的线索。 |
| `TNT` 表示用户代码转置了 C | 它更可能是内部 operand/output layout 标记，需结合 cuBLASLt descriptor 判断。 |
| tile 越大越好 | 大 tile 适合大矩阵，小矩阵会遇到资源占用和空转问题。 |
| 首次慢一定是 nvjet JIT | 首次慢可能来自 PTX JIT、NVRTC/Triton、torch.compile、cuBLASLt heuristics、内存分配或 cache warmup。 |

## 参考

- [NVIDIA cuBLASLt Heuristics Cache](https://docs.nvidia.com/cuda/cublas/index.html#heuristics-cache)
- [CUDA Programming Guide: Just-in-Time Compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#just-in-time-compilation)
- [CUDA Environment Variables: JIT Compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#jit-compilation)
- [PTX ISA: Cooperative Thread Arrays](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cooperative-thread-arrays)
- [CUDA Programming Model: Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html#thread-block-clusters)
- [NVIDIA CUTLASS: Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
