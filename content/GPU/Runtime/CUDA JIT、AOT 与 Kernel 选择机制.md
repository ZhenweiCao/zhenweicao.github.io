---
aliases:
  - CUDA JIT 入门
  - GPU JIT 与 AOT
  - Kernel 选择机制
updated: 2026-05-19
tags:
  - gpu-computing
  - cuda-programming
  - performance-profiling
---
# CUDA JIT、AOT 与 Kernel 选择机制

## 定位

这篇用于区分几个容易混在一起的词：

- AOT 编译
- PTX JIT
- NVRTC runtime compilation
- nvJitLink
- Triton / torch.compile 这类框架 JIT
- cuBLASLt heuristics / kernel selection

读 [[nvjet_kernel_naming]] 时尤其要注意：`nvjet_*` 出现在 trace 里，不等于一定发生了 CUDA JIT 编译。

相关主笔记：

- [[CUDA 编程基础]]
- [[GPU 硬件架构背景与编程范式]]
- [[CUDA GEMM 矩阵乘法优化指南]]
- [[Nsight Compute NCU 分析方法与优化思路]]

## 一句话区分

| 机制 | 它做什么 | 会不会生成新机器码 |
|------|----------|--------------------|
| AOT | 编译期用 `nvcc` 等工具提前生成 cubin/fatbin | 会，但发生在运行前 |
| PTX JIT | 运行时由 driver 把 PTX 编译成目标 GPU binary | 会 |
| NVRTC | 运行时把 CUDA C++ 字符串编译成 PTX | 会，后续还要加载或继续编译 |
| nvJitLink | 运行时链接 GPU device code，输出可加载 cubin | 会 |
| Triton / torch.compile | 框架运行时生成 IR/PTX/cubin 或调编译后端 | 通常会 |
| cuBLASLt heuristics | 在库已有 matmul 算法/kernel 中选择合适实现 | 通常不是现场编译 |

## AOT：提前编译

AOT 是 Ahead-Of-Time，意思是在程序运行前已经把 GPU 代码编译好。

典型流程：

```text
CUDA C++ source
  → nvcc
  → PTX / cubin / fatbin
  → 应用运行时加载
```

优点：

- 首次运行开销小。
- 编译结果可控，部署更稳定。
- 生产环境常用。

缺点：

- 如果没有嵌入适合未来 GPU 的 PTX，可能缺少 forward compatibility。
- 针对新架构的优化需要重新编译或升级库。

## PTX JIT：driver 在运行时编译 PTX

PTX 是 NVIDIA GPU 的虚拟 ISA。应用可以把 PTX 嵌入 fatbin；运行时如果没有当前 GPU 对应的 cubin，driver 可以把 PTX JIT 编译成目标 GPU 的 binary。

简化流程：

```text
应用包含 PTX
  → driver 在运行时看到目标 GPU
  → PTX JIT 编译成 cubin
  → 写入 ComputeCache
  → 后续运行复用缓存
```

这会让首次加载变慢，但能让旧程序在更新的 GPU 上运行，并受益于新 driver 的编译器改进。

相关环境变量：

| 变量 | 作用 |
|------|------|
| `CUDA_CACHE_DISABLE=1` | 禁用 JIT cache，每次都可能重新编译 PTX。 |
| `CUDA_CACHE_PATH` | 指定 JIT cache 路径。 |
| `CUDA_CACHE_MAXSIZE` | 控制 JIT cache 大小。 |
| `CUDA_FORCE_PTX_JIT=1` | 强制忽略已有 cubin，验证 PTX JIT 路径。 |
| `CUDA_DISABLE_PTX_JIT=1` | 禁用 PTX JIT，验证是否有兼容 cubin。 |

## NVRTC：运行时编译 CUDA C++ 源码

NVRTC 是 NVIDIA Runtime Compilation library。它接受字符串形式的 CUDA C++ device code，在运行时编译出 PTX。

典型用途：

- 需要根据运行时 shape、dtype、模板参数生成专用 kernel。
- 不希望用户机器安装完整 `nvcc`。
- 框架或库想动态生成代码。

简化流程：

```text
运行时生成 CUDA C++ 字符串
  → NVRTC 编译成 PTX
  → driver 加载 PTX
  → 可能继续 PTX JIT 成 cubin
```

所以 NVRTC 和 PTX JIT 是两层：

```text
NVRTC: CUDA C++ → PTX
PTX JIT: PTX → GPU binary
```

## nvJitLink：运行时链接 device code

nvJitLink 用于运行时链接 GPU device code。它可以接受 PTX、cubin、fatbin、LTO-IR 等输入，输出可加载的 linked cubin。

初学者不用一开始深入它，只要知道它解决的是：

```text
多个 device code 片段如何在运行时链接成一个可加载产物
```

它比单纯“编译一段 kernel 字符串”更偏链接和 LTO 路径。

## 框架 JIT：Triton、torch.compile、DeepGEMM

很多深度学习框架会把 runtime shape、dtype、stride、hardware capability 纳入编译决策。

常见例子：

- Triton 根据 Python DSL 生成 MLIR/PTX/cubin。
- `torch.compile` 通过 graph capture 和 codegen 生成专用实现。
- DeepGEMM 一类库可能针对 MoE / FP8 / FP4 GEMM 做运行时编译或预编译缓存。

这类 JIT 的典型现象：

- 第一次跑慢，后面快。
- cache 目录里出现编译产物。
- 改 shape、dtype、GPU 架构后可能重新编译。
- warmup 很重要，否则 profiling 会把编译时间混进性能数据。

## cuBLASLt heuristics：选择 kernel，不一定编译 kernel

cuBLASLt 是 NVIDIA 的轻量 GEMM 库，重点是 matmul 的灵活 API、layout、dtype、epilogue 和算法选择。

cuBLASLt 官方文档描述的是：

```text
根据 problem size、GPU 配置和其他参数，用 heuristics 选择合适 matmul kernel
```

这个选择结果可以缓存，避免每次都在 CPU 侧重复做 heuristics 计算。

这和 JIT 的区别是：

```text
JIT: 运行时生成或编译机器码
heuristics: 在已有候选实现中选择一个
```

因此看到 cuBLASLt 内部 kernel 名字，例如 `nvjet_sm100_...`，更准确的表达是：

```text
cuBLASLt 为这个 matmul 形状选择了某个内部 kernel
```

不要直接写成：

```text
cuBLASLt 现场 JIT 编译了这个 kernel
```

除非你有明确证据，例如 NVRTC/driver JIT log、cache 产物、编译耗时或库文档说明。

## Autotuning 和 Heuristics 的区别

这两个词也容易混：

| 词 | 含义 | 典型开销 |
|----|------|----------|
| Heuristics | 根据规则、模型或元数据预测一个好算法 | CPU 侧开销，通常较小 |
| Autotuning | 实际跑多个候选，测量后选最快 | 需要 benchmark，首次开销较大 |

实际系统可能混合使用：先 heuristics 过滤候选，再 benchmark 少量候选，最后缓存结果。

做笔记时建议写清楚：

```text
heuristic selection
benchmark-based autotuning
runtime compilation
```

不要把这三个都笼统叫 JIT。

## Profiling 时怎么判断是哪种“首次慢”

首次运行慢，可能来自很多地方：

| 现象 | 可能原因 | 检查方式 |
|------|----------|----------|
| 第一次进程启动慢，后续进程也慢 | PTX JIT cache 不可写或被禁用 | 查 `CUDA_CACHE_*`，看 ComputeCache |
| 第一次某个 shape 慢，换 shape 又慢 | Triton / torch.compile / DeepGEMM runtime compile | 查框架 cache、日志、warmup |
| 每种 GEMM shape 首次有几十微秒 CPU 侧开销 | cuBLASLt heuristics | 查 cuBLASLt log、heuristics cache |
| 第一轮显存分配慢 | allocator warmup | 预分配或做 warmup |
| 第一轮 HBM/L2 cache miss 多 | cache cold start | 丢弃首轮计时 |

经验上，做 kernel profiling 前至少要：

```text
warmup 多轮
固定 shape
固定 dtype/layout
确认没有编译日志继续出现
确认 profiler 只包住目标计算区间
```

## 和 `nvjet_*` 的关系

`nvjet_*` 名称里有很多像 kernel schedule 参数的字段，例如 tile、K、stage、`2cta`、layout 标记。它们有助于读 trace，但不能反推出完整实现。

最安全的判断链是：

```text
trace kernel name 是 nvjet_*
  → 调用栈显示来自 cublasLtMatmul
  → 说明 cuBLASLt 选择了一个内部 GEMM kernel
  → 结合输入 shape 判断它服务哪个 projection / FFN / MoE GEMM
```

不要跳到：

```text
nvjet_* 名字里有 jet，所以它就是 JIT 编译出来的
```

## 参考

- [CUDA Programming Guide: Just-in-Time Compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#just-in-time-compilation)
- [CUDA Environment Variables: JIT Compilation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html#jit-compilation)
- [NVIDIA NVRTC Documentation](https://docs.nvidia.com/cuda/nvrtc/index.html)
- [NVIDIA nvJitLink Documentation](https://docs.nvidia.com/cuda/nvjitlink/index.html)
- [NVIDIA cuBLASLt Heuristics Cache](https://docs.nvidia.com/cuda/cublas/index.html#heuristics-cache)
