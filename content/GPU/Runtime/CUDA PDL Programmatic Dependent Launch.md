---
aliases:
  - PDL
  - Programmatic Dependent Launch
  - CUDA Programmatic Dependent Launch
  - 可编程依赖启动
updated: 2026-05-31
tags:
  - gpu-computing
  - cuda-programming
  - performance-profiling
  - cutlass
  - blackwell-gpu
---
# CUDA PDL：Programmatic Dependent Launch

PDL 是 **Programmatic Dependent Launch**，可以译成“可编程依赖启动”。它是 CUDA 在 Hopper 及之后架构上提供的一种机制：当两个 kernel 在同一个 CUDA stream 中有依赖关系时，后一个 dependent kernel 不必总是等前一个 kernel 完全结束后才启动，而是可以提前启动并执行自己不依赖前一个结果的部分。

相关笔记：

- [[GPU 知识库索引]]
- [[CUDA 编程基础]]
- [[CUDA JIT、AOT 与 Kernel 选择机制]]
- [[CUDA Kernel 性能瓶颈定位流程]]
- [[Nsight Compute NCU 分析方法与优化思路]]
- [[CUDA GEMM 矩阵乘法优化指南]]

官方入口：

- [CUDA Programming Guide: Programmatic Dependent Launch and Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
- [CUTLASS: Dependent kernel launches](https://docs.nvidia.com/cutlass/media/docs/cpp/dependent_kernel_launch.html)
- [CUTLASS discussion: What is PDL?](https://github.com/NVIDIA/cutlass/discussions/1791)

## 先讲结论

普通 same-stream 依赖：

```text
primary kernel 完全结束
-> 所有 primary 的 global memory writes 对后续 grid 可见（CUDA stream 语义保证）
-> secondary kernel 启动
```

PDL 依赖：

```text
primary kernel 完成关键前置工作
-> primary 显式触发 cudaTriggerProgrammaticLaunchCompletion()（release 信号）
-> secondary kernel 提前启动，先做不依赖 primary 输出的 preamble
-> secondary 在读取 primary 输出前调用 cudaGridDependencySynchronize()（acquire 信号）
-> 同步保证 primary 在 trigger 点之前的所有 global memory writes 对 secondary 可见
-> secondary 继续读取 primary 输出
```

PDL 不是"让有依赖的 kernel 乱序读写"。它只是在 acquire/release 同步语义的前提下，把 secondary kernel 中**不依赖 primary 输出**的那段工作提前执行。

> 术语澄清：CUDA 的内存模型用 **acquire / release** 描述跨 grid 数据可见性，不是 cache "flush"。`cudaGridDependencySynchronize()` 提供 acquire 语义——它**不是**触发硬件 cache flush，而是保证前置 grid 在 trigger 点之前的 stores 对当前 grid 可见。本文余下段落沿用"trigger / sync"措辞，避免再用 "flush"。

![[GPU/Drawings/CUDA PDL 时序机制.svg]]

可编辑源图：[[GPU/Drawings/CUDA PDL 时序机制.excalidraw]]

## 它解决什么问题

CUDA stream 默认保证顺序语义。同一个 stream 里：

```text
kernel A
kernel B
```

通常意味着 B 要等 A 完成后才启动。这很安全，因为 A 写到 global memory 的 stores 在 A 完成后通过 CUDA stream acquire/release 语义对 B 可见。

但很多真实 kernel 不是一启动就读取前一个 kernel 的输出。它们前面会有一段 preamble / prologue，例如：

- 清零内部 buffer。
- 加载常量或参数。
- 计算索引、tile 坐标、predicate。
- 为 GEMM 预取权重、准备 TMA descriptor 或 shared memory pipeline。
- 执行与前一个 kernel 输出无关的初始化。

如果 secondary 的这些工作可以和 primary 的尾部重叠，就能隐藏一部分：

- secondary kernel launch latency。
- secondary prologue 开销。
- primary epilogue / store / tail 阶段。
- 某些内存预取或准备阶段。

这就是 PDL 的主要价值。

## API 机制

### Primary kernel：放行 secondary 的启动

primary kernel 在“已经完成 secondary 必须等待的关键前置工作”后调用：

```cpp
cudaTriggerProgrammaticLaunchCompletion();
```

示意：

```cpp
__global__ void primary_kernel(...) {
    // 1. secondary 不能越过的关键工作
    produce_some_state_or_reach_safe_point();

    // 2. 告诉 CUDA driver: secondary 可以被提前 launch
    cudaTriggerProgrammaticLaunchCompletion();

    // 3. primary 后续可与 secondary preamble 重叠的尾部工作
    epilogue_or_tail_work();
}
```

NVIDIA 文档强调：primary kernel 的 thread blocks 应在合适位置执行这个 trigger。如果 primary 没有显式调用 trigger，等价于所有 thread blocks 退出后才隐式触发；这样就退化得更接近普通串行 launch。

### Secondary kernel：读取依赖数据前必须等待

secondary kernel 可以先执行独立工作，但在真正读取 primary 输出前必须调用：

```cpp
cudaGridDependencySynchronize();
```

示意：

```cpp
__global__ void secondary_kernel(...) {
    // 1. 不依赖 primary 输出的工作，可以提前做
    preamble_independent_of_primary_output();

    // 2. acquire：保证 primary 在 trigger 点之前的所有 global memory writes
    //    对当前 grid 可见。这是 memory model acquire 语义，不是触发 cache flush。
    cudaGridDependencySynchronize();

    // 3. 现在才能安全读取 primary 写出的数据
    consume_primary_output();
}
```

这一步非常关键。PDL 允许 secondary 提前 launch，但 secondary 提前 launch 时，primary 写出的 global memory 数据不一定已经对 secondary 可见。没有 `cudaGridDependencySynchronize()` 就读取依赖数据，是 correctness bug。

### Host 侧：secondary 需要 PDL launch attribute

CUDA Programming Guide 当前的 same-stream PDL 基本示例是：

- primary kernel 可以用普通 `<<<>>>` 启动，但 primary **device 端必须**在合适位置调用 `cudaTriggerProgrammaticLaunchCompletion()`。
- secondary kernel 需要用 extensible launch API，并配置 `cudaLaunchAttributeProgrammaticStreamSerialization`。
- primary 和 secondary 仍在同一个 CUDA stream 中保持顺序关系。

完整示意：

```cpp
// === Primary kernel ===
// kernel 内部需要调用 cudaTriggerProgrammaticLaunchCompletion()
primary_kernel<<<primary_grid, primary_block, 0, stream>>>(/* args */);

// === Secondary kernel：允许 PDL ===
cudaLaunchAttribute attrSecondary{};
attrSecondary.id  = cudaLaunchAttributeProgrammaticStreamSerialization;
attrSecondary.val.programmaticStreamSerializationAllowed = 1;

cudaLaunchConfig_t configSecondary{};
configSecondary.gridDim         = grid_dim;
configSecondary.blockDim        = block_dim;
configSecondary.dynamicSmemBytes = 0;
configSecondary.stream          = stream;
configSecondary.attrs           = &attrSecondary;
configSecondary.numAttrs        = 1;

cudaLaunchKernelEx(&configSecondary, secondary_kernel, /* args */);
```

要点：

- primary 和 secondary 在 **同一个 CUDA stream** 中。
- secondary 要用 extended launch attribute 告诉 driver：这个 kernel 允许 programmatic stream serialization。
- primary 需要在 device 端显式调用 `cudaTriggerProgrammaticLaunchCompletion()`；如果 primary 没有显式 trigger，则退化为接近 primary 完全退出后才 release。
- secondary 内部负责在依赖数据前调用 `cudaGridDependencySynchronize()`。
- 更复杂的 programmatic event、跨 stream、CUDA Graph 或库封装路径可能让 primary 侧也携带 launch attribute；这种情况以对应 API / library 文档为准。

## CUDA Graph 里的 PDL

CUDA 官方文档也支持在 CUDA Graph 中表达 PDL 依赖。核心是用：

```cpp
cudaGraphDependencyTypeProgrammatic
```

连接两个 kernel node，并配合对应的 graph edge port，例如：

- `cudaGraphKernelNodePortLaunchCompletion`
- `cudaGraphKernelNodePortProgrammatic`

初学阶段先把普通 stream API 理解清楚即可。Graph 版本适合已经在用 CUDA Graph 固化推理流程的场景，例如 decode loop、固定 shape 的 serving path。

## CUTLASS 里的 PDL

CUTLASS 文档把它称为 **Dependent kernel launches**。Hopper 和 Blackwell 架构上，CUTLASS 可以通过 PDL 让两个 same-stream kernel 的部分执行重叠。

构建 CUTLASS 时，需要打开对应架构的 PDL/GDC 相关指令支持：

```bash
cmake . -DCUTLASS_ENABLE_GDC_FOR_SM90=1
```

Blackwell 对应：

```bash
cmake . -DCUTLASS_ENABLE_GDC_FOR_SM100=1
```

注意：这些宏只是让 kernel 包含 PDL 相关指令。真正 launch 时还要启用 PDL：

```cpp
gemm.run(
  /* stream = */ stream,
  /* cuda_adapter = */ nullptr,
  /* launch_with_pdl = */ true
);
```

CUTLASS 的一个典型优化场景是：GEMM 有两个输入矩阵，其中一个输入是前一个 kernel 产生的 activation，另一个输入是已经存在的 weights。GEMM 不应该提前读取 activation，但可以在等待 activation 前先预取 weights 或执行不依赖 activation 的 prologue。

```text
producer kernel:  写 activation
GEMM secondary:   提前启动，预取 weights
                  cudaGridDependencySynchronize() 等 activation 对自己可见
                  执行依赖 activation 的 mainloop
```

## 应用场景

| 场景 | 为什么可能适合 PDL |
|------|--------------------|
| RMSNorm / LayerNorm -> GEMM | GEMM 可以提前做权重预取、tile 初始化，等 activation 可见后再读 activation。 |
| activation / elementwise -> GEMM | 前一个 kernel 产出 activation，后一个 GEMM 有不依赖 activation 的 prologue。 |
| back-to-back GEMM | 第二个 GEMM 某些准备工作不依赖第一个 GEMM 的输出。 |
| attention pipeline | 某些 preamble、descriptor、tile 准备可能与上游 kernel 尾部重叠。 |
| CUDA Graph 推理路径 | 固定拓扑下，可以把 PDL edge 表达成 graph dependency。 |

适合 PDL 的基本条件：

1. 两个 kernel 有 same-stream 依赖。
2. secondary 有一段显著的 independent work。
3. primary 有一段可以和 secondary independent work 重叠的尾部。
4. secondary 能明确知道什么时候必须等待 primary 输出。
5. GPU 架构和库实现支持 PDL。

## 不适合的场景

| 场景 | 原因 |
|------|------|
| secondary 一启动就必须读 primary 输出 | 没有可提前执行的 preamble，PDL 几乎没收益。 |
| primary 很短 | launch / 同步开销可能比可隐藏部分更大。 |
| 两个 kernel 都吃满 SM 资源 | 即使允许 overlap，也可能没有资源并发执行。 |
| 正确性依赖很复杂 | 容易忘记 `cudaGridDependencySynchronize()` 或等待位置放错。 |
| 已经可用普通多 stream 并发 | 如果两个 kernel 没有数据依赖，直接用不同 stream 更自然。 |
| 通信/多进程时序敏感 | profiling 和 PDL overlap 都可能改变调度时序，需要谨慎验证。 |

## 可能的缺陷和风险

### 1. 并发不是保证

CUDA 官方文档明确指出，PDL 提供的是并发执行机会，不保证一定并发。driver 可以提前 launch secondary，但是否真的 overlap，取决于：

- SM 资源是否有余量。
- primary 和 secondary 的 block/cluster 资源需求。
- register / shared memory / barrier / cluster 占用。
- 当前 GPU 上是否还有其他工作。
- kernel 大小和调度粒度。

因此不能写依赖“必须 overlap 才能正确”的代码。依赖 overlap 还可能导致 deadlock 风险。

### 2. 正确性更容易写错

最大风险是 secondary 提前读取 primary 的结果。必须记住：

```text
secondary launch 早 != primary 输出可见
```

读取依赖数据前一定要：

```cpp
cudaGridDependencySynchronize();
```

### 3. 性能可能变差

PDL 可能引入额外同步和调度复杂度。如果 secondary 的 preamble 很短，或者 overlap 导致两个 kernel 争抢 cache、SMEM、register、L2、memory pipeline，性能可能没有提升甚至变差。

### 4. 调试和 profiling 更复杂

Nsight Systems 时间线上会看到 same-stream kernel 出现部分重叠。分析时要分清：

- secondary 的 preamble 是否真的提前执行。
- `cudaGridDependencySynchronize()` 是否造成长等待。
- primary tail 是否足够长。
- overlap 后是否引入 L2/DRAM/L1TEX 竞争。

建议先用 `nsys` 看 timeline，再用 [[Nsight Compute NCU 分析方法与优化思路]] 钻单个 kernel。

### 5. 适用架构和库版本有限

CUDA 官方文档写明，PDL 要求：

- **硬件**：compute capability ≥ 9.0（Hopper 起，含 H100/H200 + Blackwell B200/B300）；
- **CUDA Toolkit**：≥ 12.0（部分预览特性需 12.x 较新小版本）；
- **Device code**：使用 `cudaTriggerProgrammaticLaunchCompletion()` / `cudaGridDependencySynchronize()` 需要在 device 端通过 `<cuda_runtime.h>` 暴露的 device 函数路径，且通常要求 `__CUDA_ARCH__ >= 900` 才能编译通过；
- **Host code**：必须用 `cudaLaunchKernelEx` + `cudaLaunchAttribute*`，普通 `<<<>>>` 语法不携带这些属性；
- **库要求**：CUTLASS 文档把它称为 **Dependent kernel launches**，主要面向 Hopper / Blackwell，并要求构建和运行时都启用相关选项。

## 如何判断 PDL 是否值得尝试

可以按这个 checklist：

- [ ] secondary 是否有明显 independent preamble。
- [ ] primary 是否有可 overlap 的 tail / epilogue。
- [ ] secondary 是否能在读取 primary 输出前明确等待。
- [ ] 两个 kernel 是否在同一个 stream 中有顺序依赖。
- [ ] GPU 是否是 Hopper / Blackwell 或其他 CC >= 9.0 设备。
- [ ] baseline 已经用 `nsys` 确认这两个 kernel 是端到端瓶颈。
- [ ] 开启 PDL 后用 timeline 验证是否真的 overlap。
- [ ] 用端到端 latency / throughput 验证，而不是只看单 kernel 时间。

## 一个初学者版判断图

```text
两个 kernel 完全无依赖？
  -> 用不同 stream 或 graph 并发，不需要 PDL

secondary 一启动就要读 primary 输出？
  -> PDL 基本没收益

secondary 有不依赖 primary 输出的 preamble？
  -> 可以考虑 PDL

开启后是否真的 overlap，端到端是否变快？
  -> 用 nsys + benchmark 验证
```

## 和其他机制的区别

| 机制 | 解决的问题 | 和 PDL 的区别 |
|------|------------|---------------|
| 多 CUDA stream | 无依赖或显式 event 依赖的并发 | PDL 针对 same-stream 依赖 kernel 的部分 overlap。 |
| CUDA event | host 侧或 stream 之间表达依赖 | PDL 让 device kernel 内部显式 trigger / wait。 |
| CUDA Graph | 固化 launch 拓扑，降低 CPU launch overhead | Graph 可表达 PDL edge，但 PDL 关注 dependent kernel 的提前启动。 |
| Cooperative Groups / Cluster | kernel 内线程/CTA 协作 | PDL 是 kernel 与 kernel 之间的启动/同步机制。 |
| Persistent kernel | 一个 kernel 内长期调度任务 | PDL 不改变 kernel 内任务模型，只改变 dependent launch 时序。 |

## 记忆卡片

```text
PDL = same stream dependent kernels 的部分重叠机制

primary:
  cudaTriggerProgrammaticLaunchCompletion()

secondary:
  independent preamble
  cudaGridDependencySynchronize()
  dependent work

host:
  cudaLaunchKernelEx
  cudaLaunchAttributeProgrammaticStreamSerialization

收益:
  隐藏 secondary launch/prologue 或 primary tail

风险:
  并发不保证；同步位置错会有 correctness bug；资源竞争可能让性能变差
```
