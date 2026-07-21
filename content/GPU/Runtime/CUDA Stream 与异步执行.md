---
aliases:
  - CUDA Stream
  - CUDA Streams
  - GPU Stream
  - CUDA 异步执行
updated: 2026-06-14
tags:
  - gpu-computing
  - cuda-programming
  - concurrency-control
---
# CUDA Stream 与异步执行

## 定位

这篇解释 CUDA 里的 **stream**：它不是 SM、warp、block，也不是硬件上的一条固定流水线，而是 CUDA runtime 用来表达 **异步提交顺序和依赖关系** 的队列抽象。

相关主笔记：

- [[CUDA 编程基础]]
- [[CUDA 线程配置与占用率]]
- [[CUDA Kernel 性能瓶颈定位流程]]
- [[Nsight Compute NCU 分析方法与优化思路]]
- [[CUDA PDL Programmatic Dependent Launch]]
- [[GPU 硬件背景地图]]

## 一句话

**CUDA stream 是一条按序执行的 GPU 工作队列。**

Host 把 kernel、异步拷贝、event、host callback、graph launch 等操作提交到某个 stream 里；同一个 stream 内的操作按提交顺序执行，不同 stream 之间默认没有顺序关系，因此可以表达潜在并发。

```text
stream 0: H2D copy -> kernel A -> D2H copy
stream 1:             kernel B -> kernel C

同一 stream 内：按顺序
不同 stream 间：除非 event / wait / 默认流语义建立依赖，否则无顺序保证
```

![[GPU/Drawings/CUDA Stream 顺序与并发.svg]]

## Stream 不是什么

| 容易误解 | 正确理解 |
|----------|----------|
| stream 是一个 SM | 不是。SM 是硬件执行单元；stream 是 runtime 队列。 |
| stream 是一个 CPU thread | 不是。多个 CPU thread 可以提交 CUDA work；stream 只是 GPU work 的顺序容器。 |
| 一个 stream 对应一个硬件 copy engine 或计算引擎 | 不是。stream 只表达顺序和依赖，runtime / driver / hardware 决定如何调度。 |
| 多开 stream 一定更快 | 不一定。要看 kernel 是否能并发、copy 是否能异步、资源是否足够、是否被默认流或同步 API 阻塞。 |
| 同一 stream 内 kernel launch 会阻塞 CPU | 通常不会。kernel launch 对 host 是异步的；顺序约束发生在 device work 队列里。 |

可以这样放进 CUDA 层次里：

```text
Host thread
  -> CUDA Runtime API
      -> Stream（提交顺序 / 依赖）
          -> Kernel launch / memcpy / event / graph
              -> Grid / CTA / Warp / Thread 在 GPU 上执行
```

## 基本 API

创建和销毁 stream：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// stream based operations ...

cudaStreamDestroy(stream);
```

把 kernel 发到指定 stream：

```cpp
kernel<<<grid, block, shared_mem_bytes, stream>>>(args...);
```

把内存拷贝发到指定 stream：

```cpp
cudaMemcpyAsync(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice, stream);
```

同步一个 stream：

```cpp
cudaStreamSynchronize(stream);
```

只查询，不阻塞：

```cpp
cudaError_t status = cudaStreamQuery(stream);
if (status == cudaSuccess) {
    // stream 中此前提交的 work 已完成
}
```

> [!NOTE] pinned memory
> `cudaMemcpyAsync` 想真正与计算重叠，host 侧 buffer 通常需要 page-locked / pinned memory，例如 `cudaMallocHost()`。如果用普通 pageable memory，API 形式虽然是 async，但实际可能退化成同步行为，重叠收益会消失。

## 同一 stream 的顺序语义

CUDA stream 是 **in-order stream**：

```cpp
kernel1<<<grid, block, 0, s>>>();
kernel2<<<grid, block, 0, s>>>();
cudaMemcpyAsync(h, d, bytes, cudaMemcpyDeviceToHost, s);
```

在同一个 stream `s` 里：

```text
kernel1 完成
  -> kernel2 才能安全读 kernel1 的结果
      -> D2H copy 才能安全拷贝 kernel2 的结果
```

这不表示 CPU 会停在那里等。CPU 只是把 work enqueue 到 stream；device 侧按照 stream 顺序消费这些 work。

## 多 stream 的并发语义

不同 stream 之间默认没有顺序关系：

```cpp
kernelA<<<grid, block, 0, stream_a>>>(...);
kernelB<<<grid, block, 0, stream_b>>>(...);
```

这表达的是：

```text
kernelA 和 kernelB 可以并发
```

但不是保证一定并发。能不能真的 overlap，取决于：

- 两个 kernel 是否都有足够资源并发驻留。
- 单个 kernel 是否已经吃满全部 SM / register / shared memory / memory bandwidth。
- GPU 是否支持相关方向的 copy/compute overlap。
- 是否使用了 pinned host memory。
- 是否被 legacy default stream、`cudaDeviceSynchronize()`、同步 `cudaMemcpy()` 等隐式同步打断。
- stream priority 只是调度提示，不保证抢占已经运行的 work。

## Event：跨 stream 建依赖

如果两个 stream 之间有依赖，不要靠“提交顺序看起来在前面”来赌。要用 event 显式表达：

```cpp
cudaStream_t produce, consume;
cudaEvent_t ready;

cudaStreamCreate(&produce);
cudaStreamCreate(&consume);
cudaEventCreate(&ready);

producer_kernel<<<grid, block, 0, produce>>>(d_data);
cudaEventRecord(ready, produce);

cudaStreamWaitEvent(consume, ready, 0);
consumer_kernel<<<grid, block, 0, consume>>>(d_data);
```

语义是：

```text
producer_kernel 在 produce stream 中完成
  -> ready event 被 record
      -> consume stream 中 ready 后面的 work 才能开始
```

这比 `cudaDeviceSynchronize()` 细得多：只同步必要依赖，不把全设备所有 stream 都停住。

## 常见同步 API

| API | 同步范围 | 典型用途 |
|-----|----------|----------|
| `cudaDeviceSynchronize()` | 当前 device 上此前提交的所有 work | 调试、程序末尾、粗粒度全局等待 |
| `cudaStreamSynchronize(stream)` | 某一个 stream 中此前提交的 work | 等某条 pipeline 结束 |
| `cudaStreamQuery(stream)` | 非阻塞查询某 stream 是否完成 | CPU polling、服务框架中检查进度 |
| `cudaEventSynchronize(event)` | 等某个 event 完成 | 等一个中间点，而不是等完整 stream |
| `cudaStreamWaitEvent(stream, event)` | 让一个 stream 等另一个 stream 的 event | 建立跨 stream 依赖 |

经验规则：

```text
能用 event 表达局部依赖，就不要上来 cudaDeviceSynchronize()
```

`cudaDeviceSynchronize()` 很方便，但它会摊平并发，常常把潜在 overlap 直接杀掉。

## 默认 stream 的坑

如果 kernel launch 没有指定 stream：

```cpp
kernel<<<grid, block>>>(...);
```

它会进入 default stream。这里有两个语义口径：

| 默认流模式 | 语义 |
|------------|------|
| legacy default stream / NULL stream | 会和其他 blocking streams 发生隐式同步，容易让本来可并发的 work 串行化 |
| per-thread default stream | 每个 host thread 有独立默认流，不再像 legacy default stream 那样和其他 stream 广泛同步 |

legacy default stream 最容易踩坑：

```cpp
kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block>>>(...);              // 默认流
kernel3<<<grid, block, 0, stream2>>>(...);
```

在 legacy default stream 语义下，`kernel2` 可能让 `stream1` 和 `stream2` 本来可以并发的 work 串起来。

想减少这种隐式同步，常见做法：

```cpp
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
```

或者编译时启用 per-thread default stream：

```bash
nvcc --default-stream per-thread ...
```

## Copy / compute overlap

典型 pipeline 是把大数据切 chunk，让拷贝和计算重叠：

```cpp
cudaStream_t streams[2];
cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);

for (int i = 0; i < num_chunks; ++i) {
    cudaStream_t s = streams[i % 2];
    size_t offset_elems = i * chunk_elems;
    size_t bytes = chunk_elems * sizeof(float);

    cudaMemcpyAsync(d_in + offset_elems, h_in + offset_elems,
                    bytes, cudaMemcpyHostToDevice, s);

    kernel<<<grid, block, 0, s>>>(d_in + offset_elems, d_out + offset_elems);

    cudaMemcpyAsync(h_out + offset_elems, d_out + offset_elems,
                    bytes, cudaMemcpyDeviceToHost, s);
}

cudaStreamSynchronize(streams[0]);
cudaStreamSynchronize(streams[1]);
```

理想时间线：

```text
stream 0: H2D chunk0 -> kernel chunk0 -> D2H chunk0
stream 1:              H2D chunk1 -> kernel chunk1 -> D2H chunk1
```

但这类 overlap 成立需要条件：

- host buffer 用 pinned memory。
- copy size 足够大，小拷贝可能被 launch/API overhead 吃掉。
- kernel 不要把所有 SM / 带宽吃满到没有 overlap 空间。
- H2D/D2H overlap 能力和 copy engine 数量取决于 GPU。
- 避免中间插入 legacy default stream 或全设备同步。

## Stream priority

可以创建带优先级的 stream：

```cpp
int least, greatest;
cudaDeviceGetStreamPriorityRange(&least, &greatest);

cudaStream_t high_prio;
cudaStreamCreateWithPriority(&high_prio, cudaStreamNonBlocking, greatest);
```

要注意：

- priority 是 hint，不是严格实时调度。
- 通常主要影响 kernel launch 调度，不一定影响 memcpy。
- 不会抢占已经在执行的长 kernel。
- 如果要低延迟，通常还要配合短 kernel、合理切分、CUDA Graph、persistent kernel 或框架级调度。

## Stream 和 CUDA Graph

Stream 表达一串异步操作；CUDA Graph 可以把这串操作捕获成图，之后反复 launch，减少 CPU 侧提交开销。

```cpp
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

kernel1<<<grid, block, 0, stream>>>(...);
kernel2<<<grid, block, 0, stream>>>(...);

cudaGraph_t graph;
cudaStreamEndCapture(stream, &graph);
```

心智模型：

```text
stream:
  每次运行时 enqueue work

graph:
  先把 stream work 捕获成 DAG
  后续复用实例，减少重复 launch/API 开销
```

LLM 推理里 decode step 重复、kernel 很多且 shape 稳定时，CUDA Graph 往往比单纯多 stream 更关键。

## Stream 和 PDL

传统同一 stream 内，两个 dependent kernel 是串行的：

```text
primary 完成 -> secondary launch / execute
```

Hopper 起的 Programmatic Dependent Launch（PDL）允许在 same-stream dependency 下，把 secondary kernel 的部分启动开销与 primary 尾段重叠。它不是取消 stream 顺序，而是在特殊 acquire/release 机制下放宽传统 stream serialization 的一部分。详细见 [[CUDA PDL Programmatic Dependent Launch]]。

## 在 Nsight Systems 里怎么看

`nsys` timeline 里常见几类信号：

| 现象 | 可能含义 |
|------|----------|
| 多个 stream 的 kernel 色块排成一条线 | 没有真正并发；可能资源吃满、默认流同步、依赖过强 |
| kernel 之间有明显 gap | 可能是 launch overhead、CPU 调度、同步、Python 开销 |
| H2D/D2H copy 和 kernel 没有重叠 | 可能没用 `cudaMemcpyAsync`、host memory 未 pinned、copy engine/方向受限 |
| 某个默认流操作把其他 stream 前后截断 | legacy default stream 或同步 API 造成隐式同步 |
| NCCL stream 和 compute stream 没 overlap | 通信/计算依赖放置、stream wait event 或资源竞争需要检查 |

调试顺序建议：

```text
1. 先用 nsys 看 timeline
2. 确认是否真的有多个 stream
3. 看默认流和同步 API 是否插在中间
4. 看 memcpy 是否 async + pinned
5. 看 kernel 是否已经吃满 GPU，导致没有并发空间
6. 必要时再进 ncu 看单 kernel 资源占用
```

## 和 LLM 推理的关系

LLM inference 框架里经常能看到多 stream：

- compute / forward stream：主模型前向。
- copy stream：KV cache、输入输出张量、CPU/GPU 拷贝。
- communication stream：NCCL all-reduce、all-to-all、KV transfer。
- sampling 或 postprocess stream：某些实现会把采样、logits 处理拆出去。

但“用了多 stream”不等于“自动 overlap”。真正的 overlap 取决于：

- 调度器是否足够早地提交下游 work。
- stream 之间依赖是否被 event 精准表达。
- kernel 资源占用是否允许并发。
- 通信是否和计算访问同一瓶颈资源，例如 NVLink、L2、HBM。
- 框架是否在关键路径上调用了全局同步或强制取回 CPU。

所以分析推理性能时，stream 不是孤立概念，而是和 `nsys`、CUDA Graph、NCCL、KV cache 调度、CPU scheduler 一起看的。

## 常见问题

| 问题 | 答案 |
|------|------|
| stream 能让 block 之间通信吗 | 不能。stream 控制 kernel / memcpy 等操作的提交顺序，不改变 kernel 内 block 间同步语义。 |
| 同一 stream 内 kernel 能共享结果吗 | 可以依赖顺序：前一个 kernel 写 global memory，后一个 kernel 在同一 stream 中读取，通常不需要额外 event。 |
| 不同 stream 中读写同一 buffer 怎么办 | 必须用 event / wait 或其他同步建立 happens-before，否则就是数据竞争。 |
| `cudaMemcpyAsync` 一定异步吗 | 对 device 侧队列是 async 形式；若 host memory 不是 pinned，host 侧可能退化为同步，难以 overlap。 |
| stream 越多越好吗 | 不是。太多 stream 会增加调度复杂度；先用少量 stream 表达真实独立工作。 |
| stream priority 能抢占长 kernel 吗 | 通常不能。priority 是调度提示，不是抢占式实时机制。 |
| 默认 stream 安全吗 | 入门示例安全，但做并发优化时要特别小心 legacy default stream 的隐式同步。 |

## 工程 checklist

- [ ] 每个 kernel launch 是否明确传入 stream？
- [ ] 是否误用了 legacy default stream？
- [ ] 需要跨 stream 依赖的地方是否用了 event？
- [ ] 是否用了 `cudaDeviceSynchronize()` 把并发全部摊平？
- [ ] `cudaMemcpyAsync` 的 host buffer 是否 pinned？
- [ ] stream 内是否有过长 kernel 导致 priority 或 overlap 无效？
- [ ] `nsys` timeline 是否证明了真实 overlap？
- [ ] 出错时是否先 `cudaGetLastError()` 检查 launch，再在同步点检查 async error？
- [ ] 如果大量短 kernel 重复执行，是否应考虑 CUDA Graph？

## 参考

- [CUDA Programming Guide: Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
- [CUDA Programming Guide: CUDA Streams](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-streams)
- [CUDA Programming Guide: CUDA Stream Ordering](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#cuda-stream-ordering)
- [CUDA Programming Guide: Blocking and non-blocking streams and the default stream](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#blocking-and-non-blocking-streams-and-the-default-stream)
- [CUDA Programming Guide: Explicit Synchronization](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#explicit-synchronization)
