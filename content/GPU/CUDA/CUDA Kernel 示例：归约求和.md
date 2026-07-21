---
aliases:
  - reduction_sum
updated: 2026-05-20
tags:
  - gpu-computing
  - cuda-programming
  - gpu-programming
---
# CUDA Kernel 示例：归约求和

> 目标：把 reduction sum 的优化路径讲清楚，便于面试复述和工程落地。
相关主笔记：

- [[GPU 知识库索引]]
- [[CUDA 编程基础]]
- [[CUDA 线程配置与占用率]]
- [[CUDA Shared Memory 与 Bank Conflict]]

---

## 1. 问题定义

给定 `input[0..N-1]`，计算：

```cpp
float sum = 0.0f;
for (int i = 0; i < N; ++i) sum += input[i];
```

GPU 版本本质是两层归约：
- **块内归约**：每个 block 得到一个 partial sum
- **块间归约**：对 partial sum 继续归约，直到得到 1 个值

---

## 2. 优化路线图（先看全貌）

| 版本 | 核心动作 | 主要收益 | 常见瓶颈 |
|------|----------|----------|----------|
| v0 | 朴素共享内存归约 | 作为基线 | 分支发散 + 同步多 |
| v1 | 顺序地址 + 树形折半 | 分支模式更规整 | 仍有同步和共享访存开销 |
| v2 | 每线程加载 2 个元素 + grid-stride | 减少 block 数和访存压力 | 共享内存归约尾段效率一般 |
| v3 | warp shuffle 做尾归约 | 去掉最后阶段共享内存同步 | 全局带宽开始成为主瓶颈 |
| v4 | 模板展开 + 多元素累加 | 减少循环和分支开销 | 代码体积增大，可维护性下降 |

一句话记忆：**先解决发散，再减少同步，再把最后 32 个线程改成 shuffle。**

## 2.1 正确性边界

- 浮点加法不满足结合律，GPU reduction 的结果可能和 CPU 串行累加有微小差异，验证时应使用容差。
- 大数组通常需要多轮 kernel：第一轮每个 block 输出 partial sum，后续继续归约 partial sum。
- `__shfl_down_sync(mask, v, offset)` 的 `mask` 应匹配参与计算的 lane。完整 warp 可用 `0xffffffff`，尾部不足一个 warp 时要谨慎处理 active mask。
- 若输入类型是 `half` / `bf16`，通常用 FP32 累加更稳。

---

## 3. 各版本关键代码与要点

### 3.1 v0：朴素版本（仅作反例）

```cpp
for (unsigned int s = 1; s < blockDim.x; s <<= 1) {
    if (tid % (2 * s) == 0) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```

- `%` 触发非连续活跃线程，warp 利用率低。
- 作为教学起点可用，工程上不推荐。

### 3.2 v1：树形折半（更规整）

```cpp
for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

- 活跃线程连续，行为更接近硬件执行习惯。
- 仍需 `log2(blockDim.x)` 次同步。
- 这里从 `blockDim.x / 2` 开始，而不是从全局线程数开始，是因为这段代码只在**一个 block 的 shared memory** 里归约。`sdata` 的合法下标只有 `0..blockDim.x-1`，不同 block 的 `sdata` 彼此不可见，也不能在同一个 kernel 内做全局同步。
- 所以这一轮 kernel 的最终结果不是全局唯一结果，而是每个 block 一个 partial sum：块内归约完成后，只有 `threadIdx.x == 0` 把 `sdata[0]` 写到 `d_out[blockIdx.x]`。

> 初学者重点：这里的 `s` 是 **block 内 shared memory 的归约跨度**，不是全局数组或全局线程的跨度。每个 block 只能把自己的 `sdata[0..blockDim.x-1]` 归约成一个 partial sum；块内最终结果由 `threadIdx.x == 0` 写到 `d_out[blockIdx.x]`。如果 `gridDim.x > 1`，`d_out` 里会有多个 partial sums，还需要下一轮 kernel 继续归约。

### 3.3 v2：加载阶段先合并

```cpp
unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
float sum = 0.0f;
if (i < N) sum += input[i];
if (i + blockDim.x < N) sum += input[i + blockDim.x];
sdata[tid] = sum;
```

- 每线程一次处理 2 个元素，降低后续归约压力。
- 常配合 grid-stride 处理大 `N`。

### 3.4 v3：最后一 warp 改用 shuffle

```cpp
__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}
```

- warp 内寄存器直接交换，避免共享内存和 `__syncthreads()`。
- 是实战里最常见的“临门一脚”优化。

### 3.5 v4：模板展开（追求极致）

- 用 `template <unsigned int blockSize>` 让编译器静态展开。
- 常见 block 大小：`128/256/512`。
- 优点是快，缺点是代码膨胀，调试成本高。

---

## 4. 生产可用版本（推荐骨架）

```cpp
#include <cuda_runtime.h>
#include <algorithm>

__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
//    if (tid < 32) {
//        float val = sdata[tid];
//        // 利用蝴蝶型 shuffle_down 在 warp 内部求和
//        val += __shfl_down_sync(0xffffffff, val, 16);
//        val += __shfl_down_sync(0xffffffff, val, 8);
//        val += __shfl_down_sync(0xffffffff, val, 4);
//        val += __shfl_down_sync(0xffffffff, val, 2);
//        val += __shfl_down_sync(0xffffffff, val, 1);
//        // 第一个线程将本 block 的部分和写入全局内存
//        if (tid == 0) {
//            d_out[blockIdx.x] = val;
//        }
//    }
}


__global__ void reduce_sum_kernel(const float* __restrict__ d_in,
                                  float* __restrict__ d_out,
                                  int N) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    // 1) grid-stride 累加
    float local = 0.0f;
    for (int i = idx; i < N; i += stride) local += d_in[i];

    // 2) 写入共享内存后做块内归约
    sdata[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // 3) 最后一个 warp 用 shuffle 结束
    if (tid < 32) {
        float v = sdata[tid];
        if (blockDim.x >= 64) v += sdata[tid + 32];
        v = warp_reduce_sum(v);
        if (tid == 0) d_out[blockIdx.x] = v;
    }
}

float reduce_sum_host(const float* d_in, int N) {
    constexpr int block_size = 256;
    int cur_n = N;
    const float* cur_in = d_in;
    float* tmp_a = nullptr;
    float* tmp_b = nullptr;

    // 最多需要一层临时 buffer，循环直到归约成 1 个值
    cudaMalloc(&tmp_a, ((N + block_size - 1) / block_size) * sizeof(float));
    cudaMalloc(&tmp_b, ((N + block_size - 1) / block_size) * sizeof(float));

    bool use_a = true;
    while (cur_n > 1) {
        int grid = std::min(1024, (cur_n + block_size - 1) / block_size);
        float* cur_out = use_a ? tmp_a : tmp_b;
        reduce_sum_kernel<<<grid, block_size, block_size * sizeof(float)>>>(cur_in, cur_out, cur_n);
        cur_in = cur_out;
        cur_n = grid;
        use_a = !use_a;
    }

    float result = 0.0f;
    cudaMemcpy(&result, cur_in, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(tmp_a);
    cudaFree(tmp_b);
    return result;
}
```

说明：
- 这个版本避免了“第一轮在 GPU，第二轮拷回 CPU 求和”的瓶颈。
- 如果追求最高性能，可替换为 CUB 的 `DeviceReduce::Sum` 做基准对比。

---

## 5. `__shfl_down_sync` 速记

`__shfl_down_sync(mask, val, delta)`：读取当前线程下方 `delta` 号 lane 的 `val`。

- 常用掩码：`0xffffffff`（32 线程全参与）
- 归约套路：`16 -> 8 -> 4 -> 2 -> 1`
- 核心价值：**warp 内直接寄存器交换，省共享内存与同步**

---

## 6. 易错点（面试也常问）

1. **`__syncthreads()` 放错位置**：有分支时必须保证所有线程都能到达同步点。  
2. **越界访问**：加载第二个元素 `i + blockDim.x` 时必须判断。  
3. **blockSize 不是 2 的幂**：折半归约逻辑会复杂很多，通常直接约束为 2 的幂。  
4. **数值误差预期错误**：浮点加法不满足结合律，GPU/CPU 顺序不同，结果允许误差。  
5. **只测单一规模**：需要覆盖 `N=1`、`N=2^k`、`N=2^k+1`、大规模随机输入。  
6. **忽略内存带宽上限**：reduction 常是 memory-bound，后期优化收益会递减。

---

## 7. 面试高频问答（精简版）

### Q1：为什么 v0 慢？
`tid % (2*s)` 造成活跃线程稀疏，warp 执行效率低，且同步频繁。

### Q2：为什么最后 32 个元素用 shuffle？
warp 内天然锁步，shuffle 可寄存器直连，减少共享内存和同步开销。

### Q3：bank conflict 如何规避？
尽量顺序访问；必要时 padding；或在尾阶段改用 shuffle 规避共享内存冲突。

### Q4：如何支持任意大 N？
使用 grid-stride loop + 多轮归约（partial sum 继续作为输入）。

### Q5：怎么验证正确性？
对照 CPU 结果（允许容差）+ 覆盖边界输入 + 多随机种子。

### Q6：为什么块内归约从 `blockDim.x / 2` 开始，而不是全局总线程数？

因为这段树形归约处理的是当前 block 的 `sdata[0..blockDim.x-1]`。shared memory 是 block 私有的，不能用 `gridDim.x * blockDim.x` 当跨度访问，也不能让一个 block 直接读另一个 block 的 shared memory。

> 判断口诀：看到 `sdata[...]` 和 `__syncthreads()`，先默认它只在 **block 内** 生效；看到 `d_out[blockIdx.x]`，说明输出的是 **每个 block 一个 partial sum**。

最终结果的位置分两层看：

- **块内**：归约结束后，当前 block 的结果在 `sdata[0]`，由 `threadIdx.x == 0` 写出。
- **块间**：第一轮 kernel 会得到 `gridDim.x` 个 partial sums，保存在 `d_out[0..gridDim.x-1]`。这些 partial sums 还要作为下一轮 kernel 的输入继续归约，直到只剩 1 个值。

---

## 8. 复习 Checklist

- [ ] 解释清楚“块内归约 + 块间归约”两层结构  
- [ ] 写出 `grid-stride loop` 模板  
- [ ] 写出 `warp_reduce_sum` 模板  
- [ ] 说明为什么最后阶段不用 `__syncthreads()`  
- [ ] 说清楚误差来源（浮点非结合律）  

---

参考：NVIDIA CUDA Samples，Mark Harris《Optimizing Parallel Reduction》
