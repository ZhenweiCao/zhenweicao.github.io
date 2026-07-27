# Reduction Sum Kernel 优化详解

> 本文档整理 GPU 归约操作的各种优化技巧及面试常见问题。

## 1. 问题定义

实现一个 kernel，计算数组中所有元素的和：
```cpp
float sum = 0;
for (int i = 0; i < n; i++) sum += input[i];
```

## 2. 逐步优化版本

### Version 0: Naive 实现（存在线程发散）

```cpp
__global__ void reduction_v0(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // 归约：问题所在 - 使用 % 操作导致线程发散
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {  // ⚠️ 发散分支！
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

**问题分析**：
- `tid % (2*s) == 0` 导致每次迭代只有部分线程活跃
- 例如 s=1 时，tid=0,2,4,6... 活跃，tid=1,3,5... 空闲
- 造成严重的线程发散（branch divergence）

---

### Version 1: 消除线程发散（交错寻址）

```cpp
__global__ void reduction_v1(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // 改进：使用顺序寻址，相邻线程一起工作
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;  // 交错寻址
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

**改进点**：
- 相邻线程一起工作，减少 warp 内发散
- 但仍存在 `if (index < blockDim.x)` 分支

---

### Version 2: 完全消除分支（循环展开思想）

```cpp
__global__ void reduction_v2(float *input, float *output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // 每个线程加载两个元素并做第一次加法（循环展开）
    float sum = 0.0f;
    if (i < n) sum += input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // 树形归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

**关键优化**：
- 每个线程先加载 2 个元素并求和，减少 block 数量
- 第一次归约已在加载时完成（循环展开）
- 树形归约阶段 `tid < s` 确保前面线程连续工作

---

### Version 3: Warp 级优化（使用 shuffle 指令）

```cpp
// 使用 warp shuffle 进行归约
__inline__ __device__ float warp_reduce_sum(float val) {
    // shuffle down 操作
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduction_v3(float *input, float *output, int n) {
    float sum = 0.0f;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程处理多个元素（网格跨步循环）
    while (i < n) {
        sum += input[i];
        i += blockDim.x * gridDim.x;
    }

    // 存储到共享内存
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();

    // 每个 warp 先做内部归约
    sum = warp_reduce_sum(sdata[tid]);

    // warp 0 收集所有结果
    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = sum;
    }
    __syncthreads();

    // warp 0 做最终归约
    if (tid < warpSize) {
        sum = (tid < blockDim.x / warpSize) ? sdata[tid] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0) output[blockIdx.x] = sum;
    }
}
```

**核心优势**：
- `__shfl_down_sync` 在 warp 内直接交换数据，无需共享内存
- 减少共享内存访问和同步开销

---

### Version 4: 完整优化（模板 + 完全展开）

```cpp
// 完全展开的归约（需要编译时知道 block 大小）
template <unsigned int blockSize>
__device__ void warp_reduce(volatile float *sdata, int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void reduction_v4(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    float sum = 0.0f;
    // 多次加载
    while (i < n) {
        sum += input[i];
        if (i + blockSize < n) sum += input[i + blockSize];
        i += gridSize;
    }
    sdata[tid] = sum;
    __syncthreads();

    // 完全展开的归约
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }

    // 最后 warp 无需 __syncthreads()
    if (tid < 32) warp_reduce<blockSize>(sdata, tid);

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// 启动时使用模板特化
#define LAUNCH_REDUCE(BLOCK_SIZE) \
    reduction_v4<BLOCK_SIZE><<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_in, d_out, n)
```

---

## 3. 面试常见问题与解答

### Q1: 为什么 Version 0 的 `tid % (2*s)` 会导致性能问题？

**答**: 这会造成严重的 warp 内线程发散（branch divergence）。
- GPU 以 warp（32 线程）为单位执行，warp 内所有线程必须执行相同指令
- `%` 操作导致每次迭代只有部分线程活跃，其他线程空转
- 例如 s=1 时，warp 中偶数线程执行加法，奇数线程空转，50% 计算力浪费

---

### Q2: 如何解决 bank conflict？

**答**:
- **共享内存 bank 结构**: 通常分为 32 个 bank，每个 bank 每周期只能服务一个访问
- **冲突场景**: 多个线程访问同一个 bank 的不同地址（stride 为 32 的倍数）
- **解决方案**:
  1. 使用交错寻址而非顺序寻址
  2. 在共享内存分配时添加 padding: `sdata[tid + tid/32]`
  3. 使用 warp shuffle 避免共享内存访问

---

### Q3: 为什么最后要用 warp_reduce 而不是继续用共享内存？

**答**:
- **warp 内天然同步**: 一个 warp 内的线程是 SIMD 同步执行的，无需 `__syncthreads()`
- **shuffle 指令更快**: `__shfl_down_sync` 直接通过 register 交换数据，避免共享内存访问延迟
- **减少同步开销**: 最后阶段只需一个 warp 工作，其他 warp 的线程可以空闲

---

### Q4: 如何处理不同大小的输入数据？

**答**:
1. **网格跨步循环（Grid-stride loop）**:
   ```cpp
   for (int i = tid; i < n; i += blockDim.x * gridDim.x)
   ```
   让每个线程处理多个元素，适应任意大小输入

2. **多级归约**: 当 block 数量 > 1 时，将输出再作为输入递归调用

3. **尾部处理**: 对于非 2 的幂次长度，使用条件加载避免越界

---

### Q5: 模板参数 `blockSize` 的优势是什么？

**答**:
- **编译期优化**: 编译器知道具体数值，可以展开循环、优化寄存器分配
- **完全展开**: 最后的 warp reduce 可以完全展开，消除循环开销
- **类型安全**: 不同的 block 大小生成不同的函数，避免运行时分支
- **缺点**: 需要为常用 block 大小（128, 256, 512）分别实例化，增加代码体积

---

### Q6: 如何验证 kernel 的正确性？

**答**:
```cpp
// 1. 与 CPU 结果对比
float cpu_sum = 0;
for (int i = 0; i < n; i++) cpu_sum += h_input[i];

// 2. 考虑浮点精度
float tolerance = 1e-4 * n;  // 根据数据规模调整
assert(fabs(gpu_sum - cpu_sum) < tolerance);

// 3. 边界测试
// - n = 1（最小输入）
// - n = 2^k（2的幂）
// - n = 2^k + 1（非2的幂）
// - n = 0（空输入）
```

---

## 4. 性能对比总结

| 版本 | 核心优化 | 理论加速比 | 主要瓶颈 |
|------|---------|-----------|---------|
| v0 | 基础实现 | 1x | 线程发散 |
| v1 | 消除 % 操作 | ~2x | 共享内存 bank conflict |
| v2 | 循环展开+树形归约 | ~4x | 最后 warp 同步开销 |
| v3 | warp shuffle | ~8x | 全局内存带宽 |
| v4 | 模板展开+多次加载 | ~10x+ | 算术强度 |

## 5. 关键考点 checklist

- [ ] 理解 warp 执行模型和线程发散
- [ ] 掌握共享内存 bank conflict 及解决方法
- [ ] 熟悉 `__syncthreads()` 的使用场景
- [ ] 了解 warp shuffle 指令（`__shfl_down_sync`）
- [ ] 理解网格跨步循环（grid-stride loop）
- [ ] 掌握模板编译期优化技巧
- [ ] 了解多级归约的处理方式

---

*参考资料：NVIDIA CUDA Samples, Mark Harris - Optimizing Parallel Reduction*
