---
title: "第2章：CUDA 入门"
content_type: guide
maturity: reviewed
updated: 2026-07-27
publish: true
tags:
  - gpu-computing
  - cuda-programming
---
# 第2章：CUDA 入门

> 动手写第一个 CUDA 程序，理解线程模型的精髓

## 本章定位

本章负责从“知道概念”过渡到“能写并运行第一个 CUDA kernel”。读完后应能独立写 vector add，并理解 host/device memory copy、kernel launch 和错误检查。

配套主文档：

- [[CUDA 编程基础]]
- [[CUDA Kernel 示例：向量加法]]
- [[CUDA 线程配置与占用率]]

## 学习目标

学完本章后，你将能够：
- 配置 CUDA 开发环境
- 编写、编译、运行 CUDA 程序
- 理解线程索引的计算
- 掌握基本的内存操作

## 2.1 环境配置

### 2.1.1 检查你的 GPU

首先，确认你的电脑有 NVIDIA GPU：

**Windows**:
- 右键"此电脑" → 管理 → 设备管理器 → 显示适配器
- 看到 "NVIDIA GeForce RTX ..." 或类似名称

**Linux**:
```bash
nvidia-smi
```
如果看到 GPU 信息表格，说明 GPU 正常。

### 2.1.2 安装 CUDA Toolkit

**下载地址**：https://developer.nvidia.com/cuda-downloads

根据你的操作系统选择对应版本。安装过程和普通软件一样。

**验证安装**：
```bash
nvcc --version
```
看到版本信息表示安装成功。

### 2.1.3 选择开发工具

推荐使用：
- **VS Code** + CUDA 扩展（轻量，推荐新手）
- **Visual Studio**（Windows 功能强大）
- **命令行 + 任意编辑器**（最简单）

### 2.1.4 使用 Google Colab（如果没有 GPU）

1. 打开 https://colab.research.google.com/
2. 点击"更改运行时类型" → 选择"GPU"
3. 可以直接在笔记本中写 CUDA 代码

## 2.2 第一个 CUDA 程序

### 2.2.1 Hello GPU

创建文件 `hello_cuda.cu`：

```cpp
#include <stdio.h>

// 这是 Kernel 函数，在 GPU 上运行
__global__ void hello_from_gpu() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");
    
    // 启动 Kernel：1个 Block，5个线程
    hello_from_gpu<<<1, 5>>>();
    
    // 等待 GPU 完成
    cudaDeviceSynchronize();
    
    printf("Back to CPU!\n");
    return 0;
}
```

### 2.2.2 编译和运行

```bash
# 编译
nvcc hello_cuda.cu -o hello_cuda

# 运行
./hello_cuda
```

**输出**：
```text
Hello from CPU!
Hello from GPU thread 0!
Hello from GPU thread 1!
Hello from GPU thread 2!
Hello from GPU thread 3!
Hello from GPU thread 4!
Back to CPU!
```

### 2.2.3 代码解析

让我们逐行理解：

```cpp
__global__ void hello_from_gpu()
```
- `__global__`：告诉编译器这是一个 Kernel 函数
- 这个函数会在 GPU 上运行
- 可以被多个线程同时执行

```cpp
printf("Hello from GPU thread %d!\n", threadIdx.x);
```
- `threadIdx.x`：当前线程的编号
- 每个线程都有自己独特的 threadIdx.x
- 0, 1, 2, 3, 4... 每个线程打印不同的数字

```cpp
hello_from_gpu<<<1, 5>>>();
```
- 这是启动 Kernel 的语法：`kernel<<<blocks, threads>>>()`
- `1`：启动 1 个 Block
- `5`：每个 Block 有 5 个线程
- 总共 5 个线程并行执行

```cpp
cudaDeviceSynchronize();
```
- CPU 和 GPU 是异步执行的
- 这行代码让 CPU 等待 GPU 完成
- 如果不加这行，可能看不到 GPU 的输出

## 2.3 线程索引详解

### 2.3.1 一维线程

最简单的情况：一维的 Block 和线程

```cpp
__global__ void print_index_1d() {
    // 线程在 Block 内的索引
    int local_idx = threadIdx.x;
    
    // Block 在 Grid 内的索引
    int block_idx = blockIdx.x;
    
    // Block 内线程数量
    int block_size = blockDim.x;
    
    // 全局唯一索引
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    printf("Block %d, Thread %d, Global %d\n", 
           block_idx, local_idx, global_idx);
}

int main() {
    // 3 个 Block，每个 4 个线程
    print_index_1d<<<3, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

**输出**：
```text
Block 0, Thread 0, Global 0
Block 0, Thread 1, Global 1
Block 0, Thread 2, Global 2
Block 0, Thread 3, Global 3
Block 1, Thread 0, Global 4    ← blockIdx=1, threadIdx=0 → 1*4+0=4
Block 1, Thread 1, Global 5
...
```

### 2.3.2 索引计算公式（重要！）

```text
全局索引 = blockIdx.x * blockDim.x + threadIdx.x
```

这是 CUDA 编程最常用的公式，必须记住！

### 2.3.3 处理任意大小的数据

问题：如果数据量不是 Block 大小的整数倍怎么办？

```cpp
__global__ void process_array(float* data, int n) {
    // 计算全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查！非常重要！
    if (idx < n) {
        data[idx] = data[idx] * 2;
    }
}

int main() {
    int n = 1000;  // 数据量
    int threads_per_block = 256;  // 每个 Block 256 个线程
    
    // 计算需要多少个 Block
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    // (1000 + 255) / 256 = 4 个 Block
    
    // 启动 Kernel
    process_array<<<blocks, threads_per_block>>>(d_data, n);
    
    return 0;
}
```

**为什么 `(n + threads_per_block - 1) / threads_per_block`？**

这是向上取整的技巧：
- 如果 n = 1000, threads_per_block = 256
- 1000 / 256 = 3.9... 需要 4 个 Block
- (1000 + 255) / 256 = 1255 / 256 = 4 ✓

## 2.4 内存操作

### 2.4.1 CUDA 内存模型

```text
┌─────────────────┐        ┌─────────────────┐
│      CPU        │        │      GPU        │
│   (Host)        │        │   (Device)      │
├─────────────────┤        ├─────────────────┤
│  h_data        │ cudaMemcpy │  d_data      │
│  (主机内存)     │ ←───────→ │  (设备内存)    │
│                 │        │                 │
│  malloc()      │        │  cudaMalloc()   │
│  free()        │        │  cudaFree()     │
└─────────────────┘        └─────────────────┘
```

### 2.4.2 内存操作函数

| 函数 | 作用 | 例子 |
|------|------|------|
| `cudaMalloc()` | 分配 GPU 内存 | `cudaMalloc(&d_ptr, size)` |
| `cudaFree()` | 释放 GPU 内存 | `cudaFree(d_ptr)` |
| `cudaMemcpy()` | 内存拷贝 | `cudaMemcpy(dst, src, size, direction)` |
| `cudaMemset()` | 设置内存值 | `cudaMemset(d_ptr, 0, size)` |

### 2.4.3 完整示例：向量加法

```cpp
#include <stdio.h>

// 向量加法 Kernel
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000000;  // 100万个元素
    size_t size = n * sizeof(float);
    
    // 1. 分配 CPU 内存
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);
    
    // 2. 初始化数据
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 3. 分配 GPU 内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 4. 把数据从 CPU 拷贝到 GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 5. 启动 Kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    
    // 6. 把结果从 GPU 拷贝回 CPU
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 7. 验证结果
    printf("Result sample:\n");
    for (int i = 0; i < 5; i++) {
        printf("  c[%d] = %.1f + %.1f = %.1f\n", i, h_a[i], h_b[i], h_c[i]);
    }
    
    // 8. 释放内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

### 2.4.4 错误检查

CUDA 函数可能失败，需要检查错误：

```cpp
// 定义错误检查宏
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
}

// 使用
CHECK_CUDA_ERROR(cudaMalloc(&d_a, size));
CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

// 检查 Kernel 错误
vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
CHECK_CUDA_ERROR(cudaGetLastError());  // 检查 Kernel 启动错误
CHECK_CUDA_ERROR(cudaDeviceSynchronize());  // 检查 Kernel 执行错误
```

## 2.5 线程块和网格

### 2.5.1 为什么需要多个 Block？

限制：每个 Block 最多 1024 个线程

如果需要更多线程，就需要多个 Block。

### 2.5.2 二维和三维索引

CUDA 支持 1D、2D、3D 的线程组织：

```cpp
// 二维线程块
dim3 threads(16, 16);  // 16x16 = 256 个线程
dim3 blocks(32, 32);   // 32x32 = 1024 个 Block

kernel<<<blocks, threads>>>();  // 总共 32*32 * 16*16 = 262144 个线程

// 在 Kernel 中获取全局索引（正确写法）
__global__ void kernel() {
    // 必须用 block 偏移 + thread 偏移，threadIdx 只是 block 内的局部索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
}
```

> **常见错误**：以为 `int x = threadIdx.x; int y = threadIdx.y;` 是另一种写法。**这只在 grid 只有 1 个 block 时才正确**；只要 `blocks.x > 1` 或 `blocks.y > 1`，不同 block 的线程会得到相同的 `(x, y)`，导致越界、写冲突等隐蔽 bug。**全局索引必须包含 `blockIdx * blockDim` 偏移**。

### 2.5.3 二维索引的应用：图像处理

```cpp
__global__ void process_image(unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;  // 二维索引转一维
        image[idx] = 255 - image[idx];  // 反色
    }
}

int main() {
    int width = 1920, height = 1080;
    
    dim3 threads(16, 16);  // 16x16 = 256 线程/Block
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    
    process_image<<<blocks, threads>>>(d_image, width, height);
    
    return 0;
}
```

## 2.6 实践：矩阵加法

让我们把学到的知识整合起来：

```cpp
#include <stdio.h>

// 矩阵加法 Kernel
__global__ void matrix_add(float* A, float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int rows = 1024, cols = 1024;
    size_t size = rows * cols * sizeof(float);
    
    // 分配和初始化（省略详细代码）
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    
    // ... 分配内存，初始化数据，拷贝到 GPU ...
    
    // 配置线程和块
    dim3 threads(16, 16);  // 每个 Block 16x16 = 256 个线程
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
    
    // 启动 Kernel
    matrix_add<<<blocks, threads>>>(d_A, d_B, d_C, rows, cols);
    
    // ... 拷贝结果回 CPU，验证，释放内存 ...
    
    return 0;
}
```

## 2.7 调试技巧

### 2.7.1 使用 printf

在 Kernel 中使用 printf 可以调试：

```cpp
__global__ void debug_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 5) {  // 只打印前5个
        printf("data[%d] = %f\n", idx, data[idx]);
    }
}
```

### 2.7.2 检查限制

```cpp
// 检查设备属性
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Max threads dim: %d x %d x %d\n", 
       prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
```

## 💡 本章要点

1. **Kernel 是在 GPU 上运行的函数**，用 `__global__` 标记
2. **启动语法**：`kernel<<<blocks, threads>>>()`
3. **索引公式**：`idx = blockIdx.x * blockDim.x + threadIdx.x`
4. **边界检查**：始终检查 `idx < n`
5. **内存操作**：cudaMalloc, cudaMemcpy, cudaFree
6. **错误检查**：检查每个 CUDA 调用的返回值

## 📝 课后练习

1. 编写一个 Kernel，将数组每个元素加 1
2. 编写一个 Kernel，计算两个向量的点积
3. 修改向量加法程序，使用不同大小的 Block（64, 128, 256），观察性能差异
4. 编写一个 Kernel，将 256x256 的图像旋转 90 度

## 🔗 相关资源

- CUDA C++ Programming Guide（官方文档）
- CUDA Best Practices Guide

---

[上一章：基础概念 ←](第1章-基础概念.md) | [下一章：GPU 硬件 →](第3章-GPU硬件.md)
