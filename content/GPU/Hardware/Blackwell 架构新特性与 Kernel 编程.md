---
aliases:
  - Blackwell Kernel 编程
  - Blackwell 新特性
  - SM100 tcgen05
  - TMEM
  - Cluster Launch Control
  - CLC
  - GDC
updated: 2026-06-03
tags:
  - gpu-computing
  - gpu-architecture
  - cuda-programming
  - fp4-quantization
  - nvfp4-quantization
---
# Blackwell 架构新特性与 Kernel 编程

> 校对口径：本文按 CUDA 13.3 文档、PTX ISA 9.3、CUTLASS 4.x Blackwell 文档和 Transformer Engine NVFP4 文档整理。Blackwell 底层指令仍随 CUDA Toolkit 演进；写生产 kernel 时优先以当前 Toolkit 的 PTX ISA / CUTLASS 文档为准。

## 定位

这篇不是 Blackwell 规格表，而是面向 **GPU kernel 开发** 的机制笔记：看到 `tcgen05.mma`、TMEM、CLC、`2cta` / `2-SM cooperative`、NVFP4 block scaling、PDL/GDC 时，能判断它们在 kernel 里怎么组织、需要哪些同步、哪些布局约束会直接影响性能。

相关主笔记：

- [[GPU 知识库索引]]
- [[NVIDIA GPU 架构与规格]]
- [[GPU 硬件架构背景与编程范式]]
- [[CUDA CTA 与 Thread Block Cluster 入门]]
- [[CUDA GEMM 矩阵乘法优化指南]]
- [[CUDA PDL Programmatic Dependent Launch]]
- [[NVFP4 量化与反量化]]

官方入口：

- [PTX ISA: Tensor Memory and tcgen05 instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUDA C Programming Guide: Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
- [CUDA C Programming Guide: Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
- [CUTLASS: Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
- [CUTLASS: Dependent kernel launches](https://docs.nvidia.com/cutlass/media/docs/cpp/dependent_kernel_launch.html)
- [Transformer Engine: NVFP4 Training](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [NVIDIA Tensor Cores](https://www.nvidia.com/en-in/data-center/tensor-cores/)
- [CUDA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- 本地 PDF：[[GPU/References/NVIDIA Blackwell Architecture Technical Overview.pdf|NVIDIA Blackwell Architecture Technical Overview.pdf]]

## 先讲结论

Blackwell 上写高性能 GEMM / MoE / attention kernel，心智模型要从 Hopper 的：

```text
SMEM descriptors + WGMMA -> register accumulator
```

升级为：

```text
TMA/SMEM descriptors + TMEM + tcgen05.mma + block scale + cluster / scheduler policy
```

几个关键变化：

| 特性 | 它解决什么 | 写 kernel 时真正要关心什么 |
|------|------------|----------------------------|
| `tcgen05.mma` | 第五代 Tensor Core 的 PTX 指令族，覆盖 FP16/BF16/TF32/FP8/FP6/FP4 与 block-scaled MMA。 | accumulator 不再只是寄存器 fragment，而是写入 TMEM；A/B 通常由 matrix descriptor 描述；block scale 要放到 TMEM；完成信号通过 `tcgen05.commit` + `mbarrier` 观察。 |
| TMEM | 给 Tensor Core 专用的片上 accumulator / scale 存储，避免大量 accumulator 压在通用寄存器上。 | 每 SM 256KB，逻辑上 128 lanes × 512 columns × 32bit；按 column 动态分配；退出前必须 dealloc；读写要遵守 warpgroup lane ownership。 |
| `cta_group::2` / 2-SM cooperative | 让两个 CTA 作为一个 Tensor Core 工作组协作，常见于 CUTLASS 里的 `2cta` / `2sm` GEMM。 | 需要 thread block cluster；两个 CTA 必须作为 pair 协调；同一 warpgroup 内所有 `tcgen05` 指令必须使用一致的 `cta_group`；小矩阵不一定受益。 |
| NVFP4 block scaling | FP4 动态范围太小，必须用 per-block scale。NVFP4 用更细的 scale 粒度提高精度。 | 数据是 packed E2M1 FP4；每 16 个元素一个 FP8/UE4M3 scale；还有 per-tensor / per-expert FP32 global scale；硬件 block-scaled MMA 只吃符合布局要求的 scale tensor。 |
| CLC | 让已经在跑的 worker 取消尚未启动的 CTA/cluster，做硬件辅助 work stealing。 | 不是 device 端 launch 新 block，而是取消 pending ClcID；需要 `mbarrier` 等待异步结果；适合 persistent/grouped GEMM 的负载均衡。 |
| PDL/GDC | 让 same-stream dependent kernels 部分重叠：secondary 提前启动并执行独立 prologue。 | primary device 端 trigger；secondary 在读 primary 输出前 wait；CUTLASS 里 GDC 是 PTX `griddepcontrol` 路径，构建和运行都要启用。 |

### Blackwell 算力口径和输入输出格式

规格表里的 FP4、FP8/FP6、FP16/BF16、TF32、INT8、FP64/FP32 不是几块互不相关的“算力硬件”。更准确的读法是：

```text
同一 SM 内的不同执行路径
  CUDA Core / SIMT ALU: 标量、elementwise、reduction、epilogue、部分 FP32/FP64
  Tensor Core: 矩阵乘加 MMA，按 dtype / kind / sparse / scale 选择吞吐口径
  Transformer Engine: 不是第三类算力，而是选择低精度格式、scale、quant/dequant 和 Tensor Core kernel 的软硬件栈
```

在 B200 这类 Blackwell 数据中心 GPU 上，如果只看 dense Tensor Core 峰值，常见比例可以粗略理解为：

```text
FP4 : FP8/FP6 : FP16/BF16 : TF32 ~= 8 : 4 : 2 : 1
```

这个比例来自同一 Tensor Core 路径在不同输入 bit-width 下每周期能吞下更多矩阵元素。**但要吃到对应算力，必须喂对物理格式**：FP4 算力不是把 PyTorch tensor 标成 4-bit 就自动得到，而是要让 A/B、scale、layout、alignment、sparsity metadata 和输出 epilogue 都匹配目标 `tcgen05.mma kind`。

| 算力口径 | 主要入口 | A/B 输入需要什么格式 | scale / metadata | accumulator 与输出 |
|----------|----------|----------------------|------------------|--------------------|
| CUDA Core FP32 / FP64 / FP16 / BF16 | 普通 SIMT 指令、库 epilogue、elementwise kernel | 寄存器里的标量/向量值；来自 HBM 的低精度数据通常要先 unpack / cast。 | 无 Tensor Core block scale；只按普通 kernel 自己的参数处理。 | 输出先在寄存器，再写 GMEM；常用于 softmax、RMSNorm、activation、quant/dequant、GEMM epilogue。 |
| FP64 Tensor Core / HPC 路径 | HPC 库或编译器选择的 FP64 Tensor Core 路径 | A/B 为 FP64 矩阵 tile。 | 无 FP4/FP8 block scale。 | FP64 累加/输出；属于数值模拟/HPC 口径，不是 LLM FP4 推理主线。 |
| TF32 Tensor Core | `tcgen05.mma.kind::tf32` | A/B 逻辑上常来自 FP32 tensor，经库或硬件路径按 TF32 参与 MMA；layout 支持 TN/NT/NN/TT，具体以 CUTLASS 表为准。 | 可带结构化稀疏 metadata；无 block scale。 | D accumulator 在 TMEM；精确 A/B/D 类型由 `idesc` 编码，库 epilogue 再 cast/store 到 C。 |
| FP16 / BF16 Tensor Core | `tcgen05.mma.kind::f16` | A/B 为 FP16 或 BF16 tile；CUTLASS legacy 路径支持多种 A/B layout。 | 可带结构化稀疏 metadata；无 block scale。 | D 在 TMEM；推理/训练 GEMM 常用 FP32 累加，再在 epilogue 输出 FP16/BF16/FP32。 |
| INT8 Tensor Core | `tcgen05.mma.kind::i8` | A/B 为 `int8_t` 或 `uint8_t` tile，通常需要比 FP16 更高的元素对齐。 | 可带结构化稀疏 metadata；量化 scale 通常由软件 epilogue 或前后处理处理，不是这里的 block scale。 | 通常是整数累加路径，再由 epilogue 乘 scale、加 bias、cast 到目标输出。 |
| FP8 / FP6 / FP4 narrow precision，不带 block scale | `tcgen05.mma.kind::f8f6f4` | A/B 是 packed `float_e4m3`、`float_e5m2`、`float_e3m2`、`float_e2m3`、`float_e2m1` 等低 bit 浮点，subbyte 类型有额外 tensor copy alignment 要求。 | 无硬件 block scale；数值范围必须已经由上游处理好。 | D 在 TMEM；epilogue 决定输出为 FP32/BF16/FP16，或重新量化。 |
| MXFP8 / MXFP6 / MXFP4 block-scaled | `tcgen05.mma.kind::mxf8f6f4.block_scale`、`mxf4.block_scale` | A/B 是 `mx_float8_t`、`mx_float6_t`、`mx_float4_t`：低精度数据本体 + OCP microscaling。 | scale factor 是 `float_ue8m0_t`；dense scale vector size 通常 32，sparse 通常 64；scale tensor 要按硬件/CUTLASS layout 进入 TMEM。 | 数学上是 `D += (A_q * SFA) * (B_q * SFB)`；D 在 TMEM，常以 FP32 accumulator 进入 epilogue。 |
| NVFP4 block-scaled | `tcgen05.mma.kind::mxf4nvf4.block_scale` | A/B 是 packed FP4 E2M1，也就是 CUTLASS 的 `nv_float4_t` 路径；Blackwell/TE 中 NVFP4 GEMM 常见只支持 TN 快路径。 | block scale 是 `float_ue4m3_t` / FP8 E4M3，dense 每 16 个元素一个 scale，sparse 通常 32；NVFP4 还常有 FP32 global scale，但它多在 alpha/epilogue/前后处理里使用，不直接当 `tcgen05` block scale。 | D 在 TMEM；epilogue 处理 global scale、bias/activation、cast，最后输出 BF16/FP16/FP32，或生成新的 FP4 data + scale。 |
| Structured sparse Tensor Core | `tcgen05.mma.sp`，当目标 `kind` 支持 `.sp` 时使用 | A 通常是压缩后的结构化稀疏矩阵，B 是对应 dense tile。 | 需要 sparse metadata；这是同一 dtype 上的吞吐倍率修饰，不是新的输出 dtype。 | 输出格式与对应 dense `kind` 相同；规格里的 sparse 峰值不能和 dense 结果直接混用。 |

一条实用边界：

```text
规格表的算力口径 = 硬件在某种 dtype / sparse / scale 组合下的峰值
kernel 的 I/O 合同 = A/B physical layout + scale/metadata layout + D accumulator + epilogue output
模型权重格式 = checkpoint 或 runtime tensor 的存储格式
```

三者要同时对上。比如 NVFP4 权重 checkpoint 只是第一步；如果 scale layout 不是 SM100 block-scaled GEMM 期望的 512B basic-block 组织，或者输出还要单独跑 dequant/quant 小 kernel，端到端就未必能吃到 FP4 Tensor Core 峰值。

可以把 Blackwell kernel 的主线理解为：

```text
1. 用 TMA 把 A/B/scale 搬到 SMEM
2. 用 descriptor 或 tcgen05.cp 把 operand / scale 组织到 tcgen05 能消费的形态
3. 用 tcgen05.mma 把 D 累加到 TMEM
4. 用 mbarrier 观察 MMA 完成
5. 用 tcgen05.ld 把 D 从 TMEM 读回寄存器做 epilogue
6. 用 CLC / persistent scheduler / PDL 把多个 tile 或多个 kernel 的时序重叠起来
```

总图：

![[GPU/Drawings/Blackwell tcgen05 TMEM 数据通路.svg]]

## 一、从 Hopper WGMMA 到 Blackwell tcgen05

### 1.1 `tcgen05.mma` 不只是新的 `wgmma`

Hopper 的高性能 GEMM 主线是：

```text
TMA -> SMEM
SMEM descriptor -> wgmma.mma_async
accumulator -> registers
warpgroup commit/wait
epilogue -> GMEM
```

Blackwell 的 SM100 GEMM 主线变成：

```text
TMA -> SMEM
SMEM descriptor / TMEM operand / scale TMEM -> tcgen05.mma
accumulator -> TMEM
tcgen05.commit + mbarrier
tcgen05.ld -> registers
epilogue -> GMEM
```

这个变化很重要：

- **寄存器压力变化**：大 accumulator 不一定全部压在通用寄存器上，TMEM 变成 Tensor Core accumulator 的主要承载处。
- **同步对象变化**：`tcgen05.mma` 属于异步类操作，不能靠普通指令顺序判断完成；要用 `tcgen05.commit` 与 `mbarrier`。
- **operand 组织变化**：A/B 不只是寄存器 fragment 或 SMEM pointer，而是 matrix descriptor、TMEM address、instruction descriptor 的组合。
- **scale 进入主路径**：FP4/FP6/FP8 block-scaled GEMM 需要 scale tensor，且 scale tensor 的格式、布局、copy 路径会影响能否命中 Tensor Core 快路径。

### 1.2 `tcgen05.mma` 的操作数心智模型

不要先背完整 PTX 语法，先抓住操作数角色：

| 操作数 | 位置 | 作用 |
|--------|------|------|
| `D` | TMEM | accumulator / output accumulator。`enable-input-d` 决定是否读旧 D 累加。 |
| `A` | SMEM descriptor 或 TMEM | 常见 dense / narrow precision 路径由 matrix descriptor 指向 SMEM；部分路径允许 A 来自 TMEM。 |
| `B` | SMEM descriptor | B tile 通常由 matrix descriptor 指向 SMEM。 |
| `idesc` | register immediate / descriptor | instruction descriptor，编码 tile、layout、transpose、sparsity、scale 相关控制。 |
| `scale-A` / `scale-B` | TMEM | block-scaled MMA 的 scale factor 矩阵。 |
| `mbarrier` | SMEM | 观察 `tcgen05.mma` / `tcgen05.cp` 等异步操作完成。 |

伪 PTX 只表达形态：

```ptx
// 伪代码：真实 suffix、kind、shape、dtype、scale vec size 需按 PTX ISA 表选择
tcgen05.mma.cta_group::{1|2}.kind::<kind>.<shape>.<dtype>
  d_tmem,
  a_desc_or_tmem,
  b_desc,
  idesc,
  enable_input_d,
  ... optional scale operands ...;

tcgen05.commit.cta_group::{1|2}.mbarrier::arrive::one.b64 [barrier];
```

你真正需要在 kernel 里保证的是：

1. A/B/scale 的内存布局满足所选 `kind` 和 `shape`。
2. 发起 `tcgen05.mma` 前，SMEM 中由 generic proxy 写入的数据对 async proxy 可见。
3. 同一个 warpgroup 里 `tcgen05.mma` / `tcgen05.cp` / `tcgen05.shift` / `commit` 的 `cta_group` 一致。
4. 后续读取 D 前，已经用 `mbarrier` 确认 MMA 完成。
5. TMEM column 的生命周期正确：alloc -> use -> ld/st/cp/mma -> dealloc -> relinquish。

### 1.3 常见 `kind` 分组

PTX ISA 把 `tcgen05.mma` 分成多个 `kind`，可以按用途理解：

| kind | 用途 | Kernel 含义 |
|------|------|-------------|
| legacy / dense kind | FP16、BF16、TF32、INT8、FP8 等普通 dense MMA。 | 类似 WGMMA 的升级版，但 accumulator 在 TMEM。 |
| `.kind::f8f6f4` | 不带 block scale 的 8/6/4-bit narrow precision。 | 适合数据已经按指令要求编码、但不需要硬件 scale 矩阵参与的路径。 |
| `.kind::mxf8f6f4.block_scale` | OCP MXFP 系列 block-scaled MMA。 | scale factor 通常是 E8M0 / UE8M0，block size 多为 32。 |
| `.kind::mxf4.block_scale` | MXFP4 专门路径。 | FP4 block size = 32，适合 OCP MXFP4。 |
| `.kind::mxf4nvf4.block_scale` | MXFP4 / NVFP4 相关路径。 | NVFP4 的 block size = 16，scale 类型是 FP8/UE4M3，LLM 推理最常遇到。 |

这里的 `kind` 不是纯 dtype 名，它还隐含：

- A/B 数据类型和 packed 方式。
- K 维 block scaling 粒度。
- scale factor 的类型和布局。
- 合法的 transpose/layout 组合。
- 可能的 sparse metadata 约束。

因此不要把 NVFP4 kernel 写成“FP4 数据 + 任意 scale 数组 + 普通 MMA”。硬件快路径要求 scale tensor 以指令认可的方式进入 TMEM。

## 二、TMEM：Tensor Memory

### 2.1 TMEM 是什么

TMEM 是 Blackwell SM 内给第五代 Tensor Core 使用的专用片上存储。它不是 shared memory，也不是普通寄存器文件。

核心属性：

| 属性 | 说明 |
|------|------|
| 容量 | 每 SM 256KB。 |
| 逻辑形态 | 128 lanes × 512 columns × 32bit。 |
| 分配单位 | column。一个 column 覆盖 128 lanes，所以 1 column = 128 × 4B = 512B。 |
| 地址形态 | 32-bit tmem address：高 16 bit 通常表示 lane index，低 16 bit 表示 column index。 |
| 典型用途 | D accumulator、block scale matrix、某些 A operand staging。 |
| 生命周期 | CTA / CTA pair 动态申请和释放；退出前必须释放。 |

可以把 TMEM 想成：

```text
column 0        column 1        ... column 511
lane 0   [32b]  [32b]
lane 1   [32b]  [32b]
...
lane 127 [32b]  [32b]
```

它不是给任意线程随便随机读写的 scratchpad。它围绕 warpgroup 和 Tensor Core 访问模式设计。

### 2.2 TMEM allocation

TMEM 需要显式申请：

```ptx
// 伪代码：申请 ncols 个 column，起始 column 写到 shared memory 位置
tcgen05.alloc.cta_group::{1|2}.sync.aligned.shared::cta.b32 [addr], ncols;
```

规则：

| 规则 | 说明 |
|------|------|
| 申请粒度 | `ncols` 必须是 32 的幂次粒度，常见为 32、64、128、256、512。 |
| 对齐 | 起始 column 与申请大小有关，硬件保证适合后续 `tcgen05` 使用。 |
| collective | `cta_group::1` 下由一个 CTA 的指定 warp 发起；`cta_group::2` 下两个 CTA 都要参与申请/释放协议。 |
| before use | 申请返回后才可以把该 column range 用作 D 或 scale。 |
| before exit | CTA 退出前，所有 TMEM allocation 必须 dealloc；最终还要 relinquish allocation permit。 |

释放：

```ptx
// 伪代码
tcgen05.dealloc.cta_group::{1|2}.sync.aligned.b32 tmem_addr, ncols;
tcgen05.relinquish_alloc_permit.cta_group::{1|2}.sync.aligned;
```

工程建议：

- 一个 persistent worker 通常在进入主循环前申请一次 TMEM，然后在多个 tile 中复用。
- 不要在 K-loop 内反复 alloc/dealloc；TMEM allocation 是资源管理动作，不是普通寄存器赋值。
- 申请失败不是普通返回值风格，通常通过硬件 permit / waiting 处理；写手工 PTX 前先看 CUTLASS/CuTe 如何封装。

### 2.3 TMEM load / store

`tcgen05.ld` / `tcgen05.st` 在 TMEM 和寄存器之间搬数据，常用于：

- 读出 D accumulator 做 epilogue。
- 把需要进入 TMEM 的数据写入 TMEM。
- 调试或特殊数据路径。

关键点：

```text
warpgroup = 4 warps = 128 lanes
warp 0 负责 TMEM lanes 0..31
warp 1 负责 TMEM lanes 32..63
warp 2 负责 TMEM lanes 64..95
warp 3 负责 TMEM lanes 96..127
```

因此，如果要覆盖一个完整 TMEM column 的 128 lanes，通常需要整个 warpgroup 协作。只用一个 warp 访问，只覆盖它所属的 32 lanes。

同步：

| 操作 | 完成观察方式 |
|------|--------------|
| `tcgen05.ld` | `tcgen05.wait::ld` |
| `tcgen05.st` | `tcgen05.wait::st` |
| `tcgen05.mma` | `tcgen05.commit` + `mbarrier` wait |
| `tcgen05.cp` | `tcgen05.commit` + `mbarrier` wait |
| `tcgen05.shift` | `tcgen05.commit` + `mbarrier` wait |

一个常见 epilogue 形态：

```cpp
// 伪代码：等待 MMA 完成后读 TMEM D
wait_mbarrier(mma_done_barrier);

// 每个 warpgroup 从对应 TMEM lanes 读出 D fragment
tcgen05_ld_d_to_regs(...);
tcgen05_wait_ld();

// register epilogue: alpha/beta, bias, activation, cast, store
store_output_to_gmem(...);
```

### 2.4 TMEM 与 async proxy / memory fence

Blackwell 这里容易错的不是计算公式，而是 memory ordering。

需要区分几类路径：

| 路径 | 例子 | 需要注意 |
|------|------|----------|
| generic proxy | 普通 thread 写 shared memory。 | 后续 async 指令如果要读这些 SMEM 位置，需要 `fence.proxy.async`。 |
| async proxy | TMA、`tcgen05.cp`、`tcgen05.mma` 等异步路径。 | 完成通过 `mbarrier` 或对应 wait 指令观察。 |
| TMEM internal | `tcgen05.mma` 写 D，`tcgen05.ld` 读 D。 | 先 commit/wait，再 ld；跨线程同步前后用 `tcgen05.fence`。 |

经验规则：

```text
普通线程写 SMEM -> tcgen05/TMA 异步读
  需要 fence.proxy.async

tcgen05.mma/cp 写 TMEM -> 后续 thread/warpgroup 读
  需要 commit + mbarrier wait

tcgen05.ld/st 与后续普通指令交错
  需要 tcgen05.wait::ld / wait::st

跨 thread sync 前后传递 TMEM 可见性
  使用 tcgen05.fence::before_thread_sync / after_thread_sync
```

这也是为什么手写 `tcgen05` PTX 比手写 `mma.sync` 风险高：你不仅在写矩阵指令，还在写一套异步事务协议。

## 三、2-SM cooperative / `cta_group::2`

### 3.1 名字澄清

在 Blackwell GEMM 资料里会看到几种叫法：

| 名字 | 更准确的理解 |
|------|--------------|
| `2cta` | 两个 CTA 组成一个执行协作单元。 |
| `2sm` / 2-SM cooperative | 工程上通常表示两个 CTA 被调度到两个 SM 上协作处理一个 tile。 |
| `cta_group::2` | PTX `tcgen05` 指令中的参与范围 qualifier。 |

它们都不是：

- 2 个 thread。
- 2 个 warp。
- 2 个 CUDA core。
- 跨 GPU 或跨 NVLink 的通信。

它们依赖的是 **Thread Block Cluster**。cluster 内 CTA 被协同调度，并可用 cluster / DSMEM / mbarrier 等机制协作。具体 cluster 基础见 [[CUDA CTA 与 Thread Block Cluster 入门]]。

### 3.2 `cta_group::1` vs `cta_group::2`

| 维度 | `cta_group::1` | `cta_group::2` |
|------|----------------|----------------|
| 参与 CTA | 单 CTA。 | 两个 CTA，通常组成 pair。 |
| 资源范围 | 本 CTA 对应 SM 的 TMEM/SMEM 资源。 | 两个 CTA / 两个 SM 的协作资源。 |
| 典型 tile | 较小或普通 GEMM tile。 | 更大的 M tile / 更高吞吐 GEMM tile。 |
| 调度要求 | 普通 CTA 或 cluster 中单 CTA 路径。 | 需要 cluster 维度能容纳 pair，两个 CTA 都 active。 |
| 代价 | 调度较轻，适合小 tile。 | 同步更复杂，占用更多片上资源，小问题可能亏。 |

`cta_group::2` 的核心约束：

1. **两个 CTA 必须在同一个 cluster 内**。不能把两个任意 block 拼起来。
2. **peer CTA 必须 active**。一边先退出、另一边继续发 `cta_group::2` 指令会破坏协议。
3. **同一 warpgroup 的 `tcgen05` 指令 qualifier 要一致**。不要在同一个 warpgroup 流水线里混用 `cta_group::1` 和 `cta_group::2`。
4. **alloc/dealloc 是 CTA pair 级资源协议**。两个 CTA 对 TMEM 的申请、释放、fence、commit 要匹配。
5. **cluster rank 组织要稳定**。调度器通常按 `cluster_ctarank` 的某种 pair 关系把 CTA 绑定起来；写自定义 kernel 时不要假设任意 `blockIdx` 相邻就一定是 pair。

### 3.3 什么时候用 2-SM cooperative

适合：

- 大 GEMM，M/N/K 都足够大，单 CTA tile 不容易吃满 Tensor Core。
- MoE grouped GEMM 中某些 expert 的 token 数足够多，值得用更大 tile。
- 权重 tile 复用高，两个 CTA 协作能减少重复搬运或提高 Tensor Core issue 效率。
- CUTLASS / cuBLASLt 已经选择了 `2cta` kernel，profiling 显示 Tensor Core 利用率高。

不适合：

- decode 小 batch、小 M、tail wave 很多。
- expert token 数高度不均，很多 tile 不满。
- kernel 已经被 HBM/L2 或同步开销限制，而不是 Tensor Core issue 限制。
- cluster size 降低了并发，导致尾部调度更差。

经验判断：

```text
大 tile / 大 M / 计算密集
  -> 倾向尝试 2cta / 2sm

小 M / many small groups / tail-heavy
  -> 倾向 1cta、小 tile、persistent scheduler、CLC
```

### 3.4 和 DSMEM / TMA multicast 的关系

`cta_group::2` 不等于 DSMEM，但两者常一起出现。

- `cta_group::2`：Tensor Core 指令的参与范围。
- DSMEM：cluster 内跨 CTA 访问 shared memory 的地址空间。
- TMA multicast：一次 TMA copy 可以把同一个 GMEM tile multicast 到 cluster 内多个 CTA 的 SMEM。

一个高性能 GEMM 可能同时使用：

```text
TMA multicast B tile
  -> 两个 CTA 都拿到 B
  -> A tile 分别加载或按 layout 分片
  -> tcgen05.mma.cta_group::2
  -> D accumulator 分布在 TMEM
```

但你不应该把 `cta_group::2` 简化成“两个 CTA 共享 shared memory”。更准确说，它是一整套 cluster 级调度、SMEM/TMEM 地址、Tensor Core issue 和 mbarrier 协议的组合。

## 四、NVFP4 块缩放与 Blackwell block-scaled MMA

### 4.1 FP4 为什么必须有 block scale

NVFP4 的数据值本体是 FP4 E2M1。正数侧可表示集合是：

```text
0, 0.5, 1, 1.5, 2, 3, 4, 6
```

带符号后最大绝对值是 6。这个动态范围太小，不能直接承载 transformer 权重或激活。因此实际数值要写成：

```text
x_hat = q_e2m1 * s_block * s_global
```

其中：

| 项 | 常见格式 | 粒度 | 作用 |
|----|----------|------|------|
| `q_e2m1` | packed FP4 E2M1 | 每元素 4-bit，两个元素打包进 1 byte。 | 存储低精度值。 |
| `s_block` | FP8 E4M3 / UE4M3 scale | NVFP4 常见每 16 个元素一个。 | 覆盖局部动态范围。 |
| `s_global` | FP32 | per-tensor / per-expert / per-GEMM input。 | 让 block scale 自身落在 FP8 scale 可表示范围内。 |

[[NVFP4 量化与反量化]] 已经详细整理了量化公式。本文重点关注它进入 Blackwell kernel 后的约束。

### 4.2 硬件 block-scaled MMA 的数学形式

对 GEMM：

```text
D = A * B + D
```

block-scaled 视角更像：

```text
D += (A_q * SFA) * (B_q * SFB)
```

其中 `SFA`、`SFB` 是沿 K 维分块的 scale factor 矩阵。硬件不会理解你的 Python 量化对象，它只看：

- A/B 数据是否是合法 packed low-bit dtype。
- `SFA` / `SFB` 是否以指令要求的 scale layout 进入 TMEM。
- instruction descriptor 是否选择了正确的 block scale kind、vector size、transpose、sparsity。

`s_global` 通常不作为 `tcgen05.mma` 的 block scale 矩阵直接参与。工程上常见做法是：

- 权重 global scale 与激活 global scale 融进 GEMM `alpha`。
- 或在 epilogue 中乘回。
- 或在上游/下游 quant/dequant kernel 中处理。

因此写 kernel 时要把两层 scale 分清：

```text
硬件 block scale:
  每 16 或 32 个 K 元素的 SFA/SFB，参与 tcgen05 block-scaled MMA

软件/global scale:
  per tensor / per expert 的 FP32 标量，通常由 epilogue 或 GEMM alpha 处理
```

### 4.3 NVFP4 vs MXFP4

| 特性 | MXFP4 | NVFP4 |
|------|-------|-------|
| 数据本体 | FP4 E2M1 | FP4 E2M1 |
| block size | 常见 32 elements | 常见 16 elements |
| block scale | E8M0 / UE8M0 exponent-only | FP8 E4M3 / UE4M3 |
| global scale | 通常无第二层 global scale | 常见 FP32 global scale |
| 精度倾向 | 标准化，scale 更粗。 | scale 更细、更灵活，NVIDIA Blackwell / TE / TensorRT-LLM 生态重点支持。 |
| kernel 影响 | scale 元数据较少。 | scale 元数据更多，layout 和搬运更关键。 |

CUTLASS 文档中，NVFP4 类型常对应 `nv_float4_t`，scale factor 类型为 `float_ue4m3_t`，dense vector size 为 16。MXFP4 常对应 `mx_float4_t`，scale factor 类型为 `float_ue8m0_t`，dense vector size 为 32。

> 备注：scale 实际为正数，所以底层常用 `UE4M3` 这类 unsigned FP8 scale 类型；高层文档常简写成 FP8 E4M3 block scale。

### 4.4 CUTLASS 视角的关键布局约束

写 SM100 NVFP4 GEMM 时，最容易踩的是 layout。几个工程约束要先记住：

| 约束 | 含义 |
|------|------|
| 数据 layout | NVFP4 dense MMA 常见支持 `TN`：A row-major-like，B column-major-like。具体以 CUTLASS 表为准。 |
| alignment | A/B 低精度 packed 数据通常要求按 32 elements 对齐。 |
| scale vector size | NVFP4 dense = 16 elements per scale；sparse 场景通常粒度翻倍。 |
| scale layout | 不是简单 `[M, ceil(K/16)]` 连续数组；CUTLASS 使用针对 128 × 4 scale-factor basic block 的布局。 |
| scale staging | scale factor 要被搬到 TMEM，供 `tcgen05.mma` 使用。 |
| K tile | FP4 block-scaled MMA 常见 K tile 很大，例如 256，方便低 bit packing 与 scale 复用。 |

一个逻辑 layout 示意：

```text
A_q:   [M, K] packed FP4
SFA:   [M, K / 16] scale factors, but physical layout is swizzled/tiled

B_q:   [K, N] packed FP4
SFB:   [N, K / 16] or transposed physical scale layout

D/C:   [M, N] accumulator / output
```

但是物理上为了给 `tcgen05.mma` 喂数，scale 会按硬件友好的块重新排列。不要自己手写一个 naive scale array 后期待 cuTe atom 能直接吃。

### 4.5 NVFP4 GEMM 的 kernel 数据流

一个 Blackwell NVFP4 GEMM tile 可以拆成：

```text
for output tile (M_tile, N_tile):
  tmem_alloc(D accumulator columns)

  for k_tile in K:
    TMA load A_q tile -> SMEM
    TMA load B_q tile -> SMEM
    TMA load SFA/SFB tile -> SMEM

    wait TMA barriers
    fence.proxy.async if SMEM was produced by generic writes

    tcgen05.cp scale SMEM -> TMEM
    tcgen05.mma.kind::mxf4nvf4.block_scale
      D_tmem += (A_q * SFA_tmem) * (B_q * SFB_tmem)

    commit/wait or software-pipeline next stage

  tcgen05.ld D_tmem -> registers
  epilogue: global_scale, alpha/beta, bias, activation, cast
  store C

  tmem_dealloc
```

伪代码：

```cpp
// 这是组织结构，不是可直接编译的完整 CUDA。
// 真实实现建议看 CUTLASS/CuTe SM100 collective mainloop。
template <class Params>
__global__ __cluster_dims__(2, 1, 1)
void nvfp4_gemm_sm100_kernel(Params p) {
    // 1. persistent scheduler / CLC 可选
    TileCoord tile = initial_tile(blockIdx, cluster_ctarank());

    // 2. TMEM allocation：D accumulator + scale staging
    TmemHandle d_tmem = tmem_alloc(/*columns for D*/);
    TmemHandle sfa_tmem = tmem_alloc(/*columns for scale A*/);
    TmemHandle sfb_tmem = tmem_alloc(/*columns for scale B*/);

    while (tile.valid()) {
        init_accumulator_if_needed(d_tmem);

        for (int k0 = 0; k0 < p.K; k0 += K_TILE) {
            // TMA loads to SMEM; mbarrier tracks copy completion.
            tma_load_A_B_scales(tile, k0, p);
            wait_tma_stage_ready();

            // If scale/data in SMEM was produced through generic proxy, fence it.
            fence_proxy_async_if_needed();

            // Move scale factors into TMEM in the format tcgen05 expects.
            tcgen05_cp_scale_to_tmem(sfa_tmem, sfb_tmem);

            // Issue block-scaled MMA. cta_group::{1|2} must match kernel schedule.
            tcgen05_mma_nvfp4_block_scaled(
                d_tmem, A_smem_desc, B_smem_desc, sfa_tmem, sfb_tmem);

            commit_tcgen05_to_mbarrier();
            advance_pipeline_stage();
        }

        wait_mma_done();
        tcgen05_ld_accumulator_to_registers(d_tmem);
        epilogue_and_store(tile, p);

        tile = next_tile_or_clc_steal();
    }

    tmem_dealloc(d_tmem);
    tmem_dealloc(sfa_tmem);
    tmem_dealloc(sfb_tmem);
    tmem_relinquish_alloc_permit();
}
```

### 4.6 NVFP4 kernel 的常见坑

| 坑 | 表现 | 处理 |
|----|------|------|
| 把 global scale 当成 block scale | 数值整体差一个比例，或重复缩放。 | 确认 `s_block` 进入 `tcgen05`，`s_global` 在 alpha/epilogue 处理。 |
| scale layout 写错 | kernel 可以跑但结果错，或者性能走 fallback。 | 用 CUTLASS / TE / TensorRT-LLM 的 scale layout helper，不手搓 naive layout。 |
| A/B alignment 不满足 | 编译失败、运行错误或库不选 SM100 FP4 kernel。 | 按 32 elements 对齐 packed data，K/N/M tile 也按文档取合法值。 |
| 小 expert 强行 FP4 2cta | Tensor Core 利用率低，调度尾部严重。 | small-M 用小 tile / 1cta / grouped persistent。 |
| 动态 activation quant 独立成小 kernel | 省下的 GEMM 时间被 quant/dequant 和 HBM 往返吃掉。 | 尽量融合 quant、scale 计算、GEMM prologue 或使用库融合路径。 |
| 忽略 scale 带宽 | FP4 权重少了，但 scale 元数据、读写和 cache miss 变成瓶颈。 | profiling 时同时看 HBM/L2、scale load、TMA stage stall。 |

## 五、CLC：Cluster Launch Control

### 5.1 CLC 解决什么

传统 persistent kernel 的问题：

```text
launch 少量 persistent workers
workers 自己从 global queue 取 tile
优点：prologue/epilogue 少，可复用资源
缺点：work stealing 和 preemption 都靠软件 queue，调度器看不见剩余真实工作
```

普通 one-block-per-tile kernel 的问题：

```text
launch 很多 blocks
CUDA scheduler 自动分配 blocks
优点：负载均衡和 preemption 友好
缺点：每个 block 都要重复 prologue/epilogue，small tile 很亏
```

Blackwell CLC 折中：

```text
仍然 launch 足够多的 CTA / cluster
已经运行的 worker 可以 try_cancel 尚未启动的 ClcID
成功后直接接管该 tile
```

这相当于让硬件 scheduler 参与 work stealing：worker 不是从用户 global queue 弹任务，而是请求取消一个 pending CTA/cluster，并拿到它的 logical id。

![[GPU/Drawings/Blackwell CLC Work Stealing.svg]]

### 5.2 CLC 的基本语义

CLC 不是 device 端动态 launch。它只在一个已经 launch 的 grid 里工作：

| 语义 | 说明 |
|------|------|
| ClcID | 可理解为一个 launch 中尚未开始执行的 CTA / cluster 工作单元 id。 |
| try cancel | 当前 worker 请求取消某个尚未开始的 ClcID。 |
| success | 该 ClcID 不会再由原本的 CTA/cluster 启动；当前 worker 获得它并处理对应 tile。 |
| failure | 目标已经启动、不可取消或没有 pending work；当前 worker 不能再基于同一请求假设成功。 |
| async response | try_cancel 是异步的，响应写入 shared memory，通过 `mbarrier` 等待。 |

约束：

- 对同一个取消请求，必须等响应完成再 query。
- 如果请求失败，再用同一 request 继续发起新 cancel 是未定义行为。
- 在 cluster 中，建议只用一个 cluster thread 发起 CLC request，结果 multicast 给 cluster 内 CTA。
- CLC 请求和响应要配合 `mbarrier.arrive.expect_tx` / wait 语义。

### 5.3 CLC 伪代码

CUDA Programming Guide 的 sample 使用 `cuda::ptx` 里的 CLC wrapper。下面是简化后的结构：

```cpp
#include <cuda/ptx>

namespace ptx = cuda::ptx;

__global__ void persistent_gemm_with_clc(...) {
    __shared__ uint4 cancel_result;
    __shared__ uint64_t cancel_barrier;

    init_mbarrier(&cancel_barrier);

    TileId tile = tile_from_block_or_cluster_id();

    while (tile.valid()) {
        process_tile(tile);

        // 只有一个线程提交 CLC request。
        if (threadIdx.x == 0) {
            ptx::mbarrier_arrive_expect_tx(
                ptx::sem_release,
                ptx::scope_cta,
                ptx::space_shared,
                &cancel_barrier,
                sizeof(cancel_result));

            ptx::clusterlaunchcontrol_try_cancel(
                ptx::space_cluster,
                ptx::multicast_cluster,
                &cancel_result,
                next_candidate_clc_id(tile),
                &cancel_barrier);
        }

        wait_mbarrier(&cancel_barrier);

        bool ok = false;
        TileId stolen;
        if (threadIdx.x == 0) {
            ok = ptx::clusterlaunchcontrol_query_cancel_is_canceled(cancel_result);
            stolen = decode_clc_id(cancel_result);
        }

        ok = broadcast(ok);
        stolen = broadcast(stolen);
        tile = ok ? stolen : invalid_tile();
    }
}
```

这段伪代码只展示控制流。真实 GEMM 还要处理：

- cluster 内多 CTA 如何共享 stolen tile id。
- `2cta` / `2sm` worker 一次需要取消多少个 ClcID。
- pipeline 中还没完成的 TMA / MMA stage 如何 drain。
- CLC request 的 candidate 顺序如何避免抢同一个 tile。
- 如果 worker 持有 TMEM / SMEM 资源，退出前如何完整释放。

### 5.4 CLC 在 GEMM / MoE 中的用法

典型应用：grouped GEMM / MoE expert GEMM。

MoE 的问题是不同 expert 的 token 数差异很大：

```text
expert 0: 2048 tokens
expert 1:  128 tokens
expert 2: 4096 tokens
...
```

如果按静态 grid 映射：

- 大 expert 的 tile 多，尾部可能拖慢。
- 小 expert 的 tile 少，很多 worker 早退。
- 用 persistent global queue 又会引入软件调度开销和全局原子竞争。

CLC 路线：

```text
1. 仍然按所有 GEMM tiles launch 足够多的 CTA/cluster。
2. 每个 worker 先做自己原始 blockIdx 对应 tile。
3. 做完后 try_cancel 一个尚未启动的 ClcID。
4. 成功则继续处理 stolen tile。
5. 失败则说明没有合适 pending work，worker 退出。
```

对 Blackwell 2cta GEMM：

- 如果一个 logical work tile 需要两个 CTA，调度单位应按 cluster / CTA pair 设计。
- stolen ClcID 到 GEMM tile 的映射必须保证两个 CTA 对同一 tile 有一致理解。
- 一次取消一个 CTA 还是一个 cluster，取决于 launch 形态和 CLC API 使用方式；不要把单 CTA CLC 直接套到 2cta kernel。

### 5.5 CLC 适合与不适合

适合：

- tile 数远多于 SM 数，但每个 tile 工作量差异明显。
- grouped GEMM / MoE / variable-sequence attention。
- persistent kernel 的 prologue/epilogue 较重，希望 worker 复用状态。
- 需要比纯 persistent global queue 更好的调度器可见性。

不适合：

- 每个 tile 工作量几乎相同，普通 grid scheduling 已经很好。
- kernel 很短，CLC 的 mbarrier 和 query 开销占比高。
- block 间依赖复杂，取消 pending ClcID 后难以维护正确性。
- 资源占用过高，worker 常常无法并发，work stealing 没有发挥空间。

## 六、PDL / GDC：dependent kernel 的提前启动

### 6.1 PDL 和 GDC 的关系

名称容易混：

| 名称 | 层级 | 说明 |
|------|------|------|
| PDL | CUDA Runtime / Programming Guide | Programmatic Dependent Launch。same-stream dependent kernels 的提前启动机制。 |
| GDC | PTX / CUTLASS 语境 | Grid Dependency Control。CUTLASS 用 GDC 指令实现 PDL 相关路径。 |
| `griddepcontrol` | PTX 指令族 | 典型包括 launch dependents 和 wait。 |

在 CUDA C++ 层，你通常写：

```cpp
// primary kernel 内
cudaTriggerProgrammaticLaunchCompletion();

// secondary kernel 内，读取 primary 输出前
cudaGridDependencySynchronize();
```

在 CUTLASS / PTX 层，常看到：

```text
griddepcontrol.launch_dependents
griddepcontrol.wait
```

它们解决的是同一个问题：secondary kernel 不必等 primary 完全结束才开始执行，但必须在读取 primary 输出前等待。

![[GPU/Drawings/Blackwell PDL GDC 时序.svg]]

### 6.2 PDL 正确性模型

普通 same-stream 依赖：

```text
primary 完全结束
  -> primary 的 global stores 对后续 kernel 可见
  -> secondary 启动
```

PDL：

```text
primary 做完 secondary 需要等待的关键前置工作
  -> primary trigger launch completion
  -> secondary 提前 launch，先做 independent preamble
  -> secondary 在读 primary 输出前 wait
  -> wait 后才能读取 primary 输出
```

关键规则：

```text
secondary 提前 launch != primary 输出已经可见
```

必须等待：

```cpp
__global__ void secondary(...) {
    // 可以提前执行的部分：比如 descriptor 准备、权重预取、tile 坐标计算
    independent_preamble();

    // acquire：这里之后才能读 primary 写出的 global memory
    cudaGridDependencySynchronize();

    consume_primary_output();
}
```

### 6.3 Host launch 侧

CUDA Programming Guide 当前的基本 sample 是：

- primary kernel 内调用 `cudaTriggerProgrammaticLaunchCompletion()`。
- secondary kernel 用 extensible launch API 并配置 `cudaLaunchAttributeProgrammaticStreamSerialization`。
- primary 和 secondary 在同一个 stream 中保持依赖顺序。

示意：

```cpp
primary<<<grid, block, 0, stream>>>(...);

cudaLaunchAttribute attr{};
attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
attr.val.programmaticStreamSerializationAllowed = 1;

cudaLaunchConfig_t cfg{};
cfg.gridDim = secondary_grid;
cfg.blockDim = secondary_block;
cfg.dynamicSmemBytes = secondary_smem;
cfg.stream = stream;
cfg.attrs = &attr;
cfg.numAttrs = 1;

cudaLaunchKernelEx(&cfg, secondary, ...);
```

更复杂场景，例如 programmatic event、CUDA Graph 或库封装，primary 侧也可能通过 extensible launch API 携带额外属性。工程上以使用的 CUDA sample / library wrapper 为准。

### 6.4 CUTLASS 里的 GDC / PDL

CUTLASS 把 dependent kernel launch 支持编译进来需要开关：

```bash
cmake . -DCUTLASS_ENABLE_GDC_FOR_SM100=1
```

运行时还要启用：

```cpp
gemm.run(
    /* stream = */ stream,
    /* cuda_adapter = */ nullptr,
    /* launch_with_pdl = */ true);
```

这两个条件要同时满足：

| 条件 | 不满足时 |
|------|----------|
| 编译时启用 GDC 指令 | kernel 里没有相关 PTX 路径。 |
| 运行时 `launch_with_pdl=true` | 仍按普通 same-stream 依赖执行。 |

典型场景：

```text
RMSNorm / activation kernel:
  写出 normalized activation
  trigger launch completion
  继续做尾部工作或收尾 store

GEMM kernel:
  提前启动
  先预取 weights / 准备 descriptors / TMA prologue
  wait grid dependency
  再读取 activation 并进入 mainloop
```

### 6.5 PDL/GDC 适用判断

适合：

- secondary 有明显 independent preamble。
- primary 有足够 tail 可以重叠。
- 两个 kernel 在 same stream 中有真实依赖。
- secondary 能清晰划出“等待前不能读 primary 输出”的边界。
- `nsys` timeline 能看到 launch/prologue/tail 的端到端瓶颈。

不适合：

- secondary 一启动就必须读 primary 输出。
- primary 和 secondary 都吃满所有 SM，几乎没有并发空间。
- preamble 很短，PDL 同步成本超过收益。
- 通信时序或多进程同步非常敏感。
- 正确性依赖复杂，开发者无法可靠放置 wait。

## 七、把这些特性组合成 Blackwell GEMM 设计

### 7.1 设计一个 SM100 FP4 GEMM 前先回答的问题

| 问题 | 影响 |
|------|------|
| 目标 dtype 是 NVFP4、MXFP4、FP8 还是 BF16/FP16？ | 决定 `tcgen05.mma kind`、scale 类型、K tile、layout。 |
| A/B 是 TN 还是 NT？ | 决定是否有合法 block-scaled atom；NVFP4 常见只走 TN 快路径。 |
| M/N/K tile 多大？ | 决定 CTA tile、TMEM columns、SMEM staging、occupancy。 |
| 用 `cta_group::1` 还是 `cta_group::2`？ | 决定 cluster dims、TMA multicast、TMEM alloc/dealloc 协议。 |
| scale 存在什么 layout？ | 决定是否能直接用 CUTLASS atom，还是需要预处理 scale layout。 |
| activation quant 是否 fused？ | 决定是否需要额外 quant kernel，以及 PDL/GDC 是否能隐藏它。 |
| grouped/MoE 是否 load imbalance？ | 决定是否用 persistent scheduler / CLC。 |
| epilogue 做什么？ | 决定从 TMEM 读出多少 D、寄存器压力、store layout。 |

### 7.2 推荐的开发路径

对生产 kernel，不建议从手写 `tcgen05` PTX 起步。更稳的路径：

```text
1. 用 cuBLASLt / TensorRT-LLM / FlashInfer / CUTLASS 找到已有 SM100 kernel。
2. 用 Nsight Systems 确认 launch overlap、PDL、tail wave。
3. 用 Nsight Compute 看 Tensor Core、TMA、L2/HBM、SMEM stall。
4. 如果是 GEMM，先改 CUTLASS/CuTe tile、layout、dispatch policy。
5. 只有库抽象表达不了时，再手写局部 inline PTX。
```

手写 PTX 前至少准备：

- PTX ISA 对应版本。
- 编译目标，例如 `sm_100a` / 当前 Toolkit 支持的 Blackwell target。
- `tcgen05` atom 的合法 shape、dtype、scale、sparsity 组合。
- TMEM allocation plan：D、scale、临时 operand 各用多少 columns。
- SMEM layout：A/B/scale staging 是否匹配 descriptor。
- cluster dims：是否 `cta_group::2`，pair 如何组织。
- `mbarrier` plan：TMA barrier、MMA barrier、CLC response barrier 是否分开。
- epilogue plan：D 从 TMEM 读回寄存器后做什么。

### 7.3 一个完整 pipeline 心智模型

```text
Host:
  choose kernel variant
  maybe launch primary quant/norm kernel
  launch GEMM with PDL enabled if dependent

Kernel:
  cluster / CTA pair starts
  TMEM allocation
  persistent scheduler picks tile
  loop over K:
    TMA stage A/B/scale
    wait stage ready
    cp scale to TMEM
    issue tcgen05.mma
    commit to mbarrier
    overlap next TMA stage
  wait MMA completion
  load D from TMEM
  epilogue and store
  CLC try_cancel next tile
  repeat or exit
  dealloc TMEM
```

### 7.4 Profiling 时看什么

| 现象 | 可能原因 | 下一步 |
|------|----------|--------|
| Tensor Core 利用率低 | tile 太小、tail wave、scale/layout fallback、`cta_group::2` 不适配。 | 看 kernel 名称、CUTLASS dispatch policy、NCU tensor pipe 指标。 |
| HBM/L2 带宽高 | scale 元数据、A/B 重读、activation quant/dequant 分离。 | 检查 TMA load、scale layout、本地化复用。 |
| barrier stall 高 | TMA stage 不足、mbarrier phase 错、MMA wait 太早。 | 看 timeline 中 TMA/MMA overlap，调整 stages。 |
| occupancy 低 | TMEM/SMEM/register/cluster 占用过高。 | 比较 1cta vs 2cta，小 tile vs 大 tile。 |
| grouped GEMM 尾部慢 | expert token 不均，静态 tile 映射差。 | 尝试 persistent scheduler、CLC、分组排序。 |
| PDL 无收益 | secondary preamble 太短或资源无法并发。 | nsys 验证 overlap；把 wait 前工作做实或关闭 PDL。 |

## 八、常见误解

| 误解 | 更正 |
|------|------|
| `2cta` 就是两个线程或两个 warp | 它通常表示两个 CTA / 两个 SM 级别的协作单元。 |
| TMEM 是更大的 shared memory | 不是。TMEM 是 Tensor Core 专用存储，按 lane/column 组织，访问规则不同。 |
| FP4 权重小，所以一定快 | 不一定。scale layout、scale 带宽、quant/dequant、tile tail 都可能吃掉收益。 |
| NVFP4 block scale 就是一个 FP16 scale 数组 | 不是。NVFP4 常见 block scale 是 FP8/UE4M3，每 16 元素一个，并有 global scale。 |
| PDL 让 secondary 可以直接读 primary 输出 | 错。secondary 提前 launch 后必须 wait 才能读依赖数据。 |
| CLC 会动态创建新 block | 错。CLC 取消尚未启动的 pending ClcID，并把它交给当前 worker。 |
| Blackwell cluster 上限变无限大 | 不是。Thread Block Cluster 仍有 portable 和 non-portable 上限，且受 GPC 局部性约束。 |

## 九、记忆卡片

```text
tcgen05:
  Blackwell Tensor Core PTX 指令族
  D accumulator in TMEM
  completion = tcgen05.commit + mbarrier

TMEM:
  per SM 256KB
  128 lanes x 512 columns x 32-bit
  alloc/dealloc columns
  tcgen05.ld/st need wait::ld/st

cta_group::2:
  2 CTA / 2-SM cooperative path
  requires cluster-level coordination
  good for large GEMM, risky for small/tail-heavy cases

NVFP4:
  q_e2m1 * fp8/ue4m3 block scale * fp32 global scale
  block size usually 16
  scale layout is hardware/kernel contract

CLC:
  worker cancels pending ClcID
  hardware-assisted work stealing
  useful for persistent/grouped/MoE load balancing

PDL/GDC:
  primary trigger -> secondary early preamble -> secondary wait -> read dependency
  CUTLASS: enable GDC at build, launch_with_pdl at runtime
```

## 十、开放问题

- 不同 CUDA 13.x 小版本对 `sm_100a`、`sm_103`、`sm_120a` 的 low-level feature exposure 可能继续变化，写笔记和代码时应固定 Toolkit 版本。
- CUTLASS/CuTe 对 NVFP4 scale layout 的 helper 是生产上最安全的入口；如果未来 Triton/CUDA C++ 暴露更高级抽象，可再补充。
- Blackwell Ultra / B300 的 dense FP4 和 attention 加速宣传点在 kernel 层如何映射到具体 SM100/SM103 指令路径，需要用实际 NCU 与库 kernel 名称进一步核对。
- CLC 与 `cta_group::2`、persistent grouped GEMM、CUDA Graph 组合时的最优策略仍强依赖 workload 分布，需要结合 MoE token histogram 实测。
