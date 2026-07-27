---
aliases:
  - Python 进程选择性 NCU Profiling
  - 只分析指定 CUDA Kernel
  - 复杂 Python 程序 NCU 过滤手册
updated: 2026-05-30
tags:
  - gpu-computing
  - cuda-programming
  - performance-profiling
---
# 复杂 Python 进程选择性 NCU Profiling 操作手册

复杂 Python 进程通常会调用很多 CUDA kernel：PyTorch eager op、cuBLAS/cuDNN、Triton、`torch.compile` 生成 kernel、自定义 CUDA extension、NCCL/NVSHMEM 通信 kernel 等。直接 `ncu --set full python train.py` 往往会非常慢，而且报告里全是无关 kernel。

这篇手册解决一个具体问题：**我只想分析其中几个目标 kernel，应该怎么操作？**

相关笔记：

- [[Nsight Compute NCU 分析方法与优化思路]]
- [[CUDA Kernel 性能瓶颈定位流程]]
- [[NCU_ANALYSIS]]
- [[CUDA JIT、AOT 与 Kernel 选择机制]]
- [[nvjet_kernel_naming]]

## 总原则

选择性 profiling 有四种常用过滤手段，按推荐程度排序：

| 方法 | 适合场景 | 优点 | 局限 |
|------|----------|------|------|
| NVTX range 过滤 | 你能改 Python 代码，知道目标代码段在哪 | 最稳，适合复杂 Python/LLM 服务 | 需要在代码里加标记 |
| kernel name 过滤 | 已经知道 kernel 名称或名称关键词 | 命令简单，适合自定义 kernel / cuBLASLt / Triton | 框架生成 kernel 名可能长且不稳定 |
| `cudaProfilerStart/Stop` | 想用代码显式打开/关闭 profiling | 可把 profiling 限在一段代码 | 仍需配合 kernel name / launch count |
| Python call stack 过滤 | 不想加 NVTX，但知道调用函数 | 可按 Python 函数筛选 | 开销更高，CPython 版本和调用栈可见性有限 |

实践上最推荐：

```text
先用 nsys 或 NCU basic 找 kernel 名字
-> 在 Python 目标代码段加 NVTX range
-> ncu 用 --nvtx-include 限定代码段
-> 再用 --kernel-name / --launch-count 限定具体 kernel
-> 只收少量 section 迭代，最后必要时 --set full
```

## 第一步：先找到目标 kernel 名字

如果你还不知道目标 kernel 叫什么，先不要上 `--set full`。先用轻量方式拿 kernel 列表。

### 方法 A：先用 Nsight Systems 看时间线

适合端到端复杂进程：

```bash
nsys profile \
  -t cuda,nvtx,osrt \
  -o nsys_trace \
  python bench.py --args ...
```

然后看 CUDA kernel 汇总：

```bash
nsys stats --report cuda_gpu_kern_sum nsys_trace.nsys-rep
```

这样先确认：

- 哪些 kernel 耗时最多。
- 目标 kernel 大概出现在 prefill、decode、attention、GEMM、norm 还是通信阶段。
- kernel 名里有哪些稳定关键词，例如 `cutlass`、`triton`、`flash`、`ampere`、`wgmma`、`nvjet`。

### 方法 B：用 NCU basic 粗扫少量 launch

适合输入可缩小、运行时间可控的脚本：

```bash
ncu --set basic \
  --launch-count 30 \
  --page raw \
  -f -o ncu_probe \
  python bench.py --args ...
```

如果 kernel 太多，先缩短输入或只跑一个 step。目标是拿名字，不是做结论。

导出 raw 指标后也可以 grep：

```bash
ncu --import ncu_probe.ncu-rep --page raw --csv > ncu_probe.csv
```

## 方法一：按 kernel name 过滤

这是最直接的方法。

### 精确匹配一个 kernel

```bash
ncu --set full \
  --kernel-name my_kernel \
  --launch-count 1 \
  -f -o ncu_my_kernel \
  python bench.py --args ...
```

`--kernel-name my_kernel` 默认按 `function` 名匹配，也就是不带模板参数和函数参数的函数名。

### 用正则匹配一组 kernel

```bash
ncu --set full \
  --kernel-name regex:".*(flash|cutlass|triton).*" \
  --launch-count 3 \
  -f -o ncu_selected \
  python bench.py --args ...
```

注意：

- `regex:` 是 NCU 的正则前缀。
- 在 bash/zsh 里建议给正则加引号。
- `--launch-count 3` 表示只 profile 前 3 个**匹配过滤条件**的 kernel launch。

### 模板名很长时切换匹配口径

框架 kernel 常常有复杂 demangled 名称。可以用：

```bash
ncu --set full \
  --kernel-name-base demangled \
  --kernel-name regex:".*Attention.*" \
  --launch-count 1 \
  -f -o ncu_attention \
  python bench.py --args ...
```

常见口径：

| `--kernel-name-base` | 含义 | 适合 |
|----------------------|------|------|
| `function` | 函数名，不含模板参数和参数列表，默认值 | 自定义 kernel、名字较干净 |
| `demangled` | C++ demangle 后的完整名字 | CUTLASS、模板 kernel、想看类型参数 |
| `mangled` | C++ ABI mangled 名 | 极少使用，通常只在名字冲突时用 |

如果 demangled 名字太长，可以用 NCU 的 kernel renaming 功能，但初学阶段通常先用 `regex` 即可。

### `launch-skip` 和 `launch-count` 的区别

这两个参数非常重要：

```bash
ncu --set full \
  --kernel-name regex:".*decode_attention.*" \
  --launch-skip 10 \
  --launch-count 1 \
  -f -o ncu_decode_attention_11th \
  python bench.py
```

含义：

- `--launch-skip 10`：跳过前 10 个**匹配 kernel filter** 的 launch。
- `--launch-count 1`：采集接下来 1 个**匹配 kernel filter** 的 launch。

如果你想跳过程序启动阶段的所有 CUDA launch，而不管名字是否匹配，用：

```bash
--launch-skip-before-match 200
```

区别：

| 参数 | 计数对象 | 典型用途 |
|------|----------|----------|
| `--launch-skip` | 只数匹配 kernel filter 的 launch | 跳过前几个 warmup decode attention。 |
| `--launch-skip-before-match` | 数所有 kernel launch | 跳过模型初始化、JIT、allocator warmup。 |
| `--launch-count` | 只数匹配 kernel filter 的 launch | 限制报告大小和 profiling 时间。 |

采够后想让程序直接退出，可加：

```bash
--kill yes
```

例如：

```bash
ncu --set full \
  --kernel-name regex:".*decode_attention.*" \
  --launch-skip 5 \
  --launch-count 1 \
  --kill yes \
  -f -o ncu_one_decode \
  python bench.py
```

## 方法二：用 NVTX range 标记目标代码段

这是复杂 Python 进程里最推荐的方法。因为 kernel 名可能变化，但你的业务代码段通常稳定。

### 在 Python 里加 NVTX 标记

PyTorch 自带 NVTX API：

```python
import torch
from contextlib import contextmanager

@contextmanager
def ncu_range(name: str):
    # 先清空之前异步提交的 CUDA 工作，避免前一段的 kernel 混进来。
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        # 等目标段提交/完成后再退出 range，避免多 stream 场景下边界含糊。
        try:
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()

def run_decode_step(model, inputs):
    with ncu_range("decode_step"):
        return model(**inputs)
```

也可以用 context manager 简写：

```python
with torch.cuda.nvtx.range("decode_step"):
    output = model(**inputs)
```

但复杂程序里建议显式 `synchronize()`，这样边界更干净。

### NCU 只采这个 NVTX range 内的 kernel

```bash
ncu --set full \
  --nvtx \
  --nvtx-include "decode_step/" \
  --launch-count 1 \
  -f -o ncu_decode_step \
  python bench.py
```

注意这里的 `decode_step/` 结尾有 `/`。这是因为 PyTorch 的 `range_push/range_pop` 是 NVTX push/pop range，NCU 用 `/` 表示 push/pop range。

### NVTX range + kernel name 双重过滤

最常用、最稳的做法：

```bash
ncu --set full \
  --nvtx \
  --nvtx-include "decode_step/" \
  --kernel-name regex:".*(attention|flash|paged).*" \
  --launch-count 1 \
  -f -o ncu_decode_attention \
  python bench.py
```

这表示：

```text
只考虑 decode_step 这个 Python 范围内的 kernel
再从里面挑名字包含 attention/flash/paged 的 kernel
最后只 profile 第 1 个匹配项
```

### 多层 NVTX range

如果代码是：

```python
with ncu_range("decode"):
    with ncu_range("attention"):
        out = attention(q, k, v)
```

可以筛选嵌套范围：

```bash
ncu --nvtx \
  --nvtx-include "decode/*/attention" \
  --kernel-name regex:".*flash.*" \
  --launch-count 1 \
  -f -o ncu_decode_flash \
  python bench.py
```

常用写法：

| 写法 | 含义 |
|------|------|
| `"A/"` | push/pop range `A` 内的 kernel。 |
| `"A/*/B"` | `A` 和 `B` 之间允许有零个或多个中间 range。 |
| `"A]"` | `A` 在栈顶。 |
| `"[A"` | `A` 在栈底。 |
| `"Domain@A/"` | 指定 NVTX domain。PyTorch 默认通常不用 domain。 |
| `"regex:.*decode.*/"` | 用正则匹配 range 名。 |

如果你的 range 在一个线程 push、kernel 在另一个线程 launch，尝试：

```bash
--nvtx-push-pop-scope process
```

默认是 `thread` 作用域。

### 只采第 N 次 NVTX range

decode 循环里同名 range 会出现很多次。你可以用 `--range-filter` 挑第 N 次。

例如只采第 8 次 `decode_step` range 内第一个匹配 kernel：

```bash
ncu --set full \
  --nvtx \
  --nvtx-include "decode_step/" \
  --range-filter ::8 \
  --kernel-name regex:".*attention.*" \
  --launch-count 1 \
  -f -o ncu_decode_step_8_attention \
  python bench.py
```

这里 `--range-filter` 的三个槽位是：

```text
[是否在每个 start/stop range 内重新编号]:[start/stop range 实例]:[NVTX range 实例]
```

所以 `::8` 表示不筛 start/stop range，只筛第 8 个匹配的 NVTX range。

## 方法三：用 `cudaProfilerStart/Stop` 控制 profiling 开关

如果你想在代码里显式控制 profiling 开始/结束，可以用 CUDA Runtime 的 profiler API。PyTorch 可以通过 `torch.cuda.cudart()` 调用。

### Python 代码

```python
import torch
from torch.cuda import cudart, check_error

def profiled_region(fn, *args, **kwargs):
    torch.cuda.synchronize()
    check_error(cudart().cudaProfilerStart())
    try:
        out = fn(*args, **kwargs)
        return out
    finally:
        try:
            torch.cuda.synchronize()
        finally:
            check_error(cudart().cudaProfilerStop())
```

使用：

```python
for i in range(num_steps):
    if i == 8:
        out = profiled_region(run_decode_step, model, inputs)
    else:
        out = run_decode_step(model, inputs)
```

### NCU 命令

关键是加 `--profile-from-start off`：

```bash
ncu --set full \
  --profile-from-start off \
  --kernel-name regex:".*attention.*" \
  --launch-count 1 \
  -f -o ncu_profile_api_attention \
  python bench.py
```

如果不加 `--profile-from-start off`，NCU 默认从程序开始就 profile，`cudaProfilerStart()` 的意义会弱很多。

什么时候用这个方法：

- 你能改代码，但不想依赖 NVTX range 名。
- 想跳过大量初始化、warmup、JIT，只在某一步打开 profiling。
- 想和 `--range-filter` 或 `--replay-mode range` 结合。

## 方法四：按 Python call stack 过滤

NCU 支持 Python 调用栈过滤。它适合“不方便插 NVTX，但知道哪个 Python 函数触发 kernel”的场景。

示例：只 profile 从 `decode.py` 里的 `run_attention` 函数发出的 kernel：

```bash
ncu --set full \
  --call-stack-type python \
  --python-include "decode.py@run_attention" \
  --kernel-name regex:".*attention.*" \
  --launch-count 1 \
  -f -o ncu_py_stack_attention \
  python bench.py
```

也可以只按函数名：

```bash
ncu --call-stack-type python \
  --python-include "run_attention" \
  python bench.py
```

注意：

- Python call stack collection 需要 CPython 3.9+。
- 采集 call stack 有额外开销。
- 对框架内部 C++/CUDA library 发起的 kernel，Python 栈可能只告诉你上层 op，不一定能区分具体库 kernel。
- 对初学者，优先用 NVTX；call stack 过滤作为补充。

## 复杂 Python / PyTorch 常见场景

### 场景 1：只看某个 PyTorch op 触发的 kernel

Python：

```python
with ncu_range("target_matmul"):
    y = x @ w
```

命令：

```bash
ncu --set full \
  --nvtx \
  --nvtx-include "target_matmul/" \
  --kernel-name regex:".*(gemm|cutlass|cublas|nvjet).*" \
  --launch-count 1 \
  -f -o ncu_matmul \
  python bench.py
```

### 场景 2：只看 decode 第 N 步的 attention kernel

Python：

```python
for step in range(max_new_tokens):
    with ncu_range("decode_step"):
        logits = model.decode_one_token(...)
```

命令：

```bash
ncu --set full \
  --nvtx \
  --nvtx-include "decode_step/" \
  --range-filter ::16 \
  --kernel-name regex:".*(attention|flash|paged).*" \
  --launch-count 1 \
  -f -o ncu_decode16_attention \
  python bench.py
```

### 场景 3：Triton kernel 名字很多，只想抓某个 Python 函数内的 Triton

Python：

```python
with ncu_range("rmsnorm_kernel"):
    y = rmsnorm(x, weight)
```

命令：

```bash
ncu --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --nvtx \
  --nvtx-include "rmsnorm_kernel/" \
  --kernel-name regex:".*triton.*" \
  --launch-count 1 \
  -f -o ncu_rmsnorm_triton \
  python bench.py
```

### 场景 4：`torch.compile` / CUDA Graph 导致 kernel 名或 launch 方式变化

建议：

1. 先用 NVTX 标记目标 Python 区域。
2. 先用 `--set basic` 粗扫这个 range 内有哪些 kernel。
3. 如果是 CUDA Graph replay，注意 NCU 默认可以按 graph node profile；必要时显式确认：

```bash
--graph-profiling node
```

如果你想把整个 graph 当成一个 workload 看，可以改成：

```bash
--graph-profiling graph
```

初学阶段建议先用 `node`，这样仍能看到单个 kernel 节点。

### 场景 5：Python 进程会拉起子进程

例如 `torchrun`、multiprocessing、worker 进程。显式加：

```bash
--target-processes all
```

示例：

```bash
ncu --set full \
  --target-processes all \
  --target-processes-filter regex:".*python.*" \
  --nvtx \
  --nvtx-include "decode_step/" \
  --kernel-name regex:".*attention.*" \
  --launch-count 1 \
  -f -o ncu_torchrun_decode \
  torchrun --nproc_per_node=1 bench.py
```

多 GPU / DDP 时，先只抓一个 rank，减少噪声：

```bash
CUDA_VISIBLE_DEVICES=0 ncu --devices 0 \
  --target-processes all \
  --nvtx \
  --nvtx-include "decode_step/" \
  --kernel-name regex:".*attention.*" \
  --launch-count 1 \
  -f -o ncu_rank0_decode \
  torchrun --nproc_per_node=1 bench.py
```

如果必须多 rank 同时 profile，报告会变复杂，且 profiling 可能改变进程间时序。通信相关问题通常先用 `nsys`，再对局部 kernel 用 NCU。

## 推荐命令模板

### 模板 1：最常用，NVTX + kernel name

```bash
ncu --set full \
  --nvtx \
  --nvtx-include "<range_name>/" \
  --kernel-name regex:"<kernel_keyword>" \
  --launch-count 1 \
  -f -o ncu_target \
  python bench.py
```

### 模板 2：迭代优化时只收关键 section

```bash
ncu \
  --section SpeedOfLight \
  --section LaunchStats \
  --section Occupancy \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --nvtx \
  --nvtx-include "<range_name>/" \
  --kernel-name regex:"<kernel_keyword>" \
  --launch-count 1 \
  -f -o ncu_iter \
  python bench.py
```

### 模板 3：只靠 kernel name

```bash
ncu --set full \
  --kernel-name-base demangled \
  --kernel-name regex:".*(cutlass|flash|triton).*" \
  --launch-skip 5 \
  --launch-count 1 \
  -f -o ncu_by_name \
  python bench.py
```

### 模板 4：代码里用 profiler API 开关

```bash
ncu --set full \
  --profile-from-start off \
  --kernel-name regex:".*target.*" \
  --launch-count 1 \
  -f -o ncu_start_stop \
  python bench.py
```

## 实战排错

### 没采到任何 kernel

检查：

- `--kernel-name` 正则是否过窄。先去掉 kernel filter，只保留 NVTX。
- push/pop range 是否写成了 `"range/"`，不是 `"range"`。
- Python 是否真的执行到了目标代码段。
- CUDA work 是否在另一个子进程里，必要时加 `--target-processes all`。
- 如果用了 `cudaProfilerStart/Stop`，是否加了 `--profile-from-start off`。

### 采到了太多 kernel

处理：

- 加 `--kernel-name regex:"..."`。
- 加 `--launch-count 1`。
- 加 `--range-filter ::N` 只抓第 N 次 range。
- 把 NVTX range 缩小到更靠近目标 op。

### kernel 名字每次都变

常见于 Triton、`torch.compile`、模板库。处理：

- 优先用 NVTX range。
- 用宽松关键词，例如 `triton`、`cutlass`、`flash`、`gemm`。
- 先用 `--set basic` 在 range 内探测，再写更准的正则。

### `--launch-skip` 与 `--kernel-name` 组合的常见坑

这两个参数搭配时容易踩三个坑：

1. **`--launch-skip` 是 0-based 还是基于"匹配后"的编号**：`--launch-skip N` 指的是**跳过前 N 个匹配 `--kernel-name` filter 的 launch**，并不是跳过前 N 个所有 kernel launch。如果你看到第 11 次 attention launch 而以为它对应 `--launch-skip 11`，实际应该用 `--launch-skip 10`（跳过前 10 个，采集第 11 个）。
2. **`--kernel-name` 改了，编号会变**：如果先用 `regex:".*attention.*"` 数到目标在第 8 次，再改为 `regex:".*flash_attention.*"`，编号要重新数——因为新正则匹配的 launch 集合不同。
3. **加了 `--profile-from-start off` 后 `--launch-skip` 仍从 0 开始**：`--profile-from-start off` 只影响何时**开始记录**，`--launch-skip` 的计数是在 filter 命中之后；二者不会互相抵消。常见误区是以为 `cudaProfilerStart` 之前的 launch 不计入 skip。

排错套路：

```bash
# 第一步：在 range / NVTX 圈定后，用 --set basic + 大 --launch-count 列出所有匹配 launch
ncu --set basic --launch-count 100 --page raw \
    --kernel-name regex:".*attention.*" \
    -o list.ncu-rep python ...

# 看 raw page 中编号，确认目标 launch 是匹配后的第几个，再回填到 --launch-skip
```

### profiling 太慢

处理：

- 从 `--set basic` 或少量 `--section` 开始。
- `--launch-count 1`。
- 减小输入规模，但保持目标 kernel shape 接近真实场景。
- 先用 NCU 做单 kernel，端到端用 `nsys`。

### 指标看起来和真实运行不一致

NCU 可能会 replay kernel，多次采集不同 metric；默认还可能控制 cache 和 clock。结论应该关注瓶颈方向和相对变化，不要把 NCU 下的 wall time 当线上真实延迟。

如果你要尽量接近真实 cache 行为，可以研究：

```bash
--cache-control none
```

但这会降低跨 replay 的稳定性。初学阶段先用默认值。

## 小抄

| 目标 | 推荐命令 |
|------|----------|
| 找 kernel 名字 | `ncu --set basic --launch-count 30 --page raw ...` |
| 只抓名字匹配的 kernel | `--kernel-name regex:".*xxx.*"` |
| 模板 kernel 用完整 demangled 名 | `--kernel-name-base demangled` |
| 跳过前 N 个匹配 kernel | `--launch-skip N` |
| 跳过前 N 个所有 kernel launch | `--launch-skip-before-match N` |
| 只抓 1 个匹配 kernel | `--launch-count 1` |
| 采完就结束程序 | `--kill yes` |
| 按 Python 代码段过滤 | `--nvtx --nvtx-include "range/"` |
| 只抓第 N 次 NVTX range | `--range-filter ::N` |
| 代码里显式开关 profiling | `--profile-from-start off` + `cudaProfilerStart/Stop` |
| 子进程也 profile | `--target-processes all` |
| 只 profile 某些进程名 | `--target-processes-filter regex:".*python.*"` |
| Python 调用栈过滤 | `--call-stack-type python --python-include "file.py@func"` |

## 参考

- [NVIDIA Nsight Compute CLI: Kernel filtering, NVTX filtering, launch count/skip](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)
- [CUDA Runtime API: cudaProfilerStart / cudaProfilerStop](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html)
- [PyTorch CUDA NVTX API](https://docs.pytorch.org/docs/main/cuda.html#nvidia-tools-extension-nvtx)
- [PyTorch torch.cuda.cudart](https://docs.pytorch.org/docs/stable/generated/torch.cuda.cudart.html)
