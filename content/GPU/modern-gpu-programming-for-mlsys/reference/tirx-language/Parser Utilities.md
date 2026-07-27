---
title: "Parser utilities"
content_type: reference
maturity: stable
updated: 2026-07-21
lang: en
publish: true
aliases:
  - "Parser utilities"
source: https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/8950d661e8499008546e3520c667c1cacec9af21/tirx_guide/language_reference/cuda/parser_utils.rst
source_commit: 8950d661e8499008546e3520c667c1cacec9af21
imported: 2026-07-21
tags:
  - gpu-computing
  - gpu-kernel
  - tutorial-note
  - tirx
  - reference-note
---
# Parser utilities

A few helpers act at **parse time** (when TVMScript is turned into TIRx), letting you inline Python-computed values, factor out reusable fragments, and bundle parser-side state.

## `T.meta_var` — inline a Python value

`T.meta_var(x)` tells the parser to treat `x` — a value computed in **Python** — as a compile-time *meta* value and inline it directly into the IR, rather than parse it as a script variable. It avoids a throwaway local, and it drives metaprogramming: a plain Python `for` over a meta value unrolls in the parser.

```python
n = T.meta_var(4)              # n is a Python int, inlined
for j in range(n):            # unrolled at parse time
    acc[0] = acc[0] + A[tx, j]

```
## `@T.inline` — inline functions

`@T.inline` defines a function whose body is **inlined at each call site** during parsing — no call appears in the generated code. It follows Python's lexical (LEGB) scoping with late binding, so a parameter shadows an enclosing variable:

```python
@T.inline
def add_into(acc, x):
    acc[0] = acc[0] + x

add_into(acc, A[tx, j])       # inlined -> acc[0] = acc[0] + A[tx, j]

```
## `@T.meta_class` — parser-side state objects

`@T.meta_class` marks a plain Python class whose **instances are parser meta values**: their fields can hold buffers and scalars, so you can bundle related allocations and state into one object and use it in the kernel body.

```python
@T.meta_class
class State:
    def __init__(self, smem):
        self.acc = T.alloc_local([1], "float32")
        self.buf = T.decl_buffer([64], "float16", smem, scope="shared.dyn")

s = State(smem.data)
s.acc[0] = T.float32(0.0)     # use its fields like ordinary buffers
# ... s.buf[i] ...

```
This is handy for grouping a kernel's pipeline state (barriers, accumulators, scratch views) instead of threading many separate locals through the body.

## `T.constexpr`

`T.constexpr` marks a compile-time kernel parameter, baked in by `@T.jit`'s `.specialize(...)`. See [[GPU/modern-gpu-programming-for-mlsys/part-2-tirx/10 Introduction to TIRx|Introduction to TIRx]] for the details.

---

[[GPU/modern-gpu-programming-for-mlsys/reference/TIRx Language Reference|Previous]] · [[GPU/modern-gpu-programming-for-mlsys/README|Book index]] · [[GPU/modern-gpu-programming-for-mlsys/reference/tirx-language/Data Types and Expressions|Next]]
