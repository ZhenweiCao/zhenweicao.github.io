---
aliases:
  - "Source and Import Metadata"
source: https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/blob/8950d661e8499008546e3520c667c1cacec9af21/README.md
source_commit: 8950d661e8499008546e3520c667c1cacec9af21
imported: 2026-07-21
tags:
  - gpu-computing
  - gpu-kernel
  - tutorial-note
  - source-note
  - reference-note
---
# Source and Import Metadata

## Provenance

- Upstream repository: [mlc-ai/modern-gpu-programming-for-mlsys](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys)
- Local source: `/Users/meimei/repo/Github/modern-gpu-programming-for-mlsys`
- Imported commit: `8950d661e8499008546e3520c667c1cacec9af21`
- Imported on: `2026-07-21`
- Scope: English edition only; the `zh/` tree and Chinese-only assets were intentionally excluded.

## Obsidian adaptations

- Sphinx/MyST `toctree` blocks became an explicit reading index.
- Sphinx cross-references became Obsidian wikilinks, including section-level links.
- MyST admonitions became Obsidian callouts.
- RST reference pages became Markdown, including code blocks, notes, and list tables.
- Images were copied locally and embedded from `assets/images/`.
- English interactive pages were copied to `assets/interactive/`; shared JavaScript and CSS remain one directory above them so their original relative URLs continue to work.
- Raw iframe directives became Embed HTML wikilink embeds with explicit width and height.

## Imported assets

- Markdown/RST source documents: 26
- Local images: 23
- Interactive HTML pages: 23

## Source mapping

- `index.md` → `README.md`
- `chapter_background/index.md` → `part-1-gpu/01 GPU Execution Model.md`
- `chapter_performance/index.md` → `part-1-gpu/02 What Makes a Kernel Fast.md`
- `chapter_data_layout/index.md` → `part-1-gpu/03 Data Layout and Its Notation.md`
- `chapter_layout_generations/index.md` → `part-1-gpu/04 The Evolution of Tensor Core Data Layouts.md`
- `chapter_tma/index.md` → `part-1-gpu/05 Async Data Movement - TMA.md`
- `chapter_tensor_cores/index.md` → `part-1-gpu/06 Blackwell Tensor Core - tcgen05 MMA.md`
- `chapter_tmem/index.md` → `part-1-gpu/07 Tensor Memory - TMEM.md`
- `chapter_async_barriers/index.md` → `part-1-gpu/08 Async Coordination - mbarrier.md`
- `chapter_clc/index.md` → `part-1-gpu/09 Advanced Scheduling - Cluster Launch Control.md`
- `chapter_intro_tirx/index.md` → `part-2-tirx/10 Introduction to TIRx.md`
- `chapter_tirx_layout_api/index.md` → `part-2-tirx/11 TIRx Layout API.md`
- `chapter_gemm_basics/index.md` → `part-3-gemm/12 Building a Tiled GEMM.md`
- `chapter_gemm_async/index.md` → `part-3-gemm/13 Pipelining GEMM with TMA.md`
- `chapter_gemm_advanced/index.md` → `part-3-gemm/14 Scaling GEMM with Warp Specialization and Clusters.md`
- `chapter_flash_attention/index.md` → `part-4-flash-attention/15 Flash Attention 4.md`
- `appendix/index.md` → `reference/Reference.md`
- `appendix/debugging_warp_specialized.md` → `reference/Debugging Warp-Specialized Kernels.md`
- `tirx_guide/language_reference/index.rst` → `reference/TIRx Language Reference.md`
- `tirx_guide/language_reference/cuda/parser_utils.rst` → `reference/tirx-language/Parser Utilities.md`
- `tirx_guide/language_reference/cuda/data_types.rst` → `reference/tirx-language/Data Types and Expressions.md`
- `tirx_guide/language_reference/cuda/buffers.rst` → `reference/tirx-language/Buffers and Memory.md`
- `tirx_guide/language_reference/cuda/control_flow.rst` → `reference/tirx-language/Control Flow.md`
- `tirx_guide/language_reference/cuda/threads_sync.rst` → `reference/tirx-language/CUDA and PTX Intrinsics.md`
- `tirx_guide/arch/index.rst` → `reference/Compiler Internals.md`
- `tirx_guide/arch/lowering_pipeline.rst` → `reference/TIRx Lowering Pipeline.md`
