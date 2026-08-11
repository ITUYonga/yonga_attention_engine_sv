# sv-yae: A Streaming BF16 Self-Attention Accelerator on FPGA

![architecture](images/architecture.png)

**Team:** İTÜ Yonga — Yonga Attention Engine (YAE) | Can Ertürk, Belinay, Hasan, Taha
**Date:** August 2026 | **Platform:** Xilinx Vivado / SystemVerilog | **Competition:** FPT26

Self-attention, the mechanism that lets a transformer decide which words matter to which other words, computed not in software but directly in hardware, one systolic array cell at a time.

**sv-yae** is a SystemVerilog RTL implementation of the self-attention block used in LLaMA-style transformer models. Every stage of the pipeline — projection, positional encoding, similarity, softmax, output — is a dedicated hardware module. Data moves between them as a continuous BF16 stream over AXI-Stream-style valid/ready handshakes, and the single most expensive piece of hardware (the multiply-accumulate array) is time-shared between two different matrix multiplications instead of being duplicated.

Key highlights:

- **Streaming BF16 datapath:** every value on the internal bus is 16 bits of bfloat16 plus a 1-bit source tag, moving one element per clock, no big parallel buses.
- **RoPE (Rotary Positional Encoding):** Q and K vectors are rotated by a position-dependent angle pulled from a ROM, using the standard base-10000 frequency formula.
- **GQA (Grouped Query Attention):** multiple query heads share a single key/value head, cutting KV storage without cutting query resolution.
- **Time-shared systolic array:** the same SIZE×SIZE PE array computes both Q·Kᵀ and score·V, arbitrated by a tag bit instead of built twice.
- **Valid/ready everywhere:** every module boundary in the design, from the AXI-Stream front end down to a single multiply-accumulate step, uses the same handshake discipline.
- **Ping-pong double buffering:** new tokens load into one buffer while the previous one is still being processed.
- **Zero DSP-block dependency where it matters:** the BF16 adder/multiplier are built from plain logic, not vendor macros, so the design stays portable.

## Navigation

- [Why It Matters](#why-it-matters)
- [What is Self-Attention?](#what-is-self-attention)
- [Mathematical Background](#mathematical-background)
- [Architecture](#architecture)
- [Algorithm Explanations](#algorithm-explanations)
- [Performance Analysis](#performance-analysis)
- [Directory Structure](#directory-structure)
- [Modules](#modules)
- [Simulation](#simulation)
- [Physical Interpretation](#physical-interpretation)
- [Real-World Applications](#real-world-applications)
- [Known Limitations / What's Next](#known-limitations--whats-next)
- [References](#references)

---

## Why It Matters

Transformer inference is dominated by attention, and attention is dominated by two big matrix multiplications (Q·Kᵀ and score·V) plus a row-wise softmax in between. On a CPU or GPU, this is a well-optimized library call; on the edge, at low power, or as a fixed accelerator block feeding a larger SoC, it has to be built from scratch in gates.

| | Software (CPU/GPU library) | This project (FPGA) |
|---|---|---|
| Execution model | Kernel calls into optimized BLAS/cuBLAS | Dedicated hardware pipeline, one stage per operation |
| Latency character | Throughput-oriented, batches amortize overhead | Deterministic, streaming, one token in flight at a time |
| Precision | FP16/FP32/INT8 depending on runtime | Fixed BF16 throughout, chosen once at design time |
| Resource reuse | Implicit, managed by the runtime scheduler | Explicit: one systolic array does both Q·Kᵀ and score·V, arbitrated by a tag bit |
| Power/area envelope | General-purpose silicon, always-on overhead | Only the instantiated logic exists; nothing else is paid for |
| Where it wins | Large batch, data-center inference | Small, fixed-shape, always-on, or power/area constrained inference |

The point isn't that hardware beats a GPU on raw throughput — it doesn't, not at this scale. The point is that a purpose-built accelerator can do attention with no runtime, no scheduler, and no more logic than the four stages actually require.

## What is Self-Attention?

Self-attention answers one question for every token in a sequence: *which other tokens should I pay attention to, and how much?*

Every token is projected into three vectors — a **Query** (what am I looking for), a **Key** (what do I offer), and a **Value** (what do I actually contribute). A token's attention score against another token is the dot product of its Query with that token's Key. Those scores are scaled, optionally masked (so a token can't look at future tokens), and turned into a probability distribution with softmax. The final output for a token is the weighted sum of every Value vector, weighted by those probabilities.

```
Attention(Q, K, V) = softmax( Q·Kᵀ / sqrt(d_k) + mask ) · V
```

This project implements exactly that formula as a hardware pipeline, plus two refinements used by modern LLMs: **RoPE** (so the model knows token *order* without adding position vectors to the embeddings) and **GQA** (so multiple query heads can share one key/value head, cutting the KV cache size).

## Mathematical Background

**Scaled dot-product attention.** For a query matrix `Q`, key matrix `K`, and value matrix `V`, each row being one token:

```
scores = Q · Kᵀ / sqrt(d_k)
weights = softmax(scores)   (row-wise, with an optional causal mask)
output = weights · V
```

**RoPE.** For a vector pair `(x1, x2)` at position `pos`, rotated by angle `theta`:

```
y1 = x1*cos(theta) - x2*sin(theta)
y2 = x1*sin(theta) + x2*cos(theta)
theta(pos, i) = pos * base^(-2i/d_model),   base = 10000
```

This project pairs element `i` with element `i + D_MODEL/2` (the "rotate-half" convention used by LLaMA), rather than with its immediate neighbor.

**GQA.** With `NUM_Q_HEADS` query heads and `NUM_KV_HEADS` key/value heads (`NUM_Q_HEADS` divisible by `NUM_KV_HEADS`), query head `q` reads from key/value head `q / (NUM_Q_HEADS / NUM_KV_HEADS)`. No data is duplicated or moved — only the read address changes.

**BF16 format.** Every value on the internal bus is `{sign[1], exponent[8], mantissa[7]}`, the same exponent range as FP32 truncated to 8 mantissa bits. On top of the 16-bit value, this design carries one extra tag bit (bit 16) through the shared arithmetic units, identifying which logical stream produced the value — this is what lets one physical PE array serve two different matrix multiplications without mixing up their operands.

## Architecture

*(Full data-flow diagram to be added here — `images/architecture.png`, exported from `architecture.drawio`.)*

`architecture.drawio` (open with the draw.io / diagrams.net extension) has five pages: the system-level data flow below, plus one FSM diagram each for `qkv_proj`, `rope`, `softmax`, and the `uram_pingpong_controller` read side — generated from the actual `case`/`always_comb` state logic in those files by `scripts/gen_diagrams.py`, not hand-drawn from memory. Re-run that script after changing a state machine and the diagram stays in sync; the `.drawio` file itself is still freely editable by hand afterward (add the exported PNGs under `images/` once you're happy with the layout).

```
AXI-Stream in (FP32) → bf16_comb → dbuf (ping-pong) → dbuf_read_streamer
                                                              │
                                          token vector, one element/cycle
                                                              ▼
                                                    projection_block
                                       ┌───────────┬───────────┬───────────┐
                                       │  qkv_proj │  qkv_proj │  qkv_proj │  (Wq / Wk / Wv)
                                       │   + rope  │   + rope  │  (no rope)│
                                       │    (Q)    │    (K)    │    (V)    │
                                       └─────┬─────┴─────┬─────┴─────┬─────┘
                                             Q            K            V
                                             │            └──────┬─────┘
                                             │                   ▼
                                             │            URAM ping-pong (Q/K/V store)
                                             │                   │
                                             ▼                   ▼
                                    ┌───────────────────────────────────────┐
                                    │   Module A — shared SIZE×SIZE PE array │
                                    │   (input arbiter picks Q·Kᵀ vs         │
                                    │    score·V by tag, output router      │
                                    │    sends the result back out by tag)  │
                                    └───────────────────┬────────────────────┘
                                                         │
                                     Q·Kᵀ result ────────┤
                                                         ▼
                                          scale_mask → softmax (exp/recip LUTs)
                                                         │
                                     softmax weights ────┘ (fed back into Module A for score·V)
                                                         │
                                                         ▼
                                          qkv_proj (Wo) → tx_fifo → bf16_convert
                                                         │
                                                         ▼
                                                AXI-Stream out (FP32)
```

Module A is instantiated **once**. The same physical array computes Q·Kᵀ on one pass and score·V on the next, distinguishing the two data streams with a 1-bit tag carried alongside every value — this is the single biggest resource-saving decision in the design.

## Algorithm Explanations

**QKV Projection (`qkv_proj.sv`).** A single parameterized matrix-vector multiply module, instantiated four times (Wq, Wk, Wv, Wo) with a different weight matrix loaded into each. Implemented as a simple state machine reusing one multiplier and one adder per instance rather than a parallel array — correctness first, a wider design can follow once the team confirms it's needed.

**RoPE (`rope.sv`).** Rotates Q and K (never V) using sine/cosine values pulled from a ROM addressed by `(position, pair index)`. Because there's no dedicated subtractor yet, the "minus" in the rotation formula is done by flipping the sign bit of the second operand before feeding the shared adder.

**GQA Mapper (`gqa_mapper.sv`).** Purely combinational index translation — given the current query head, it returns which key/value head that query head should read from. It never touches vector data itself.

**Shared Q·Kᵀ / score·V array (`qk_array.sv`, `qk_pe.sv`, wrapped by `mod_a_wrapper.sv`).** A `SIZE×SIZE` output-stationary systolic array. Query/key elements stream in serially along the depth dimension; once a full depth slice has been accumulated, the array switches to a parallel read-out phase. An input arbiter (`mod_a_input_arbiter.sv`) gives priority to score·V requests arriving from softmax, then accepts new Q·Kᵀ tiles — this keeps the softmax-to-output path from stalling behind newly arriving queries. An output router (`mod_a_output_router.sv`) sends the result back to softmax or to the output projection depending on the same tag bit.

**Scale, Mask & Softmax (`scale_mask.sv`, `softmax.sv`, `exp_lut.sv`, `recip_lut.sv`, `row_fifo.sv`).** Raw similarity scores are scaled and (per-element) masked to `-infinity` in a 3-stage backpressure pipeline, then normalized by a 4-state softmax FSM (accumulate → invert → normalize) that trades throughput for numerical accuracy: exponentials and reciprocals both come from small LUTs rather than an iterative approximation.

**Memory & Data Routing (`dbuf.sv`, `dbuf_read_streamer.sv`, `uram_memory.sv`, `uram_pingpong_controller.sv`, `tx_fifo.sv`, `axi_stream_if.sv`).** `dbuf` lets a new token vector load while the previous one is still being processed. Q/K/V are held in ping-pong URAM banks so Module A can read one bank while the next token's projections write the other. `tx_fifo` and `axi_stream_if` carry finished output vectors back out over AXI-Stream, doing the BF16 ↔ FP32 conversion at the boundary.

## Performance Analysis

### Resource Utilization

Synthesized and implemented on a Xilinx Artix-7 (Nexys4 DDR, xc7a100tcsg324-1) with Vivado 2024.1. Numbers below are from the post-synthesis/post-implementation utilization report:

| Resource | Used | Available | % |
|---|---|---|---|
| Slice LUTs | 1,486 | 63,400 | 2.34 |
| Slice FFs | 510 | 126,800 | 0.40 |
| CARRY4 | 21 | 15,850 | 0.13 |
| DSP48E1 | 0 | 240 | 0.00 |
| BRAM (Block RAM Tile) | 0 | 135 | 0.00 |

The 1,486 LUTs split roughly into 918 used as distributed-RAM (`RAMD64E`, mostly `weight_mem`/`x_mem`/`y_mem` and the RoPE ROMs across the four `qkv_proj` and two `rope` instances) and 568 as combinational logic (`LUT6`/`LUT5`/`LUT4`/`LUT3`/`LUT2`/`LUT1`, implementing the BF16 arithmetic and every module's control FSM). Of the 510 flip-flops, 272 are `FDRE` (reset-only) and 238 are `FDCE` (clock-enable + reset), matching the mix of free-running shift/counter registers and gated pipeline registers throughout the design. DSP48E1 and Block RAM usage are both zero — matching the "zero DSP-block dependency" design goal, everything runs through plain LUT logic and 21 `CARRY4` chains rather than vendor arithmetic macros.

*(An earlier run of this table showed only ~73 LUTs — that was a real bug, not a small design: `axi_stream_if.sv`'s `m_axis_tdata` output port was declared 16 bits wide instead of 32, which broke the AXI output data path and let Vivado prove almost the entire pipeline had no observable effect, so it synthesized nearly all of it away. Fixed; see commit history.)*

![floorplan](images/floorplan.png)

*Post-implementation device view (clock regions X0Y0–X1Y3, Nexys4 DDR / xc7a100t). Note: re-export this screenshot from the corrected build — the one currently in `images/` was captured before the `m_axis_tdata` width fix above and under-represents the real utilization.*

along with achieved `f_max`, cycles per token at a given `D_MODEL`/sequence length, and a breakdown of where the LUT/FF budget actually goes (the BF16 adder/multiplier trees are expected to dominate, same as the reference FDTD project's adder-chain-heavy profile).

## Directory Structure

```
rtl/
  compute/     Belinay — bf16_add, bf16_mul, bf16_comb, bf16_convert, qk_array, qk_pe
  projection/  Can     — qkv_proj, rope, gqa_mapper, projection_block (+ generated RoPE ROMs)
  softmax/     Hasan   — scale_mask, softmax, exp_lut, recip_lut, row_fifo, wrapper (+ LUT hex tables)
  control/     Taha    — axi_stream_if, dbuf, dbuf_read_streamer, uram_*, tx_fifo, mod_a_*
  top.sv       full system integration
tb/            standalone testbenches, one per module/subsystem, plus a shared BF16 <-> real helper
scripts/       gen_rope_rom.py    — generates the RoPE sin/cos ROM content
               gen_diagrams.py    — regenerates architecture.drawio from the RTL's own state logic
architecture.drawio   system data-flow + per-module FSM diagrams (multi-page, draw.io format)
```

## Modules

| File | Owner | Purpose |
|---|---|---|
| `rtl/compute/bf16_add.sv` | Belinay | 2-stage pipelined BF16 adder, 17-bit bus (value + tag) |
| `rtl/compute/bf16_mul.sv` | Belinay | 2-stage pipelined BF16 multiplier, same bus convention |
| `rtl/compute/bf16_comb.sv` | Belinay | FP32 → BF16 (truncate or round) |
| `rtl/compute/bf16_convert.sv` | Belinay | BF16 → FP32 |
| `rtl/compute/qk_array.sv` | Belinay | Time-shared `SIZE×SIZE` systolic PE array (Q·Kᵀ and score·V) |
| `rtl/compute/qk_pe.sv` | Belinay | Single processing element of the array |
| `rtl/projection/qkv_proj.sv` | Can | y = W·x projection, instantiated 4× (Wq/Wk/Wv/Wo) |
| `rtl/projection/rope.sv` | Can | Rotary positional encoding for Q and K |
| `rtl/projection/gqa_mapper.sv` | Can | Query-head → KV-head index translation |
| `rtl/projection/projection_block.sv` | Can | Wraps the above four instances behind 5 fixed ports |
| `rtl/softmax/scale_mask.sv` | Hasan | Per-element scale + causal mask, 3-stage pipeline |
| `rtl/softmax/softmax.sv` | Hasan | Row-wise softmax, accumulate/invert/normalize FSM |
| `rtl/softmax/exp_lut.sv` / `recip_lut.sv` | Hasan | LUT-based exponential and reciprocal |
| `rtl/softmax/row_fifo.sv` | Hasan | Holds a row's exponentials between passes |
| `rtl/softmax/scale_mask_softmax_wrapper.sv` | Hasan | Wraps scale_mask + softmax behind one AXI-Stream pair |
| `rtl/control/axi_stream_if.sv` | Taha | FP32 AXI-Stream ↔ internal BF16 boundary |
| `rtl/control/dbuf.sv` / `dbuf_read_streamer.sv` | Taha | Ping-pong token buffer and its read-side streamer |
| `rtl/control/uram_memory.sv` / `uram_pingpong_controller.sv` | Taha | Q/K/V storage and its ping-pong read/write scheduling |
| `rtl/control/mod_a_input_arbiter.sv` / `mod_a_output_router.sv` | Taha | Arbitration and routing around the shared PE array |
| `rtl/control/mod_a_wrapper.sv` | Taha | Streaming wrapper around `qk_array` (serializer/deserializer) |
| `rtl/control/tx_fifo.sv` | Taha | Output-side FIFO before the AXI-Stream boundary |
| `rtl/top.sv` | Taha | Full system integration |

## Simulation

Each module has a standalone, self-checking testbench under `tb/` — no full-system simulation is required to validate an individual stage.

| Testbench | Covers |
|---|---|
| `tb_bf16_add.sv`, `tb_bf16_mul.sv` | Pipelined BF16 arithmetic, exact-representable operands |
| `tb_gqa_mapper.sv` | Exhaustive head-index mapping check |
| `tb_qkv_proj.sv` | Weight loading + W·x MAC pipeline |
| `tb_rope.sv` | ROM loading, rotate-half pairing, identity rotation at pos=0, tolerance check against `$sin`/`$cos` at pos=4 |
| `tb_qk_array.sv` | Shared systolic array through `mod_a_wrapper`, verified against a hand-computed Q·Kᵀ |
| `tb_scale_mask_softmax.sv` | Full scale→mask→softmax chain against the real LUT tables, uniform and causally-masked rows |

Run with Vivado's simulator from the repo root (paths in the RTL are relative to repo root — see the note below):

```
cd tb
xvlog -sv -f filelist_rtl.f tb_bf16_add.sv
xelab tb_bf16_add -s tb_bf16_add_sim
xsim tb_bf16_add_sim -runall
```

Repeat with the other `tb_*.sv` files. `rope.sv` and `scale_mask_softmax_wrapper.sv` load `.mem`/`.hex` files with paths relative to the repo root (`rtl/projection/rope_*_rom.mem`, `rtl/softmax/*_table.hex`) — if Vivado's GUI simulation runs from a different working directory, either point `xsim.simulate.runtime` at the repo root or copy those four files into the simulation run directory.

## Physical Interpretation

Every wire in this design corresponds to something concrete:

- **The 17th bit on every internal bus** isn't part of the number — it's a label saying which logical operation produced this value, so the one shared PE array and the one shared adder/multiplier can serve two different consumers without their results getting crossed.
- **`valid`/`ready` on every boundary** is a literal, physical backpressure signal: nothing moves unless both sides agree, which is what lets a slow softmax stage stall a fast projection stage without losing data.
- **Ping-pong buffers** are two independent memories with a single-bit selector — while hardware writes into one, hardware reads from the other, and they swap roles when a vector finishes.
- **The systolic array's "output-stationary" phase** is the literal moment `SIZE × SIZE` partial sums, one per PE, all become valid at once — that's the hardware equivalent of finishing an entire tile of the Q·Kᵀ matrix in a single parallel step after `DEPTH` serial cycles of accumulation.

## Real-World Applications

- **Edge LLM inference:** a fixed-function attention block that never leaves an idle GPU running, for always-on or power-constrained assistants.
- **Prefill/decode acceleration in a larger SoC:** as one accelerator tile among several, offloading the attention block specifically while a CPU/DSP handles everything else.
- **Teaching hardware-software co-design:** the same self-attention formula every ML course teaches, but with every stage forced to justify its own logic, latency, and memory footprint.

## Known Limitations / What's Next

Honest status, not a marketing list:

- **GQA head addressing is not wired end-to-end.** `gqa_mapper` produces a correct KV-head index, but the current URAM controller has no per-head addressing concept yet — the integrated design currently exercises a single head/KV-group path.
- **RoPE ROM content now exists** (`scripts/gen_rope_rom.py`, base-10000 frequency formula) but has not been cross-checked against a Python/NumPy golden model across many positions — only two positions have been spot-checked in `tb_rope.sv`.
- **`en_scale_mask` is a per-element signal, not a static enable** — it must be driven per data element by real causal-mask logic; tying it to a constant will mask an entire row.
- **No weight-loading lock/handshake yet** — nothing currently prevents a weight write from racing a computation using that same weight.
- **Score·V tag alignment is a flagged assumption**, not a confirmed team decision — see the comment in `mod_a_input_arbiter.sv`.
- **No full end-to-end (`top.sv`) simulation yet** — module-level testbenches pass, system-level integration testing is next once the items above are resolved.
- **Synthesis/utilization numbers are pending** — see [Performance Analysis](#performance-analysis).

## References

[1] A. Vaswani et al., "Attention Is All You Need," in *Proc. NeurIPS*, 2017.
[2] J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv:2104.09864, 2021.
[3] J. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints," arXiv:2305.13245, 2023.
[4] Google Brain, "bfloat16: The secret to high performance on Cloud TPUs," Google Cloud Blog, 2019.
