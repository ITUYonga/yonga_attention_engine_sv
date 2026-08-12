# We Merged Four People's Hardware Designs Into One Chip. Synthesis Quietly Deleted 95% of It.

Attention is one of the oldest tricks evolution ever came up with. Long before anything had a brain, before there was an eye to see with, chemistry itself already had a crude version of it: some reactions simply proceeded faster in one direction than another, some molecules bound more readily to some surfaces than others. It took billions of years for that raw asymmetry to become something we'd recognize as looking somewhere on purpose, but once nervous systems existed, the outcome was unmistakable. A frog's eye does not send its brain an equally weighted report of everything in its visual field. It reports the fly. Attention, in the biological sense, is a mechanism for deciding, moment to moment, what is worth processing at all, so that a small amount of computation can behave as if it were a much larger one.

Take that same problem and hand it to a language model reading a sentence. Every word in the input could, in principle, be relevant to every other word, but treating all of them as equally important would be both computationally wasteful and semantically wrong. Self-attention is the mechanism transformer models use to solve exactly this problem: for every word, decide which other words actually deserve its attention, and by how much. It does this with a surprisingly small amount of arithmetic, three vectors per token, a similarity score between every pair of tokens, and a weighted sum, and that simple operation, repeated at scale, is most of what makes a modern language model work at all.

Over the last few weeks, our team at İTÜ Yonga, Belinay Güler, Can Ertürk, Hasan Sancak, and M. Taha Aydemir, built that operation from scratch. Not as a Python function. Not as a CUDA kernel. As physical logic gates on an FPGA. This is the story of what that actually took, including the part where synthesis quietly deleted almost the entire chip and none of us noticed for a while.

## What self-attention actually computes

Every token gets projected into three vectors: a Query (what am I looking for), a Key (what do I offer), and a Value (what do I actually contribute). A token's attention score against another token is the dot product of its Query with that token's Key. Those scores get scaled, optionally masked so a token can't look at future tokens, and turned into a probability distribution with softmax. The final output for a token is the weighted sum of every Value vector, weighted by that distribution.

Attention(Q, K, V) = softmax(QKᵀ / √d_k + mask) · V

That is the entire mathematical content of self-attention. Everything else, the depth, the parameter counts, the behavior people find remarkable, is this same operation happening many times over. Our job was to take that formula and turn it into a pipeline of hardware modules that compute it one clock edge at a time, with a fixed amount of silicon.

**[INSERT IMAGE: images/architecture.png : caption: "The full YAE data path: AXI-Stream input, QKV projection with RoPE, the shared Q·Kᵀ / score·V systolic array, and the scale/mask/softmax stage."]**

## The architecture

A token enters as FP32, gets truncated to bfloat16 (16 bits instead of 32, same dynamic range as FP32, fewer mantissa bits), and is projected into Query, Key, and Value vectors.

**Rotary positional encoding.** A transformer has no built-in sense of word order unless something gives it one. RoPE encodes position by rotating each Query and Key vector by an angle that depends on both the token's position in the sequence and which pair of dimensions is being rotated, using the standard base-10000 frequency formula. We pair element i with element i + D_MODEL/2, the rotate-half convention most LLaMA-style models use, rather than with its immediate neighbor. In hardware, one rotation is two multiplies, a subtract, and an add, reusing the same shared multiplier and adder the rest of the pipeline already needed instead of adding a dedicated block for it.

**Grouped query attention.** Every extra key/value head a transformer keeps around costs memory bandwidth and storage, and query heads don't strictly need one each. In our configuration, eight query heads share two key/value heads. The mapping from a given query head to its key/value head is a single combinational lookup. It never moves or duplicates data, it only changes which address downstream logic reads from.

The most resource-hungry part of self-attention is the pair of matrix multiplications, Q·Kᵀ and score·V. A naive design builds two separate multiply-accumulate arrays for them. We built one. A single SIZE×SIZE systolic processing-element array runs the Q·Kᵀ pass, then runs a second time on the score·V pass, using the exact same physical multipliers, distinguished by a single tag bit that rides alongside every value on the bus. An arbiter and a router make sure the two passes never get their data crossed. That one decision, sharing the most expensive block instead of duplicating it, is the single biggest resource-saving choice in the whole design.

**[INSERT IMAGE: article/figures/fig2_systolic_array.jpeg : caption: "The shared SIZE×SIZE systolic array. Q and K stream in serially along the array's depth, and once a full depth slice has accumulated, every PE becomes valid on the same clock edge."]**

The array itself is output-stationary. Instead of streaming partial sums in and out, every processing element holds its own running sum in place while Q and K elements stream past it serially, one depth index at a time. Once a full depth slice has accumulated, all SIZE × SIZE partial sums become valid on the same clock edge, the hardware equivalent of finishing an entire tile of the similarity matrix in one parallel step after a fixed number of serial cycles.

**Softmax.** The similarity scores this array produces get scaled, causally masked (set to negative infinity wherever a token isn't allowed to look ahead), and normalized by softmax before the second, score·V pass runs on the same array. This is where we chose accuracy over raw throughput deliberately: instead of a parallel reduction tree, softmax is a four-state pipeline that processes one row element at a time. ACCUMULATE streams each element through a lookup-table-based exponential unit and keeps a running sum. INVERT spends exactly one cycle turning that sum into a reciprocal through a second lookup table. DIVIDE_NORMALIZE streams the row's exponentials back out of a small FIFO, each one multiplied by that reciprocal, before the machine returns to idle for the next row.

**[INSERT IMAGE: images/diagram_fsm_softmax.png : caption: "softmax's four-state machine: two full passes over every row, accumulate then normalize, trading throughput for a numerically accurate result."]**

Every module boundary in the pipeline uses the same valid/ready handshake, so a slow stage like softmax can stall a fast one like projection without losing a single value.

## Four people, four branches, and a chip that had never actually met itself

Here is the part nobody puts in the architecture diagram. This project was built by four people working on four separate pieces: the BF16 arithmetic core and the systolic array, the QKV projection and RoPE, the softmax pipeline, and the memory and control infrastructure that ties everything together. Each piece was written, and each piece looked reasonable, in isolation.

Then we tried to actually connect them into one `top.sv`, and it turned out that four independently written branches, each built against assumptions about the others' interfaces that had never been checked against real code, is not the same thing as a working chip. It's four plausible-looking RTL directories. The gap between those two things is where the real engineering happened, and it is genuinely more instructive than the architecture itself.

**The bug that deleted 95% of the design.** One AXI interface module declared its output data port as 16 bits wide instead of 32. Sixteen bits. One line. Vivado's synthesizer, being extremely good at its job, looked at that mismatch, correctly proved that the truncated output path carried no observable information, and deleted almost the entire design as dead logic. Utilization report: 73 LUTs. Not 73% of a smaller design, 73 individual look-up tables, out of a design that should have used roughly 1,500. The chip "compiled." It even routed. It just didn't compute anything. Fixing the port width brought the whole pipeline back.

**The wrapper that couldn't tell you when it was done.** One module computed its own "last element of this operation" flag internally and then simply never exposed it as an output port. Downstream logic that needed to know when a tile of data had finished had no way to ask.

**The masking bug that would have zeroed every attention score.** The causal mask control signal was wired to a constant 1 in the top-level integration. By the softmax module's own logic, a constant 1 means "mask every element of every row," which would have made every attention weight collapse to the same fixed distribution regardless of input, a bug that produces a chip that runs, produces plausible-looking numbers, and is completely wrong.

None of these were exotic mistakes. They were the ordinary consequence of four people building interfaces against a shared plan instead of against each other's actual code. Reading the code side by side would not have caught any of them. Only actually elaborating the whole design and simulating it did.

## Testing four separate hardware modules is not like testing four separate functions

Once the pipeline elaborated, we built a standalone, self-checking testbench for every module, so each stage could be verified without needing the full chip assembled. This sounds simple. It was not, and the reason why is worth explaining because it is a genuinely common trap in RTL verification.

**[INSERT IMAGE: images/tb_bf16_add.png : caption: "First real hardware bug: seven adder checks, and the sixth one was reading the previous check's result instead of its own."]**

Our very first testbench had a bug that took a long time to root-cause: every check was reading the result of the *previous* check, off by exactly one. We tried longer waits, we tried extra delays, nothing fixed it, some attempts made it worse. The actual cause was a classic race condition: driving a DUT's inputs with a blocking assignment immediately before `@(posedge clk)` can race against the DUT's own always_ff block sampling those same signals at that same edge, and which one "wins" is a simulator implementation detail, not something the Verilog LRM guarantees. The fix, once found, was universal: switch to nonblocking assignment with an explicit synchronizing edge beforehand. We had to apply that same fix to every single testbench in the suite.

**[INSERT IMAGE: images/tb_qk_array.png : caption: "The shared systolic array, verified against a hand-computed 2×2 case. All four expected outputs matched."]**

The shared Q·Kᵀ / score·V array had a subtler bug: it would accept the first three of four expected inputs and then simply hang. Tracing internal signals cycle by cycle (not guessing, actually printing `slice_inflight`, `pe_ready_vector`, and friends every clock edge) showed the testbench's own stimulus loop was silently overwriting a data element that had not yet been acknowledged by the DUT, because the loop assumed "I waited for ready, therefore my data was accepted" without ever confirming the handshake actually completed on that exact edge. The array itself was fine. The way we were talking to it wasn't.

**[INSERT IMAGE: images/tb_scale_softmax.png : caption: "Softmax fully verified: a uniform row normalizes to 0.25 each, a causally masked row correctly collapses the masked element to 0."]**

Softmax was the deepest rabbit hole. Getting it to correctly handle a masked row (the actual causal-attention use case, not just a uniform unmasked row) surfaced four separate, real bugs stacked on top of each other: a pipeline stage that could silently drop an element under backpressure, a "valid" signal that stayed stuck high and made a multiplier think it was receiving a continuous stream of new operations instead of one, an off-by-one in exactly which cycle an output counter incremented on, and, most satisfying to finally find, a reciprocal look-up table whose exponent formula was correct only when the input happened to be an exact power of two, silently returning answers exactly double what they should have been otherwise. Four bugs, each one hiding behind the previous one, each only visible once the one in front of it was fixed.

By the end, seven for seven testbenches passed: the BF16 adder and multiplier, the GQA head mapper (exhaustively checked across all head combinations), the QKV projection pipeline, RoPE against a `$sin`/`$cos` reference, the shared systolic array, and the full scale/mask/softmax chain against both a uniform and a causally masked row.

## What we actually measured

**[INSERT IMAGE: images/device.png : caption: "Post-implementation device view on a Xilinx Artix-7 (Nexys4 DDR)."]**

Synthesized and implemented on a Xilinx Artix-7 (Nexys4 DDR, xc7a100t) in Vivado 2024.1, the corrected design uses 1,486 LUTs (2.34% of the device), 510 flip-flops (0.40%), and, notably, zero DSP48E1 slices and zero Block RAM tiles. Every multiply and add in this design runs on plain LUT logic, not vendor arithmetic macros, which was a deliberate choice to keep the core portable across FPGA families rather than locked to one vendor's DSP primitives.

## What's still open, because a project post that hides its own limitations isn't worth much

The RoPE rotation is implemented and passes a targeted correctness check, but hasn't yet been cross-checked against a full Python/NumPy golden model across many positions. GQA head mapping computes the correct index but isn't wired into the memory controller's addressing yet, so the integrated design currently exercises a single head/KV-group path. There's no weight-loading lock yet. We haven't run a full end-to-end simulation of the assembled top-level design, only every module in isolation, which is the next real milestone. And we haven't measured timing closure or maximum clock frequency at all yet.

## Why the integration story matters just as much

The architecture above is real, and we're proud of the systolic array sharing trick in particular. But a post that only showed the block diagram would miss the part just as worth remembering: that four people can each write correct-looking RTL, and the system they produce together can still not work, silently, past synthesis, past place and route, all the way to a utilization report that looks fine until you actually check the number. The fix was never clever. It was running the thing, reading what it actually did instead of what we assumed it did, and being willing to trace a signal cycle by cycle until the disagreement between our mental model and the hardware's actual behavior became visible.

That is, in the end, not so different from what the frog's eye is doing. Deciding, deliberately, what to actually look at.

---

*Built by Belinay Güler, Can Ertürk, Hasan Sancak, and M. Taha Aydemir at İTÜ Yonga. Full RTL, testbenches, and technical writeup [link to repo, if going public].*
