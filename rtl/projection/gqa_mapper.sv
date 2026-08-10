// FUNC_GQA_MAPPER : Tells a Q head which KV head it is allowed to use
// Can Erturk  21.07.2026
//
//   Purpose:  In grouped query attention there are more Q heads than KV
//             heads, groups of Q heads share the same K and V. Example
//             from my notes, 8 Q heads and 2 KV heads means Q head 0 to
//             3 all use KV head 0, and Q head 4 to 7 all use KV head 1.
//             This module just does that index translation, it does not
//             move any vector data itself. Whoever reads from Taha's
//             kv_cache is expected to use kv_head_idx_o as the read
//             address so the actual K and V vectors "get broadcast" by
//             simply reading the same cache entry for every Q head in
//             that group.
//
//             this is an assumption on my part, plan doc calls this
//             "head broadcast/select muxing" which could also mean it
//             physically muxes K/V vector buses instead of just giving
//             out an index. going with the index only version since it
//             is way simpler and kv_cache read address is already how
//             Taha's module is structured (address in, data out). easy
//             to change later if this assumption is wrong
//
//   parameters:
//           NUM_Q_HEADS:   how many query heads the model has
//           NUM_KV_HEADS:  how many key/value heads the model has, must
//                         divide NUM_Q_HEADS evenly
//
//   inputs:
//           q_head_idx_i:  which Q head is currently being processed
//   output:
//           kv_head_idx_o: which KV head that Q head should read from

module gqa_mapper #(
    parameter int NUM_Q_HEADS = 8,
    parameter int NUM_KV_HEADS = 2
)(
    input logic [$clog2(NUM_Q_HEADS)-1:0] q_head_idx_i,
    output logic [$clog2(NUM_KV_HEADS)-1:0] kv_head_idx_o
);

    localparam int GROUP_SIZE = NUM_Q_HEADS / NUM_KV_HEADS;

    // plain division here, only cheap in hardware if GROUP_SIZE ends up
    // being a power of 2 (then the tool turns it into a shift by
    // itself), keep NUM_Q_HEADS / NUM_KV_HEADS as powers of 2 to be safe
    assign kv_head_idx_o = q_head_idx_i / GROUP_SIZE;

endmodule
