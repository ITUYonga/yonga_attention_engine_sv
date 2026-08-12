// FUNC_PROJECTION_BLOCK : Wraps qkv_proj, rope, and gqa_mapper into one
// hardwired top level unit with 5 fixed outputs
//
//   Purpose:  Bundles four qkv_proj instances (Wq, Wk, Wv, Wo), two rope
//             instances (Q and K paths), and one gqa_mapper into a single
//             block with a small, fixed set of external ports, so
//             whoever wires this into the top level chip does not need
//             to know anything about how many internal modules there are
//             or how they connect to each other. Every output goes to
//             exactly one destination, decided purely by wiring, there is
//             no routing or tag based decision anywhere in this module:
//             q_data_o always goes to the shared systolic array, k_data_o
//             and v_data_o always go to the KV cache, kv_head_idx_o
//             always goes to the KV cache's read address, and out_data_o
//             always goes to the AXI-Stream output path.
//
//   parameters:
//           DATA_WIDTH:               bit width of one bf16 element
//           D_MODEL:                  length of the input token vector
//           D_OUT_Q:                  length of the Q output vector
//           D_OUT_KV:                 length of the K and V output
//                                     vectors, can be smaller than
//                                     D_OUT_Q under grouped query
//                                     attention
//           MAX_POS:                  largest token position RoPE's ROM
//                                     supports
//           NUM_Q_HEADS, NUM_KV_HEADS: passed straight through to
//                                     gqa_mapper
//
//   inputs:
//           clk_i, rst_ni:         clock and active low reset
//           wq_data_i/wq_addr_i/wq_we_i, wk_..., wv_..., wo_...:
//                                  weight load ports, one independent set
//                                  per projection instance
//           token_data_i/token_valid_i/token_last_i:
//                                  the raw token vector from dbuf, feeds
//                                  Wq, Wk, and Wv all at once
//           pos_i:                 token position, forwarded to both
//                                  RoPE instances
//           q_ready_i, k_ready_i, v_ready_i:
//                                  backpressure from each output's
//                                  destination
//           q_head_idx_i:          current query head index, forwarded
//                                  to gqa_mapper
//           attn_data_i/attn_valid_i/attn_last_i:
//                                  the attention output vector coming
//                                  back from the shared array, feeds Wo
//           out_ready_i:           backpressure from the AXI-Stream
//                                  output path
//   output:
//           token_ready_o:         AND of Wq, Wk, and Wv's own
//                                  x_ready_o, so a new token is only
//                                  accepted once all three are actually
//                                  ready to receive it
//           q_data_o/q_valid_o/q_last_o:
//                                  rotated Q vector, to the shared array
//           k_data_o/k_valid_o/k_last_o:
//                                  rotated K vector, to the KV cache
//           v_data_o/v_valid_o/v_last_o:
//                                  V vector, unrotated, to the KV cache
//           kv_head_idx_o:         which KV head the current Q head
//                                  should read from
//           attn_ready_o:          backpressure back to whatever feeds
//                                  attn_data_i
//           out_data_o/out_valid_o/out_last_o:
//                                  final projected output vector
//
//   notes:
//           found and fixed a real bug here, token_ready_o used to be
//           wired only to the Wq instance's own x_ready_o, leaving Wk and
//           Wv's readiness unused. Since all three share the same
//           token_valid_i/token_data_i, if either one ever fell out of
//           lockstep with Wq (staying busy computing or draining while
//           Wq had already returned to loading), a new token could be
//           accepted based on Wq alone while Wk or Wv silently missed it
//           entirely, since each instance only listens to its input
//           while its own state machine is actually in the loading
//           state. token_ready_o is now the AND of all three

module projection_block #(
    parameter DATA_WIDTH  = 16,
    parameter D_MODEL     = 64,
    parameter D_OUT_Q     = 64,
    parameter D_OUT_KV    = 64,
    parameter MAX_POS     = 512,
    parameter NUM_Q_HEADS = 8,
    parameter NUM_KV_HEADS = 2
)(
    input clk_i,
    input rst_ni,

    // weight yukleme
    input [DATA_WIDTH-1:0] wq_data_i,
    input [$clog2(D_MODEL*D_OUT_Q)-1:0] wq_addr_i,
    input wq_we_i,

    input [DATA_WIDTH-1:0] wk_data_i,
    input [$clog2(D_MODEL*D_OUT_KV)-1:0] wk_addr_i,
    input wk_we_i,

    input [DATA_WIDTH-1:0] wv_data_i,
    input [$clog2(D_MODEL*D_OUT_KV)-1:0] wv_addr_i,
    input wv_we_i,

    input [DATA_WIDTH-1:0] wo_data_i,
    input [$clog2(D_OUT_Q*D_MODEL)-1:0] wo_addr_i,
    input wo_we_i,

    // token girisi dbuf'tan geliyor
    input [DATA_WIDTH:0] token_data_i,
    input token_valid_i,
    input token_last_i,
    output token_ready_o,

    // pozisyon rope icin
    input [$clog2(MAX_POS)-1:0] pos_i,

    // Q cikisi belinaya gidiyor
    output [DATA_WIDTH:0] q_data_o,
    output q_valid_o,
    output q_last_o,
    input q_ready_i,

    // K cikisi tahaya gidiyor kv_cache
    output [DATA_WIDTH:0] k_data_o,
    output k_valid_o,
    output k_last_o,
    input k_ready_i,

    // V cikisi tahaya gidiyor kv_cache rope yok
    output [DATA_WIDTH:0] v_data_o,
    output v_valid_o,
    output v_last_o,
    input v_ready_i,

    // gqa mapper
    input [$clog2(NUM_Q_HEADS)-1:0] q_head_idx_i,
    output [$clog2(NUM_KV_HEADS)-1:0] kv_head_idx_o,

    // output projection belinaydan geri donus
    input [DATA_WIDTH:0] attn_data_i,
    input attn_valid_i,
    input attn_last_i,
    output attn_ready_o,

    output [DATA_WIDTH:0] out_data_o,
    output out_valid_o,
    output out_last_o,
    input out_ready_i
);

// q proj -> q rope arasi kablolar
wire [DATA_WIDTH:0] q_proj_out;
wire q_proj_out_v,q_proj_out_l,q_proj_out_r;

// k proj -> k rope arasi kablolar
wire [DATA_WIDTH:0] k_proj_out;
wire k_proj_out_v,k_proj_out_l,k_proj_out_r;

// Wq/Wk/Wv ayni token_valid_i/token_data_i'yi paylasiyor, ucu de ayni
// cycle'da ST_LOAD'dan cikip tekrar ST_LOAD'a donmezse (mesela biri
// k_ready_i/v_ready_i backpressure'i yuzunden ST_DRAIN'de takilirsa) lockstep
// bozulur. token_ready_o'yu tek instance'a (eskiden sadece Q) baglamak
// digerlerinin hazir olmadigi bir cycle'da token kabul edip o instance'larin
// veriyi sessizce kacirmasina yol acardi, o yuzden ucunun x_ready_o'su da
// AND'lenip disariya oyle veriliyor
wire q_proj_rdy, k_proj_rdy, v_proj_rdy;
assign token_ready_o = q_proj_rdy && k_proj_rdy && v_proj_rdy;

// -------- Q yolu --------
// token -> qkv_proj(Wq) -> rope -> q_data_o (belinaya)

qkv_proj #(.DATA_WIDTH(DATA_WIDTH),.D_MODEL(D_MODEL),.D_OUT(D_OUT_Q)) u_proj_q(
    .clk_i(clk_i),.rst_ni(rst_ni),
    .w_data_i(wq_data_i),.w_addr_i(wq_addr_i),.w_we_i(wq_we_i),
    .x_data_i(token_data_i),.x_valid_i(token_valid_i),
    .x_last_i(token_last_i),.x_ready_o(q_proj_rdy),
    .y_data_o(q_proj_out),.y_valid_o(q_proj_out_v),
    .y_last_o(q_proj_out_l),.y_ready_i(q_proj_out_r),
    .busy_o()
);

rope #(.DATA_WIDTH(DATA_WIDTH),.D_MODEL(D_OUT_Q),.MAX_POS(MAX_POS)) u_rope_q(
    .clk_i(clk_i),.rst_ni(rst_ni),
    .pos_i(pos_i),
    .x_data_i(q_proj_out),.x_valid_i(q_proj_out_v),
    .x_last_i(q_proj_out_l),.x_ready_o(q_proj_out_r),
    .y_data_o(q_data_o),.y_valid_o(q_valid_o),
    .y_last_o(q_last_o),.y_ready_i(q_ready_i)
);

// -------- K yolu --------
// token -> qkv_proj(Wk) -> rope -> k_data_o (tahaya kv_cache)

qkv_proj #(.DATA_WIDTH(DATA_WIDTH),.D_MODEL(D_MODEL),.D_OUT(D_OUT_KV)) u_proj_k(
    .clk_i(clk_i),.rst_ni(rst_ni),
    .w_data_i(wk_data_i),.w_addr_i(wk_addr_i),.w_we_i(wk_we_i),
    .x_data_i(token_data_i),.x_valid_i(token_valid_i),
    .x_last_i(token_last_i),.x_ready_o(k_proj_rdy),
    .y_data_o(k_proj_out),.y_valid_o(k_proj_out_v),
    .y_last_o(k_proj_out_l),.y_ready_i(k_proj_out_r),
    .busy_o()
);

rope #(.DATA_WIDTH(DATA_WIDTH),.D_MODEL(D_OUT_KV),.MAX_POS(MAX_POS)) u_rope_k(
    .clk_i(clk_i),.rst_ni(rst_ni),
    .pos_i(pos_i),
    .x_data_i(k_proj_out),.x_valid_i(k_proj_out_v),
    .x_last_i(k_proj_out_l),.x_ready_o(k_proj_out_r),
    .y_data_o(k_data_o),.y_valid_o(k_valid_o),
    .y_last_o(k_last_o),.y_ready_i(k_ready_i)
);

// -------- V yolu --------
// token -> qkv_proj(Wv) -> direkt cikis rope yok

qkv_proj #(.DATA_WIDTH(DATA_WIDTH),.D_MODEL(D_MODEL),.D_OUT(D_OUT_KV)) u_proj_v(
    .clk_i(clk_i),.rst_ni(rst_ni),
    .w_data_i(wv_data_i),.w_addr_i(wv_addr_i),.w_we_i(wv_we_i),
    .x_data_i(token_data_i),.x_valid_i(token_valid_i),
    .x_last_i(token_last_i),.x_ready_o(v_proj_rdy),
    .y_data_o(v_data_o),.y_valid_o(v_valid_o),
    .y_last_o(v_last_o),.y_ready_i(v_ready_i),
    .busy_o()
);

// -------- GQA --------
// sadece index cevirisi veri tasimaz

gqa_mapper #(.NUM_Q_HEADS(NUM_Q_HEADS),.NUM_KV_HEADS(NUM_KV_HEADS)) u_gqa(
    .q_head_idx_i(q_head_idx_i),
    .kv_head_idx_o(kv_head_idx_o)
);

// -------- output projection --------
// belinaydan gelen attention sonucunu Wo ile carpip tahaya veriyor

qkv_proj #(.DATA_WIDTH(DATA_WIDTH),.D_MODEL(D_OUT_Q),.D_OUT(D_MODEL)) u_proj_out(
    .clk_i(clk_i),.rst_ni(rst_ni),
    .w_data_i(wo_data_i),.w_addr_i(wo_addr_i),.w_we_i(wo_we_i),
    .x_data_i(attn_data_i),.x_valid_i(attn_valid_i),
    .x_last_i(attn_last_i),.x_ready_o(attn_ready_o),
    .y_data_o(out_data_o),.y_valid_o(out_valid_o),
    .y_last_o(out_last_o),.y_ready_i(out_ready_i),
    .busy_o()
);

endmodule
