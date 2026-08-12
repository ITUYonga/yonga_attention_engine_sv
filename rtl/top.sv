`timescale 1ns / 1ps

// FUNC_ATTENTION_ENGINE_TOP : Wires every module in the design into one
// chip, from the fp32 AXI-Stream boundary to the fp32 AXI-Stream boundary
//
//   Purpose:  Instantiates and connects the whole pipeline in order.
//             axi_stream_if and dbuf take a token sequence in off the
//             external AXI bus. projection_block turns each token into
//             rotated Q, rotated K, and plain V, feeding Q straight to
//             the shared systolic array and K/V into the ping pong URAM
//             banks by way of uram_pingpong_controller. Once a whole
//             matrix's worth of Q and K is stored, mod_a_input_arbiter
//             feeds it into mod_a_wrapper (wrapping qk_array) depth slice
//             by depth slice for the Q times K transpose pass,
//             mod_a_output_router sends the result on to
//             scale_mask_softmax_wrapper, and softmax's normalized
//             weights come back around through the same arbiter, paired
//             with a V read, for the array's second, score times V pass.
//             That result goes back into projection_block's Wo instance,
//             and the finished output vector leaves through tx_fifo and
//             axi_stream_if. Every module boundary uses the same
//             valid/ready handshake, and the 1 bit source tag riding
//             alongside every 17 bit value is how the arbiter and routers
//             tell the two passes through the shared array apart.
//
//   parameters:
//           DATA_WIDTH_f32:    bit width of the external AXI data bus
//           DATA_WIDTH_bf16:   bit width of one internal bf16 element
//           DBUF_ADDR_WIDTH:   address width of dbuf's own banks
//           D_MODEL:           length of one token vector
//           MATRIX_SIZE:       total elements in one full Q/K/V matrix,
//                              passed to the URAM controller
//           MAX_POS:           largest token position RoPE's ROM
//                              supports
//
//   inputs:
//           clk, rst_n:         clock and active low reset
//           select_bf16:        forwarded to axi_stream_if's fp32 to
//                               bf16 conversion select
//           s_axis_tdata/tvalid/tlast:
//                               external AXI-Stream slave side, the
//                               token sequence coming in
//           m_axis_tready:      external AXI-Stream master side, whether
//                               the outside world can accept output
//           w_data_i/w_addr_i:  shared weight value/address bus for
//                               loading all four projection weight
//                               matrices
//           w_we_q, w_we_k, w_we_v, w_we_o:
//                               which weight matrix w_data_i/w_addr_i
//                               is currently targeting
//   output:
//           s_axis_tready:      external AXI-Stream slave side ready
//           m_axis_tdata/tvalid/tlast:
//                               external AXI-Stream master side, the
//                               finished output vector going out
//
//   notes:
//           this is the level every module level testbench stops short
//           of. A first full simulation of this exact assembled pipeline
//           found and fixed four further integration bugs that no single
//           module's own testbench could reach

module attention_engine_top #(
    parameter DATA_WIDTH_f32  = 32,
    parameter DATA_WIDTH_bf16 = 16,
    parameter DBUF_ADDR_WIDTH = 10,  
    parameter D_MODEL         = 64,  
    parameter MATRIX_SIZE     = 1024,
    parameter MAX_POS         = 512
)(
    input  logic clk,
    input  logic rst_n,
    input  logic select_bf16,

    // AXI-STREAM GİRİŞ (FP32)
    input  logic [DATA_WIDTH_f32-1:0] s_axis_tdata,
    input  logic                      s_axis_tvalid,
    output logic                      s_axis_tready,
    input  logic                      s_axis_tlast,

    // AXI-STREAM ÇIKIŞ (FP32)
    output logic [DATA_WIDTH_f32-1:0] m_axis_tdata,
    output logic                      m_axis_tvalid,
    input  logic                      m_axis_tready,
    output logic                      m_axis_tlast,

    // AĞIRLIK YÜKLEME (AXI-Lite)
    input  logic [DATA_WIDTH_bf16-1:0] w_data_i,
    input  logic [$clog2(D_MODEL*D_MODEL)-1:0] w_addr_i,
    input  logic                       w_we_q, w_we_k, w_we_v, w_we_o
);

    // internal wires
    logic a_out_last, a_w0_last;
    logic [DBUF_ADDR_WIDTH-1:0] dbuf_read_addr;
    logic [DATA_WIDTH_bf16-1:0] dbuf_read_data; 
    logic dbuf_v, dbuf_r, dbuf_last;
    logic [DATA_WIDTH_bf16-1:0] dbuf_d; 
    
    logic [$clog2(MAX_POS)-1:0] pos_cnt;
    logic a_w0_v, a_w0_r;
    logic [16:0] a_w0_d; 

    logic [16:0] q_proj_d, k_proj_d, v_proj_d;
    logic q_proj_v, k_proj_v, v_proj_v;
    logic q_proj_last, k_proj_last, v_proj_last;
    logic q_proj_r, k_proj_r, v_proj_r;

    logic [16:0] out_proj_d;
    logic out_proj_v, out_proj_last, out_proj_r;

    logic q_uram_ren, k_uram_ren, v_uram_ren;
    // URAM Adresleri Ping-Pong için 1 bit genişletildi (Örn: 11 bit)
    logic [DBUF_ADDR_WIDTH:0] uram_waddr; 
    logic [DBUF_ADDR_WIDTH:0] q_uram_raddr, k_uram_raddr, v_uram_raddr;
    
    logic q_uram_rvalid, k_uram_rvalid, v_uram_rvalid;
    logic [16:0] q_rdata, k_rdata, v_rdata; 

    logic a_in_v, a_in_r;
    logic [16:0] a_in_q, a_in_k;
    logic a_out_v, a_out_r;
    logic [16:0] a_out_d;

    logic fifo_out_v, fifo_out_r;
    // bit 17 = real end-of-vector last (tx_fifo's if_last), bits 16:0 = tagged bf16 value
    logic [17:0] fifo_out_d;

    // Modül C (Softmax) Bağlantıları
    logic c_in_v, c_in_r, c_in_last; // last pini eklendi
    logic [16:0] c_in_d;

    logic c_v, c_r, c_out_last;      // last pini eklendi
    logic [16:0] c_d;

    // AXI, dbuf and the read streamer
    logic dbuf_rx_we, dbuf_rx_full, internal_rx_last;
    logic [DATA_WIDTH_bf16-1:0] dbuf_rx_data;
    logic swap_buffers;

    assign swap_buffers = (dbuf_rx_we && internal_rx_last);

    axi_stream_if #(.DATA_WIDTH_bf16(16), .DATA_WIDTH_f32(32)) u_axi_if (
        .select_bf16(select_bf16),
        .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready), .s_axis_tlast(s_axis_tlast),
        .dbuf_rx_data(dbuf_rx_data), .dbuf_rx_we(dbuf_rx_we), .dbuf_rx_full(dbuf_rx_full), .internal_rx_last(internal_rx_last),
        .internal_tx_data(fifo_out_d), .internal_tx_valid(fifo_out_v), .internal_tx_ready(fifo_out_r), 
        .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast)
    );

    dbuf #(.DATA_WIDTH(16), .ADDR_WIDTH(DBUF_ADDR_WIDTH)) u_dbuf (
        .clk(clk), .rst_n(rst_n),
        .rx_data(dbuf_rx_data), .rx_we(dbuf_rx_we), .rx_full(dbuf_rx_full),
        .read_addr(dbuf_read_addr), .read_data(dbuf_read_data), .swap_buffers(swap_buffers)
    );

    dbuf_read_streamer #(.DATA_WIDTH(16), .ADDR_WIDTH(DBUF_ADDR_WIDTH), .SEQ_LENGTH(MATRIX_SIZE), .D_MODEL(D_MODEL)) u_dbuf_streamer (
        .clk(clk), .rst_n(rst_n), .swap_buffers(swap_buffers),
        .read_addr(dbuf_read_addr), .read_data(dbuf_read_data),
        .m_valid(dbuf_v), .m_data(dbuf_d), .m_last(dbuf_last), .m_ready(dbuf_r)
    );

    // RoPE position counter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pos_cnt <= '0;
        end else if (dbuf_v && dbuf_r && dbuf_last) begin
            if (pos_cnt == MAX_POS - 1) pos_cnt <= '0;
            else pos_cnt <= pos_cnt + 1'b1;
        end
    end

    // Module B (projection_block), never stalls
    projection_block #(
        .DATA_WIDTH(16), .D_MODEL(D_MODEL), .D_OUT_Q(D_MODEL), .D_OUT_KV(D_MODEL),
        .MAX_POS(MAX_POS), .NUM_Q_HEADS(8), .NUM_KV_HEADS(2)
    ) u_projection_block (
        .clk_i(clk), .rst_ni(rst_n),
        .wq_data_i(w_data_i), .wq_addr_i(w_addr_i), .wq_we_i(w_we_q),
        .wk_data_i(w_data_i), .wk_addr_i(w_addr_i), .wk_we_i(w_we_k),
        .wv_data_i(w_data_i), .wv_addr_i(w_addr_i), .wv_we_i(w_we_v),
        .wo_data_i(w_data_i), .wo_addr_i(w_addr_i), .wo_we_i(w_we_o),
        
        .token_data_i({1'b0, dbuf_d}), .token_valid_i(dbuf_v), .token_last_i(dbuf_last), .token_ready_o(dbuf_r),
        .pos_i(pos_cnt),
        
        .q_data_o(q_proj_d), .q_valid_o(q_proj_v), .q_last_o(q_proj_last), .q_ready_i(q_proj_r),
        .k_data_o(k_proj_d), .k_valid_o(k_proj_v), .k_last_o(k_proj_last), .k_ready_i(k_proj_r),
        .v_data_o(v_proj_d), .v_valid_o(v_proj_v), .v_last_o(v_proj_last), .v_ready_i(v_proj_r),

        .q_head_idx_i('0), .kv_head_idx_o(),
        
        .attn_data_i(a_w0_d), .attn_valid_i(a_w0_v), .attn_last_i(a_w0_last), .attn_ready_o(a_w0_r),
        .out_data_o(out_proj_d), .out_valid_o(out_proj_v), .out_last_o(out_proj_last), .out_ready_i(out_proj_r)
    );

    // URAM ping-pong controller and memories
    logic q_uram_rdy, k_uram_rdy, v_uram_rdy;
    assign q_proj_r = q_uram_rdy;
    assign k_proj_r = k_uram_rdy;
    assign v_proj_r = v_uram_rdy;

    uram_pingpong_controller #(.ADDR_WIDTH(DBUF_ADDR_WIDTH), .MATRIX_SIZE(MATRIX_SIZE), .D_MODEL(D_MODEL)) u_uram_ctrl (
        .clk(clk), .rst_n(rst_n),
        .write_valid(q_proj_v), .write_last(q_proj_last), .waddr(uram_waddr),
        .softmax_valid(c_v),
        .qk_data_valid(qk_valid), .qk_ready(qk_ready),
        .q_ren(q_uram_ren), .q_raddr(q_uram_raddr),
        .k_ren(k_uram_ren), .k_raddr(k_uram_raddr),
        .v_ren(v_uram_ren), .v_raddr(v_uram_raddr)
    );

    // Derinlik 2 katına çıktı! (Ping-Pong için)
    uram_memory #(.DATA_WIDTH(17), .ADDR_WIDTH(DBUF_ADDR_WIDTH+1), .DEPTH(2*MATRIX_SIZE)) u_uram_q (
        .clk(clk), .rst_n(rst_n), 
        .wr_en(q_proj_v), .wr_addr(uram_waddr), .wr_data(q_proj_d), .ready(q_uram_rdy), 
        .rd_en(q_uram_ren), .rd_addr(q_uram_raddr), .rdata(q_rdata), .rvalid(q_uram_rvalid)
    );
    uram_memory #(.DATA_WIDTH(17), .ADDR_WIDTH(DBUF_ADDR_WIDTH+1), .DEPTH(2*MATRIX_SIZE)) u_uram_k (
        .clk(clk), .rst_n(rst_n), 
        .wr_en(k_proj_v), .wr_addr(uram_waddr), .wr_data(k_proj_d), .ready(k_uram_rdy), 
        .rd_en(k_uram_ren), .rd_addr(k_uram_raddr), .rdata(k_rdata), .rvalid(k_uram_rvalid)
    );
    uram_memory #(.DATA_WIDTH(17), .ADDR_WIDTH(DBUF_ADDR_WIDTH+1), .DEPTH(2*MATRIX_SIZE)) u_uram_v (
        .clk(clk), .rst_n(rst_n), 
        .wr_en(v_proj_v), .wr_addr(uram_waddr), .wr_data(v_proj_d), .ready(v_uram_rdy), 
        .rd_en(v_uram_ren), .rd_addr(v_uram_raddr), .rdata(v_rdata), .rvalid(v_uram_rvalid)
    );

    // Module A input arbiter
    logic qk_valid, qk_ready;
    assign qk_valid = q_uram_rvalid && k_uram_rvalid; 

    mod_a_input_arbiter #(.DATA_WIDTH(17)) u_a_arb (
        .clk(clk), .rst_n(rst_n),
        .c_valid(c_v), .c_data(c_d), .c_ready(c_r), .v_rdata(v_rdata),
        .qk_valid(qk_valid), .q_data(q_rdata), .k_data(k_rdata), .qk_ready(qk_ready),
        .mod_a_valid(a_in_v), .mod_a_data_l(a_in_q), .mod_a_data_t(a_in_k), .mod_a_tag(), .mod_a_ready(a_in_r)
    );

    // Module A (systolic array) and its wrapper
    mod_a_wrapper #(.SIZE(4), .DEPTH(D_MODEL)) u_mod_a_wrap (
        .clk_i(clk), .rst_ni(rst_n),
        .s_valid(a_in_v), .s_q_data(a_in_q), .s_k_data(a_in_k), .s_ready(a_in_r),
        .m_valid(a_out_v), .m_data(a_out_d), .m_ready(a_out_r), .m_last(a_out_last)
    );

    mod_a_output_router #(.DATA_WIDTH(17)) u_a_router (
        .mod_a_valid(a_out_v), .mod_a_data(a_out_d), .mod_a_tag(a_out_d[16]), .mod_a_ready(a_out_r),
        .c_valid(c_in_v), .c_data(c_in_d), .c_last(c_in_last), .c_ready(c_in_r),
        .b_valid(a_w0_v), .b_data(a_w0_d), .b_last(a_w0_last), .b_ready(a_w0_r), .mod_a_last(a_out_last)
    );

    // Module C (scale + mask + softmax wrapper)
    scale_mask_softmax_wrapper #(
        .DATA_WIDTH(16),
        .MAX_ROW_LEN(D_MODEL) 
    ) u_mod_c_softmax (
        .clk(clk), 
        .rst_n(rst_n),

        // 1 / sqrt(64) = 0.125, bf16 16'h3E00
        .scale_factor(16'h3E00),
        // real per-element causal mask generator doesn't exist yet; a
        // constant 1 (mask everything to -inf) synthesizes as a fixed
        // value, so Vivado proves most of softmax/Module A "dead" and
        // deletes it, and breaks softmax functionally too. 0 (no masking)
        // is the more correct default until real causal mask logic exists
        .en_scale_mask(1'b0),
        .ext_stall(1'b0),

        .s_axis_tdata(c_in_d[15:0]),
        .s_axis_tvalid(c_in_v),
        .s_axis_tready(c_in_r),
        .s_axis_tlast(c_in_last),

        .m_axis_tdata(c_d[15:0]),
        .m_axis_tvalid(c_v),
        .m_axis_tready(c_r),
        .m_axis_tlast(c_out_last)
    );

    // pack the outgoing value with a 17-bit tag (1 = Wo projection target)
    assign c_d[16] = 1'b1;

    // output TX FIFO
    logic tx_full, tx_empty;
    assign out_proj_r = !tx_full;
    assign fifo_out_v = !tx_empty;

    tx_fifo #(.DATA_WIDTH(17), .ADDR_WIDTH(4)) u_tx_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_data(out_proj_d), .wr_en(out_proj_v && !tx_full), .full(tx_full), .if_last(out_proj_last),
        .rd_data(fifo_out_d), .rd_en(fifo_out_r && !tx_empty), .empty(tx_empty)
    );

endmodule
