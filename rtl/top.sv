`timescale 1ns / 1ps

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

    // =========================================================
    // İÇ KABLOLAR
    // =========================================================
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

    // =========================================================
    // 1. AXI, DBUF VE STREAMER
    // =========================================================
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

    // =========================================================
    // 2. RoPE POZİSYON SAYACI
    // =========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pos_cnt <= '0;
        end else if (dbuf_v && dbuf_r && dbuf_last) begin
            if (pos_cnt == MAX_POS - 1) pos_cnt <= '0;
            else pos_cnt <= pos_cnt + 1'b1;
        end
    end

    // =========================================================
    // 3. MODÜL B (projection_block) -> HİÇ DURMAZ!
    // =========================================================
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

    // =========================================================
    // 4. URAM PING-PONG YÖNETİCİSİ VE BELLEKLER
    // =========================================================
    logic q_uram_rdy, k_uram_rdy, v_uram_rdy;
    assign q_proj_r = q_uram_rdy;
    assign k_proj_r = k_uram_rdy;
    assign v_proj_r = v_uram_rdy;

    uram_pingpong_controller #(.ADDR_WIDTH(DBUF_ADDR_WIDTH), .MATRIX_SIZE(MATRIX_SIZE)) u_uram_ctrl (
        .clk(clk), .rst_n(rst_n),
        .write_valid(q_proj_v), .write_last(q_proj_last), .waddr(uram_waddr),
        .softmax_valid(c_v), 
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

    // =========================================================
    // 5. MODÜL A HAKEMİ 
    // =========================================================
    logic qk_valid, qk_ready;
    assign qk_valid = q_uram_rvalid && k_uram_rvalid; 

    mod_a_input_arbiter #(.DATA_WIDTH(17)) u_a_arb (
        .clk(clk), .rst_n(rst_n),
        .c_valid(c_v), .c_data(c_d), .c_ready(c_r), .v_rdata(v_rdata),
        .qk_valid(qk_valid), .q_data(q_rdata), .k_data(k_rdata), .qk_ready(qk_ready),
        .mod_a_valid(a_in_v), .mod_a_data_l(a_in_q), .mod_a_data_t(a_in_k), .mod_a_tag(), .mod_a_ready(a_in_r)
    );

    // =========================================================
    // 6. MODÜL A (SİSTOLİK DİZİ) VE WRAPPER
    // =========================================================
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

  // =========================================================
    // 7. MODÜL C (SCALE + MASK + SOFTMAX WRAPPER)
    // =========================================================
    scale_mask_softmax_wrapper #(
        .DATA_WIDTH(16),
        .MAX_ROW_LEN(D_MODEL) 
    ) u_mod_c_softmax (
        .clk(clk), 
        .rst_n(rst_n),

        // --- Hardcoded Konfigürasyon ---
        // 1 / sqrt(64) = 0.125 (BF16 Karşılığı: 16'h3E00)
        .scale_factor(16'h3E00), 
        // GECICI: gercek eleman-bazli causal mask ureticisi henuz yok.
        // 1'b1 (her elemani hep -inf'e maskele) sentezde sabit deger
        // oldugu icin Vivado softmax/Module A'nin buyuk bolumunu "olu
        // mantik" sayip siliyordu (utilization raporunu anlamsizlastiriyordu),
        // ustune ustluk softmax'i da fonksiyonel olarak kirar. 0 (hic
        // maskeleme yok) gercek causal mask gelene kadar daha dogru bir
        // varsayilan.
        .en_scale_mask(1'b0),
        .ext_stall(1'b0),        // Otoyol mantığında dışarıdan durdurma yok

        // --- Giriş AXI-Stream ---
        .s_axis_tdata(c_in_d[15:0]), 
        .s_axis_tvalid(c_in_v), 
        .s_axis_tready(c_in_r),
        .s_axis_tlast(c_in_last),    

        // --- Çıkış AXI-Stream ---
        .m_axis_tdata(c_d[15:0]), 
        .m_axis_tvalid(c_v), 
        .m_axis_tready(c_r),
        .m_axis_tlast(c_out_last)    
    );

    // Çıkan veriyi 17-bit Tag (1: W0 Projeksiyonu Hedefi) ile paketle
    assign c_d[16] = 1'b1;

    // =========================================================
    // 8. ÇIKIŞ FIFO (TX)
    // =========================================================
    logic tx_full, tx_empty;
    assign out_proj_r = !tx_full;
    assign fifo_out_v = !tx_empty;

    tx_fifo #(.DATA_WIDTH(17), .ADDR_WIDTH(4)) u_tx_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_data(out_proj_d), .wr_en(out_proj_v && !tx_full), .full(tx_full), .if_last(out_proj_last),
        .rd_data(fifo_out_d), .rd_en(fifo_out_r && !tx_empty), .empty(tx_empty)
    );

endmodule
