`timescale 1ns / 1ps

module attention_engine_top #(
    parameter DATA_WIDTH_f32  = 32,
    parameter DATA_WIDTH_bf16 = 16,
    parameter DBUF_ADDR_WIDTH = 10,
    parameter D_MODEL         = 64,      // Token uzunluğu
    parameter MATRIX_SIZE     = 1024     // Toplam matris uzunluğu
)(
    input  logic clk,
    input  logic rst_n,
    input  logic select_bf16,

    // ==========================================
    // 1. AXI-STREAM GİRİŞ (FP32)
    // ==========================================
    input  logic [DATA_WIDTH_f32-1:0] s_axis_tdata,
    input  logic                      s_axis_tvalid,
    output logic                      s_axis_tready,
    input  logic                      s_axis_tlast,

    // ==========================================
    // 2. AXI-STREAM ÇIKIŞ (FP32)
    // ==========================================
    output logic [DATA_WIDTH_f32-1:0] m_axis_tdata,
    output logic                      m_axis_tvalid,
    input  logic                      m_axis_tready,
    output logic                      m_axis_tlast,

    // ==========================================
    // 3. AĞIRLIK YÜKLEME (AXI-Lite üzerinden Can'ın modülüne bağlanacak)
    // ==========================================
    input  logic [DATA_WIDTH_bf16-1:0] w_data_i,
    input  logic [$clog2(D_MODEL*D_MODEL)-1:0] w_addr_i,
    input  logic                       w_we_i
);

    // =========================================================
    // İÇ KABLO (WIRE) TANIMLAMALARI (Otoyol tamamen 17-Bit)
    // =========================================================
    
    // Dbuf <-> Streamer
    logic [DBUF_ADDR_WIDTH-1:0] dbuf_read_addr;
    logic [DATA_WIDTH_bf16-1:0] dbuf_read_data; 
    
    // Streamer -> Modül B Hakemi
    logic dbuf_v, dbuf_r, dbuf_last;
    logic [DATA_WIDTH_bf16-1:0] dbuf_d; 
    
    // Modül A (Yönlendirici) -> Modül B Hakemi (W0 Dönüş Hattı)
    logic a_w0_v, a_w0_r;
    logic [16:0] a_w0_d; 

    // Modül B Giriş ve Çıkışları
    logic b_in_v, b_in_r, b_in_last; 
    logic [16:0] b_in_d; // Can'ın güncellediği 17-bit giriş
    logic b_out_v, b_out_r, b_out_last; 
    logic [16:0] b_out_d; 
    
    // Modül B Yönlendiricisi -> URAM'ler
    logic q_uram_we, k_uram_we, v_uram_we;
    logic q_uram_rdy, k_uram_rdy, v_uram_rdy;
    logic [16:0] uram_wdata; 
    
    // URAM Okuma Sinyalleri (Dışarıdan bir kontrolcü bağlayana kadar sıfırda tutulur)
    logic q_uram_ren = 1'b0, k_uram_ren = 1'b0, v_uram_ren = 1'b0;
    logic [DBUF_ADDR_WIDTH-1:0] q_uram_raddr = '0, k_uram_raddr = '0, v_uram_raddr = '0;
    logic q_uram_rvalid, k_uram_rvalid, v_uram_rvalid;
    logic [16:0] q_rdata, k_rdata, v_rdata; 

    // Modül C (Softmax) -> Modül A Hakemi
    logic c_v, c_r; 
    logic [16:0] c_d; 

    // Modül A Hakemi -> Modül A (Wrapper)
    logic a_in_v, a_in_r; 
    logic [16:0] a_in_q, a_in_k; 

    // Modül A (Wrapper) -> Modül A Yönlendiricisi
    logic a_out_v, a_out_r; 
    logic [16:0] a_out_d; 

    // Modül A Yönlendiricisi -> Modül C (Softmax)
    logic c_in_v, c_in_r;
    logic [16:0] c_in_d;

    // Modül B Yönlendiricisi -> TX FIFO
    logic fifo_in_v, fifo_in_r; 
    logic [16:0] fifo_in_d; 
    logic fifo_out_v, fifo_out_r; 
    logic [16:0] fifo_out_d; 

    // =========================================================
    // 1. AXI & DBUF (Giriş Alt Sistemi - Ping Pong)
    // =========================================================
    logic dbuf_rx_we, dbuf_rx_full, internal_rx_last;
    logic [DATA_WIDTH_bf16-1:0] dbuf_rx_data;
    logic swap_buffers;

    assign swap_buffers = (dbuf_rx_we && internal_rx_last);

    axi_stream_if #(
        .DATA_WIDTH_bf16(DATA_WIDTH_bf16),
        .DATA_WIDTH_f32(DATA_WIDTH_f32)
    ) u_axi_if (
        .select_bf16(select_bf16),
        .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready), .s_axis_tlast(s_axis_tlast),
        .dbuf_rx_data(dbuf_rx_data), .dbuf_rx_we(dbuf_rx_we), .dbuf_rx_full(dbuf_rx_full), .internal_rx_last(internal_rx_last),
        
        .internal_tx_data(fifo_out_d), .internal_tx_valid(fifo_out_v), .internal_tx_ready(fifo_out_r), 
        .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast)
    );

    dbuf #(
        .DATA_WIDTH(DATA_WIDTH_bf16),
        .ADDR_WIDTH(DBUF_ADDR_WIDTH)
    ) u_dbuf (
        .clk(clk), .rst_n(rst_n),
        .rx_data(dbuf_rx_data), .rx_we(dbuf_rx_we), .rx_full(dbuf_rx_full),
        .read_addr(dbuf_read_addr), .read_data(dbuf_read_data), .swap_buffers(swap_buffers)
    );

    dbuf_read_streamer #(
        .DATA_WIDTH(DATA_WIDTH_bf16), .ADDR_WIDTH(DBUF_ADDR_WIDTH), .SEQ_LENGTH(MATRIX_SIZE), .D_MODEL(D_MODEL)
    ) u_dbuf_streamer (
        .clk(clk), .rst_n(rst_n), .swap_buffers(swap_buffers),
        .read_addr(dbuf_read_addr), .read_data(dbuf_read_data),
        .m_valid(dbuf_v), .m_data(dbuf_d), .m_last(dbuf_last), .m_ready(dbuf_r)
    );

    // =========================================================
    // 2. MODÜL B GİRİŞ HAKEMİ (16 Bit Veriyi 17 Bit'e Çevirir)
    // =========================================================
    always_comb begin
        a_w0_r = 1'b0; dbuf_r = 1'b0; b_in_v = 1'b0; b_in_d = '0; b_in_last = 1'b0;
        
        if (a_w0_v) begin // Otoyolu boşaltmak için öncelik Modül A'dan (W0) dönende
            b_in_v    = 1'b1; 
            b_in_d    = a_w0_d; // a_w0_d halihazırda 17 bittir (16. biti 1)
            b_in_last = 1'b0;   // W0 için last kullanımını kendi ihtiyacınıza göre uyarlayabilirsiniz
            a_w0_r    = b_in_r;
        end else if (dbuf_v) begin // Sonra taze Q/V/K verileri
            b_in_v    = 1'b1; 
            b_in_d    = {1'b0, dbuf_d}; // 16 bitlik verinin yanına (17. bit olarak) 0 bas!
            b_in_last = dbuf_last;
            dbuf_r    = b_in_r;
        end
    end

    // =========================================================
    // 3. CAN'IN MODÜL B'Sİ (qkv_proj - 17 Bit Güncellemeli)
    // =========================================================
    mod_b #(
        .DATA_WIDTH(DATA_WIDTH_bf16), .D_MODEL(D_MODEL), .D_OUT(D_MODEL)
    ) u_mod_b (
        .clk_i(clk), .rst_ni(rst_n),
        .w_data_i(w_data_i), .w_addr_i(w_addr_i), .w_we_i(w_we_i),
        
        .x_data_i(b_in_d),     // Tamamen 17-bit paket olarak giriyor
        .x_valid_i(b_in_v),
        .x_last_i(b_in_last),
        .x_ready_o(b_in_r),

        .y_data_o(b_out_d),    // 17-bit çıkıyor (16. bit tag)
        .y_valid_o(b_out_v),
        .y_last_o(b_out_last),
        .y_ready_i(b_out_r),
        .busy_o()
    );

    // =========================================================
    // 4. MODÜL B ÇIKIŞ YÖNLENDİRİCİSİ (Otomatik Sıralı Q -> V -> K)
    // =========================================================
    mod_b_output_router #(
        .DATA_WIDTH(17), // Kablo 17-bit taşıyor, yönlendirme için b_out_d[16]'ya bakacak
        .MATRIX_SIZE(MATRIX_SIZE)
    ) u_b_router (
        .clk(clk), .rst_n(rst_n),
        .mod_b_valid(b_out_v), .mod_b_data(b_out_d), .mod_b_tag(b_out_d[16]), .mod_b_ready(b_out_r),
        
        .q_uram_valid(q_uram_we), .k_uram_valid(k_uram_we), .v_uram_valid(v_uram_we),
        .uram_data(uram_wdata), 
        .q_uram_ready(q_uram_rdy), .k_uram_ready(k_uram_rdy), .v_uram_ready(v_uram_rdy),
        
        .fifo_valid(fifo_in_v), .fifo_data(fifo_in_d), .fifo_ready(fifo_in_r)
    );

    // =========================================================
    // 5. URAM BELLEKLERİ (UltraRAM - 17 Bit)
    // =========================================================
    uram_memory #(.DATA_WIDTH(17), .ADDR_WIDTH(DBUF_ADDR_WIDTH), .DEPTH(MATRIX_SIZE)) u_uram_q (
        .clk(clk), .rst_n(rst_n), 
        .wr_en(q_uram_we), .wr_data(uram_wdata), .ready(q_uram_rdy), 
        .rd_en(q_uram_ren), .rd_addr(q_uram_raddr), .rdata(q_rdata), .rvalid(q_uram_rvalid)
    );
    uram_memory #(.DATA_WIDTH(17), .ADDR_WIDTH(DBUF_ADDR_WIDTH), .DEPTH(MATRIX_SIZE)) u_uram_k (
        .clk(clk), .rst_n(rst_n), 
        .wr_en(k_uram_we), .wr_data(uram_wdata), .ready(k_uram_rdy), 
        .rd_en(k_uram_ren), .rd_addr(k_uram_raddr), .rdata(k_rdata), .rvalid(k_uram_rvalid)
    );
    uram_memory #(.DATA_WIDTH(17), .ADDR_WIDTH(DBUF_ADDR_WIDTH), .DEPTH(MATRIX_SIZE)) u_uram_v (
        .clk(clk), .rst_n(rst_n), 
        .wr_en(v_uram_we), .wr_data(uram_wdata), .ready(v_uram_rdy), 
        .rd_en(v_uram_ren), .rd_addr(v_uram_raddr), .rdata(v_rdata), .rvalid(v_uram_rvalid)
    );

    // =========================================================
    // 6. MODÜL A GİRİŞ HAKEMİ 
    // =========================================================
    logic qk_valid, qk_ready;
    assign qk_valid = q_uram_rvalid && k_uram_rvalid; // URAM okumaları senkronize edilir

    mod_a_input_arbiter #(.DATA_WIDTH(17)) u_a_arb (
        .clk(clk), .rst_n(rst_n),
        .c_valid(c_v), .c_data(c_d), .c_ready(c_r), .v_rdata(v_rdata), 
        .qk_valid(qk_valid), .q_data(q_rdata), .k_data(k_rdata), .qk_ready(qk_ready), 
        .mod_a_valid(a_in_v), .mod_a_data_q(a_in_q), .mod_a_data_k(a_in_k), .mod_a_ready(a_in_r)
    );

    // =========================================================
    // 7. SENİN MODÜL A WRAPPER'IN (Serileştirici ve Auto-Start/Last)
    // =========================================================
    mod_a_wrapper #(
        .SIZE(4), .DEPTH(D_MODEL)
    ) u_mod_a_wrap (
        .clk_i(clk), .rst_ni(rst_n),
        .s_valid(a_in_v), .s_q_data(a_in_q), .s_k_data(a_in_k), .s_ready(a_in_r),
        .m_valid(a_out_v), .m_data(a_out_d), .m_ready(a_out_r)
    );

    // =========================================================
    // 8. MODÜL A ÇIKIŞ YÖNLENDİRİCİSİ (Yönlendirmeyi 17. bitten okur)
    // =========================================================
    mod_a_output_router #(.DATA_WIDTH(17)) u_a_router (
        .mod_a_valid(a_out_v), .mod_a_data(a_out_d), .mod_a_tag(a_out_d[16]), .mod_a_ready(a_out_r),
        .c_valid(c_in_v), .c_data(c_in_d), .c_ready(c_in_r), 
        .b_valid(a_w0_v), .b_data(a_w0_d), .b_ready(a_w0_r)  
    );

    // =========================================================
    // 9. MODÜL C (Softmax: Scale & Mask)
    // =========================================================
    mod_c u_mod_c_softmax (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tdata(c_in_d[15:0]), // Softmax genelde Tag'e ihtiyaç duymaz, sade veriyi işler
        .s_axis_tvalid(c_in_v), 
        .s_axis_tready(c_in_r),
        
        // Çıkan veriyi tekrar 17-bit Tag (1: W0 Projeksiyonu) ile paketle
        .m_axis_tdata(c_d[15:0]), 
        .m_axis_tvalid(c_v), 
        .m_axis_tready(c_r)
    );
    assign c_d[16] = 1'b1; // Softmax'ten çıkan veri B modülünden geçip dışarı çıkacak

    // =========================================================
    // 10. ÇIKIŞ (TX) FIFO'SU (17-Bit AXI-Stream Arayüzü İçin)
    // =========================================================
    logic tx_full, tx_empty;
    assign fifo_in_r = !tx_full;
    assign fifo_out_v = !tx_empty;

    tx_fifo #(.DATA_WIDTH(17), .ADDR_WIDTH(4)) u_tx_fifo (
        .clk(clk), .rst_n(rst_n),
        .wr_data(fifo_in_d), .wr_en(fifo_in_v && !tx_full), .full(tx_full),
        .rd_data(fifo_out_d), .rd_en(fifo_out_r && !tx_empty), .empty(tx_empty)
    );

endmodule
