`timescale 1ns / 1ps

module mod_a_wrapper #(
    parameter int SIZE = 4,
    parameter int DEPTH = 64 // Matrisin iç çarpım derinliği (D_MODEL)
)(
    input  logic        clk_i,
    input  logic        rst_ni,

    // ==========================================
    // 1. DATAFLOW GİRİŞİ (Otoyoldan Gelen)
    // ==========================================
    input  logic        s_valid,
    input  logic [16:0] s_q_data, // 17-Bit (Tag + BF16)
    input  logic [16:0] s_k_data, // 17-Bit (Tag + BF16)
    output logic        s_ready,

    // ==========================================
    // 2. DATAFLOW ÇIKIŞI (Otoyola Giden)
    // ==========================================
    output logic        m_valid,
    output logic [16:0] m_data,   // 17-Bit (Tag + BF16)
    input  logic        m_ready
);

    localparam int TOTAL_INPUT_PAIRS = SIZE * DEPTH;

    // --- İç Sinyaller ---
    logic        enable_i;
    logic        start_i;
    logic        last_i;
    logic        ready_o;
    logic        valid_o;
    logic        done_o;
    logic        busy_o;
    logic [16:0] sum_out [0:SIZE-1][0:SIZE-1];

    // --- Giriş Sayacı (Auto Start/Last Üretici) ---
    logic [$clog2(TOTAL_INPUT_PAIRS)-1:0] in_cnt;

    // --- Çıkış Serileştirici (Serializer) Sinyalleri ---
    logic        drain_active;
    logic [$clog2(SIZE)-1:0] row_cnt;
    logic [$clog2(SIZE)-1:0] col_cnt;
    logic [16:0] out_buffer [0:SIZE-1][0:SIZE-1];

    // ==========================================
    // MODÜL A (qk_array) INSTANCE
    // ==========================================
    qk_array #(
        .SIZE(SIZE)
    ) u_qk_array (
        .clk_i    (clk_i),
        .rst_ni   (rst_ni),
        .enable_i (enable_i),
        .start_i  (start_i),
        .valid_i  (s_valid && !drain_active), // Boşaltım (drain) varken içeri veri alma
        .last_i   (last_i),
        .q_in     (s_q_data),
        .k_in     (s_k_data),
        .sum_out  (sum_out),
        .ready_o  (ready_o),
        .valid_o  (valid_o),
        .done_o   (done_o),
        .busy_o   (busy_o)
    );

    // ==========================================
    // GİRİŞ KONTROL MANTIĞI
    // ==========================================
    // Çıkış boşaltılırken (drain_active) Modül A'yı dondur.
    assign enable_i = !drain_active; 
    
    // Hazır sinyalini otoyola ilet
    assign s_ready  = ready_o && !drain_active;

    // Sayaca bağlı dinamik sinyaller
    assign start_i = (in_cnt == '0) && s_valid;
    assign last_i  = (in_cnt == TOTAL_INPUT_PAIRS - 1) && s_valid;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            in_cnt <= '0;
        end else if (s_valid && s_ready) begin
            if (in_cnt == TOTAL_INPUT_PAIRS - 1)
                in_cnt <= '0;
            else
                in_cnt <= in_cnt + 1'b1;
        end
    end

    // ==========================================
    // ÇIKIŞ SERİLEŞTİRİCİ MANTIĞI (Serializer)
    // ==========================================
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            drain_active <= 1'b0;
            row_cnt      <= '0;
            col_cnt      <= '0;
        end else begin
            // Modül A çarpımı bitirdiğinde devasa matrisi yakala
            if (done_o) begin
                drain_active <= 1'b1;
                out_buffer   <= sum_out;
                row_cnt      <= '0;
                col_cnt      <= '0;
            end 
            // Yakalanan matrisi seri olarak otoyola aktar
            else if (drain_active && m_ready) begin
                if (col_cnt == SIZE - 1) begin
                    col_cnt <= '0;
                    if (row_cnt == SIZE - 1) begin
                        drain_active <= 1'b0; // Tüm elemanlar gönderildi, boşaltım bitti
                        row_cnt      <= '0;
                    end else begin
                        row_cnt <= row_cnt + 1'b1;
                    end
                end else begin
                    col_cnt <= col_cnt + 1'b1;
                end
            end
        end
    end

    // Otoyola Çıkış Atamaları
    assign m_valid = drain_active;
    assign m_data  = drain_active ? out_buffer[row_cnt][col_cnt] : 17'b0;

endmodule