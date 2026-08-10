`timescale 1ns / 1ps

module dbuf_read_streamer #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 10,
    parameter SEQ_LENGTH = 1024, // Dbuf içindeki toplam veri sayısı
    parameter D_MODEL    = 64    // Can'ın beklediği Token (Vektör) uzunluğu
)(
    input  logic                  clk, 
    input  logic                  rst_n,

    // -----------------------------------------
    // DBUF Arayüzü (Bellek Okuma)
    // -----------------------------------------
    input  logic                  swap_buffers, 
    output logic [ADDR_WIDTH-1:0] read_addr,
    input  logic [DATA_WIDTH-1:0] read_data,    // 1 cycle gecikmeli gelir

    // -----------------------------------------
    // Modül B'ye Giden Doğrudan Veri Yolu
    // -----------------------------------------
    output logic                  m_valid,
    output logic [DATA_WIDTH-1:0] m_data,
    output logic                  m_last,       // Can'ın token_last_i pini için
    input  logic                  m_ready
);

    // Artık Q, K, V diye ayrı ayrı okumaya gerek yok, sadece TEK BİR okuma durumu var
    typedef enum logic {
        IDLE      = 1'b0, 
        STREAMING = 1'b1
    } state_t;
    
    state_t state;
    
    logic [ADDR_WIDTH-1:0]      addr_cnt;
    logic [$clog2(D_MODEL)-1:0] token_cnt; 
    logic                       read_en;

    // Gecikme Yönetimi İçin Pipeline Yazmaçları
    logic pipe_valid, pipe_last;

    // =========================================================
    // ADRES ÜRETİCİ VE TEK TUR SAYACI
    // =========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= IDLE;
            addr_cnt   <= '0;
            token_cnt  <= '0;
            pipe_valid <= 1'b0;
            pipe_last  <= 1'b0;
        end else begin
            
            if (m_ready || !m_valid) begin
                pipe_valid <= read_en;
                // Token sayacı D_MODEL - 1'e ulaştıysa bu token'ın son verisidir
                pipe_last  <= (token_cnt == D_MODEL - 1); 
            end

            case (state)
                IDLE: begin
                    // Yeni veri paketi geldiğinde akışı başlat
                    if (swap_buffers) begin
                        state     <= STREAMING;
                        addr_cnt  <= '0;
                        token_cnt <= '0;
                    end
                end
                
                STREAMING: begin
                    if (m_ready || !m_valid) begin 
                        
                        // İç Token Sayacı (Her D_MODEL elemanda bir sıfırlanır)
                        if (token_cnt == D_MODEL - 1)
                            token_cnt <= '0;
                        else
                            token_cnt <= token_cnt + 1'b1;

                        // Dış Matris Sayacı (Tüm veri bittiğinde tek tur atıp durur)
                        if (addr_cnt == SEQ_LENGTH - 1) begin
                            addr_cnt <= '0; 
                            state    <= IDLE; // Sadece 1 tur okudu ve durdu!
                        end else begin
                            addr_cnt <= addr_cnt + 1'b1;
                        end
                    end
                end
            endcase
        end
    end

    assign read_en   = (state == STREAMING);
    assign read_addr = addr_cnt;

    // =========================================================
    // ÇIKIŞ YAZMAÇLARI (Bellek Gecikmesi Senkronizasyonu)
    // =========================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_valid <= 1'b0;
            m_last  <= 1'b0;
            m_data  <= '0;
        end else if (m_ready || !m_valid) begin
            m_valid <= pipe_valid;
            m_last  <= pipe_last; 
            m_data  <= read_data; 
        end
    end

endmodule
