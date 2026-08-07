module dbuf_read_streamer #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 10,
    parameter SEQ_LENGTH = 1024,
    parameter D_MODEL    = 64    // Can'ın beklediği Token uzunluğu
)(
    input  logic                  clk, 
    input  logic                  rst_n,

    // DBUF Arayüzü
    input  logic                  swap_buffers, 
    output logic [ADDR_WIDTH-1:0] read_addr,
    input  logic [DATA_WIDTH-1:0] read_data,

    // Modül B Hakemine Giden Dataflow Arayüzü
    output logic                  m_valid,
    output logic [DATA_WIDTH-1:0] m_data,
    output logic                  m_last,       // YENİ: Can'ın x_last_i pini için
    input  logic                  m_ready
);

    typedef enum logic [1:0] {READ_Q, READ_K, READ_V, IDLE} state_t;
    state_t state;
    
    logic [ADDR_WIDTH-1:0] addr_cnt;
    logic [$clog2(D_MODEL)-1:0] token_cnt; // YENİ: Her bir token içindeki elemanları sayar
    logic read_en;
    
    // Gecikme Yönetimi İçin Pipeline Yazmaçları
    logic pipe_valid, pipe_last;

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
                // Eğer token sayacı D_MODEL - 1'e ulaştıysa bu token'ın son verisidir
                pipe_last  <= (token_cnt == D_MODEL - 1); 
            end

            case (state)
                IDLE: begin
                    if (swap_buffers) begin
                        state     <= READ_Q;
                        addr_cnt  <= '0;
                        token_cnt <= '0;
                    end
                end
                
                READ_Q, READ_K, READ_V: begin
                    if (m_ready || !m_valid) begin 
                        
                        // İç Token Sayacı (Her 64 elemanda bir sıfırlanır)
                        if (token_cnt == D_MODEL - 1)
                            token_cnt <= '0;
                        else
                            token_cnt <= token_cnt + 1'b1;

                        // Dış Matris Sayacı (1024 elemanda bir makas değiştirir)
                        if (addr_cnt == SEQ_LENGTH - 1) begin
                            addr_cnt <= '0; 
                            
                            if (state == READ_Q)      state <= READ_V; // Senin istediğin Q -> V -> K sırası
                            else if (state == READ_V) state <= READ_K;
                            else                      state <= IDLE;   
                        end else begin
                            addr_cnt <= addr_cnt + 1'b1;
                        end
                    end
                end
            endcase
        end
    end

    assign read_en   = (state != IDLE);
    assign read_addr = addr_cnt;

    // Çıkış Yazmaçları
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_valid <= 1'b0;
            m_last  <= 1'b0;
            m_data  <= '0;
        end else if (m_ready || !m_valid) begin
            m_valid <= pipe_valid;
            m_last  <= pipe_last; // Can'ın modülüne 1 saat vuruşu senkronize olarak gider
            m_data  <= read_data; 
        end
    end

endmodule