`timescale 1ns / 1ps

module uram_read_controller #(
    parameter ADDR_WIDTH = 10,
    parameter MATRIX_SIZE = 1024
)(
    input  logic clk,
    input  logic rst_n,

    // Dış Kontrol (FSM veya Top-Level'dan gelir)
    input  logic start_qk, // Q x K^T işlemi için oku
    input  logic start_sv, // Softmax x V işlemi için oku

    // URAM_Q Okuma Portu
    output logic                  q_ren,
    output logic [ADDR_WIDTH-1:0] q_raddr,

    // URAM_K Okuma Portu
    output logic                  k_ren,
    output logic [ADDR_WIDTH-1:0] k_raddr,

    // URAM_V Okuma Portu
    output logic                  v_ren,
    output logic [ADDR_WIDTH-1:0] v_raddr,

    // Durum Çıkışları
    output logic busy
);

    typedef enum logic [1:0] {IDLE, READ_QK, READ_V} state_t;
    state_t state;
    logic [ADDR_WIDTH-1:0] addr_cnt;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= IDLE;
            addr_cnt <= '0;
        end else begin
            case (state)
                IDLE: begin
                    addr_cnt <= '0;
                    if (start_qk)      state <= READ_QK;
                    else if (start_sv) state <= READ_V;
                end
                
                READ_QK, READ_V: begin
                    if (addr_cnt == MATRIX_SIZE - 1) begin
                        state    <= IDLE;
                        addr_cnt <= '0;
                    end else begin
                        addr_cnt <= addr_cnt + 1'b1;
                    end
                end
            endcase
        end
    end

    // Adres ve Enable Atamaları
    assign q_ren   = (state == READ_QK);
    assign q_raddr = addr_cnt;

    assign k_ren   = (state == READ_QK);
    assign k_raddr = addr_cnt;

    assign v_ren   = (state == READ_V);
    assign v_raddr = addr_cnt;

    assign busy    = (state != IDLE);

endmodule
