`timescale 1ns / 1ps
`include "bf16_test_utils.svh"

// Exercises Belinay's qk_array through Taha's mod_a_wrapper with a tiny
// SIZE=2, DEPTH=2 case chosen so QK^T comes out equal to Q itself:
//   Q = [[1,2],[3,4]],  K = [[1,0],[0,1]]  (K acts as identity)
//   sum_out[row][col] = sum_d Q[row][d]*K[col][d] = Q[row][col]
// Input order per qk_array's own header comment: for each depth d, send
// SIZE (q,k) pairs row by row.
module tb_qk_array;

    localparam int SIZE = 2;
    localparam int DEPTH = 2;

    logic clk_i = 0, rst_ni = 0;
    logic s_valid;
    logic [16:0] s_q_data, s_k_data;
    logic s_ready;
    logic m_valid;
    logic [16:0] m_data;
    logic m_ready;
    logic m_last;

    real q_mat [0:SIZE-1][0:DEPTH-1] = '{'{1.0, 2.0}, '{3.0, 4.0}};
    real k_mat [0:SIZE-1][0:DEPTH-1] = '{'{1.0, 0.0}, '{0.0, 1.0}};

    real expected [0:SIZE-1][0:SIZE-1];
    real got [0:SIZE*SIZE-1];
    int got_cnt;
    int errors = 0;

    mod_a_wrapper #(.SIZE(SIZE), .DEPTH(DEPTH)) dut (
        .clk_i(clk_i), .rst_ni(rst_ni),
        .s_valid(s_valid), .s_q_data(s_q_data), .s_k_data(s_k_data), .s_ready(s_ready),
        .m_valid(m_valid), .m_data(m_data), .m_ready(m_ready)
    );

    always #5 clk_i = ~clk_i;
    assign m_ready = 1'b1;

    always @(posedge clk_i) begin
        if (rst_ni && m_valid && m_ready) begin
            if (got_cnt < SIZE*SIZE) got[got_cnt] = bf16_to_real(m_data[15:0]);
            got_cnt++;
        end
    end

    initial begin
        for (int row = 0; row < SIZE; row++)
            for (int col = 0; col < SIZE; col++) begin
                expected[row][col] = 0.0;
                for (int d = 0; d < DEPTH; d++) expected[row][col] += q_mat[row][d]*k_mat[col][d];
            end

        rst_ni = 0;
        s_valid = 0; s_q_data = '0; s_k_data = '0;
        got_cnt = 0;
        repeat (3) @(posedge clk_i);
        rst_ni = 1;
        @(posedge clk_i);

        for (int d = 0; d < DEPTH; d++) begin
            for (int row = 0; row < SIZE; row++) begin
                wait (s_ready == 1'b1);
                s_q_data = {1'b0, real_to_bf16(q_mat[row][d])};
                s_k_data = {1'b0, real_to_bf16(k_mat[row][d])};
                s_valid = 1'b1;
                @(posedge clk_i);
            end
        end
        s_valid = 1'b0;

        wait (got_cnt >= SIZE*SIZE);
        @(posedge clk_i);

        for (int row = 0; row < SIZE; row++) begin
            for (int col = 0; col < SIZE; col++) begin
                real g = got[row*SIZE+col];
                if (g != expected[row][col]) begin
                    errors++;
                    $display("FAIL sum_out[%0d][%0d]: expected %0f got %0f", row, col, expected[row][col], g);
                end else begin
                    $display("PASS sum_out[%0d][%0d] = %0f", row, col, g);
                end
            end
        end

        if (errors == 0) $display("TB_QK_ARRAY: PASS");
        else $display("TB_QK_ARRAY: FAIL (%0d errors)", errors);
        $finish;
    end

    initial begin
        #200000;
        $display("TB_QK_ARRAY: TIMEOUT");
        $finish;
    end

endmodule
