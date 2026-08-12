`timescale 1ns / 1ps
`include "bf16_test_utils.svh"

// Standalone check for Belinay's bf16_mul: exact-representable operands only
// (powers of two, small integers) so results stay exact in bf16.
module tb_bf16_mul;

    logic clk_i = 0, rst_ni = 0;
    logic valid_i;
    logic [16:0] a_in, b_in;
    logic [16:0] result_mul_o;
    logic valid_o;

    int errors = 0;
    int checked = 0;

    bf16_mul dut (
        .clk_i(clk_i), .rst_ni(rst_ni), .valid_i(valid_i),
        .a_in(a_in), .b_in(b_in),
        .result_mul_o(result_mul_o), .valid_o(valid_o)
    );

    always #5 clk_i = ~clk_i;

    // Nonblocking, clock-synchronized stimulus -- see tb_bf16_add.sv for why
    // a blocking-assign-before-@(posedge) pattern here races with the DUT's
    // always_ff reading the same signals at the same edge.
    task automatic check_mul(real a, real b);
        real expected, got;
        begin
            @(posedge clk_i);          // sync point
            a_in <= {1'b0, real_to_bf16(a)};
            b_in <= {1'b0, real_to_bf16(b)};
            valid_i <= 1'b1;
            @(posedge clk_i);          // stage 0 -> stage 1 captures a_in/b_in
            valid_i <= 1'b0;
            @(posedge clk_i);          // stage 1 -> stage 2
            @(posedge clk_i);          // stage 2 -> valid_o/result_mul_o now updated
            #1;                        // let the NBA update settle before reading
            expected = a * b;
            got = bf16_to_real(result_mul_o[15:0]);
            checked++;
            if (got != expected) begin
                errors++;
                $display("FAIL mul(%0f, %0f): expected %0f got %0f (tag=%0b)", a, b, expected, got, result_mul_o[16]);
            end else begin
                $display("PASS mul(%0f, %0f) = %0f", a, b, got);
            end
        end
    endtask

    initial begin
        rst_ni = 0;
        valid_i = 0;
        a_in = '0; b_in = '0;
        repeat (3) @(posedge clk_i);
        rst_ni = 1;

        check_mul(2.0, 3.0);
        check_mul(-2.0, 4.0);
        check_mul(0.5, 0.5);
        check_mul(-1.0, -1.0);
        check_mul(8.0, 0.125);
        check_mul(0.0, 5.0);

        $display("---- tb_bf16_mul: %0d checked, %0d failed ----", checked, errors);
        if (errors == 0) $display("TB_BF16_MUL: PASS");
        else $display("TB_BF16_MUL: FAIL");
        $finish;
    end

endmodule
