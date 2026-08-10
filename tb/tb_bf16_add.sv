`timescale 1ns / 1ps
`include "bf16_test_utils.svh"

// Standalone check for Belinay's bf16_add: exact-representable operands only
// (integers/halves) so there is no rounding ambiguity to argue about.
module tb_bf16_add;

    logic clk_i = 0, rst_ni = 0;
    logic valid_i;
    logic [16:0] a_in, b_in;
    logic [16:0] result_sum_o;
    logic valid_o;

    int errors = 0;
    int checked = 0;

    bf16_add dut (
        .clk_i(clk_i), .rst_ni(rst_ni), .valid_i(valid_i),
        .a_in(a_in), .b_in(b_in),
        .result_sum_o(result_sum_o), .valid_o(valid_o)
    );

    always #5 clk_i = ~clk_i;

    task automatic check_add(real a, real b);
        real expected, got;
        begin
            a_in = {1'b0, real_to_bf16(a)};
            b_in = {1'b0, real_to_bf16(b)};
            valid_i = 1'b1;
            @(posedge clk_i);
            valid_i = 1'b0;
            wait (valid_o == 1'b1);
            @(negedge clk_i);
            expected = a + b;
            got = bf16_to_real(result_sum_o[15:0]);
            checked++;
            if (got != expected) begin
                errors++;
                $display("FAIL add(%0f, %0f): expected %0f got %0f (tag=%0b)", a, b, expected, got, result_sum_o[16]);
            end else begin
                $display("PASS add(%0f, %0f) = %0f", a, b, got);
            end
            @(posedge clk_i);
        end
    endtask

    initial begin
        rst_ni = 0;
        valid_i = 0;
        a_in = '0; b_in = '0;
        repeat (3) @(posedge clk_i);
        rst_ni = 1;
        @(posedge clk_i);

        check_add(1.0, 2.0);
        check_add(4.0, -1.0);
        check_add(-3.0, -5.0);
        check_add(0.5, 0.5);
        check_add(8.0, 8.0);
        check_add(0.0, 7.0);
        check_add(-2.0, 2.0);

        $display("---- tb_bf16_add: %0d checked, %0d failed ----", checked, errors);
        if (errors == 0) $display("TB_BF16_ADD: PASS");
        else $display("TB_BF16_ADD: FAIL");
        $finish;
    end

endmodule
