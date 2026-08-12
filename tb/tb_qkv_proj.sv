`timescale 1ns / 1ps
`include "bf16_test_utils.svh"

// y = W*x with a tiny D_MODEL=4 / D_OUT=2 matrix, all-integer values so bf16
// arithmetic is exact and the expected result can be computed in plain real
// math without worrying about rounding.
//
//   x        = [1, 2, 3, 4]
//   W row 0  = [1, 1, 1, 1]  -> y0 = 10
//   W row 1  = [2, 0, 0, 0]  -> y1 = 2
module tb_qkv_proj;

    localparam int DATA_WIDTH = 16;
    localparam int D_MODEL = 4;
    localparam int D_OUT = 2;

    logic clk_i = 0, rst_ni = 0;

    logic [DATA_WIDTH-1:0] w_data_i;
    logic [$clog2(D_MODEL*D_OUT)-1:0] w_addr_i;
    logic w_we_i;

    logic [DATA_WIDTH:0] x_data_i;
    logic x_valid_i, x_last_i;
    logic x_ready_o;

    logic [DATA_WIDTH:0] y_data_o;
    logic y_valid_o, y_last_o;
    logic y_ready_i;
    logic busy_o;

    real x_vec [0:D_MODEL-1] = '{1.0, 2.0, 3.0, 4.0};
    real w_mat [0:D_OUT-1][0:D_MODEL-1] = '{'{1.0, 1.0, 1.0, 1.0}, '{2.0, 0.0, 0.0, 0.0}};
    real expected_y [0:D_OUT-1];

    int errors = 0;
    int out_idx;

    qkv_proj #(.DATA_WIDTH(DATA_WIDTH), .D_MODEL(D_MODEL), .D_OUT(D_OUT)) dut (
        .clk_i(clk_i), .rst_ni(rst_ni),
        .w_data_i(w_data_i), .w_addr_i(w_addr_i), .w_we_i(w_we_i),
        .x_data_i(x_data_i), .x_valid_i(x_valid_i), .x_last_i(x_last_i), .x_ready_o(x_ready_o),
        .y_data_o(y_data_o), .y_valid_o(y_valid_o), .y_last_o(y_last_o), .y_ready_i(y_ready_i),
        .busy_o(busy_o)
    );

    always #5 clk_i = ~clk_i;

    initial begin
        for (int o = 0; o < D_OUT; o++) begin
            expected_y[o] = 0.0;
            for (int m = 0; m < D_MODEL; m++) expected_y[o] += w_mat[o][m] * x_vec[m];
        end

        rst_ni = 0;
        w_we_i = 0; w_data_i = '0; w_addr_i = '0;
        x_valid_i = 0; x_last_i = 0; x_data_i = '0;
        y_ready_i = 1;
        repeat (3) @(posedge clk_i);
        rst_ni = 1;
        @(posedge clk_i);

        // load weights: address = out_idx*D_MODEL + mac_idx
        // Nonblocking so the DUT's own always_ff (which samples w_addr_i/
        // w_data_i/w_we_i at this same edge) can never race a blocking
        // assign made right before @(posedge) -- see tb_bf16_add.sv.
        for (int o = 0; o < D_OUT; o++) begin
            for (int m = 0; m < D_MODEL; m++) begin
                w_addr_i <= (o*D_MODEL+m);
                w_data_i <= real_to_bf16(w_mat[o][m]);
                w_we_i <= 1'b1;
                @(posedge clk_i);
            end
        end
        w_we_i <= 1'b0;
        @(posedge clk_i);

        // stream the token vector in, one element per cycle
        for (int m = 0; m < D_MODEL; m++) begin
            wait (x_ready_o == 1'b1);
            x_data_i <= {1'b0, real_to_bf16(x_vec[m])};
            x_valid_i <= 1'b1;
            x_last_i <= (m == D_MODEL-1);
            @(posedge clk_i);
        end
        x_valid_i <= 1'b0;
        x_last_i <= 1'b0;

        // drain the output vector, checking against the expected values
        out_idx = 0;
        while (out_idx < D_OUT) begin
            @(posedge clk_i);
            if (y_valid_o) begin
                real got;
                got = bf16_to_real(y_data_o[DATA_WIDTH-1:0]);
                if (got != expected_y[out_idx]) begin
                    errors++;
                    $display("FAIL y[%0d]: expected %0f got %0f (tag=%0b)", out_idx, expected_y[out_idx], got, y_data_o[DATA_WIDTH]);
                end else begin
                    $display("PASS y[%0d] = %0f", out_idx, got);
                end
                out_idx++;
            end
        end

        if (errors == 0) $display("TB_QKV_PROJ: PASS");
        else $display("TB_QKV_PROJ: FAIL (%0d errors)", errors);
        $finish;
    end

    initial begin
        #100000;
        $display("TB_QKV_PROJ: TIMEOUT");
        $finish;
    end

endmodule
