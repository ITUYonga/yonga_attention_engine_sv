`timescale 1ns / 1ps
`include "bf16_test_utils.svh"

// Drives Hasan's scale_mask_softmax_wrapper directly (bypassing top.sv) with
// en_scale_mask driven PER ELEMENT, the way the module's handshake actually
// expects it. This also demonstrates the bug flagged during integration:
// top.sv currently ties en_scale_mask to a constant 1, which (per this same
// RTL) would force every element of every row to -infinity. Driving it
// correctly here shows the module itself is fine; the bug is in how top.sv
// calls it.
module tb_scale_mask_softmax;

    localparam int DATA_WIDTH = 16;
    localparam int MAX_ROW_LEN = 16;
    localparam int ROW_LEN = 4;

    logic clk = 0, rst_n = 0;
    logic [DATA_WIDTH-1:0] scale_factor;
    logic en_scale_mask, ext_stall;
    logic [DATA_WIDTH-1:0] s_axis_tdata;
    logic s_axis_tvalid, s_axis_tready, s_axis_tlast;
    logic [DATA_WIDTH-1:0] m_axis_tdata;
    logic m_axis_tvalid, m_axis_tready, m_axis_tlast;

    real got_row [0:ROW_LEN-1];
    int got_cnt;
    int errors = 0;

    scale_mask_softmax_wrapper #(.DATA_WIDTH(DATA_WIDTH), .MAX_ROW_LEN(MAX_ROW_LEN)) dut (
        .clk(clk), .rst_n(rst_n),
        .scale_factor(scale_factor), .en_scale_mask(en_scale_mask), .ext_stall(ext_stall),
        .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready), .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast)
    );

    always #5 clk = ~clk;
    assign m_axis_tready = 1'b1;

    // Collector: grabs every accepted output element.
    always @(posedge clk) begin
        if (rst_n && m_axis_tvalid && m_axis_tready) begin
            if (got_cnt < ROW_LEN) got_row[got_cnt] = bf16_to_real(m_axis_tdata);
            got_cnt++;
        end
    end

    task automatic run_row(input real x_vals [0:ROW_LEN-1], input bit mask_vals [0:ROW_LEN-1]);
        begin
            got_cnt = 0;
            // Nonblocking so the DUT's own always_ff (sampling s_axis_tdata/
            // s_axis_tvalid at this same edge) can never race a blocking
            // assign made right before @(posedge) -- see tb_bf16_add.sv.
            //
            // Hold each element constant and re-check (valid && ready) at
            // every edge until it is actually accepted, instead of assuming
            // one wait-then-advance is enough -- see tb_qk_array.sv, where
            // that assumption silently dropped a data element whenever
            // tready fell after the wait already succeeded.
            for (int i = 0; i < ROW_LEN; i++) begin
                s_axis_tdata  <= real_to_bf16(x_vals[i]);
                en_scale_mask <= mask_vals[i];
                s_axis_tvalid <= 1'b1;
                s_axis_tlast  <= (i == ROW_LEN-1);
                do begin
                    @(posedge clk);
                end while (!(s_axis_tvalid && s_axis_tready));
            end
            s_axis_tvalid <= 1'b0;
            s_axis_tlast <= 1'b0;
            en_scale_mask <= 1'b0;
            wait (got_cnt >= ROW_LEN);
            @(posedge clk);
        end
    endtask

    initial begin
        real x4 [0:ROW_LEN-1];
        bit no_mask [0:ROW_LEN-1];
        bit last_mask [0:ROW_LEN-1];
        real tol;

        for (int i = 0; i < ROW_LEN; i++) begin x4[i] = 0.0; no_mask[i] = 0; last_mask[i] = 0; end
        last_mask[ROW_LEN-1] = 1;

        rst_n = 0;
        scale_factor = 16'h3F80; // 1.0, scaling is a no-op for this test
        en_scale_mask = 0; ext_stall = 0;
        s_axis_tvalid = 0; s_axis_tlast = 0; s_axis_tdata = '0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        // Test 1: uniform row, no masking -> expect 0.25 each
        run_row(x4, no_mask);
        tol = 0.02;
        for (int i = 0; i < ROW_LEN; i++) begin
            if (!(got_row[i] > 0.25-tol && got_row[i] < 0.25+tol)) begin
                errors++;
                $display("FAIL uniform row[%0d]: expected ~0.25 got %0f", i, got_row[i]);
            end else begin
                $display("PASS uniform row[%0d] = %0f", i, got_row[i]);
            end
        end

        repeat (10) @(posedge clk);

        // Test 2: last element masked -> expect ~0.333 for the first three,
        // ~0.0 for the masked one
        run_row(x4, last_mask);
        tol = 0.03;
        for (int i = 0; i < ROW_LEN-1; i++) begin
            if (!(got_row[i] > (1.0/3.0)-tol && got_row[i] < (1.0/3.0)+tol)) begin
                errors++;
                $display("FAIL masked row[%0d]: expected ~0.333 got %0f", i, got_row[i]);
            end else begin
                $display("PASS masked row[%0d] = %0f", i, got_row[i]);
            end
        end
        if (!(got_row[ROW_LEN-1] > -tol && got_row[ROW_LEN-1] < tol)) begin
            errors++;
            $display("FAIL masked row[%0d] (should be ~0): got %0f", ROW_LEN-1, got_row[ROW_LEN-1]);
        end else begin
            $display("PASS masked row[%0d] (masked) = %0f", ROW_LEN-1, got_row[ROW_LEN-1]);
        end

        if (errors == 0) $display("TB_SCALE_MASK_SOFTMAX: PASS");
        else $display("TB_SCALE_MASK_SOFTMAX: FAIL (%0d errors)", errors);
        $finish;
    end

    initial begin
        #2000000;
        $display("TB_SCALE_MASK_SOFTMAX: TIMEOUT");
        $finish;
    end

endmodule
