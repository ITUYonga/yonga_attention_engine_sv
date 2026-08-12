`timescale 1ns / 1ps
`include "bf16_test_utils.svh"
`include "bf16_bitexact_utils.svh"

// End-to-end testbench for attention_engine_top (rtl/top.sv), covering the
// Q.K^T half of the pipeline: AXI-Stream token input -> QKV projection ->
// RoPE -> ping-pong URAM -> shared systolic array (Q.K^T pass) -> scale ->
// softmax. It checks the 16 normalized attention weights softmax produces
// (dut.c_v / dut.c_d) against a bit-exact golden model that replays the
// same bf16 arithmetic the RTL does, using the same ROM/LUT files.
//
// Deliberately out of scope: the score.V pass and the output projection
// that follow softmax. Tracing the design turned up a second, separate
// ordering mismatch there (softmax streams a row's scores out in the same
// order Module A produced them, but the array's second pass over the
// shared systolic grid needs them, and the paired V elements, in a
// different order), which needs its own fix and its own testbench. See the
// report's Current Limitations section.
//
// Scenario: exactly NUM_TOKENS=4 tokens, matching mod_a_wrapper's SIZE=4
// tile hardcoded in top.sv (this is a hard requirement, not a test choice:
// the shared array can only complete one Q.K^T operation per SIZE tokens).
// D_MODEL is left at the RTL's default of 64 so RoPE's existing
// rope_sin_rom.mem / rope_cos_rom.mem stay valid (they are addressed
// pos*D_MODEL/2 + pair_idx for a D_MODEL=64 layout).
//
// Token t is the standard basis vector e_t (1.0 at dim t, 0 elsewhere).
// Wq is the identity matrix, Wk is 2*identity, so Q_proj[t] = token t and
// K_proj[t] = 2*token t exactly, before RoPE rotates each token's single
// nonzero pair by its position's angle. This keeps the projection and
// Q.K^T stages easy to reason about (off-diagonal scores are exactly zero,
// only the diagonal is a real, RoPE-rotated bf16 value) while still
// exercising the real hardware datapath: the full 64-step MAC loop, the
// full RoPE ROM, and the full 64-deep systolic accumulation.
module tb_top;

    localparam int D_MODEL     = 64;
    localparam int PAIR_CNT    = D_MODEL / 2;
    localparam int MAX_POS     = 512;
    localparam int NUM_TOKENS  = 4;              // must equal mod_a_wrapper's SIZE in top.sv
    localparam int MATRIX_SIZE = NUM_TOKENS * D_MODEL;
    localparam int DBUF_ADDR_WIDTH = 10;

    logic clk = 1'b0;
    logic rst_n = 1'b0;
    logic select_bf16 = 1'b0;

    logic [31:0] s_axis_tdata;
    logic        s_axis_tvalid, s_axis_tready, s_axis_tlast;

    logic [31:0] m_axis_tdata;
    logic        m_axis_tvalid, m_axis_tready, m_axis_tlast;

    logic [15:0] w_data_i;
    logic [$clog2(D_MODEL*D_MODEL)-1:0] w_addr_i;
    logic w_we_q, w_we_k, w_we_v, w_we_o;

    attention_engine_top #(
        .DATA_WIDTH_f32(32),
        .DATA_WIDTH_bf16(16),
        .DBUF_ADDR_WIDTH(DBUF_ADDR_WIDTH),
        .D_MODEL(D_MODEL),
        .MATRIX_SIZE(MATRIX_SIZE),
        .MAX_POS(MAX_POS)
    ) dut (
        .clk(clk), .rst_n(rst_n), .select_bf16(select_bf16),
        .s_axis_tdata(s_axis_tdata), .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready), .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata), .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast),
        .w_data_i(w_data_i), .w_addr_i(w_addr_i),
        .w_we_q(w_we_q), .w_we_k(w_we_k), .w_we_v(w_we_v), .w_we_o(w_we_o)
    );

    always #5 clk = ~clk;

    // Not exercised by this test (see file header); just keep it drained
    // so nothing upstream of it ever blocks on m_axis backpressure.
    assign m_axis_tready = 1'b1;

    // golden-model ROMs/LUTs, loaded from the same files the DUT reads
    logic [15:0] sin_rom        [0:MAX_POS*PAIR_CNT-1];
    logic [15:0] cos_rom        [0:MAX_POS*PAIR_CNT-1];
    logic [15:0] exp_table_rom  [0:255];
    logic [6:0]  recip_table_rom[0:127];

    initial begin
        $readmemh("rtl/projection/rope_sin_rom.mem", sin_rom);
        $readmemh("rtl/projection/rope_cos_rom.mem", cos_rom);
        $readmemh("rtl/softmax/exp_table.hex", exp_table_rom);
        $readmemh("rtl/softmax/recip_table.hex", recip_table_rom);
    end

    task automatic rope_pair(
        input  logic [15:0] x1, input logic [15:0] x2,
        input  logic [15:0] cos_val, input logic [15:0] sin_val,
        output logic [15:0] y1, output logic [15:0] y2
    );
        logic [15:0] tmp1, tmp3, mulr;
        begin
            tmp1 = bf16_mul_bits(x1, cos_val);
            mulr = bf16_mul_bits(x2, sin_val);
            y1   = bf16_add_bits(tmp1, {~mulr[15], mulr[14:0]}); // subtract via sign flip, same as rope.sv
            tmp3 = bf16_mul_bits(x1, sin_val);
            mulr = bf16_mul_bits(x2, cos_val);
            y2   = bf16_add_bits(tmp3, mulr);
        end
    endtask

    logic [15:0] token_vec      [0:NUM_TOKENS-1][0:D_MODEL-1];
    logic [15:0] wq_bits        [0:D_MODEL-1][0:D_MODEL-1];
    logic [15:0] wk_bits        [0:D_MODEL-1][0:D_MODEL-1];
    logic [15:0] q_proj_bits    [0:NUM_TOKENS-1][0:D_MODEL-1];
    logic [15:0] k_proj_bits    [0:NUM_TOKENS-1][0:D_MODEL-1];
    logic [15:0] q_rope_bits    [0:NUM_TOKENS-1][0:D_MODEL-1];
    logic [15:0] k_rope_bits    [0:NUM_TOKENS-1][0:D_MODEL-1];
    logic [15:0] raw_score      [0:NUM_TOKENS-1][0:NUM_TOKENS-1];
    logic [15:0] scaled_score   [0:NUM_TOKENS-1][0:NUM_TOKENS-1];
    logic [15:0] exp_score      [0:NUM_TOKENS-1][0:NUM_TOKENS-1];
    logic [15:0] row_sum        [0:NUM_TOKENS-1];
    logic [15:0] row_recip      [0:NUM_TOKENS-1];
    logic [15:0] expected_weight[0:NUM_TOKENS-1][0:NUM_TOKENS-1];

    logic [15:0] ONE_BF16, TWO_BF16, ZERO_BF16, SCALE_BF16;

    task automatic build_golden_model();
        int t, out_i, in_i, d, p, row, col;
        logic [15:0] acc, prod, cosv, sinv, y1, y2;
        logic         lut_is_special;
        logic [15:0]  lut_special_val;
        logic [6:0]   recip_mant_in;
        logic         recip_sign_in;
        logic [7:0]   recip_expo_out;
        begin
            ONE_BF16   = real_to_bf16(1.0);
            TWO_BF16   = real_to_bf16(2.0);
            ZERO_BF16  = real_to_bf16(0.0);
            SCALE_BF16 = real_to_bf16(0.125); // matches top.sv's hardcoded scale_factor (16'h3E00)

            for (t = 0; t < NUM_TOKENS; t++) begin
                for (d = 0; d < D_MODEL; d++) begin
                    token_vec[t][d] = (d == t) ? ONE_BF16 : ZERO_BF16;
                end
            end

            for (out_i = 0; out_i < D_MODEL; out_i++) begin
                for (in_i = 0; in_i < D_MODEL; in_i++) begin
                    wq_bits[out_i][in_i] = (out_i == in_i) ? ONE_BF16 : ZERO_BF16;
                    wk_bits[out_i][in_i] = (out_i == in_i) ? TWO_BF16 : ZERO_BF16;
                end
            end

            // y = W @ x, replaying qkv_proj.sv's MAC accumulation order exactly
            for (t = 0; t < NUM_TOKENS; t++) begin
                for (out_i = 0; out_i < D_MODEL; out_i++) begin
                    acc = 16'b0;
                    for (in_i = 0; in_i < D_MODEL; in_i++) begin
                        prod = bf16_mul_bits(wq_bits[out_i][in_i], token_vec[t][in_i]);
                        acc  = bf16_add_bits(acc, prod);
                    end
                    q_proj_bits[t][out_i] = acc;

                    acc = 16'b0;
                    for (in_i = 0; in_i < D_MODEL; in_i++) begin
                        prod = bf16_mul_bits(wk_bits[out_i][in_i], token_vec[t][in_i]);
                        acc  = bf16_add_bits(acc, prod);
                    end
                    k_proj_bits[t][out_i] = acc;
                end
            end

            // RoPE, position = token index (pos_cnt starts at 0 and increments
            // once per completed token, so token t lands at position t)
            for (t = 0; t < NUM_TOKENS; t++) begin
                for (p = 0; p < PAIR_CNT; p++) begin
                    cosv = cos_rom[t*PAIR_CNT + p];
                    sinv = sin_rom[t*PAIR_CNT + p];
                    rope_pair(q_proj_bits[t][p], q_proj_bits[t][p+PAIR_CNT], cosv, sinv, y1, y2);
                    q_rope_bits[t][p]          = y1;
                    q_rope_bits[t][p+PAIR_CNT] = y2;
                    rope_pair(k_proj_bits[t][p], k_proj_bits[t][p+PAIR_CNT], cosv, sinv, y1, y2);
                    k_rope_bits[t][p]          = y1;
                    k_rope_bits[t][p+PAIR_CNT] = y2;
                end
            end

            // Q.K^T, replaying qk_pe.sv's running accumulate exactly
            for (row = 0; row < NUM_TOKENS; row++) begin
                for (col = 0; col < NUM_TOKENS; col++) begin
                    acc = 16'b0;
                    for (d = 0; d < D_MODEL; d++) begin
                        prod = bf16_mul_bits(q_rope_bits[row][d], k_rope_bits[col][d]);
                        acc  = bf16_add_bits(acc, prod);
                    end
                    raw_score[row][col] = acc;
                end
            end

            // scale (masking disabled, en_scale_mask=0 in top.sv), then softmax
            // per row, replaying softmax.sv's ACCUMULATE/INVERT/DIVIDE_NORMALIZE
            // sequence (elements arrive col-major within a row, exp values
            // accumulate new+running in that same order)
            for (row = 0; row < NUM_TOKENS; row++) begin
                acc = 16'b0;
                for (col = 0; col < NUM_TOKENS; col++) begin
                    scaled_score[row][col] = bf16_mul_bits(raw_score[row][col], SCALE_BF16);

                    exp_lut_special(scaled_score[row][col], lut_is_special, lut_special_val);
                    if (lut_is_special)
                        exp_score[row][col] = lut_special_val;
                    else
                        exp_score[row][col] = exp_table_rom[scaled_score[row][col][7:0]];

                    acc = bf16_add_bits(exp_score[row][col], acc);
                end
                row_sum[row] = acc;

                recip_lut_exponent(acc, lut_is_special, lut_special_val, recip_mant_in, recip_sign_in, recip_expo_out);
                if (lut_is_special)
                    row_recip[row] = lut_special_val;
                else
                    row_recip[row] = {recip_sign_in, recip_expo_out, recip_table_rom[recip_mant_in]};

                for (col = 0; col < NUM_TOKENS; col++) begin
                    expected_weight[row][col] = bf16_mul_bits(exp_score[row][col], row_recip[row]);
                end
            end
        end
    endtask

    int errors = 0;
    int got_count = 0;

    // driver
    initial begin
        int out_i, in_i, t, d;

        build_golden_model();
        $display("[%0t] DEBUG golden model built", $time);

        rst_n = 1'b0; select_bf16 = 1'b0;
        s_axis_tvalid = 1'b0; s_axis_tlast = 1'b0; s_axis_tdata = '0;
        w_data_i = '0; w_addr_i = '0;
        w_we_q = 1'b0; w_we_k = 1'b0; w_we_v = 1'b0; w_we_o = 1'b0;
        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);
        $display("[%0t] DEBUG reset released", $time);

        // load Wq = identity
        for (out_i = 0; out_i < D_MODEL; out_i++) begin
            for (in_i = 0; in_i < D_MODEL; in_i++) begin
                w_addr_i <= out_i * D_MODEL + in_i;
                w_data_i <= (out_i == in_i) ? ONE_BF16 : ZERO_BF16;
                w_we_q   <= 1'b1;
                @(posedge clk);
            end
        end
        w_we_q <= 1'b0;
        @(posedge clk);
        $display("[%0t] DEBUG Wq loaded", $time);

        // load Wk = 2*identity
        for (out_i = 0; out_i < D_MODEL; out_i++) begin
            for (in_i = 0; in_i < D_MODEL; in_i++) begin
                w_addr_i <= out_i * D_MODEL + in_i;
                w_data_i <= (out_i == in_i) ? TWO_BF16 : ZERO_BF16;
                w_we_k   <= 1'b1;
                @(posedge clk);
            end
        end
        w_we_k <= 1'b0;
        @(posedge clk);
        $display("[%0t] DEBUG Wk loaded", $time);

        // Wv/Wo deliberately left unloaded: qkv_proj's FSM and bf16_mul/
        // bf16_add's valid_o timing depend only on valid_i, never on operand
        // values, so an unloaded weight_mem (X) only ever corrupts V/output
        // NUMBERS this test does not check, never the handshake timing the
        // Q.K^T -> softmax path this test does check depends on.

        // stream 4 tokens in, one bf16-exact FP32 element per cycle. token t
        // is the standard basis vector e_t: 1.0 at dim t, 0.0 everywhere else
        for (t = 0; t < NUM_TOKENS; t++) begin
            for (d = 0; d < D_MODEL; d++) begin
                s_axis_tdata  <= (d == t) ? 32'h3F800000 : 32'h00000000;
                s_axis_tvalid <= 1'b1;
                s_axis_tlast  <= (t == NUM_TOKENS-1) && (d == D_MODEL-1);
                do begin
                    @(posedge clk);
                end while (!(s_axis_tvalid && s_axis_tready));
            end
            $display("[%0t] DEBUG token %0d sent", $time, t);
        end
        s_axis_tvalid <= 1'b0;
        s_axis_tlast  <= 1'b0;
        $display("[%0t] DEBUG all tokens streamed", $time);
    end

    // internal pipeline tracer: watches softmax.sv's own FSM (state,
    // add_valid_o, last_d1/last_d2, element_count, fifo empty/full,
    // mul_valid_o) end to end
    logic [1:0] sm_state_prev;
    initial sm_state_prev = 2'bxx;
    initial begin
        forever begin
            @(posedge clk);
            if (dut.u_mod_c_softmax.u_softmax.current_state !== sm_state_prev) begin
                $display("[%0t] DEBUG softmax STATE %0d -> %0d  add_busy=%b last_d1=%b last_d2=%b element_count=%0d fifo_empty=%b fifo_full=%b sum_acc=%h",
                    $time, sm_state_prev, dut.u_mod_c_softmax.u_softmax.current_state,
                    dut.u_mod_c_softmax.u_softmax.add_busy,
                    dut.u_mod_c_softmax.u_softmax.last_d1,
                    dut.u_mod_c_softmax.u_softmax.last_d2,
                    dut.u_mod_c_softmax.u_softmax.element_count,
                    dut.u_mod_c_softmax.u_softmax.fifo_empty,
                    dut.u_mod_c_softmax.u_softmax.fifo_full,
                    dut.u_mod_c_softmax.u_softmax.sum_acc);
                sm_state_prev = dut.u_mod_c_softmax.u_softmax.current_state;
            end
            if (dut.u_mod_c_softmax.u_softmax.add_valid_o) begin
                $display("[%0t] DEBUG softmax add_valid_o add_result=%h last_d1=%b last_d2=%b state=%0d element_count=%0d",
                    $time, dut.u_mod_c_softmax.u_softmax.add_result,
                    dut.u_mod_c_softmax.u_softmax.last_d1,
                    dut.u_mod_c_softmax.u_softmax.last_d2,
                    dut.u_mod_c_softmax.u_softmax.current_state,
                    dut.u_mod_c_softmax.u_softmax.element_count);
            end
            if (dut.u_mod_c_softmax.u_softmax.mul_valid_o) begin
                $display("[%0t] DEBUG softmax mul_valid_o mul_result=%h state=%0d fifo_empty=%b element_count=%0d",
                    $time, dut.u_mod_c_softmax.u_softmax.mul_result,
                    dut.u_mod_c_softmax.u_softmax.current_state,
                    dut.u_mod_c_softmax.u_softmax.fifo_empty,
                    dut.u_mod_c_softmax.u_softmax.element_count);
            end
            if (dut.c_in_v)
                $display("[%0t] DEBUG c_in_v=1 c_in_d=%h c_in_last=%b c_in_r=%b", $time, dut.c_in_d, dut.c_in_last, dut.c_in_r);
            if (dut.c_v)
                $display("[%0t] DEBUG c_v=1 c_d=%h c_r=%b", $time, dut.c_d, dut.c_r);
        end
    end

    // checker: watches softmax's final output (dut.c_v / dut.c_d) directly,
    // since m_axis is not reached by this test's scope
    initial begin
        int row, col;
        logic [15:0] got;
        longint cyc;

        row = 0; col = 0; cyc = 0;
        while (got_count < NUM_TOKENS*NUM_TOKENS) begin
            @(posedge clk);
            cyc++;
            if (cyc % 20000 == 0)
                $display("[%0t] DEBUG heartbeat: %0d cycles waited, %0d/%0d weights so far", $time, cyc, got_count, NUM_TOKENS*NUM_TOKENS);
            if (dut.c_v) begin
                got = dut.c_d[15:0];
                if (got !== expected_weight[row][col]) begin
                    errors++;
                    $display("FAIL weight[%0d][%0d]: expected %h (%f) got %h (%f)",
                        row, col,
                        expected_weight[row][col], bf16_to_real(expected_weight[row][col]),
                        got, bf16_to_real(got));
                end else begin
                    $display("PASS weight[%0d][%0d] = %h (%f)", row, col, got, bf16_to_real(got));
                end
                got_count++;
                if (col == NUM_TOKENS-1) begin
                    col = 0;
                    row = row + 1;
                end else begin
                    col = col + 1;
                end
            end
        end

        if (errors == 0) $display("TB_TOP: PASS (%0d/%0d attention weights matched)", got_count, NUM_TOKENS*NUM_TOKENS);
        else $display("TB_TOP: FAIL (%0d errors)", errors);
        $finish;
    end

    initial begin
        #5000000;
        $display("TB_TOP: TIMEOUT (%0d/%0d attention weights received)", got_count, NUM_TOKENS*NUM_TOKENS);
        $finish;
    end

endmodule
