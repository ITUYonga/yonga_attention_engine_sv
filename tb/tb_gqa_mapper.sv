`timescale 1ns / 1ps

// Exhaustive check: 8 Q heads / 2 KV heads -> heads 0-3 map to KV 0, 4-7 to KV 1.
module tb_gqa_mapper;

    localparam int NUM_Q_HEADS  = 8;
    localparam int NUM_KV_HEADS = 2;

    logic [$clog2(NUM_Q_HEADS)-1:0]  q_head_idx_i;
    logic [$clog2(NUM_KV_HEADS)-1:0] kv_head_idx_o;

    int errors = 0;

    gqa_mapper #(.NUM_Q_HEADS(NUM_Q_HEADS), .NUM_KV_HEADS(NUM_KV_HEADS)) dut (
        .q_head_idx_i(q_head_idx_i),
        .kv_head_idx_o(kv_head_idx_o)
    );

    initial begin
        for (int q = 0; q < NUM_Q_HEADS; q++) begin
            int expected;
            q_head_idx_i = q[$clog2(NUM_Q_HEADS)-1:0];
            #1;
            expected = q / (NUM_Q_HEADS / NUM_KV_HEADS);
            if (kv_head_idx_o !== expected[$clog2(NUM_KV_HEADS)-1:0]) begin
                errors++;
                $display("FAIL q_head=%0d: expected kv=%0d got kv=%0d", q, expected, kv_head_idx_o);
            end else begin
                $display("PASS q_head=%0d -> kv_head=%0d", q, kv_head_idx_o);
            end
        end

        if (errors == 0) $display("TB_GQA_MAPPER: PASS");
        else $display("TB_GQA_MAPPER: FAIL (%0d errors)", errors);
        $finish;
    end

endmodule
