`timescale 1ns / 1ps

// FUNC_QK_PE : One multiply accumulate cell of the shared systolic array
//
//   Purpose:  Computes one running dot product term, pe_sum_out plus
//             equals q_in times k_in, every time a fresh q_in and k_in
//             pair arrives. Internally this is a tiny state machine
//             (IDLE, MUL, ADD) that issues one bf16_mul then feeds its
//             result into one bf16_add together with the running sum,
//             since neither of those units is 0 cycle anymore. start_i
//             clears the running sum for a new matrix operation, done_o
//             pulses once the accumulation that included a last flagged
//             pair has finished.
//
//   parameters:
//           none
//
//   inputs:
//           clk_i:        clock
//           rst_ni:       active low reset
//           start_i:      begin a new accumulation, clears pe_sum_out,
//                         may arrive on the same cycle as the first
//                         valid q_in/k_in pair
//           q_in:         one Q element, bit 16 is the source tag, bits
//                         15 down to 0 are the bf16 value
//           k_in:         one K element, same layout as q_in
//           q_valid_i:    q_in is valid this cycle
//           k_valid_i:    k_in is valid this cycle
//           q_last_i:     this q_in is the last term of the dot product
//           k_last_i:     this k_in is the last term of the dot product
//   output:
//           q_out, k_out, q_valid_o, k_valid_o, q_last_o, k_last_o:
//                         one cycle delayed forwarding of the inputs,
//                         kept for interface compatibility with the
//                         original design, the serial buffered qk_array
//                         does not use these
//           ready_o:      high while idle, able to accept a new pair
//           pe_sum_out:   the running dot product sum so far, 16 bit bf16
//           done_o:       pulses one cycle after a pair flagged as the
//                         last term finished accumulating
//
//   notes:
//           q_valid_i and k_valid_i are expected to always agree, an
//           assertion (simulation only) checks that q_in and k_in carry
//           the same source tag bit whenever a multiply is issued, since
//           mixing tags across the two operands would mean the array is
//           being fed two unrelated data streams at once

module qk_pe (
    input  logic        clk_i,
    input  logic        rst_ni,
    input  logic        start_i,

    input  logic [16:0] q_in,
    input  logic [16:0] k_in,
    input  logic        q_valid_i,
    input  logic        k_valid_i,
    input  logic        q_last_i,
    input  logic        k_last_i,

    output logic [16:0] q_out,
    output logic [16:0] k_out,
    output logic        q_valid_o,
    output logic        k_valid_o,
    output logic        q_last_o,
    output logic        k_last_o,

    output logic        ready_o,
    output logic [15:0] pe_sum_out,
    output logic        done_o
);

    // Forwarding registers are retained for compatibility with the original
    // PE interface. The serial-buffered qk_array does not use these outputs.
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            q_out     <= 17'b0;
            k_out     <= 17'b0;
            q_valid_o <= 1'b0;
            k_valid_o <= 1'b0;
            q_last_o  <= 1'b0;
            k_last_o  <= 1'b0;
        end
        else begin
            q_out     <= q_in;
            k_out     <= k_in;
            q_valid_o <= q_valid_i;
            k_valid_o <= k_valid_i;
            q_last_o  <= q_last_i;
            k_last_o  <= k_last_i;
        end
    end

    typedef enum logic [1:0] {
        IDLE,
        MUL,
        ADD
    } state_t;

    state_t state;
    state_t state_next;

    logic [15:0] sum_reg;
    logic        last_pend;

    logic        mul_valid_i;
    logic [16:0] mul_result;
    logic        mul_valid_o;

    logic        add_valid_i;
    logic [16:0] add_result;
    logic        add_valid_o;

    bf16_mul u_bf16_mul (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .valid_i      (mul_valid_i),
        .a_in         (q_in),
        .b_in         (k_in),
        .result_mul_o (mul_result),
        .valid_o      (mul_valid_o)
    );

    bf16_add u_bf16_add (
        .clk_i        (clk_i),
        .rst_ni       (rst_ni),
        .valid_i      (add_valid_i),
        .a_in         ({mul_result[16], sum_reg}),
        .b_in         (mul_result),
        .result_sum_o (add_result),
        .valid_o      (add_valid_o)
    );

    assign mul_valid_i = (state == IDLE) && q_valid_i && k_valid_i;
    assign add_valid_i = (state == MUL)  && mul_valid_o;

    assign ready_o    = (state == IDLE);
    assign pe_sum_out = sum_reg;

    always_comb begin
        state_next = state;

        case (state)
            IDLE: begin
                if (q_valid_i && k_valid_i) begin
                    state_next = MUL;
                end
            end

            MUL: begin
                if (mul_valid_o) begin
                    state_next = ADD;
                end
            end

            ADD: begin
                if (add_valid_o) begin
                    state_next = IDLE;
                end
            end

            default: begin
                state_next = IDLE;
            end
        endcase
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state     <= IDLE;
            sum_reg   <= 16'b0;
            last_pend <= 1'b0;
            done_o    <= 1'b0;
        end
        else if (start_i) begin
            // start_i may be asserted with the first valid Q/K pair.
            sum_reg <= 16'b0;
            done_o  <= 1'b0;

            if (q_valid_i && k_valid_i) begin
                state     <= MUL;
                last_pend <= q_last_i && k_last_i;
            end
            else begin
                state     <= IDLE;
                last_pend <= 1'b0;
            end
        end
        else begin
            state  <= state_next;
            done_o <= 1'b0;

            if ((state == IDLE) && q_valid_i && k_valid_i) begin
                last_pend <= q_last_i && k_last_i;
            end

            if ((state == ADD) && add_valid_o) begin
                sum_reg <= add_result[15:0];

                if (last_pend) begin
                    done_o <= 1'b1;
                end
            end
        end
    end

`ifndef SYNTHESIS
    always_ff @(posedge clk_i) begin
        if (rst_ni && mul_valid_i) begin
            assert (q_in[16] == k_in[16])
                else $error("qk_pe: Q and K tag bits differ");
        end
    end
`endif

endmodule
