`timescale 1ns / 1ps

// FUNC_MOD_A_WRAPPER : Turns qk_array's serial in, parallel out interface
// into a plain valid/ready stream on both sides
//
//   Purpose:  qk_array itself wants DEPTH serial pairs per row times SIZE
//             rows in on one side, and hands back a whole SIZE by SIZE
//             matrix at once on the other. This wrapper hides both of
//             those shapes behind an ordinary one element per cycle
//             valid/ready stream. On the input side, an internal counter
//             watches how many pairs have been accepted and generates
//             qk_array's own start_i and last_i automatically, so the
//             caller just streams SIZE times DEPTH pairs and does not
//             have to track array internals. On the output side, once
//             done_o fires the whole result matrix is latched into
//             out_buffer and then drained out one element at a time in
//             row major order.
//
//   parameters:
//           SIZE:    matches qk_array's own SIZE, the array is SIZE by
//                    SIZE PEs
//           DEPTH:   how many serial pairs make up one row, normally
//                    D_MODEL
//
//   inputs:
//           clk_i, rst_ni:   clock and active low reset
//           s_valid:         s_q_data/s_k_data are valid this cycle
//           s_q_data:        one Q element, 17 bit tagged
//           s_k_data:        one K element, 17 bit tagged
//           m_ready:         downstream can accept an output element
//                            this cycle
//   output:
//           s_ready:         this module can accept one serial pair this
//                            cycle
//           m_valid:         m_data is valid this cycle
//           m_data:          one element of the drained result matrix
//           m_last:          this element is the last one being drained
//
//   notes:
//           enable_i is held low while draining (drain_active) so the
//           array cannot start a new matrix operation until the previous
//           result has been fully read out
//
//           m_last currently only pulses on the very last element of the
//           whole matrix, not once per row. A per row version was tried
//           (asserting m_last whenever col_cnt wraps) so downstream
//           softmax could normalize each row on its own instead of
//           treating the whole matrix as one giant row, but that change
//           introduced a second, not yet diagnosed deadlock between
//           softmax and this drain logic and was reverted

module mod_a_wrapper #(
    parameter int SIZE = 4,
    parameter int DEPTH = 64 // Matrisin iç çarpım derinliği (D_MODEL)
)(
    input  logic        clk_i,
    input  logic        rst_ni,

    // dataflow input
    input  logic        s_valid,
    input  logic [16:0] s_q_data, // 17-bit (tag + bf16)
    input  logic [16:0] s_k_data, // 17-bit (tag + bf16)
    output logic        s_ready,

    // dataflow output
    output logic        m_valid,
    output logic [16:0] m_data,   // 17-Bit (Tag + BF16)
    output logic        m_last,
    input  logic        m_ready
);

    localparam int TOTAL_INPUT_PAIRS = SIZE * DEPTH;

    logic        enable_i;
    logic        start_i;
    logic        last_i;
    logic        ready_o;
    logic        valid_o;
    logic        done_o;
    logic        busy_o;
    logic [16:0] sum_out [0:SIZE-1][0:SIZE-1];

    // input counter, generates start_i/last_i automatically
    logic [$clog2(TOTAL_INPUT_PAIRS)-1:0] in_cnt;

    // output serializer state
    logic        drain_active;
    logic [$clog2(SIZE)-1:0] row_cnt;
    logic [$clog2(SIZE)-1:0] col_cnt;
    logic [16:0] out_buffer [0:SIZE-1][0:SIZE-1];

    // Module A (qk_array) instance
    qk_array #(
        .SIZE(SIZE)
    ) u_qk_array (
        .clk_i    (clk_i),
        .rst_ni   (rst_ni),
        .enable_i (enable_i),
        .start_i  (start_i),
        .valid_i  (s_valid && !drain_active), // Boşaltım (drain) varken içeri veri alma
        .last_i   (last_i),
        .q_in     (s_q_data),
        .k_in     (s_k_data),
        .sum_out  (sum_out),
        .ready_o  (ready_o),
        .valid_o  (valid_o),
        .done_o   (done_o),
        .busy_o   (busy_o)
    );

    // freeze Module A while the previous result is draining
    assign enable_i = !drain_active;
    assign s_ready  = ready_o && !drain_active;

    assign start_i = (in_cnt == '0) && s_valid;
    assign last_i  = (in_cnt == TOTAL_INPUT_PAIRS - 1) && s_valid;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            in_cnt <= '0;
        end else if (s_valid && s_ready) begin
            if (in_cnt == TOTAL_INPUT_PAIRS - 1)
                in_cnt <= '0;
            else
                in_cnt <= in_cnt + 1'b1;
        end
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            drain_active <= 1'b0;
            row_cnt      <= '0;
            col_cnt      <= '0;
        end else begin
            // Module A finished, latch the whole result matrix
            if (done_o) begin
                drain_active <= 1'b1;
                out_buffer   <= sum_out;
                row_cnt      <= '0;
                col_cnt      <= '0;
            end
            // drain the latched matrix out serially
            else if (drain_active && m_ready) begin
                if (col_cnt == SIZE - 1) begin
                    col_cnt <= '0;
                    if (row_cnt == SIZE - 1) begin
                        drain_active <= 1'b0;
                        row_cnt      <= '0;
                    end else begin
                        row_cnt <= row_cnt + 1'b1;
                    end
                end else begin
                    col_cnt <= col_cnt + 1'b1;
                end
            end
        end
    end

    assign m_valid = drain_active;
    assign m_data  = drain_active ? out_buffer[row_cnt][col_cnt] : 17'b0;
    // REVERTED 2026-08-12: tried changing this to pulse once per row
    // (col_cnt==SIZE-1 alone, matching softmax.sv's "end of row" semantics
    // for tlast -- softmax was treating the whole SIZE*SIZE drain as one
    // giant row and normalizing every element by the combined total instead
    // of its own row's sum). That change made softmax correctly finish row
    // 0's ACCUMULATE/INVERT/DIVIDE_NORMALIZE cycle, but it then never
    // returned to IDLE/ACCUMULATE to accept row 1, deadlocking the whole
    // pipeline (scale_mask's single in-flight slot filled and never
    // drained). Root cause of THAT deadlock not yet found. Reverted to the
    // known-non-hanging single-pulse-at-the-very-end behavior so the
    // pipeline still completes; the "softmax treats all 16 scores as one
    // row" numeric bug is still open, tracked separately.
    assign m_last  = drain_active && (row_cnt == SIZE - 1) && (col_cnt == SIZE - 1);
endmodule
