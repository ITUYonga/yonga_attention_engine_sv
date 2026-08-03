`timescale 1ns / 1ps

// ASSUMPTION (confirm with teammates before using):
// exp_lut, bf16_add, recip_lut, bf16_mul are now internally pipelined and
// each exposes a clk/rst_n/in_valid/out_valid handshake:
//   input  in_valid   - pulse: operand(s) are valid this cycle, start op
//   output out_valid  - pulse: result is valid this cycle
// If your teammates' actual ports differ (e.g. fixed LATENCY parameter
// instead of a valid signal), tell me and this only needs a small change,
// not a rewrite.
//
// NOTE ON THROUGHPUT: row-sum accumulation is inherently serial (each add
// needs the previous running sum), so this version processes one element
// at a time through exp -> add, waiting for each stage's out_valid before
// issuing the next. That's a correctness-first design, not the fastest
// possible one — acceptable for now, revisit later if timing/throughput
// numbers demand a reduction-tree accumulator instead.

module softmax #(
    parameter DATA_WIDTH = 16
    )(
        input  logic clk,
        input  logic rst_n,

      //AXI-4 Slave (scale_mask --> softmax)
        input  logic [DATA_WIDTH-1:0]    s_axis_tdata,
        input  logic                     s_axis_tvalid,
        output logic                     s_axis_tready,
        input  logic                     s_axis_tlast,

      //AXI-4 Master (softmax --> Score x V)
        output logic [DATA_WIDTH-1:0]    m_axis_tdata,
        output logic                     m_axis_tvalid,
        input  logic                     m_axis_tready,
        output logic                     m_axis_tlast
     );

//2) FSM state enumeration & internal signals
    typedef enum logic [3:0] {
        IDLE,            // waiting for a new element to arrive
        EXP_WAIT,        // waiting for exp_lut's out_valid
        ADD_WAIT,        // waiting for bf16_add's out_valid
        INVERT_ISSUE,    // kick off recip_lut for the row's final sum
        INVERT_WAIT,     // waiting for recip_lut's out_valid
        NORMALIZE_ISSUE, // pop one element from fifo, kick off bf16_mul
        NORMALIZE_WAIT   // waiting for bf16_mul's out_valid, then output it
    } state_type;

    state_type current_state, next_state;

    logic [DATA_WIDTH-1:0]    sum_acc;
    logic [DATA_WIDTH-1:0]    inv_sum;
    logic [DATA_WIDTH-1:0]    sum_next;
    logic [DATA_WIDTH-1:0]    recip_out;

    logic                     held_last;      // was the in-flight element the row's last?
    logic [15:0]              element_count;  // # elements stored in fifo, awaiting normalize

    logic [DATA_WIDTH-1:0]    exp_data;
    logic [DATA_WIDTH-1:0]    fifo_dout;
    logic                     fifo_empty, fifo_full;

    logic exp_out_valid, add_out_valid, recip_out_valid, mul_out_valid;
    logic mul_result_valid;
    logic [DATA_WIDTH-1:0] mul_result;

//3) sub-block instantiations (in_valid/out_valid handshake assumed)
    exp_lut exp_val (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (current_state == IDLE && s_axis_tvalid && s_axis_tready),
        .x_in      (s_axis_tdata),
        .out_valid (exp_out_valid),
        .y_out     (exp_data)
    );

    bf16_add accu (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (current_state == EXP_WAIT && exp_out_valid),
        .a_in      (exp_data),
        .b_in      (sum_acc),
        .out_valid (add_out_valid),
        .result_sum_o (sum_next)
    );

    recip_lut recip_inst (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (current_state == INVERT_ISSUE),
        .x_in      (sum_acc),
        .out_valid (recip_out_valid),
        .y_out     (recip_out)
    );

    bf16_mul norm_multiplier (
        .clk       (clk),
        .rst_n     (rst_n),
        .in_valid  (current_state == NORMALIZE_ISSUE && !fifo_empty),
        .a_in      (fifo_dout),
        .b_in      (inv_sum),
        .out_valid (mul_out_valid),
        .result_mul_o (mul_result)
    );

    row_fifo mem_buffer (
        .clk    (clk),
        .rst_n  (rst_n),
        .wr_data(exp_data),
        .wr_en  (current_state == EXP_WAIT && exp_out_valid),
        .rd_en  (current_state == NORMALIZE_ISSUE && !fifo_empty),
        .rd_data(fifo_dout),
        .empty  (fifo_empty),
        .full   (fifo_full)
    );

//4) Sequential register updates
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state  <= IDLE;
            sum_acc        <= '0;
            inv_sum        <= '0;
            held_last      <= '0;
            element_count  <= '0;
            m_axis_tvalid  <= 1'b0;
            m_axis_tlast   <= 1'b0;
            m_axis_tdata   <= '0;
        end else begin
            current_state <= next_state;

            // latch which element was tagged tlast, so ADD_WAIT knows
            // whether to go to INVERT_ISSUE or back to IDLE afterward
            if (current_state == IDLE && s_axis_tvalid && s_axis_tready)
                held_last <= s_axis_tlast;

            // capture accumulator result once bf16_add finishes
            if (current_state == ADD_WAIT && add_out_valid) begin
                sum_acc       <= sum_next;
                element_count <= element_count + 1'b1;
            end

            // capture reciprocal once recip_lut finishes; reset sum_acc
            // for cleanliness (not reused until next row anyway)
            if (current_state == INVERT_WAIT && recip_out_valid) begin
                inv_sum <= recip_out;
                sum_acc <= '0;
            end

            // register the normalized output once bf16_mul finishes
            m_axis_tvalid <= 1'b0; // default low unless set below
            if (current_state == NORMALIZE_WAIT && mul_out_valid) begin
                m_axis_tdata  <= mul_result;
                m_axis_tvalid <= 1'b1;
                m_axis_tlast  <= (element_count == 16'd1);
                element_count <= element_count - 1'b1;
            end
        end
    end

//5) Next-state logic
    always_comb begin
        next_state = current_state;
        case (current_state)
            IDLE:
                if (s_axis_tvalid && s_axis_tready) next_state = EXP_WAIT;

            EXP_WAIT:
                if (exp_out_valid) next_state = ADD_WAIT;

            ADD_WAIT:
                if (add_out_valid) next_state = held_last ? INVERT_ISSUE : IDLE;

            INVERT_ISSUE:
                next_state = INVERT_WAIT;

            INVERT_WAIT:
                if (recip_out_valid) next_state = NORMALIZE_ISSUE;

            NORMALIZE_ISSUE:
                next_state = NORMALIZE_WAIT;

            NORMALIZE_WAIT:
                if (mul_out_valid) begin
                    next_state = (element_count == 16'd1) ? IDLE : NORMALIZE_ISSUE;
                end

            default:
                next_state = IDLE;
        endcase
    end

//6) Flow control
    // only accept a new input element while idle and the exp/add pipe
    // for the previous element has fully drained (this design serializes
    // one element at a time through exp->add, see NOTE above)
    assign s_axis_tready = (current_state == IDLE) && !fifo_full;

endmodule
