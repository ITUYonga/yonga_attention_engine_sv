`timescale 1ns / 1ps

// Serial-input, output-stationary QK^T array.
//
// For each depth index d, send SIZE Q/K pairs in this order:
//   pair 0: q_in = Q[0][d], k_in = K[0][d]
//   pair 1: q_in = Q[1][d], k_in = K[1][d]
//   ...
//   pair SIZE-1: q_in = Q[SIZE-1][d], k_in = K[SIZE-1][d]
//
// After SIZE accepted pairs, all SIZE*SIZE PEs process the buffered depth
// slice in parallel:
//   sum_out[row][col] += Q[row][d] * K[col][d]
//
// enable_i behavior:
//   enable_i = 1: serial input and new PE dispatch are allowed.
//   enable_i = 0: no new input is accepted and no new slice is dispatched.
//                 An arithmetic operation already inside a PE is allowed to
//                 finish. Its valid/done event is saved and presented when
//                 enable_i returns to 1, so the FSM cannot miss completion.
module qk_array #(
    parameter int SIZE = 4
)(
    input  logic        clk_i,
    input  logic        rst_ni,

    // Controlled by the external FSM.
    input  logic        enable_i,
    input  logic        start_i,
    input  logic        valid_i,
    input  logic        last_i,

    // Bit 16 is the unchanged FSM/source tag.
    // Bits 15:0 contain BF16 data.
    input  logic [16:0] q_in,
    input  logic [16:0] k_in,

    // Complete QK^T matrix. Bit 16 carries the operation tag.
    output logic [16:0] sum_out [0:SIZE-1][0:SIZE-1],

    // The module can accept one serial q_in/k_in pair when high.
    output logic        ready_o,

    // Pulses once for every completed depth slice.
    output logic        valid_o,

    // Pulses when the final depth slice has completed.
    output logic        done_o,

    // High from an accepted start_i until the final done event is delivered.
    output logic        busy_o
);

    localparam int INDEX_W  = (SIZE <= 1) ? 1 : $clog2(SIZE);
    localparam int PE_COUNT = SIZE * SIZE;

    logic [16:0] q_buffer [0:SIZE-1];
    logic [16:0] k_buffer [0:SIZE-1];

    logic [INDEX_W-1:0] load_index;
    logic               buffer_full;
    logic               buffered_last;

    logic operation_active;
    logic first_slice_pending;
    logic final_slice_dispatched;
    logic operation_tag;
    logic operation_tag_valid;

    logic pe_dispatch;
    logic pe_start;
    logic dispatched_last;

    logic [PE_COUNT-1:0] pe_ready_vector;
    logic [PE_COUNT-1:0] pe_done_vector;
    logic [15:0]         pe_sum_matrix [0:SIZE-1][0:SIZE-1];

    logic pe_all_ready;
    logic pe_all_done;

    logic slice_inflight;
    logic slice_seen_busy;
    logic slice_complete_event;

    logic valid_pending;
    logic done_pending;

    logic start_allowed;
    logic serial_accept;

    assign pe_all_ready = &pe_ready_vector;
    assign pe_all_done  = &pe_done_vector;

    assign busy_o = operation_active || valid_pending || done_pending;

    // A new operation cannot overwrite a completion event that the FSM has
    // not consumed yet.
    assign start_allowed = !busy_o;

    // ready_o is forced low while the FSM pauses the interface.
    assign ready_o = enable_i &&
                     operation_active &&
                     !buffer_full &&
                     !final_slice_dispatched;

    // The first serial pair may be supplied together with start_i.
    assign serial_accept = enable_i && valid_i &&
                           ((start_i && start_allowed) ||
                            (!start_i && ready_o));

    // Dispatch only a complete buffered slice and only while enabled.
    assign pe_dispatch = enable_i &&
                         operation_active &&
                         buffer_full &&
                         !slice_inflight &&
                         !final_slice_dispatched &&
                         (first_slice_pending || pe_all_ready);

    assign pe_start = pe_dispatch && first_slice_pending;

    // Do not treat the old ready=1 level on the dispatch edge as completion.
    // At least one cycle with PEs busy must first be observed.
    assign slice_complete_event = slice_inflight &&
                                  slice_seen_busy &&
                                  pe_all_ready;

    integer i;

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            load_index             <= '0;
            buffer_full            <= 1'b0;
            buffered_last          <= 1'b0;
            operation_active       <= 1'b0;
            first_slice_pending    <= 1'b0;
            final_slice_dispatched <= 1'b0;
            operation_tag          <= 1'b0;
            operation_tag_valid    <= 1'b0;

            slice_inflight         <= 1'b0;
            slice_seen_busy        <= 1'b0;
            dispatched_last        <= 1'b0;

            valid_pending          <= 1'b0;
            done_pending           <= 1'b0;
            valid_o                <= 1'b0;
            done_o                 <= 1'b0;

            for (i = 0; i < SIZE; i = i + 1) begin
                q_buffer[i] <= 17'b0;
                k_buffer[i] <= 17'b0;
            end
        end
        else begin
            // Output handshakes are one-cycle pulses.
            valid_o <= 1'b0;
            done_o  <= 1'b0;

            // If the FSM paused while a slice completed, present the saved
            // event once it enables the array again.
            if (enable_i && valid_pending) begin
                valid_o       <= 1'b1;
                valid_pending <= 1'b0;

                if (done_pending) begin
                    done_o           <= 1'b1;
                    done_pending     <= 1'b0;
                    operation_active <= 1'b0;
                end
            end

            // Start a new matrix operation. start_i may carry pair zero.
            if (enable_i && start_i && start_allowed) begin
                load_index             <= '0;
                buffer_full            <= 1'b0;
                buffered_last          <= 1'b0;
                operation_active       <= 1'b1;
                first_slice_pending    <= 1'b1;
                final_slice_dispatched <= 1'b0;
                operation_tag_valid    <= 1'b0;

                slice_inflight         <= 1'b0;
                slice_seen_busy        <= 1'b0;
                dispatched_last        <= 1'b0;

                if (valid_i) begin
                    q_buffer[0] <= q_in;
                    k_buffer[0] <= k_in;

                    operation_tag       <= q_in[16];
                    operation_tag_valid <= 1'b1;
                    buffered_last       <= last_i;

                    if (SIZE == 1) begin
                        load_index  <= '0;
                        buffer_full <= 1'b1;
                    end
                    else begin
                        load_index <= {{(INDEX_W-1){1'b0}}, 1'b1};
                    end
                end
            end
            else begin
                // Load one serial Q/K pair whenever the FSM presents a valid
                // transfer while ready_o is high.
                if (serial_accept) begin
                    q_buffer[load_index] <= q_in;
                    k_buffer[load_index] <= k_in;
                    buffered_last        <= buffered_last || last_i;

                    if (!operation_tag_valid) begin
                        operation_tag       <= q_in[16];
                        operation_tag_valid <= 1'b1;
                    end

                    if (load_index == SIZE-1) begin
                        load_index  <= '0;
                        buffer_full <= 1'b1;
                    end
                    else begin
                        load_index <= load_index + 1'b1;
                    end
                end

                // All PEs consume the buffered depth slice on this edge.
                if (pe_dispatch) begin
                    buffer_full         <= 1'b0;
                    buffered_last       <= 1'b0;
                    load_index          <= '0;
                    first_slice_pending <= 1'b0;

                    slice_inflight  <= 1'b1;
                    slice_seen_busy <= 1'b0;
                    dispatched_last <= buffered_last;

                    if (buffered_last) begin
                        final_slice_dispatched <= 1'b1;
                    end
                end

                if (slice_inflight && !pe_all_ready) begin
                    slice_seen_busy <= 1'b1;
                end

                // One complete depth slice has been accumulated by all PEs.
                if (slice_complete_event) begin
                    slice_inflight  <= 1'b0;
                    slice_seen_busy <= 1'b0;

                    if (enable_i) begin
                        valid_o <= 1'b1;

                        if (dispatched_last) begin
                            done_o           <= 1'b1;
                            operation_active <= 1'b0;
                        end
                    end
                    else begin
                        valid_pending <= 1'b1;

                        if (dispatched_last) begin
                            done_pending <= 1'b1;
                        end
                    end
                end

                // pe_all_done is an extra consistency path for the final
                // transaction. It prevents a final done event from being lost
                // if PE-ready timing changes during later revisions.
                if (pe_all_done && dispatched_last) begin
                    if (enable_i) begin
                        valid_o          <= 1'b1;
                        done_o           <= 1'b1;
                        operation_active <= 1'b0;
                    end
                    else begin
                        valid_pending <= 1'b1;
                        done_pending  <= 1'b1;
                    end
                end
            end
        end
    end

    genvar row;
    genvar col;

    generate
        for (row = 0; row < SIZE; row = row + 1) begin : GEN_ROWS
            for (col = 0; col < SIZE; col = col + 1) begin : GEN_COLS
                localparam int PE_INDEX = row * SIZE + col;

                logic [16:0] q_unused;
                logic [16:0] k_unused;
                logic        q_valid_unused;
                logic        k_valid_unused;
                logic        q_last_unused;
                logic        k_last_unused;

                qk_pe u_qk_pe (
                    .clk_i      (clk_i),
                    .rst_ni     (rst_ni),
                    .start_i    (pe_start),

                    .q_in       (q_buffer[row]),
                    .k_in       (k_buffer[col]),
                    .q_valid_i  (pe_dispatch),
                    .k_valid_i  (pe_dispatch),
                    .q_last_i   (buffered_last),
                    .k_last_i   (buffered_last),

                    .q_out      (q_unused),
                    .k_out      (k_unused),
                    .q_valid_o  (q_valid_unused),
                    .k_valid_o  (k_valid_unused),
                    .q_last_o   (q_last_unused),
                    .k_last_o   (k_last_unused),

                    .ready_o    (pe_ready_vector[PE_INDEX]),
                    .pe_sum_out (pe_sum_matrix[row][col]),
                    .done_o     (pe_done_vector[PE_INDEX])
                );

                // The PE arithmetic result is 16-bit BF16. The array restores
                // the unchanged operation tag in output bit 16.
                assign sum_out[row][col] = {
                    operation_tag,
                    pe_sum_matrix[row][col]
                };
            end
        end
    endgenerate

endmodule
