// FUNC_QKV_PROJ : Turns one token vector into one projected output vector
// Can Erturk  21.07.2026
//
//   Purpose:  Multiplies the input token vector with a weight matrix and
//             adds the results up to get the projected vector. This is
//             just a matrix times vector operation, done one multiply add
//             at a time with a simple state machine, not a parallel array.
//             Same module gets instantiated 3 times for Q K V projection
//             and once more for the output projection, just with a
//             different weight matrix loaded into each instance.
//
//   parameters:
//           DATA_WIDTH:   bit width of one bf16 number, should stay 16
//           D_MODEL:      length of the input token vector
//           D_OUT:        length of the output vector, can be smaller
//                         than D_MODEL when this instance is used for the
//                         K or V projection in a grouped query attention
//                         setup where K and V have fewer heads than Q
//
//   inputs:
//           clk_i:        clock
//           rst_ni:       active low reset
//           w_data_i:     one weight value to write into the weight memory
//           w_addr_i:     address of the weight value being written
//           w_we_i:       write enable for the weight memory
//           x_data_i:     one element of the input token vector, 17 bits
//                         wide (bit DATA_WIDTH is the source tag from
//                         upstream, bits DATA_WIDTH-1:0 are the bf16
//                         value). the tag is stripped on entry and not
//                         used internally, this module tags its own
//                         outputs with SRC_TAG_CAN
//           x_valid_i:    x_data_i is valid this cycle
//           x_last_i:     x_data_i is the last element of this vector
//           y_ready_i:    downstream can accept an output element this cycle
//   output:
//           x_ready_o:    module can accept an input element this cycle
//           y_data_o:     one projected output element, bit DATA_WIDTH is
//                         the source tag (see below), bits DATA_WIDTH-1:0
//                         are the bf16 value
//           y_valid_o:    y_data_o is valid this cycle
//           y_last_o:     y_data_o is the last element of this vector
//           busy_o:       module is currently loading or computing, only
//                         here to make waveform debug easier
//
//   notes:
//           weight loading has no protection right now, it just assumes
//           the weights get written once before x_valid_i ever goes high
//           and never change again while the module is running. good
//           enough for a first working version, needs a real load/lock
//           handshake later
//
//           weight_mem is read with a plain array index (w_rd_addr) which
//           is an asynchronous read just like the one flagged in Taha's
//           dbuf.sv. same BRAM inference risk applies here, left as is
//           for now since this is the first pass and correctness matters
//           more than resource usage at this point
//
//           source tag (bit DATA_WIDTH of every 17-bit bus): Belinay's
//           bf16_mul/bf16_add (e295872) now carry a 1-bit tag through
//           their pipeline unchanged, used to tell apart which side fed a
//           shared resource (agreed with Belinay: 0 = came from my block,
//           1 = came from hers). x_data_i now arrives as 17 bits but the
//           incoming tag is stripped immediately (only the lower 16 bits
//           are stored in x_mem). whatever this module outputs was
//           computed here, so y_data_o always gets tagged SRC_TAG_CAN.
//           internal MAC operands are tagged SRC_TAG_CAN too, it doesn't
//           matter for the math, the bus width just requires something
//           there.
//
//           bf16_mul and bf16_add are no longer 0-cycle combinational
//           (Belinay's e295872 update made both 2-stage pipelined with a
//           valid_i/valid_o handshake). the MAC loop below issues valid_i
//           and waits for valid_o instead of assuming the result is ready
//           the same cycle, this adds latency per MAC step but is
//           functionally the same multiply-accumulate as before
//
//   changelog:
//           07.08.2026 - x_data_i widened from [DATA_WIDTH-1:0] to
//                        [DATA_WIDTH:0] (16-bit -> 17-bit) so the port
//                        width matches the 17-bit buses used everywhere
//                        else in the design (Belinay's bf16 units, dbuf,
//                        axi_stream_if). the incoming tag bit is stripped
//                        at x_mem write time, internal computation stays
//                        16-bit, output is re-tagged with SRC_TAG_CAN.

module qkv_proj #(
    parameter int DATA_WIDTH = 16,
    parameter int D_MODEL = 64,
    parameter int D_OUT = 64
)(
    input logic clk_i,
    input logic rst_ni,

    // weight load port, simple write only
    input logic [DATA_WIDTH-1:0] w_data_i,
    input logic [$clog2(D_MODEL*D_OUT)-1:0] w_addr_i,
    input logic w_we_i,

    // input token vector, streamed in one element per cycle
    // [DATA_WIDTH]   = source tag from upstream (stripped, not used)
    // [DATA_WIDTH-1:0] = bf16 value (stored in x_mem)
    input logic [DATA_WIDTH:0] x_data_i,
    input logic x_valid_i,
    input logic x_last_i,
    output logic x_ready_o,

    // output vector, streamed out one element per cycle
    output logic [DATA_WIDTH:0] y_data_o,
    output logic y_valid_o,
    output logic y_last_o,
    input logic y_ready_i,

    output logic busy_o
);

    localparam int IN_IDX_W = $clog2(D_MODEL);
    localparam int OUT_IDX_W = $clog2(D_OUT);
    localparam int W_ADDR_W = $clog2(D_MODEL*D_OUT);

    // agreed with Belinay: 0 = this value was produced by my (Can's) block,
    // 1 = produced by hers. only meaningful on a bus shared between us,
    // see note above.
    localparam logic SRC_TAG_CAN = 1'b0;

    // the things this module is doing, one at a time. ST_COMPUTE from the
    // original single-cycle-MAC version is now split into an issue/wait
    // pair for the multiply and another for the add, since both now take
    // multiple cycles to produce a result
    typedef enum logic [2:0] {
        ST_LOAD,      // waiting for / collecting the input vector
        ST_MUL_ISSUE, // pulse valid_i into bf16_mul for the current mac step
        ST_MUL_WAIT,  // wait for bf16_mul's valid_o
        ST_ADD_ISSUE, // pulse valid_i into bf16_add for the current mac step
        ST_ADD_WAIT,  // wait for bf16_add's valid_o
        ST_DRAIN      // streaming the finished output vector out
    } state_t;

    state_t state_q, state_d;

    // memories
    logic [DATA_WIDTH-1:0] weight_mem [0:D_MODEL*D_OUT-1];
    logic [DATA_WIDTH-1:0] x_mem [0:D_MODEL-1];
    logic [DATA_WIDTH-1:0] y_mem [0:D_OUT-1];

    // counters
    logic [IN_IDX_W-1:0] in_ptr_q, in_ptr_d; // which x element we are loading
    logic [OUT_IDX_W-1:0] out_idx_q, out_idx_d; // which output element we are computing
    logic [IN_IDX_W-1:0] mac_idx_q, mac_idx_d; // which input element the mac loop is on
    logic [OUT_IDX_W-1:0] drain_ptr_q, drain_ptr_d; // which output element we are sending out

    logic [DATA_WIDTH-1:0] acc_q, acc_d; // running sum for the current output element
    logic [DATA_WIDTH-1:0] prod_q, prod_d; // product captured out of bf16_mul, waiting for the add step

    // weight memory write, always available, see notes above
    always_ff @(posedge clk_i) begin
        if (w_we_i) begin
            weight_mem[w_addr_i] <= w_data_i;
        end
    end

    // mac datapath, reused for every single multiply add in the whole module
    logic [W_ADDR_W-1:0] w_rd_addr;
    logic [DATA_WIDTH-1:0] mac_a, mac_b;

    assign w_rd_addr = out_idx_q * D_MODEL + mac_idx_q;
    assign mac_a = weight_mem[w_rd_addr];
    assign mac_b = x_mem[mac_idx_q];

    logic mul_valid_i, mul_valid_o;
    logic [DATA_WIDTH:0] mul_result;

    bf16_mul u_bf16_mul (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .valid_i(mul_valid_i),
        .a_in({SRC_TAG_CAN, mac_a}),
        .b_in({SRC_TAG_CAN, mac_b}),
        .result_mul_o(mul_result),
        .valid_o(mul_valid_o)
    );

    logic add_valid_i, add_valid_o;
    logic [DATA_WIDTH:0] add_result;

    bf16_add u_bf16_add (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .valid_i(add_valid_i),
        .a_in({SRC_TAG_CAN, acc_q}),
        .b_in({SRC_TAG_CAN, prod_q}),
        .result_sum_o(add_result),
        .valid_o(add_valid_o)
    );

    assign busy_o = (state_q != ST_LOAD) || (in_ptr_q != '0);

    // next state and datapath control logic, all in one block since this
    // is a simple design, not trying to be fancy about splitting it up
    always_comb begin
        state_d = state_q;
        in_ptr_d = in_ptr_q;
        out_idx_d = out_idx_q;
        mac_idx_d = mac_idx_q;
        drain_ptr_d = drain_ptr_q;
        acc_d = acc_q;
        prod_d = prod_q;

        x_ready_o = 1'b0;
        y_valid_o = 1'b0;
        y_last_o = 1'b0;
        y_data_o = {SRC_TAG_CAN, y_mem[drain_ptr_q]};

        mul_valid_i = 1'b0;
        add_valid_i = 1'b0;

        case (state_q)

            ST_LOAD: begin
                x_ready_o = 1'b1;
                if (x_valid_i) begin
                    if (x_last_i) begin
                        in_ptr_d = '0;
                        out_idx_d = '0;
                        mac_idx_d = '0;
                        acc_d = '0;
                        state_d = ST_MUL_ISSUE;
                    end else begin
                        in_ptr_d = in_ptr_q + 1'b1;
                    end
                end
            end

            ST_MUL_ISSUE: begin
                mul_valid_i = 1'b1;
                state_d = ST_MUL_WAIT;
            end

            ST_MUL_WAIT: begin
                if (mul_valid_o) begin
                    prod_d = mul_result[DATA_WIDTH-1:0];
                    state_d = ST_ADD_ISSUE;
                end
            end

            ST_ADD_ISSUE: begin
                add_valid_i = 1'b1;
                state_d = ST_ADD_WAIT;
            end

            ST_ADD_WAIT: begin
                if (add_valid_o) begin
                    if (mac_idx_q == D_MODEL-1) begin
                        mac_idx_d = '0;
                        acc_d = '0;
                        if (out_idx_q == D_OUT-1) begin
                            drain_ptr_d = '0;
                            state_d = ST_DRAIN;
                        end else begin
                            out_idx_d = out_idx_q + 1'b1;
                            state_d = ST_MUL_ISSUE;
                        end
                    end else begin
                        mac_idx_d = mac_idx_q + 1'b1;
                        acc_d = add_result[DATA_WIDTH-1:0];
                        state_d = ST_MUL_ISSUE;
                    end
                end
            end

            ST_DRAIN: begin
                y_valid_o = 1'b1;
                y_last_o = (drain_ptr_q == D_OUT-1);
                if (y_ready_i) begin
                    if (drain_ptr_q == D_OUT-1) begin
                        state_d = ST_LOAD;
                    end else begin
                        drain_ptr_d = drain_ptr_q + 1'b1;
                    end
                end
            end

            default: state_d = ST_LOAD;

        endcase
    end

    // state and counter registers
    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_LOAD;
            in_ptr_q <= '0;
            out_idx_q <= '0;
            mac_idx_q <= '0;
            drain_ptr_q <= '0;
            acc_q <= '0;
            prod_q <= '0;
        end else begin
            state_q <= state_d;
            in_ptr_q <= in_ptr_d;
            out_idx_q <= out_idx_d;
            mac_idx_q <= mac_idx_d;
            drain_ptr_q <= drain_ptr_d;
            acc_q <= acc_d;
            prod_q <= prod_d;
        end
    end

    // x_mem write: strip the source tag (bit DATA_WIDTH), store only the
    // 16-bit bf16 value. the tag carried by the upstream bus is not
    // meaningful inside this module, we re-tag everything on output.
    always_ff @(posedge clk_i) begin
        if (state_q == ST_LOAD && x_valid_i) begin
            x_mem[in_ptr_q] <= x_data_i[DATA_WIDTH-1:0];
        end
    end

    // y_mem is written once per finished output element, right when the
    // add step for the last mac_idx closes out that element
    always_ff @(posedge clk_i) begin
        if (state_q == ST_ADD_WAIT && add_valid_o && mac_idx_q == D_MODEL-1) begin
            y_mem[out_idx_q] <= add_result[DATA_WIDTH-1:0];
        end
    end

endmodule
