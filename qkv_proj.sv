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
//           x_data_i:     one element of the input token vector
//           x_valid_i:    x_data_i is valid this cycle
//           x_last_i:     x_data_i is the last element of this vector
//           y_ready_i:    downstream can accept an output element this cycle
//   output:
//           x_ready_o:    module can accept an input element this cycle
//           y_data_o:     one element of the projected output vector
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
//           bf16_mul and bf16_add are currently fully combinational
//           (0 cycle latency). this design assumes that stays true. if
//           Belinay adds pipeline stages inside them later this state
//           machine needs extra wait cycles added in ST_COMPUTE

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
    input logic [DATA_WIDTH-1:0] x_data_i,
    input logic x_valid_i,
    input logic x_last_i,
    output logic x_ready_o,

    // output vector, streamed out one element per cycle
    output logic [DATA_WIDTH-1:0] y_data_o,
    output logic y_valid_o,
    output logic y_last_o,
    input logic y_ready_i,

    output logic busy_o
);

    localparam int IN_IDX_W = $clog2(D_MODEL);
    localparam int OUT_IDX_W = $clog2(D_OUT);
    localparam int W_ADDR_W = $clog2(D_MODEL*D_OUT);

    // the three things this module is doing, one at a time
    typedef enum logic [1:0] {
        ST_LOAD,    // waiting for / collecting the input vector
        ST_COMPUTE, // running the multiply add loop
        ST_DRAIN    // streaming the finished output vector out
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

    // weight memory write, always available, see notes above
    always_ff @(posedge clk_i) begin
        if (w_we_i) begin
            weight_mem[w_addr_i] <= w_data_i;
        end
    end

    // mac datapath, reused for every single multiply add in the whole module
    logic [W_ADDR_W-1:0] w_rd_addr;
    logic [DATA_WIDTH-1:0] mac_a, mac_b, mac_prod, mac_sum;

    assign w_rd_addr = out_idx_q * D_MODEL + mac_idx_q;
    assign mac_a = weight_mem[w_rd_addr];
    assign mac_b = x_mem[mac_idx_q];

    bf16_mul u_bf16_mul (
        .a_in(mac_a),
        .b_in(mac_b),
        .result_mul_o(mac_prod)
    );

    bf16_add u_bf16_add (
        .a_in(acc_q),
        .b_in(mac_prod),
        .result_sum_o(mac_sum)
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

        x_ready_o = 1'b0;
        y_valid_o = 1'b0;
        y_last_o = 1'b0;
        y_data_o = y_mem[drain_ptr_q];

        case (state_q)

            ST_LOAD: begin
                x_ready_o = 1'b1;
                if (x_valid_i) begin
                    // note, this write happens combinationally into x_mem
                    // in the ST_LOAD handling below with a nonblocking
                    // assign outside this always_comb block, see the
                    // always_ff for x_mem further down
                    if (x_last_i) begin
                        in_ptr_d = '0;
                        out_idx_d = '0;
                        mac_idx_d = '0;
                        acc_d = '0;
                        state_d = ST_COMPUTE;
                    end else begin
                        in_ptr_d = in_ptr_q + 1'b1;
                    end
                end
            end

            ST_COMPUTE: begin
                // one mac step per clock cycle, mac_sum is the combinational
                // result of acc_q + weight*x for the current mac_idx_q
                if (mac_idx_q == D_MODEL-1) begin
                    // last multiply add for this output element, done
                    mac_idx_d = '0;
                    acc_d = '0;
                    if (out_idx_q == D_OUT-1) begin
                        drain_ptr_d = '0;
                        state_d = ST_DRAIN;
                    end else begin
                        out_idx_d = out_idx_q + 1'b1;
                    end
                end else begin
                    mac_idx_d = mac_idx_q + 1'b1;
                    acc_d = mac_sum;
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
        end else begin
            state_q <= state_d;
            in_ptr_q <= in_ptr_d;
            out_idx_q <= out_idx_d;
            mac_idx_q <= mac_idx_d;
            drain_ptr_q <= drain_ptr_d;
            acc_q <= acc_d;
        end
    end

    // x_mem is written straight from the input handshake, kept in its own
    // always_ff instead of the big case above just to keep that block
    // readable
    always_ff @(posedge clk_i) begin
        if (state_q == ST_LOAD && x_valid_i) begin
            x_mem[in_ptr_q] <= x_data_i;
        end
    end

    // y_mem is written once per finished output element, right when the
    // mac loop closes out that element
    always_ff @(posedge clk_i) begin
        if (state_q == ST_COMPUTE && mac_idx_q == D_MODEL-1) begin
            y_mem[out_idx_q] <= mac_sum;
        end
    end

endmodule
