// FUNC_ROPE : Adds position info to a vector by rotating it with sin/cos
//
//   Purpose:  Takes one Q vector or one K vector (this module gets used
//             twice, once wired to Q and once wired to K, V never goes
//             through this module at all) and rotates it using position
//             dependent sin/cos values pulled from a ROM. This is how the
//             model learns whether a token was the 3rd word or the 7th
//             word in the sentence, otherwise attention has no idea about
//             word order at all.
//
//             pairing scheme used here is the rotate half style, meaning
//             element i is paired with element i + D_MODEL/2, not with
//             its neighbor i+1. this is an assumption on my part since
//             the plan doc does not say which convention the golden
//             model script uses, needs double checking once that script
//             exists
//
//             for every pair (x1, x2) at position pos with angle theta
//             y1 = x1*cos(theta) - x2*sin(theta)
//             y2 = x1*sin(theta) + x2*cos(theta)
//
//   parameters:
//           DATA_WIDTH:   bit width of one bf16 number, should stay 16
//           D_MODEL:      length of the vector, must be an even number
//           MAX_POS:      largest token position the rom is built for,
//                         basically the max sequence length supported
//
//   inputs:
//           clk_i:        clock
//           rst_ni:       active low reset
//           pos_i:        position of the token this vector belongs to,
//                         sampled once at the start of a new vector, has
//                         to stay valid for that first cycle only
//           x_data_i:     one element of the input vector, 17 bits wide
//                         (bit DATA_WIDTH is the source tag from upstream,
//                         bits DATA_WIDTH-1:0 are the bf16 value). the
//                         tag is stripped on entry and not used internally,
//                         this module tags its own outputs with SRC_TAG_CAN
//           x_valid_i:    x_data_i is valid this cycle
//           x_last_i:     x_data_i is the last element of this vector
//           y_ready_i:    downstream can accept an output element
//   output:
//           x_ready_o:    module can accept an input element this cycle
//           y_data_o:     one rotated output element, bit DATA_WIDTH is
//                         the source tag (see below), bits DATA_WIDTH-1:0
//                         are the bf16 value
//           y_valid_o:    y_data_o is valid this cycle
//           y_last_o:     y_data_o is the last element of this vector
//
//   notes:
//           sin_rom and cos_rom are loaded with $readmemh from files that
//           do not exist yet, someone needs to write a small python script
//           that dumps sin(pos*freq_i) and cos(pos*freq_i) in bf16 hex for
//           every pos 0..MAX_POS-1 and every pair index 0..D_MODEL/2-1,
//           using the usual base 10000 rope frequency formula. until that
//           file exists this module will just read x'es (undefined memory)
//
//           there is no bf16_sub module from Belinay yet, so subtraction
//           here is done by flipping the sign bit of the second operand
//           and feeding it into bf16_add, works fine for bf16 since sign
//           is just the top bit same as normal floating point
//
//           source tag (bit DATA_WIDTH of every 17-bit bus): Belinay's
//           bf16_mul/bf16_add (e295872) now carry a 1-bit tag through
//           their pipeline unchanged, used to tell apart which side fed a
//           shared resource (agreed with Belinay: 0 = came from my block,
//           1 = came from hers). x_data_i now arrives as 17 bits but the
//           incoming tag is stripped immediately (only the lower 16 bits
//           are stored in x_mem). whatever this module outputs was
//           computed here, so y_data_o always gets tagged SRC_TAG_CAN.
//           internal mul/add operands are tagged SRC_TAG_CAN too, the
//           value doesn't matter for the math, the bus width just
//           requires something there. Q and K both go through this same
//           module so both keep the same tag, which matters downstream:
//           Belinay's qk_pe asserts its two operands carry the same tag
//
//           bf16_mul and bf16_add are no longer 0-cycle combinational
//           (Belinay's e295872 update made both 2-stage pipelined with a
//           valid_i/valid_o handshake). each of the 4 micro steps per
//           pair now issues a mul, waits for its valid_o, and the two
//           steps that need a subtract/add (mstep 1 and 3) go on to issue
//           an add and wait for that valid_o too, instead of assuming
//           either result is ready the same cycle it was issued
//
//           x_data_i is 17 bits wide, not 16, so its port width matches
//           the 17-bit buses used everywhere else in the design
//           (Belinay's bf16 units, dbuf, axi_stream_if). the incoming tag
//           bit is stripped at x_mem write time, internal computation
//           stays 16-bit, output is re-tagged with SRC_TAG_CAN

module rope #(
    parameter int DATA_WIDTH = 16,
    parameter int D_MODEL = 64,
    parameter int MAX_POS = 512,
    parameter string SIN_ROM_FILE = "rtl/projection/rope_sin_rom.mem",
    parameter string COS_ROM_FILE = "rtl/projection/rope_cos_rom.mem"
)(
    input logic clk_i,
    input logic rst_ni,

    input logic [$clog2(MAX_POS)-1:0] pos_i,

    // input vector, streamed in one element per cycle
    // [DATA_WIDTH]     = source tag from upstream (stripped, not used)
    // [DATA_WIDTH-1:0] = bf16 value (stored in x_mem)
    input logic [DATA_WIDTH:0] x_data_i,
    input logic x_valid_i,
    input logic x_last_i,
    output logic x_ready_o,

    output logic [DATA_WIDTH:0] y_data_o,
    output logic y_valid_o,
    output logic y_last_o,
    input logic y_ready_i
);

    localparam int PAIR_CNT = D_MODEL / 2;
    localparam int PAIR_IDX_W = $clog2(PAIR_CNT);
    localparam int IN_IDX_W = $clog2(D_MODEL);
    localparam int POS_W = $clog2(MAX_POS);
    localparam int ROM_ADDR_W = $clog2(MAX_POS * PAIR_CNT);

    // agreed with Belinay: 0 = this value was produced by my (Can's) block,
    // 1 = produced by hers. only meaningful on a bus shared between us,
    // see note above.
    localparam logic SRC_TAG_CAN = 1'b0;

    // ST_COMPUTE from the original single-cycle version is now split into
    // an issue/wait pair for the multiply, and another for the add, since
    // neither is 0-cycle anymore. mstep_q still tracks which of the 4
    // micro steps of the current pair we are on.
    typedef enum logic [2:0] {
        ST_LOAD,      // get the vector in
        ST_MUL_ISSUE, // pulse valid_i into bf16_mul for the current mstep
        ST_MUL_WAIT,  // wait for bf16_mul's valid_o
        ST_ADD_ISSUE, // pulse valid_i into bf16_add (only mstep 1 and 3)
        ST_ADD_WAIT,  // wait for bf16_add's valid_o
        ST_DRAIN      // send it out
    } state_t;

    state_t state_q, state_d;

    // memories
    logic [DATA_WIDTH-1:0] x_mem [0:D_MODEL-1];
    logic [DATA_WIDTH-1:0] y_mem [0:D_MODEL-1];
    logic [DATA_WIDTH-1:0] sin_rom [0:MAX_POS*PAIR_CNT-1];
    logic [DATA_WIDTH-1:0] cos_rom [0:MAX_POS*PAIR_CNT-1];

    initial begin
        $readmemh(SIN_ROM_FILE, sin_rom);
        $readmemh(COS_ROM_FILE, cos_rom);
    end

    // counters
    logic [IN_IDX_W-1:0] in_ptr_q, in_ptr_d; // loading x_mem
    logic [POS_W-1:0] pos_q, pos_d; // token position
    logic [PAIR_IDX_W-1:0] pair_idx_q, pair_idx_d; // which pair
    logic [1:0] mstep_q, mstep_d; // 0 to 3, which sub step of the pair
    logic [IN_IDX_W-1:0] drain_ptr_q, drain_ptr_d;

    logic [DATA_WIDTH-1:0] tmp1_q, tmp1_d; // x1*cos, saved for later
    logic [DATA_WIDTH-1:0] tmp3_q, tmp3_d; // x1*sin, saved for later
    logic [DATA_WIDTH-1:0] mul_res_q, mul_res_d; // last mul result, held for the add step

    // rom address for the pair we are on
    logic [ROM_ADDR_W-1:0] rom_addr;
    assign rom_addr = pos_q * PAIR_CNT + pair_idx_q;

    logic [DATA_WIDTH-1:0] sin_val, cos_val;
    assign sin_val = sin_rom[rom_addr];
    assign cos_val = cos_rom[rom_addr];

    logic [DATA_WIDTH-1:0] x1, x2;
    assign x1 = x_mem[pair_idx_q];
    assign x2 = x_mem[pair_idx_q + PAIR_CNT];

    // one multiplier, reused 4 times per pair
    //   mstep 0 : x1*cos -> tmp1
    //   mstep 1 : x2*sin -> tmp1 - this = y1
    //   mstep 2 : x1*sin -> tmp3
    //   mstep 3 : x2*cos -> tmp3 + this = y2
    logic [DATA_WIDTH-1:0] mul_a_val, mul_b_val;

    always_comb begin
        case (mstep_q)
            2'd0: begin mul_a_val = x1; mul_b_val = cos_val; end
            2'd1: begin mul_a_val = x2; mul_b_val = sin_val; end
            2'd2: begin mul_a_val = x1; mul_b_val = sin_val; end
            default: begin mul_a_val = x2; mul_b_val = cos_val; end
        endcase
    end

    logic mul_valid_i, mul_valid_o;
    logic [DATA_WIDTH:0] mul_result;

    bf16_mul u_bf16_mul (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .valid_i(mul_valid_i),
        .a_in({SRC_TAG_CAN, mul_a_val}),
        .b_in({SRC_TAG_CAN, mul_b_val}),
        .result_mul_o(mul_result),
        .valid_o(mul_valid_o)
    );

    // one adder, mstep 1 is a subtract, done by flipping the sign bit,
    // mstep 3 is a normal add. both use mul_res_q, the result the
    // multiplier produced for this same mstep
    logic [DATA_WIDTH-1:0] add_a_val, add_b_val;
    logic [DATA_WIDTH-1:0] mul_res_neg;

    assign mul_res_neg = {~mul_res_q[DATA_WIDTH-1], mul_res_q[DATA_WIDTH-2:0]};

    always_comb begin
        if (mstep_q == 2'd1) begin
            add_a_val = tmp1_q;
            add_b_val = mul_res_neg; // subtract
        end else begin
            add_a_val = tmp3_q;
            add_b_val = mul_res_q; // add
        end
    end

    logic add_valid_i, add_valid_o;
    logic [DATA_WIDTH:0] add_result;

    bf16_add u_bf16_add (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .valid_i(add_valid_i),
        .a_in({SRC_TAG_CAN, add_a_val}),
        .b_in({SRC_TAG_CAN, add_b_val}),
        .result_sum_o(add_result),
        .valid_o(add_valid_o)
    );

    // next state / control logic
    always_comb begin
        state_d = state_q;
        in_ptr_d = in_ptr_q;
        pos_d = pos_q;
        pair_idx_d = pair_idx_q;
        mstep_d = mstep_q;
        drain_ptr_d = drain_ptr_q;
        tmp1_d = tmp1_q;
        tmp3_d = tmp3_q;
        mul_res_d = mul_res_q;

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
                    if (in_ptr_q == '0) begin
                        pos_d = pos_i; // first element, save the position
                    end
                    if (x_last_i) begin
                        in_ptr_d = '0;
                        pair_idx_d = '0;
                        mstep_d = 2'd0;
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
                    mul_res_d = mul_result[DATA_WIDTH-1:0];
                    case (mstep_q)
                        2'd0: begin
                            // x1*cos, no add needed, save straight to tmp1
                            tmp1_d = mul_result[DATA_WIDTH-1:0];
                            mstep_d = 2'd1;
                            state_d = ST_MUL_ISSUE;
                        end
                        2'd2: begin
                            // x1*sin, no add needed, save straight to tmp3
                            tmp3_d = mul_result[DATA_WIDTH-1:0];
                            mstep_d = 2'd3;
                            state_d = ST_MUL_ISSUE;
                        end
                        default: begin
                            // mstep 1 (x2*sin) or mstep 3 (x2*cos), both
                            // need to go through the adder next
                            state_d = ST_ADD_ISSUE;
                        end
                    endcase
                end
            end

            ST_ADD_ISSUE: begin
                add_valid_i = 1'b1;
                state_d = ST_ADD_WAIT;
            end

            ST_ADD_WAIT: begin
                if (add_valid_o) begin
                    // y1 (mstep 1) or y2 (mstep 3) ready now, saved to
                    // y_mem below
                    if (mstep_q == 2'd1) begin
                        mstep_d = 2'd2;
                        state_d = ST_MUL_ISSUE;
                    end else begin
                        // mstep 3, pair done
                        if (pair_idx_q == PAIR_CNT-1) begin
                            drain_ptr_d = '0;
                            state_d = ST_DRAIN;
                        end else begin
                            pair_idx_d = pair_idx_q + 1'b1;
                            mstep_d = 2'd0;
                            state_d = ST_MUL_ISSUE;
                        end
                    end
                end
            end

            ST_DRAIN: begin
                y_valid_o = 1'b1;
                y_last_o = (drain_ptr_q == D_MODEL-1);
                if (y_ready_i) begin
                    if (drain_ptr_q == D_MODEL-1) begin
                        state_d = ST_LOAD;
                    end else begin
                        drain_ptr_d = drain_ptr_q + 1'b1;
                    end
                end
            end

            default: state_d = ST_LOAD;

        endcase
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin
        if (!rst_ni) begin
            state_q <= ST_LOAD;
            in_ptr_q <= '0;
            pos_q <= '0;
            pair_idx_q <= '0;
            mstep_q <= '0;
            drain_ptr_q <= '0;
            tmp1_q <= '0;
            tmp3_q <= '0;
            mul_res_q <= '0;
        end else begin
            state_q <= state_d;
            in_ptr_q <= in_ptr_d;
            pos_q <= pos_d;
            pair_idx_q <= pair_idx_d;
            mstep_q <= mstep_d;
            drain_ptr_q <= drain_ptr_d;
            tmp1_q <= tmp1_d;
            tmp3_q <= tmp3_d;
            mul_res_q <= mul_res_d;
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

    // y1 saved when mstep 1's add completes, y2 when mstep 3's add
    // completes
    always_ff @(posedge clk_i) begin
        if (state_q == ST_ADD_WAIT && add_valid_o) begin
            if (mstep_q == 2'd1) begin
                y_mem[pair_idx_q] <= add_result[DATA_WIDTH-1:0];
            end else begin
                y_mem[pair_idx_q + PAIR_CNT] <= add_result[DATA_WIDTH-1:0];
            end
        end
    end

endmodule
