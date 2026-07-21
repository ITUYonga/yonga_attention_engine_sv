// FUNC_ROPE : Adds position info to a vector by rotating it with sin/cos
// Can Erturk  21.07.2026
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
//           x_data_i:     one element of the input vector
//           x_valid_i:    x_data_i is valid this cycle
//           x_last_i:     x_data_i is the last element of this vector
//           y_ready_i:    downstream can accept an output element
//   output:
//           x_ready_o:    module can accept an input element this cycle
//           y_data_o:     one element of the rotated output vector
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

module rope #(
    parameter int DATA_WIDTH = 16,
    parameter int D_MODEL = 64,
    parameter int MAX_POS = 512
)(
    input logic clk_i,
    input logic rst_ni,

    input logic [$clog2(MAX_POS)-1:0] pos_i,

    input logic [DATA_WIDTH-1:0] x_data_i,
    input logic x_valid_i,
    input logic x_last_i,
    output logic x_ready_o,

    output logic [DATA_WIDTH-1:0] y_data_o,
    output logic y_valid_o,
    output logic y_last_o,
    input logic y_ready_i
);

    localparam int PAIR_CNT = D_MODEL / 2;
    localparam int PAIR_IDX_W = $clog2(PAIR_CNT);
    localparam int IN_IDX_W = $clog2(D_MODEL);
    localparam int POS_W = $clog2(MAX_POS);
    localparam int ROM_ADDR_W = $clog2(MAX_POS * PAIR_CNT);

    typedef enum logic [1:0] {
        ST_LOAD,    // get the vector in
        ST_COMPUTE, // rotate it
        ST_DRAIN    // send it out
    } state_t;

    state_t state_q, state_d;

    // memories
    logic [DATA_WIDTH-1:0] x_mem [0:D_MODEL-1];
    logic [DATA_WIDTH-1:0] y_mem [0:D_MODEL-1];
    logic [DATA_WIDTH-1:0] sin_rom [0:MAX_POS*PAIR_CNT-1];
    logic [DATA_WIDTH-1:0] cos_rom [0:MAX_POS*PAIR_CNT-1];

    initial begin
        // TODO these files do not exist yet, need a gen script, see notes above
        $readmemh("rope_sin_rom.mem", sin_rom);
        $readmemh("rope_cos_rom.mem", cos_rom);
    end

    // counters
    logic [IN_IDX_W-1:0] in_ptr_q, in_ptr_d; // loading x_mem
    logic [POS_W-1:0] pos_q, pos_d; // token position
    logic [PAIR_IDX_W-1:0] pair_idx_q, pair_idx_d; // which pair
    logic [1:0] mstep_q, mstep_d; // 0 to 3, which sub step of the pair
    logic [IN_IDX_W-1:0] drain_ptr_q, drain_ptr_d;

    logic [DATA_WIDTH-1:0] tmp1_q, tmp1_d; // x1*cos, saved for later
    logic [DATA_WIDTH-1:0] tmp3_q, tmp3_d; // x1*sin, saved for later

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
    logic [DATA_WIDTH-1:0] mul_a, mul_b, mul_result;

    always_comb begin
        case (mstep_q)
            2'd0: begin mul_a = x1; mul_b = cos_val; end
            2'd1: begin mul_a = x2; mul_b = sin_val; end
            2'd2: begin mul_a = x1; mul_b = sin_val; end
            default: begin mul_a = x2; mul_b = cos_val; end
        endcase
    end

    bf16_mul u_bf16_mul (
        .a_in(mul_a),
        .b_in(mul_b),
        .result_mul_o(mul_result)
    );

    // one adder, mstep 1 is a subtract, done by flipping the sign bit,
    // mstep 3 is a normal add
    logic [DATA_WIDTH-1:0] add_a, add_b, add_result;
    logic [DATA_WIDTH-1:0] mul_result_neg;

    assign mul_result_neg = {~mul_result[DATA_WIDTH-1], mul_result[DATA_WIDTH-2:0]};

    always_comb begin
        if (mstep_q == 2'd1) begin
            add_a = tmp1_q;
            add_b = mul_result_neg; // subtract
        end else begin
            add_a = tmp3_q;
            add_b = mul_result; // add
        end
    end

    bf16_add u_bf16_add (
        .a_in(add_a),
        .b_in(add_b),
        .result_sum_o(add_result)
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

        x_ready_o = 1'b0;
        y_valid_o = 1'b0;
        y_last_o = 1'b0;
        y_data_o = y_mem[drain_ptr_q];

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
                        state_d = ST_COMPUTE;
                    end else begin
                        in_ptr_d = in_ptr_q + 1'b1;
                    end
                end
            end

            ST_COMPUTE: begin
                // 1 clock per micro step, 4 steps per pair
                case (mstep_q)
                    2'd0: begin
                        tmp1_d = mul_result;
                        mstep_d = 2'd1;
                    end
                    2'd1: begin
                        // y1 ready now, saved to y_mem below
                        mstep_d = 2'd2;
                    end
                    2'd2: begin
                        tmp3_d = mul_result;
                        mstep_d = 2'd3;
                    end
                    default: begin
                        // y2 ready now, saved to y_mem below
                        if (pair_idx_q == PAIR_CNT-1) begin
                            drain_ptr_d = '0;
                            state_d = ST_DRAIN;
                        end else begin
                            pair_idx_d = pair_idx_q + 1'b1;
                            mstep_d = 2'd0;
                        end
                    end
                endcase
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
        end else begin
            state_q <= state_d;
            in_ptr_q <= in_ptr_d;
            pos_q <= pos_d;
            pair_idx_q <= pair_idx_d;
            mstep_q <= mstep_d;
            drain_ptr_q <= drain_ptr_d;
            tmp1_q <= tmp1_d;
            tmp3_q <= tmp3_d;
        end
    end

    // fills x_mem, kept in its own block so the case above stays readable
    always_ff @(posedge clk_i) begin
        if (state_q == ST_LOAD && x_valid_i) begin
            x_mem[in_ptr_q] <= x_data_i;
        end
    end

    // y1 saved at mstep 1, y2 saved at mstep 3
    always_ff @(posedge clk_i) begin
        if (state_q == ST_COMPUTE && mstep_q == 2'd1) begin
            y_mem[pair_idx_q] <= add_result;
        end
        if (state_q == ST_COMPUTE && mstep_q == 2'd3) begin
            y_mem[pair_idx_q + PAIR_CNT] <= add_result;
        end
    end

endmodule
