`timescale 1ns / 1ps

//1) module header & port definitions
module scale_mask #(
    parameter DATA_WIDTH  = 16,
    parameter MUL_LATENCY = 2   // set this to match bf16_mul's actual pipeline depth
)(
  //port definitions
  //basic ports
    input  logic clk,
    input  logic rst_n,

  //AXI-4 Slave (PE Array --> scale_mask)
    input  logic [DATA_WIDTH-1:0]    s_axis_tdata,
    input  logic                     s_axis_tvalid,
    output logic                     s_axis_tready,
    input  logic                     s_axis_tlast,

  //Configuration & Control
    input  logic [DATA_WIDTH-1:0]    scale_factor,
    input  logic                     en_scale_mask,

  //AXI-4 Master (scale_mask --> softmax)
    output logic [DATA_WIDTH-1:0]    m_axis_tdata,
    output logic                     m_axis_tvalid,
    input  logic                     m_axis_tready,
    output logic                     m_axis_tlast
);

//2) constants and internal wires
    localparam logic [DATA_WIDTH-1:0]    BF16_NEG_INF = 'hFF80;

    logic [DATA_WIDTH-1:0]      scaled_score;   // bf16_mul result, MUL_LATENCY cycles late
    logic [DATA_WIDTH-1:0]      masked_score;   // masked version of scaled_score

    // single shared enable: freezes every register in the pipeline together
    // when downstream can't accept data this cycle.
    logic pipe_en;
    assign pipe_en = m_axis_tready;

//3) datapath: bf16_mul (assumed to have its own MUL_LATENCY-cycle pipeline
//   internally, enabled by pipe_en so it stalls in lockstep with everything else)
    bf16_mul multiplier_inst(
        .clk          (clk),
        .rst_n        (rst_n),
        .en           (pipe_en),
        .a_in         (s_axis_tdata),
        .b_in         (scale_factor),
        .result_mul_o (scaled_score)
    );

//4) shadow register: carries tvalid/tlast through the same MUL_LATENCY
//   stages that scaled_score travels through inside bf16_mul, so when
//   scaled_score finally appears, its matching valid/last bits appear
//   at the exact same cycle.
    logic [MUL_LATENCY-1:0] valid_shadow;
    logic [MUL_LATENCY-1:0] last_shadow;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_shadow <= '0;
            last_shadow  <= '0;
        end else if (pipe_en) begin
            valid_shadow <= {valid_shadow[MUL_LATENCY-2:0], s_axis_tvalid};
            last_shadow  <= {last_shadow[MUL_LATENCY-2:0],  s_axis_tlast && s_axis_tvalid};
        end
    end

    // the bits that just fell off the end of the shadow register are the
    // ones aligned with scaled_score emerging from bf16_mul this cycle
    logic scaled_valid, scaled_last;
    assign scaled_valid = valid_shadow[MUL_LATENCY-1];
    assign scaled_last  = last_shadow[MUL_LATENCY-1];

//5) masking - combinational, applied to the now-aligned scaled_score
    always_comb begin
        if (en_scale_mask)
            masked_score = BF16_NEG_INF;
        else
            masked_score = scaled_score;
    end

//6) Stream Flow-Control Logic
    assign s_axis_tready = pipe_en;

//7) Sequential Output Pipeline Register
//   same shape as your original block, but now driven by the *delayed*
//   valid/last from the shadow register instead of the raw input signals.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axis_tdata  <= '0;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
        end else if (pipe_en) begin
            m_axis_tvalid <= scaled_valid;
            m_axis_tlast  <= scaled_last;
            if (scaled_valid) begin
                m_axis_tdata <= masked_score;
            end
        end
    end

endmodule
