`timescale 1ns / 1ps

// FUNC_SCALE_MASK_SOFTMAX_WRAPPER : Chains scale_mask straight into
// softmax behind one shared AXI-Stream interface
//
//   Purpose:  scale_mask and softmax are two separate pipelines that
//             always run back to back, this module just wires
//             scale_mask's output directly into softmax's input and
//             exposes only the outer, whole chain interface, so the rest
//             of the chip (top.sv) can treat scale plus mask plus
//             softmax as one block instead of two.
//
//   parameters:
//           DATA_WIDTH:    bit width of one bf16 element, should stay 16
//           MAX_ROW_LEN:   longest row softmax's internal FIFO and
//                          counters need to support, forwarded straight
//                          to softmax
//
//   inputs:
//           clk, rst_n:      clock and active low reset
//           scale_factor:    forwarded to scale_mask
//           en_scale_mask:   forwarded to scale_mask
//           ext_stall:       forwarded to scale_mask
//           s_axis_tdata/tvalid/tlast:
//                            raw scores in, scale_mask's own slave side
//           m_axis_tready:   backpressure from whatever reads softmax's
//                            normalized output
//   output:
//           s_axis_tready:   backpressure back to whatever feeds raw
//                            scores in
//           m_axis_tdata/tvalid/tlast:
//                            normalized attention weights out, softmax's
//                            own master side
//
//   notes:
//           the wires between scale_mask and softmax are entirely
//           internal, nothing outside this module ever sees them

module scale_mask_softmax_wrapper #(
    parameter int DATA_WIDTH  = 16,
    parameter int MAX_ROW_LEN = 1024
)(
    // System Signals
    input  logic                    clk,
    input  logic                    rst_n,

    // Configuration & Control Signals (for scale_mask)
    input  logic [DATA_WIDTH-1:0]   scale_factor,
    input  logic                    en_scale_mask,
    input  logic                    ext_stall,

    // Upstream AXI-Stream Input Interface (PE Array -> Wrapper -> scale_mask)
    input  logic [DATA_WIDTH-1:0]   s_axis_tdata,
    input  logic                    s_axis_tvalid,
    output logic                    s_axis_tready,
    input  logic                    s_axis_tlast,

    // Downstream AXI-Stream Output Interface (Softmax -> Wrapper -> Score x V)
    output logic [DATA_WIDTH-1:0]   m_axis_tdata,
    output logic                    m_axis_tvalid,
    input  logic                    m_axis_tready,
    output logic                    m_axis_tlast
);

    // intermediate AXI-Stream wires, scale_mask -> softmax
    logic [DATA_WIDTH-1:0] inter_axis_tdata;
    logic                  inter_axis_tvalid;
    logic                  inter_axis_tready;
    logic                  inter_axis_tlast;

    scale_mask #(
        .DATA_WIDTH(DATA_WIDTH)
    ) u_scale_mask (
        .clk            (clk),
        .rst_n          (rst_n),

        // Upstream Interface
        .s_axis_tdata   (s_axis_tdata),
        .s_axis_tvalid  (s_axis_tvalid),
        .s_axis_tready  (s_axis_tready),
        .s_axis_tlast   (s_axis_tlast),

        // Controls
        .scale_factor   (scale_factor),
        .en_scale_mask  (en_scale_mask),
        .ext_stall      (ext_stall),

        // Interconnect Output (Master side)
        .m_axis_tdata   (inter_axis_tdata),
        .m_axis_tvalid  (inter_axis_tvalid),
        .m_axis_tready  (inter_axis_tready),
        .m_axis_tlast   (inter_axis_tlast)
    );

    softmax #(
        .DATA_WIDTH (DATA_WIDTH),
        .MAX_ROW_LEN(MAX_ROW_LEN)
    ) u_softmax (
        .clk            (clk),
        .rst_n          (rst_n),

        // Interconnect Input (Slave side)
        .s_axis_tdata   (inter_axis_tdata),
        .s_axis_tvalid  (inter_axis_tvalid),
        .s_axis_tready  (inter_axis_tready),
        .s_axis_tlast   (inter_axis_tlast),

        // Downstream Interface
        .m_axis_tdata   (m_axis_tdata),
        .m_axis_tvalid  (m_axis_tvalid),
        .m_axis_tready  (m_axis_tready),
        .m_axis_tlast   (m_axis_tlast)
    );

endmodule