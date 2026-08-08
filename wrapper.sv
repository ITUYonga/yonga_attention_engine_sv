`timescale 1ns / 1ps

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

    // =========================================================================
    // Internal Intermediate AXI-Stream Wires (scale_mask -> softmax)
    // =========================================================================
    logic [DATA_WIDTH-1:0] inter_axis_tdata;
    logic                  inter_axis_tvalid;
    logic                  inter_axis_tready;
    logic                  inter_axis_tlast;

    // =========================================================================
    // Scale & Mask Sub-module Instantiation
    // =========================================================================
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

    // =========================================================================
    // Softmax Sub-module Instantiation
    // =========================================================================
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