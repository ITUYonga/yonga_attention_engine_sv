// FUNC_AXI_STREAM_IF : Boundary between the outside fp32 AXI stream and
// the chip's internal bf16/17 bit buses
//
//   Purpose:  On the way in, converts each incoming fp32 word to bf16
//             through bf16_comb and forwards it to dbuf, driving
//             dbuf_rx_we whenever the external master has valid data and
//             the internal side can accept it. On the way out, takes the
//             17 bit tagged data waiting in the output FIFO, drops the
//             tag, widens the bf16 value back to fp32 through
//             bf16_convert, and presents it on the external AXI master
//             side. select_bf16 exists so the fp32 conversion step can be
//             bypassed for raw bf16 in, bf16 in style testing.
//
//   parameters:
//           DATA_WIDTH_bf16:   bit width of the internal bf16 bus,
//                              should stay 16
//           DATA_WIDTH_f32:    bit width of the external AXI data bus,
//                              should stay 32
//
//   inputs:
//           select_bf16:       0 converts fp32 down to bf16 on entry, 1
//                              is reserved for a future raw bf16 path
//           s_axis_tdata/tvalid/tlast:
//                              external AXI-Stream slave side, data
//                              coming into the chip
//           dbuf_rx_full:      dbuf cannot accept another element
//                              this cycle
//           internal_tx_data:  17 bit tagged word waiting in the output
//                              FIFO, bit 17 is the real end of vector
//                              last flag, bit 16 is the source tag, bits
//                              15 down to 0 are the bf16 value
//           internal_tx_valid: internal_tx_data is valid this cycle
//           m_axis_tready:     external AXI-Stream master side, the
//                              outside world can accept an element
//   output:
//           dbuf_rx_data/dbuf_rx_we/internal_rx_last:
//                              converted bf16 element and its handshake
//                              going into dbuf
//           s_axis_tready:     external AXI-Stream slave side ready,
//                              simply the inverse of dbuf_rx_full
//           m_axis_tdata/tvalid/tlast:
//                              external AXI-Stream master side, data
//                              leaving the chip
//           internal_tx_ready: passthrough of m_axis_tready back to
//                              whatever feeds internal_tx_data
//
//   notes:
//           the tag bit inside internal_tx_data is simply discarded here,
//           it only matters to internal routing, the outside world never
//           sees it

module axi_stream_if #(
    parameter DATA_WIDTH_bf16 = 16,
    parameter DATA_WIDTH_f32 = 32

)(
    input logic select_bf16,

    input  logic [DATA_WIDTH_f32-1:0] s_axis_tdata,   // data in
    input  logic        s_axis_tvalid,  // pc data var sinyali al
    output logic        s_axis_tready,  // pc data alabiliriz sinyali ver
    input  logic        s_axis_tlast,   // data bitti sinyali al

    
    output logic [DATA_WIDTH_bf16-1:0] dbuf_rx_data,   // iceriye gidecek veri 16-bit (bf16)
    output logic        dbuf_rx_we,     // dbuf'a Write Enable sinyali
    input  logic        dbuf_rx_full,    // dbuf doldu sinyali
    output logic        internal_rx_last ,//FSM son data geldi sinyali ver


    output logic [DATA_WIDTH_f32-1:0] m_axis_tdata,   // output data
    output logic        m_axis_tvalid,  // PC data var sinyali ver
    input  logic        m_axis_tready,  // PC data alabilirim sinyali al
    output logic        m_axis_tlast,   // PC Data bitti sinayi ver


    // bit [DATA_WIDTH_bf16+1] = real end-of-vector last flag (tx_fifo's if_last),
    // bit [DATA_WIDTH_bf16]   = source tag (unused here, always 0 for the
    //                           output-projection path), bits [DATA_WIDTH_bf16-1:0]
    //                           = bf16 value
    input  logic [DATA_WIDTH_bf16+1:0] internal_tx_data,  // FIFO'dan gelen data
    input  logic        internal_tx_valid, // FSM data hazir sinyal
    output logic        internal_tx_ready // FIFO data alabilirim sinyali
);
    //f32-bf16
    bf16_comb bf16(
        .fp32_in(s_axis_tdata),
        .sel_i(select_bf16),
        .bf16_out(dbuf_rx_data)
    );
    //RX
    assign s_axis_tready = ~dbuf_rx_full;
    assign dbuf_rx_we    = s_axis_tvalid && s_axis_tready;
    assign internal_rx_last = s_axis_tlast;
    //TX
    assign m_axis_tvalid = internal_tx_valid;
    assign m_axis_tlast  = internal_tx_data[DATA_WIDTH_bf16+1];
    assign internal_tx_ready = m_axis_tready;
    //bf16-f32
    bf16_convert fp32(
        .bf16_in(internal_tx_data[DATA_WIDTH_bf16-1:0]),
        .fp32_out(m_axis_tdata)
    );

endmodule