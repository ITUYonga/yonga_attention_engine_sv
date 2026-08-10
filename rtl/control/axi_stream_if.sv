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


    output logic [DATA_WIDTH_bf16-1:0] m_axis_tdata,   // output data
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