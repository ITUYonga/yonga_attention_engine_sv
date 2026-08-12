// FUNC_TX_FIFO : Small output FIFO between the output projection and the
// external AXI-Stream master
//
//   Purpose:  Lets the output projection push finished elements out as
//             soon as they are ready without waiting for the outside
//             world to actually accept them, since the external AXI
//             master's readiness has nothing to do with the chip's own
//             clock timing. wr_data is stored together with its own
//             if_last flag in the same memory word, so the real end of
//             vector marker survives sitting in the queue and comes back
//             out attached to the right element later.
//
//   parameters:
//           DATA_WIDTH:   bit width of one stored element, not counting
//                         the extra if_last bit
//           ADDR_WIDTH:   FIFO depth is 2 to the power of ADDR_WIDTH
//
//   inputs:
//           clk, rst_n:   clock and active low reset
//           wr_data:      one element to push in
//           wr_en:        wr_data is valid this cycle, push it
//           if_last:      this element is the real last element of its
//                         vector, stored alongside wr_data
//           rd_en:        pop one element this cycle
//   output:
//           full:         no room left to push another element
//           rd_data:      the element at the front of the queue, one bit
//                         wider than DATA_WIDTH, top bit is the stored
//                         if_last flag
//           empty:        nothing left to pop
//
//   notes:
//           a plain circular buffer with one extra pointer bit used only
//           to tell empty and full apart, no other backpressure logic

module tx_fifo #(
    parameter DATA_WIDTH = 16, 
    parameter ADDR_WIDTH = 4 )(
    input  logic                  clk,
    input  logic                  rst_n,

    input  logic [DATA_WIDTH-1:0] wr_data,  //B'den gelen data
    input  logic                  wr_en,   
    output logic                  full,     // FIFO full sinayli B'ye ver
    input logic                   if_last,  //FSM last mi sinyali

    output logic [DATA_WIDTH:0] rd_data,  // AXI'ye verilecek data
    input  logic                  rd_en,    // AXI okumak istiyo sinyali
    output logic                  empty     // FIFO empty sinyali AXI'ye ver validi 0 yapsin
);

    
    logic [DATA_WIDTH:0] memory [0:(1<<ADDR_WIDTH)-1];

    logic [ADDR_WIDTH:0] wr_ptr;
    logic [ADDR_WIDTH:0] rd_ptr;

    assign empty = (wr_ptr == rd_ptr);
    
    assign full  = (wr_ptr[ADDR_WIDTH] != rd_ptr[ADDR_WIDTH]) &&  (wr_ptr[ADDR_WIDTH-1:0] == rd_ptr[ADDR_WIDTH-1:0]);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 'b0;
        end else if (wr_en && !full) begin
            memory[wr_ptr[ADDR_WIDTH-1:0]] <= {if_last,wr_data};
            wr_ptr <= wr_ptr + 1'b1;
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_ptr <= '0;
            rd_data <= '0;
        end else if (rd_en && !empty) begin
            rd_data <= memory[rd_ptr[ADDR_WIDTH-1:0]];
            rd_ptr  <= rd_ptr + 1'b1;
        end
    end

endmodule