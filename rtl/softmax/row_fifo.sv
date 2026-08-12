// FUNC_ROW_FIFO : Holds one row's worth of exponentials between softmax's
// ACCUMULATE and DIVIDE_NORMALIZE passes
//
//   Purpose:  softmax needs every element's own e to the power of scaled
//             score value twice, once to add into the row's running sum,
//             and again later, after the reciprocal of that sum is
//             known, to multiply by. This little FIFO is exactly that
//             second copy, values go in during ACCUMULATE as each
//             element's exponential is computed and go back out again,
//             in the same order, during DIVIDE_NORMALIZE.
//
//   parameters:
//           DATA_WIDTH:   bit width of one stored element
//           ADDR_WIDTH:   FIFO depth is 2 to the power of ADDR_WIDTH,
//                         needs to cover the longest row softmax will
//                         ever see
//
//   inputs:
//           clk, rst_n:   clock and active low reset
//           wr_data:      one exponential value to push in
//           wr_en:        wr_data is valid this cycle, push it
//           rd_en:        pop one element this cycle
//   output:
//           full:         no room left to push another element
//           rd_data:      the element at the front of the queue,
//                         combinational, always reflects whatever rd_ptr
//                         currently points at
//           empty:        nothing left to pop
//
//   notes:
//           a plain circular buffer, write and read can happen on the
//           same cycle since they use independent pointers into the same
//           memory array

module row_fifo #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 4 //# of tokens
    )(
    //Write Domain / Intake
    input  logic                     clk, 
    input  logic                     rst_n,
    input  logic                     wr_en ,
    input  logic [DATA_WIDTH-1:0]    wr_data, 
    
    output logic                     full,
    
    //Read Domain / Discharge
    input  logic                     rd_en,   
    
    output logic [DATA_WIDTH-1:0]    rd_data,
    output logic                     empty
    );
    
    //Calculating the total # of phys. memory rows.
    localparam int DEPTH = 1 << ADDR_WIDTH; //left-shift to gain 2^addr_width
    
    // Dual-Port Memory Array, Storage Gri
    // Can be written and read at the same time
    logic [DATA_WIDTH-1:0] mem [0: DEPTH-1]; //16X16 matrix
    
    //Pointer and Synchronizer Register Decleration
    // pointers are 1-bit wider to count the laps
    logic [ADDR_WIDTH:0] wr_ptr, rd_ptr;
        
    

    
    //Memory Write and Read Logic
    always_ff @(posedge clk) begin
        if(wr_en && !full) 
            mem[wr_ptr[ADDR_WIDTH-1:0]] <= wr_data;   //part-select slicing we only look th elowest 4 bits for the equal row in mem.
                                                      //5'b0_0000 => row 0, 5'b0_1110 => row 14, 5'b1_0010 => row 2(loops back to start)                                                                                               
    end
    
    // Cont. Assignment reads the data pointed to by the read register
    // it is combinational logic => the door is kept open 
    // no need to wait a cc. to read data
    assign rd_data = mem[rd_ptr[ADDR_WIDTH-1:0]];
    
    //Write-Read  Engine
    always_ff @(posedge clk or negedge rst_n) begin
        if(!rst_n) begin
            wr_ptr             <= '0;
            rd_ptr             <= '0;
        end else begin
            if(wr_en && !full) begin
                wr_ptr       <=  wr_ptr + 1'b1;
            end
            if(rd_en && !empty) begin
                rd_ptr <= rd_ptr + 1'b1;
            end           
        end
    end
    

    
    //Status Flag Geneative Logic
    assign empty = (wr_ptr == rd_ptr);
    assign full = (wr_ptr[ADDR_WIDTH-1:0] == rd_ptr[ADDR_WIDTH-1:0]) && 
               (wr_ptr[ADDR_WIDTH] != rd_ptr[ADDR_WIDTH]);
    


endmodule