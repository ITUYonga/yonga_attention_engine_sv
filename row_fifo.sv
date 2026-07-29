//Module Definiton, 
//Declaration of parameters and ports

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