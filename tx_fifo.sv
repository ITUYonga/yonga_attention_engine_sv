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