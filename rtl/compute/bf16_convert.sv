`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 18.07.2026 16:52:13
// Design Name: 
// Module Name: bf16_convert
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module bf16_convert(
    input logic [15:0] bf16_in,
    output logic [31:0] fp32_out

    );
    
    assign fp32_out = {bf16_in, 16'b0};
endmodule
