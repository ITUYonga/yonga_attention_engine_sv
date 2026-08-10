`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 18.07.2026 13:47:13
// Design Name: 
// Module Name: bf16_comb
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


module bf16_comb(
    input logic [31:0] fp32_in, // input
    input logic sel_i, // 0 truncation, 1 rounding
    output logic[15:0] bf16_out // output
    );
    
    logic [15:0] bits; // the bits we take from the input 
    logic round_bit; // this will be used in the case of rounding
    
    assign bits = fp32_in[31:16];
    assign round_bit = fp32_in[15];
    
    always_comb begin 
        
        if(sel_i) begin 
        bf16_out = bits + {15'b0, round_bit};
        end
        
        else begin 
        bf16_out = bits;
        end
    
    end
    

endmodule
