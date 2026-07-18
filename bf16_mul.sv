`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 18.07.2026 16:38:03
// Design Name: 
// Module Name: bf16_mul
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


module bf16_mul(
    input logic [15:0] a_in,
    input logic [15:0] b_in,
    output logic [15:0] result_mul_o

    );
    
    logic sign_a = a_in[15]; // sign bits
    logic sign_b = b_in[15];
    
    logic [7:0] expo_a = a_in [14:7];// exponents
    logic [7:0] expo_b = b_in [14:7];
    
    logic [7:0] mant_a;// mantissa
    logic [7:0] mant_b;
    
    assign mant_a = (|expo_a)? {1'b1, a_in[6:0]}: 8'b0;
    assign mant_b = (|expo_b)? {1'b1, b_in[6:0]}: 8'b0;
    
    logic final_sign;
    logic [15:0] final_mant;
    
    assign final_sign = sign_a ^ sign_b;
    assign final_mant = mant_a * mant_b;
    
    always_comb begin 
        if(final_mant[15]) begin // if there is a  carry(?) increase the exponent by 1 and take the highest possible bit
            result_mul_o = {final_sign, (expo_a + expo_b - 'd127 + 'd1), final_mant[14:8]};
        end
        
        else begin // since there is no carry(?) bit exponent stays the same, but the mantissa's upper index drops by one because it is not possible that there is a 1 at the 15th bit without a carry
            result_mul_o = {final_sign, (expo_a + expo_b - 'd127), final_mant[13:7]};       
        end
      
    end
    
    
endmodule
