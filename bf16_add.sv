`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 18.07.2026 14:52:36
// Design Name: 
// Module Name: bf16_add
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


module bf16_add(
    input logic [15:0] a_in,
    input logic [15:0] b_in,
    output logic [15:0] result_sum_o

    );
    
    logic sign_a = a_in[15]; // sign bits
    logic sign_b = b_in[15];
    
    logic [7:0] expo_a = a_in [14:7];// exponents
    logic [7:0] expo_b = b_in [14:7];
    
    logic [7:0] mant_a;// mantissa
    logic [7:0] mant_b;
    
    assign mant_a = (|expo_a)? {1'b1, a_in[6:0]}: 8'b0;
    assign mant_b = (|expo_b)? {1'b1, b_in[6:0]}: 8'b0;
    // if there is no exponent mantissa is zero, if there is exponent bits then we add the secret 1 in front of the mantissa bits 
    
    logic [7:0] expo_dif;
    logic [7:0] expo_shared;
    logic [7:0] lined_mant_a;
    logic [7:0] lined_mant_b;
    
    
    always_comb begin 
        if(expo_a > expo_b) begin // if the exponent bit is higher on one number we have to align the bits according to that.
            expo_dif = expo_a - expo_b;
            expo_shared = expo_a; // shared one should be the bigger one
            lined_mant_a = mant_a; // and the mantissa of the bigger one does not change we align the smaller one according to that.
            lined_mant_b = (expo_dif >'d8)? 8'b0: (mant_b >> expo_dif); // if the difference is greater than eight bits which is the total bit of the mantissa then all of the bits should be zero, if not then we shift the mantissa to right according to the difference of the exponent
        end 
        
        else begin 
            expo_dif = expo_b - expo_a;
            expo_shared = expo_b;
            lined_mant_b = mant_b;
            lined_mant_a = (expo_dif >'d8)? 8'b0: (mant_a >> expo_dif);
        end

    end
    
    logic signed [9:0] signed_a, signed_b, sum;
    assign signed_a = sign_a ? -{2'b0, lined_mant_a} : {2'b0, lined_mant_a};// if the sign is negative then we put a minus in front of the nımber if not it stays the same 
    assign signed_b = sign_b ? -{2'b0, lined_mant_b} : {2'b0, lined_mant_b};// the reason for the two bit gap is that 1 bit is for the carry and the other bit is for the sign (negative is 1, positive is 0)
    assign sum  = signed_a + signed_b;
    
    logic final_sign;
    logic [8:0] final_sum;
    logic [6:0] final_mant;
    logic [7:0] final_expo;
    
    always_comb begin 
        final_sign = sum[9]; // the final sign of the addition is the 10th bit of the sum
        final_sum = (final_sign)? -sum[8:0] : sum[8:0]; // if the sign is negative convert the 2's complement, if not keep it the same (without the sign)  
    end
    
    always_comb begin 
        if(final_sum == 9'b0) begin 
            final_mant = 7'b0;
            final_expo = 8'b0;     
        end
        
        else if (final_sum[8]) begin // if there is a carry
            final_mant = final_sum[7:1]; // shift the mantissa 1 bit to the right
            final_expo = expo_shared + 1'b1; // increase the exponent by 1
        end
        
        else if(final_sum[7]) begin 
            final_mant = final_sum[6:0]; // no need for shifting because the exponent completely aligns
            final_expo = expo_shared;
        end
    
        else if (final_sum[6]) begin 
            final_mant = {final_sum [5:0], 1'b0}; // since the exponent is decreased by 1 bit, shift the mantissa 1 bit
            final_expo = expo_shared - 'd1; // does not perfectly align so decrease the exponent by 1
        end
        
        else if(final_sum[5]) begin 
            final_mant = {final_sum[4:0], 2'b0}; // same thing continues as the previous else if but incerasing by 1 by 1
            final_expo = expo_shared - 'd2;
        end
        
        else if(final_sum[4]) begin 
            final_mant = {final_sum[3:0], 3'b0};
            final_expo = expo_shared - 'd3;
        end
        
        else if (final_sum[3]) begin 
            final_mant = {final_sum[2:0], 4'b0};
            final_expo = expo_shared - 'd4;
        end
        
        else begin // if the number is too small then ignore it
            final_mant = 7'b0; 
            final_expo = 8'b0;
        end
    
    end
  
  assign result_sum_o = {final_sign, final_expo, final_mant};  
    
    
    
endmodule
