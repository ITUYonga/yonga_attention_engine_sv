`timescale 1ns / 1ps

// FUNC_BF16_ADD : Adds two bf16 numbers through a 2 stage pipeline
//
//   Purpose:  Computes a_in plus b_in in bfloat16 (sign, 8 bit exponent,
//             7 bit mantissa). The smaller operand's mantissa is shifted
//             to align with the larger operand's exponent, the signed
//             mantissas are added, then the sum is renormalized and the
//             exponent re biased for the result. Subnormal results are
//             flushed to zero and exponent overflow saturates to
//             infinity. Every value on the bus is 17 bits wide, bit 16 is
//             a source tag that rides through the pipeline unchanged and
//             plays no part in the arithmetic, it only lets whoever reads
//             result_sum_o tell which side of a shared resource produced
//             the value.
//
//   parameters:
//           none
//
//   inputs:
//           clk_i:          clock
//           rst_ni:         active low reset
//           valid_i:        a_in and b_in are valid this cycle, starts a
//                           new add
//           a_in:           first operand, bit 16 is the source tag, bits
//                           15 down to 0 are the bf16 value
//           b_in:           second operand, same layout as a_in
//   output:
//           result_sum_o:   a_in plus b_in, same 17 bit layout, valid two
//                           cycles after valid_i
//           valid_o:        result_sum_o is valid this cycle
//
//   notes:
//           the tag carried into the result always comes from a_in's tag
//           bit, b_in's tag bit is never read, so every caller is
//           expected to place matching tags on both operands
//
//           this is a fixed two cycle pipeline with no backpressure of
//           its own, once valid_i pulses a result is guaranteed exactly
//           two cycles later whether or not anything downstream is ready
//           to accept it

module bf16_add(
    input logic clk_i,
    input logic rst_ni,
    input logic valid_i,

    input logic [16:0] a_in, // added 1-bit a source tag
    input logic [16:0] b_in,

    output logic [16:0] result_sum_o,
    output logic valid_o
   );
    
    logic sign_a; // sign bits
    logic sign_b;
    
    logic [7:0] expo_a;// exponents
    logic [7:0] expo_b;
    
    logic [7:0] mant_a;// mantissa
    logic [7:0] mant_b;
    
    assign sign_a = a_in[15];
    assign sign_b = b_in[15];

    assign expo_a = a_in [14:7];
    assign expo_b = b_in [14:7];

    assign mant_a = (|expo_a)? {1'b1, a_in[6:0]}: 8'b0;
    assign mant_b = (|expo_b)? {1'b1, b_in[6:0]}: 8'b0;
    // if there is no exponent mantissa is zero, if there is exponent bits then we add the secret 1 in front of the mantissa bits 
    
    logic [7:0] expo_dif;
    logic [7:0] expo_shared;
    logic [7:0] lined_mant_a;
    logic [7:0] lined_mant_b;
    logic signed [9:0] signed_a;
    logic signed [9:0] signed_b;  
    
    always_comb begin 
        expo_dif = 8'b0;
        expo_shared = 8'b0;
        lined_mant_a = 8'b0;
        lined_mant_b = 8'b0;

        if(expo_a > expo_b) begin // if the exponent bit is higher on one number we have to align the bits according to that.
            expo_dif = expo_a - expo_b;
            expo_shared = expo_a; // shared one should be the bigger one                lined_mant_a = mant_a; // and the mantissa of the bigger one does not change we align the smaller one according to that.
            lined_mant_a = mant_a;
            lined_mant_b = (expo_dif >= 8'd8)? 8'b0: (mant_b >> expo_dif); // if the difference is greater than eight bits which is the total bit of the mantissa then all of the bits should be zero, if not then we shift the mantissa to right according to the difference of the exponent
        end 
            
        else begin 
            expo_dif = expo_b - expo_a;
            expo_shared = expo_b;
            lined_mant_b = mant_b;
            lined_mant_a = (expo_dif >= 8'd8)? 8'b0: (mant_a >> expo_dif);
         end

    end
    
    assign signed_a = sign_a ? -$signed({2'b00, lined_mant_a}) : $signed({2'b00, lined_mant_a});// if the sign is negative then we put a minus in front of the nımber if not it stays the same 
    assign signed_b = sign_b ? -$signed({2'b00, lined_mant_b}) : $signed({2'b00, lined_mant_b});// the reason for the two bit gap is that 1 bit is for the carry and the other bit is for the sign (negative is 1, positive is 0)
   
    logic signed [9:0] s1_signed_a;
    logic signed [9:0] s1_signed_b;
    logic [7:0] s1_expo_shared;
    logic s1_tag;
    logic s1_valid;
    
    logic signed [9:0] sum;
    logic [9:0] abs_sum;
    logic final_sign;
    
    always_comb begin 
        sum = s1_signed_a + s1_signed_b;
        final_sign = sum[9]; // the final sign of the addition is the 10th bit of the sum

        if(final_sign) begin 
            abs_sum = $unsigned(-sum);
        end  

        else begin 
            abs_sum = $unsigned(sum);
        end
    end
    
    logic [8:0] s2_final_sum;
    logic s2_final_sign;
    logic [7:0] s2_expo_shared;
    logic s2_tag;
    logic s2_valid;

    logic [3:0] norm_shift;
    logic [6:0] normalized_mant;
    logic signed [9:0] normalized_expo;
    logic [16:0] result;

    always_comb begin
        norm_shift     = 4'd0;
        normalized_mant = 7'b0;
        normalized_expo = 10'sd0;
        result = {s2_tag, 16'b0};

        if (s2_final_sum == 9'b0) begin
            result = {s2_tag, 16'b0};
        end

        else if (s2_final_sum[8]) begin
            // Mantissa addition overflow: shift right and increment exponent.
            if (s2_expo_shared >= 8'd254) begin
                // Exponent overflow: infinity.
                result = {s2_tag, s2_final_sign, 8'hFF, 7'b0}; 
            end
            else begin
                result = {s2_tag, s2_final_sign, s2_expo_shared + 8'd1, s2_final_sum[7:1]};
            end
        end

        else begin
            if (s2_final_sum[7]) begin
                norm_shift      = 4'd0;
                normalized_mant = s2_final_sum[6:0];
            end
            else if (s2_final_sum[6]) begin
                norm_shift      = 4'd1;
                normalized_mant = {s2_final_sum[5:0], 1'b0};
            end
            else if (s2_final_sum[5]) begin
                norm_shift      = 4'd2;
                normalized_mant = {s2_final_sum[4:0], 2'b0};
            end
            else if (s2_final_sum[4]) begin
                norm_shift      = 4'd3;
                normalized_mant = {s2_final_sum[3:0], 3'b0};
            end
            else if (s2_final_sum[3]) begin
                norm_shift      = 4'd4;
                normalized_mant = {s2_final_sum[2:0], 4'b0};
            end
            else if (s2_final_sum[2]) begin
                norm_shift      = 4'd5;
                normalized_mant = {s2_final_sum[1:0], 5'b0};
            end
            else if (s2_final_sum[1]) begin
                norm_shift      = 4'd6;
                normalized_mant = {s2_final_sum[0], 6'b0};
            end
            else begin
                // final_sum[0] is the leading one.
                norm_shift      = 4'd7;
                normalized_mant = 7'b0;
            end

            normalized_expo =
                $signed({2'b00, s2_expo_shared}) -$signed({6'b0, norm_shift});

            // Subnormal outputs are flushed to zero.
            if (normalized_expo <= 10'sd0) begin
                result = {s2_tag, 16'b0};
            end
            else begin
                result = {s2_tag, s2_final_sign, normalized_expo[7:0], normalized_mant };
            end
        end
    end
  
  always_ff @(posedge clk_i or negedge rst_ni) begin 
    if(!rst_ni) begin 
        s1_signed_a <= 10'sd0;
        s1_signed_b <= 10'sd0;
        s1_expo_shared <= 8'b0;
        s1_tag <= 1'b0;
        s1_valid <= 1'b0;

        s2_final_sum <= 9'b0;
        s2_final_sign <= 1'b0;
        s2_expo_shared <= 8'b0;
        s2_tag <= 1'b0;
        s2_valid <= 1'b0;

        result_sum_o <= 17'b0;
        valid_o <= 1'b0;
    end
    else begin 
        valid_o <= s2_valid;

        if(s2_valid) begin 
            result_sum_o <= result;
        end

        s2_valid <= s1_valid;

        if(s1_valid) begin 
            s2_final_sum <= abs_sum[8:0];
            s2_final_sign <= final_sign;
            s2_expo_shared <= s1_expo_shared;
            s2_tag <= s1_tag;
        end

        s1_valid <= valid_i;

        if(valid_i) begin 
            s1_signed_a <= signed_a;
            s1_signed_b <= signed_b;
            s1_expo_shared <= expo_shared;
            s1_tag <= a_in[16]; 
        end
    end
  end 
    
    
endmodule
