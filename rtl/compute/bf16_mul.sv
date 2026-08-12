`timescale 1ns / 1ps

// FUNC_BF16_MUL : Multiplies two bf16 numbers through a 2 stage pipeline
//
//   Purpose:  Computes a_in times b_in in bfloat16 (sign, 8 bit exponent,
//             7 bit mantissa). Signs are xored, exponents are added and
//             re biased, mantissas are multiplied, and the product is
//             renormalized by one bit if the multiply carried into the
//             top of the mantissa. Exponent overflow saturates to
//             infinity. Every value on the bus is 17 bits wide, bit 16 is
//             a source tag that rides through the pipeline unchanged and
//             plays no part in the arithmetic, it only lets whoever reads
//             result_mul_o tell which side of a shared resource produced
//             the value.
//
//   parameters:
//           none
//
//   inputs:
//           clk_i:          clock
//           rst_ni:         active low reset
//           valid_i:        a_in and b_in are valid this cycle, starts a
//                           new multiply
//           a_in:           first operand, bit 16 is the source tag, bits
//                           15 down to 0 are the bf16 value
//           b_in:           second operand, same layout as a_in
//   output:
//           result_mul_o:   a_in times b_in, same 17 bit layout, valid
//                           two cycles after valid_i
//           valid_o:        result_mul_o is valid this cycle
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

module bf16_mul(
    input logic clk_i,
    input logic rst_ni,
    input logic valid_i,

    input logic [16:0] a_in,
    input logic [16:0] b_in,

    output logic [16:0] result_mul_o,
    output logic valid_o
    );
    
    logic sign_a, sign_b; // sign bits   
    logic [7:0] expo_a, expo_b;// exponents 
    logic [7:0] mant_a, mant_b;// mantissa
    logic final_sign;
    logic [15:0] final_mant;
    logic signed [9:0] exponent_temp;
    logic zero_flag;
    logic fsm_tag;

    assign sign_a = a_in[15];
    assign sign_b = b_in[15];

    assign expo_a = a_in [14:7];
    assign expo_b = b_in [14:7];

    assign mant_a = (|expo_a)? {1'b1, a_in[6:0]}: 8'b0;
    assign mant_b = (|expo_b)? {1'b1, b_in[6:0]}: 8'b0;
    
    assign exponent_temp = $signed({2'b00, expo_a}) + $signed({2'b00, expo_b}) - 10'sd127 + (final_mant[15] ? 10'sd1 : 10'sd0);
    
    assign final_sign = sign_a ^ sign_b;
    assign final_mant = mant_a * mant_b;

    assign zero_flag = (expo_a == 8'b0) || (expo_b == 8'b0);
    assign fsm_tag = a_in[16];
    
    // First stage registers
    logic s1_final_sign;
    logic [15:0] s1_final_mant;
    logic signed [9:0] s1_exponent_temp;
    logic s1_zero_flag;
    logic s1_tag;
    logic s1_valid;
    logic [16:0] s1_result;

    logic [16:0] s2_result;
    logic s2_valid;

    always_comb begin 
        s1_result = {s1_tag, 16'b0};

        if (s1_zero_flag) begin
            s1_result = {s1_tag,16'b0};
         end

        else if (s1_exponent_temp <= 10'sd0) begin 
            s1_result = {s1_tag,16'b0};    
        end

        else if (s1_exponent_temp >= 10'sd255) begin 
             s1_result = {s1_tag, s1_final_sign, 8'hFF, 7'b0};    
        end
       
        else if (s1_final_mant[15]) begin // if there is a  normalization bit increase the exponent by 1 and take the highest possible bit
            s1_result = {s1_tag, s1_final_sign, s1_exponent_temp[7:0], s1_final_mant[14:8]};
        end
        
        else begin // since there is no normalization bit exponent stays the same, but the mantissa's upper index drops by one because it is not possible that there is a 1 at the 15th bit without a normalization bit
            s1_result = {s1_tag, s1_final_sign, s1_exponent_temp[7:0], s1_final_mant[13:7]};       
        end
      
    end

    always_ff @(posedge clk_i or negedge rst_ni) begin 
        if(!rst_ni) begin 
            s1_final_sign <= 1'b0;
            s1_final_mant <= 16'b0;
            s1_exponent_temp <= 10'sd0;
            s1_zero_flag <= 1'b0;
            s1_tag <= 1'b0;
            s1_valid <= 1'b0;

            result_mul_o <= 17'b0;
            valid_o <= 1'b0;

            s2_result <= 17'b0;
            s2_valid <= 1'b0;
        end

        else begin 
           valid_o <= s2_valid;

           if(s2_valid) begin
                result_mul_o <= s2_result;
           end

           s2_valid <= s1_valid;

           if(s1_valid) begin 
                s2_result <= s1_result;
           end

           s1_valid <= valid_i;

           if(valid_i) begin 
                s1_final_sign <= final_sign;
                s1_final_mant <= final_mant;
                s1_exponent_temp <= exponent_temp;
                s1_zero_flag <= zero_flag;
                s1_tag <= fsm_tag; 
           end
        end        
    end    
    
endmodule
