`timescale 1ns / 1ps

// FUNC_BF16_COMB : Converts an fp32 value down to bf16
//
//   Purpose:  bfloat16 shares fp32's sign and exponent width, it only
//             drops the bottom 16 bits of the mantissa, so the
//             conversion is just keeping the upper 16 bits of the fp32
//             word. sel_i chooses between plain truncation and a round
//             to nearest using the fp32 word's own top dropped bit,
//             which is cheap because it only needs one add instead of a
//             real rounding unit. Purely combinational, no clock needed.
//
//   parameters:
//           none
//
//   inputs:
//           fp32_in:   32 bit IEEE-754 single precision value
//           sel_i:     0 selects truncation, 1 selects round to nearest
//   output:
//           bf16_out:  16 bit bf16 value
//
//   notes:
//           the round to nearest here only looks at the single bit right
//           below the bf16 mantissa (fp32_in[15]), it is not a full
//           round to nearest even, ties always round up

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
