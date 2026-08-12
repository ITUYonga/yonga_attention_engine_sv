`timescale 1ns / 1ps

// FUNC_BF16_CONVERT : Widens a bf16 value back up to fp32
//
//   Purpose:  bf16 and fp32 share the same sign and exponent layout, bf16
//             is simply fp32 with the bottom 16 mantissa bits missing, so
//             going the other way is just placing the bf16 word in the
//             upper 16 bits of an fp32 word and filling the lower 16
//             mantissa bits with zero. Purely combinational, no clock
//             needed.
//
//   parameters:
//           none
//
//   inputs:
//           bf16_in:    16 bit bf16 value
//   output:
//           fp32_out:   32 bit IEEE-754 single precision value, exact
//                       widening of bf16_in
//
//   notes:
//           this is an exact, lossless widening, every bf16 value has a
//           unique fp32 representation this way, it is only the other
//           direction (bf16_comb.sv) that loses precision

module bf16_convert(
    input logic [15:0] bf16_in,
    output logic [31:0] fp32_out

    );
    
    assign fp32_out = {bf16_in, 16'b0};
endmodule
