`timescale 1ns / 1ps

// FUNC_RECIP_LUT : Looks up 1 divided by x for one bf16 value
//
//   Purpose:  softmax needs the reciprocal of a row's summed exponentials
//             exactly once per row, so a small ROM plus a bit of exponent
//             arithmetic is enough, no need for a real divider. The
//             exponent is transformed directly (254 minus the input
//             exponent, or 253 minus it when the mantissa is not an exact
//             power of two, since a non power of two reciprocal needs one
//             extra renormalizing decrement), and the mantissa comes
//             straight out of the table.
//
//   parameters:
//           HEX_FILE_PATH:   path to the hex file $readmemh loads the
//                            128 entry mantissa table from
//
//   inputs:
//           clk:      clock
//           x_in:     one bf16 value, must be nonzero and finite
//   output:
//           y_out:    1 divided by x_in in bf16, purely combinational
//
//   notes:
//           an exponent of all zero bits (zero or subnormal input) is
//           treated as divide by zero and returns infinity rather than
//           indexing the table
//
//           found and fixed a real bug in the exponent transform, it used
//           to always be 254 minus the input exponent, correct only when
//           the input's mantissa was exactly zero (an exact power of
//           two). For every other input, 1 over 1.mantissa lands between
//           0.5 and 1, which needs one more exponent decrement once
//           renormalized back into the 1 to 2 range, and the table's own
//           mantissa values already assumed that decrement had happened.
//           Every non power of two reciprocal used to come out exactly
//           double what it should have been
//
//           this must stay purely combinational for the same reason as
//           exp_lut.sv, softmax.sv latches inv_sum_reg from this output
//           on the very cycle sum_acc first holds its final value, a
//           registered y_out here would make the reciprocal lag by a
//           cycle and capture the previous, incomplete sum instead

module recip_lut #(
    parameter string HEX_FILE_PATH = "rtl/softmax/recip_table.hex"
)(
    input  logic        clk,
    input  logic [15:0] x_in,   // 16-bit bf16 sum_acc
    output logic [15:0] y_out   // 16-bit bf16 reciprocal 1/sum_acc (1-cycle latency)
);


    logic       sign_in;
    logic [7:0] exp_in;
    logic [6:0] mant_in;

    assign sign_in = x_in[15];
    assign exp_in  = x_in[14:7];
    assign mant_in = x_in[6:0];

    // Exponent transform: E_out = 254 - E_in, correct only when the input
    // mantissa is exactly 0 (exact power of two, 1/(1.mantissa) = 1.0
    // exactly). For every other input, 1/(1.mantissa) lands in (0.5, 1),
    // which needs one extra exponent decrement once renormalized back into
    // [1, 2), matching the extra decrement already baked into the ROM's
    // mantissa values.
    logic [7:0] exp_out;
    assign exp_out = (mant_in == 7'b0) ? (8'd254 - exp_in) : (8'd253 - exp_in);

    // Mantissa ROM lookup
    logic [6:0] mant_rom [0:127];

    initial begin
        $readmemh(HEX_FILE_PATH, mant_rom);
    end

    logic [6:0] mant_out;
    assign mant_out = mant_rom[mant_in];

    // Reconstruct BF16 output
    logic [15:0] recip_comb;

    always_comb begin
        if (exp_in == 8'h00) begin
            // Divide-by-Zero or Subnormal -> Return Infinity
            recip_comb = {sign_in, 8'hFF, 7'b0}; 
        end else begin
            // Concatenate: {sign, new_exponent, inverted_mantissa}
            recip_comb = {sign_in, exp_out, mant_out};
        end
    end

    // Purely combinational: softmax.sv latches inv_sum_reg <= recip_out on
    // the very cycle sum_acc first holds its final row total (the INVERT
    // state), assuming both are available together. A registered y_out
    // here used to make recip_out lag x_in by 1 cycle, so inv_sum_reg
    // captured the reciprocal of the PREVIOUS (incomplete) sum_acc instead
    // of the final one -- the same latency-vs-consumer mismatch already
    // found and fixed in exp_lut.sv.
    assign y_out = recip_comb;

endmodule